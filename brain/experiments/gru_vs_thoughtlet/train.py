"""GRU vs Thoughtlet training — matched-state, matched-budget comparison.

Trains Model A (reactive), Model B (GRU-384), Model C (Thoughtlet-384x12)
with identical encoders, training budgets, and seeds.

Key: Uses sequence-based training so recurrent models actually use their
recurrence. Includes scheduled sampling to bridge train/eval distribution shift.
"""
from __future__ import annotations

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

BRAIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRAIN_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import make_model, count_parameters, N_FRAMES, ACTION_CLASSES


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to torch.device, supporting cuda, dml (AMD), and cpu."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            import torch_directml
            if torch_directml.device_count() > 0:
                return torch_directml.device()
        except ImportError:
            pass
        return torch.device("cpu")
    if device_str == "dml":
        import torch_directml
        return torch_directml.device()
    if device_str.startswith("dml:"):
        import torch_directml
        idx = int(device_str.split(":")[1])
        return torch_directml.device(idx)
    return torch.device(device_str)


# --- Corpus dataset: sequence-based ---

class CorpusSequenceDataset(Dataset):
    """Yields fixed-length sequences of consecutive timesteps from the same episode.

    Each sample: (frames, actions) where
      frames:  (T, N_FRAMES, 3, 16, 16) float32 [0,1]
      actions: (T,) int64 — the ground-truth action at each timestep
    """

    def __init__(self, corpus_dir: str, seq_len: int = 32, extra_dirs: list = None):
        self.corpus_dir = Path(corpus_dir)
        self.seq_len = seq_len
        manifest = json.loads((self.corpus_dir / "manifest.json").read_text())
        self.episode_names = [ep["name"] for ep in manifest["episodes"]]

        # Preload all episodes
        self.frames = []
        self.actions = []
        self.is_extra = []
        for name in self.episode_names:
            fr = np.load(self.corpus_dir / "frames" / f"{name}.npy")
            act = np.load(self.corpus_dir / "actions" / f"{name}.npy")
            self.frames.append(fr)
            self.actions.append(act)
            self.is_extra.append(False)
        # DAgger / extra episodes (same frames/actions layout + manifest schema)
        for d in extra_dirs or []:
            d = Path(d)
            manifest2 = json.loads((d / "manifest.json").read_text())
            for ep in manifest2["episodes"]:
                name = ep["name"]
                self.frames.append(np.load(d / "frames" / f"{name}.npy"))
                self.actions.append(np.load(d / "actions" / f"{name}.npy"))
                self.is_extra.append(True)

        # Build sample indices: (episode_idx, start_t)
        # Include t=0 with padded buffers so eval's initial repeated-frame buffer is in-distribution
        self.indices = []
        for e, fr in enumerate(self.frames):
            T = fr.shape[0]
            min_start = 0
            max_start = T - seq_len
            if max_start < 0:
                continue
            for t in range(min_start, max_start + 1):
                self.indices.append((e, t))

    @classmethod
    def extra_only(cls, extra_dirs: list, seq_len: int = 32):
        """Dataset containing only extra (DAgger) episodes, for balanced mixing."""
        obj = cls.__new__(cls)
        obj.corpus_dir = Path(extra_dirs[0])
        obj.seq_len = seq_len
        obj.frames, obj.actions, obj.is_extra = [], [], []
        obj.episode_names = []
        for d in extra_dirs:
            d = Path(d)
            manifest2 = json.loads((d / "manifest.json").read_text())
            for ep in manifest2["episodes"]:
                name = ep["name"]
                obj.frames.append(np.load(d / "frames" / f"{name}.npy"))
                obj.actions.append(np.load(d / "actions" / f"{name}.npy"))
                obj.is_extra.append(True)
                obj.episode_names.append(name)
        obj.indices = []
        for e, fr in enumerate(obj.frames):
            T = fr.shape[0]
            max_start = T - obj.seq_len
            if max_start < 0:
                continue
            for t in range(0, max_start + 1):
                obj.indices.append((e, t))
        return obj

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        e, t0 = self.indices[idx]
        fr = self.frames[e]   # (T, 16, 16, 3) uint8
        act = self.actions[e] # (T, 2) int8

        T = self.seq_len
        # Build frame windows: for each timestep t0..t0+T-1, stack N_FRAMES prior frames
        frames_win = []
        for dt in range(T):
            t = t0 + dt
            win = []
            for i in range(N_FRAMES):
                ti = max(t - (N_FRAMES - 1 - i), 0)
                win.append(fr[ti])
            frames_win.append(np.stack(win))  # (N_FRAMES, 16, 16, 3)

        # (T, N_FRAMES, 16, 16, 3) → (T, N_FRAMES, 3, 16, 16) float32
        frames = np.stack(frames_win).transpose(0, 1, 4, 2, 3).astype(np.float32) / 255.0
        # Ground-truth actions for the sequence
        actions = act[t0:t0 + T, 1].astype(np.int64)  # current action label

        return {
            "frames": torch.from_numpy(frames),      # (T, N_FRAMES, 3, 16, 16)
            "actions": torch.from_numpy(actions),     # (T,)
        }


def collate_sequences(batch):
    """Stack into (B, T, ...)."""
    return {
        "frames": torch.stack([b["frames"] for b in batch]),   # (B, T, N_FRAMES, 3, 16, 16)
        "actions": torch.stack([b["actions"] for b in batch]), # (B, T)
    }


# --- Sequence-aware evaluation ---

@torch.no_grad()
def evaluate(model, dataloader, device, max_batches: int = 100) -> Dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    n_batches = 0

    for batch in dataloader:
        if n_batches >= max_batches:
            break
        frames = batch["frames"].to(device)   # (B, T, ...)
        actions = batch["actions"].to(device) # (B, T)
        B, T = actions.shape

        is_reactive = model.__class__.__name__ == "ReactiveModel"

        h = None
        prev_actions = torch.zeros(B, dtype=torch.long, device=device)
        batch_loss = 0.0
        batch_correct = 0
        batch_tokens = 0

        for t in range(T):
            frame_t = frames[:, t]  # (B, N_FRAMES, 3, 16, 16)
            action_t = actions[:, t]

            if is_reactive:
                logits, _ = model(frame_t, prev_actions, None)
            elif model.__class__.__name__ == "GRUModel":
                if h is None and hasattr(model, "init_hidden"):
                    h = model.init_hidden(B, device)
                logits, h = model(frame_t, prev_actions, h)
            else:
                if h is None and hasattr(model, "init_thoughts"):
                    h = model.init_thoughts(B, device)
                logits, h = model(frame_t, prev_actions, h)

            loss = F.cross_entropy(logits, action_t)
            batch_loss += loss.item()
            batch_correct += (logits.argmax(dim=-1) == action_t).sum().item()
            batch_tokens += B
            prev_actions = action_t

        total_loss += batch_loss / T
        total_correct += batch_correct
        total_tokens += batch_tokens
        n_batches += 1

    model.train()
    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_correct / max(total_tokens, 1),
    }


# --- Training ---

def train_model(
    model_type: str,
    corpus_dir: str,
    n_steps: int = 6000,
    batch_size: int = 16,
    seq_len: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    seed: int = 42,
    save_dir: str = "runs/gru_vs_thoughtlet",
    eval_every: int = 500,
    device: str = "auto",
    scheduled_sampling: bool = True,
    extra_dirs: list = None,
    init_checkpoint: str = None,
    ckpt_name: str = None,
    extra_ratio: float = 0.0,
    model_kwargs: dict = None,
) -> Dict:
    # Maximize CPU utilization — respect TORCH_NUM_THREADS env for parallel jobs
    try:
        import os
        n_cores = int(os.environ.get("TORCH_NUM_THREADS", os.cpu_count() or 8))
        torch.set_num_threads(n_cores)
        torch.set_num_interop_threads(max(1, n_cores // 2))
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = get_device(device)
    print(f"  Device: {dev} | batch_size={batch_size} seq_len={seq_len} | threads={torch.get_num_threads()}")

    model = make_model(model_type, **(model_kwargs or {})).to(dev)
    if model_kwargs:
        print(f"  Model kwargs: {model_kwargs}")
    if init_checkpoint:
        ckpt = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  Init from checkpoint: {init_checkpoint}")
    param_info = count_parameters(model)
    print(f"  Model: {model_type} | Params: {param_info['total']:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    dataset = CorpusSequenceDataset(corpus_dir, seq_len=seq_len, extra_dirs=extra_dirs)
    n_extra = sum(dataset.is_extra)
    print(f"  Dataset: {len(dataset)} seqs ({n_extra} from extra dirs)")
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_sequences, num_workers=0,
    )
    # Balanced DAgger mixing: separate extra-only loader sampled with prob extra_ratio
    extra_loader_iter = None
    if extra_dirs and extra_ratio > 0:
        extra_ds = CorpusSequenceDataset.extra_only(extra_dirs, seq_len=seq_len)
        extra_loader = DataLoader(
            extra_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate_sequences, num_workers=0,
        )
        extra_loader_iter = iter(extra_loader)
        print(f"  DAgger mixing: extra_ratio={extra_ratio} ({len(extra_ds)} extra seqs)")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    stem = ckpt_name or f"seed_{seed}_{model_type}"
    checkpoint_path = save_path / f"{stem}.pt"
    log_path = save_path / f"{stem}_log.json"

    model.train()
    steps_log = []
    start_time = time.time()

    is_reactive = model_type == "reactive"
    use_scheduled = scheduled_sampling

    dataloader_iter = iter(dataloader)
    for step in range(1, n_steps + 1):
        # With prob extra_ratio, draw a pure-DAgger batch so expert corrections
        # are not drowned out by the 500k-sequence corpus
        if extra_loader_iter is not None and torch.rand(1).item() < extra_ratio:
            try:
                batch = next(extra_loader_iter)
            except StopIteration:
                extra_loader_iter = iter(extra_loader)
                batch = next(extra_loader_iter)
        else:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

        frames = batch["frames"].to(dev)   # (B, T, N_FRAMES, 3, 16, 16)
        actions = batch["actions"].to(dev) # (B, T)
        B, T = actions.shape

        # Scheduled sampling: ramp from 0 → 1 over first 30% of training
        if use_scheduled:
            ss_prob = min(1.0, step / (n_steps * 0.3))
        else:
            ss_prob = 0.0

        # Initialize recurrent state and previous action
        h = None
        prev_actions = torch.zeros(B, dtype=torch.long, device=dev)

        total_loss = torch.tensor(0.0, device=dev)
        for t in range(T):
            frame_t = frames[:, t]  # (B, N_FRAMES, 3, 16, 16)
            action_t = actions[:, t]

            # Duck-typed recurrent init: GRUModel -> init_hidden, ThoughtletModel
            # family -> init_thoughts, others (ablation variants) init zeros
            # internally when h=None.
            def _init_h():
                if hasattr(model, "init_hidden"):
                    return model.init_hidden(B, dev)
                if hasattr(model, "init_thoughts"):
                    return model.init_thoughts(B, dev)
                return None

            # Per-timestep scheduled sampling: decide independently for each t
            if use_scheduled and torch.rand(1).item() < ss_prob:
                # Use model's own prediction as prev_action (detached)
                with torch.no_grad():
                    if is_reactive:
                        pred_logits, _ = model(frame_t, prev_actions, None)
                    else:
                        h_detached = h.detach() if h is not None else _init_h()
                        pred_logits, _ = model(frame_t, prev_actions, h_detached)
                prev_actions = pred_logits.argmax(dim=-1)

            # Forward pass (with gradients)
            if is_reactive:
                logits, _ = model(frame_t, prev_actions, None)
            else:
                if h is None:
                    h = _init_h()
                logits, h = model(frame_t, prev_actions, h)

            loss = loss_fn(logits, action_t)
            total_loss = total_loss + loss

            # Use ground-truth action as next prev_action (teacher forcing)
            prev_actions = action_t

        avg_loss = total_loss / T
        optimizer.zero_grad()
        avg_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if step % 200 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step:5d}/{n_steps} | loss={avg_loss.item():.4f} | {elapsed:.1f}s")

        if step % eval_every == 0 or step == n_steps:
            eval_metrics = evaluate(model, dataloader, dev, max_batches=100)
            steps_log.append({
                "step": step,
                "train_loss": avg_loss.item(),
                "eval_loss": eval_metrics["loss"],
                "eval_acc": eval_metrics["accuracy"],
                "ss_prob": ss_prob,
            })
            print(f"    eval_loss={eval_metrics['loss']:.4f} | eval_acc={eval_metrics['accuracy']:.3f}")

    total_time = time.time() - start_time
    torch.save({
        "model_type": model_type,
        "seed": seed,
        "state_dict": model.state_dict(),
        "param_info": param_info,
    }, checkpoint_path)

    final_metrics = {
        "model_type": model_type,
        "seed": seed,
        "param_info": param_info,
        "n_steps": n_steps,
        "total_time_s": total_time,
        "final_eval_loss": steps_log[-1]["eval_loss"] if steps_log else float("nan"),
        "final_eval_acc": steps_log[-1]["eval_acc"] if steps_log else float("nan"),
        "steps": steps_log,
    }
    with open(log_path, "w") as f:
        json.dump(final_metrics, f, indent=2)

    print(f"  DONE | eval_loss={final_metrics['final_eval_loss']:.4f} | eval_acc={final_metrics['final_eval_acc']:.3f} | {total_time:.1f}s")
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="GRU vs Thoughtlet training")
    parser.add_argument("--corpus", default=str(BRAIN_ROOT / "datasets" / "embodied-corpus-v1"))
    parser.add_argument("--models", nargs="+", default=["reactive", "gru", "thoughtlet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 142, 242, 342])
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--device", default="auto", help="cpu, cuda, dml, or auto")
    parser.add_argument("--cycles", type=int, default=None, help="thoughtlet thinking cycles (default per model)")
    parser.add_argument("--K", type=int, default=None, help="thoughtlet slot count")
    parser.add_argument("--thought-size", type=int, default=None, help="thoughtlet slot dim")
    parser.add_argument("--extra-dirs", nargs="*", default=None, help="extra corpus-compat data dirs (DAgger)")
    parser.add_argument("--extra-ratio", type=float, default=0.0, help="prob of pure-extra batch per step")
    parser.add_argument("--init-checkpoint", default=None, help="warm-start weights (fine-tune)")
    parser.add_argument("--ckpt-name", default=None, help="override checkpoint filename")
    parser.add_argument("--save-dir", default=str(BRAIN_ROOT / "runs" / "gru_vs_thoughtlet"))
    args = parser.parse_args()

    all_results = {}
    start = time.time()

    for model_type in args.models:
        for seed in args.seeds:
            print(f"\n=== Training {model_type} seed={seed} ===")
            model_kwargs = {}
            if args.cycles is not None:
                model_kwargs["cycles"] = args.cycles
            if args.K is not None:
                model_kwargs["K"] = args.K
            if args.thought_size is not None:
                model_kwargs["thought_size"] = args.thought_size
            result = train_model(
                model_type=model_type,
                corpus_dir=args.corpus,
                model_kwargs=model_kwargs,
                n_steps=args.steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                lr=args.lr,
                seed=seed,
                save_dir=args.save_dir,
                eval_every=args.eval_every,
                device=args.device,
                extra_dirs=args.extra_dirs,
                init_checkpoint=args.init_checkpoint,
                ckpt_name=args.ckpt_name,
                extra_ratio=args.extra_ratio,
            )
            all_results[f"{model_type}_seed{seed}"] = {
                "final_eval_loss": result["final_eval_loss"],
                "final_eval_acc": result["final_eval_acc"],
                "param_info": result["param_info"],
                "total_time_s": result["total_time_s"],
            }

    total_time = time.time() - start
    print(f"\n=== ALL DONE | {total_time:.1f}s ===")
    print(f"{'Model':12s} {'Seed':>5s} {'Eval Loss':>10s} {'Eval Acc':>10s} {'Params':>8s}")
    print("-" * 55)
    for key, val in all_results.items():
        model_type, seed_str = key.rsplit("_seed", 1)
        print(f"{model_type:12s} {seed_str:>5s} {val['final_eval_loss']:10.4f} {val['final_eval_acc']:10.3f} {val['param_info']['total']:>8d}")

    summary_path = Path(args.save_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
