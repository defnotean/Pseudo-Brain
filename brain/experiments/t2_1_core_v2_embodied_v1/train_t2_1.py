"""T2-1: Train Core V2 (CONFIG_A / B / C) on the shared TRAIN corpus.

Preregistration: brain/docs/preregistrations/2026-08-31-t2-1-core-v2-embodied-v1.md

Models:
  CONFIG_A: Core V2 (W120/K32/C3) consequence_learning=False — recurrent single-head baseline
  CONFIG_B: Core V2 (W120/K32/C3) consequence_learning=True, world_model=False
  CONFIG_C: Core V2 (W120/K32/C3) consequence_learning=True, world_model=True

All trained with deployed_decision_loss (5-class CE on teacher's applied action)
on the shared TRAIN corpus (embodied-corpus-v1, sha 59c41b6dc71f…).

Uses stable dynamics (normalized_mixture_v1 braincell, convex_gated_v1 belief)
and truncated BPTT over random subsequences for efficiency.

Usage (from brain/):
  py -3.11 experiments/t2_1_core_v2_embodied_v1/train_t2_1.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import zlib
from hashlib import sha256
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

_HERE = Path(__file__).resolve()
BRAIN_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(BRAIN_ROOT / "src"))
sys.path.insert(0, str(BRAIN_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from run_provenance import apply_deterministic_mode, provenance  # noqa: E402

apply_deterministic_mode()

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
try:
    torch.set_num_threads(1)
except RuntimeError:
    pass

try:
    from ctypes import windll  # type: ignore
    windll.kernel32.SetPriorityClass(
        windll.kernel32.GetCurrentProcess(), 0x00004000
    )
except Exception:
    pass

from irene_brain.v2.core import CoreV2Model  # noqa: E402
from irene_brain.v2.config import CoreV2Config, FeatureFlags  # noqa: E402
from irene_brain.v2.losses import deployed_decision_loss  # noqa: E402
from irene_brain.v2.state import CoreV2State  # noqa: E402

# ── Frozen constants (preregistration §3) ─────────────────────────────────

TOTAL_STEPS = 6000
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_MAX_NORM = 1.0
SEEDS = (42, 142, 242, 342)
CORPUS_RELPATH = BRAIN_ROOT / "datasets" / "embodied-corpus-v1"
CORPUS_SHA256_PREFIX = "59c41b6dc71f"
ACTION_CLASSES = 5
BURN_IN = 16
SEQ_LEN = 64  # subsequence length for truncated BPTT

RESULT_DIR = BRAIN_ROOT / "runs" / "t2-1-core-v2-embodied-v1"

# ── Stable dynamics config (matches V2.1 runner) ─────────────────────────

STABLE_CORE_CONFIG = CoreV2Config(
    decision_aggregation="direct_mean_logits_v1",
    braincell_dynamics="normalized_mixture_v1",
    belief_dynamics="convex_gated_v1",
    hazard_parameterization="probability_sigmoid_v1",
    latent_comparison="cosine_distance_v1",
    reward_comparison="raw_mse_v0",
    outcome_action_conditioning="thought_only_v0",
    outcome_architecture="legacy_hypothesis_world_v0",
    hazard_outcome_path="shared_outcome_v0",
    reward_prediction="scalar_v0",
    prediction_error_fusion="latent_only_v0",
)

# ── Model configurations ──────────────────────────────────────────────────

CONFIG_A_FLAGS = FeatureFlags(
    prediction_error_feedback=False,
    episodic_memory=False,
    consequence_learning=False,
    world_model_learning=False,
    session_adaptation=False,
)

CONFIG_B_FLAGS = FeatureFlags(
    prediction_error_feedback=True,
    episodic_memory=False,
    consequence_learning=True,
    world_model_learning=False,
    session_adaptation=False,
)

CONFIG_C_FLAGS = FeatureFlags(
    prediction_error_feedback=True,
    episodic_memory=False,
    consequence_learning=True,
    world_model_learning=True,
    session_adaptation=False,
)

CONFIGS = {
    "config_a": CONFIG_A_FLAGS,
    "config_b": CONFIG_B_FLAGS,
    "config_c": CONFIG_C_FLAGS,
}


# ── Corpus loading ────────────────────────────────────────────────────────

def stable_seed(name: str) -> int:
    return int(zlib.crc32(name.encode()) & 0xFFFFFFFF)


def load_corpus() -> tuple[list[np.ndarray], list[np.ndarray]]:
    manifest = json.loads((CORPUS_RELPATH / "manifest.json").read_text())
    frames, actions = [], []
    for rec in manifest["episodes"]:
        frames.append(np.load(CORPUS_RELPATH / "frames" / f"{rec['name']}.npy"))
        actions.append(np.load(CORPUS_RELPATH / "actions" / f"{rec['name']}.npy"))
    return frames, actions


# ── Frame → tensor (upscale 16x16 → 32x32) ───────────────────────────────

def frames_to_tensor(
    frames_np: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Convert [T, 16, 16, 3] uint8 frames to [1, T, 3, 32, 32] float32 tensor."""
    tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float().div_(255.0)
    tensor = F.interpolate(tensor, size=(32, 32), mode="nearest")
    return tensor.unsqueeze(0).to(device=device)


def param_fingerprint(model: torch.nn.Module) -> str:
    h = sha256()
    for p in model.parameters():
        h.update(p.detach().to("cpu").numpy().tobytes())
    return h.hexdigest()


# ── Build subsequence sample list ─────────────────────────────────────────

def build_subsequence_indices(
    frames: list[np.ndarray],
    seq_len: int,
) -> list[tuple[int, int, int]]:
    """Return (episode, start_tick, end_tick) tuples for all valid subsequences."""
    indices: list[tuple[int, int, int]] = []
    for e, fr in enumerate(frames):
        T = fr.shape[0]
        if T < seq_len + 1:
            continue
        # Sample start ticks such that the subsequence fits in the episode
        for start in range(0, T - seq_len, seq_len // 2):
            indices.append((e, start, start + seq_len))
    return indices


# ── Training: truncated BPTT over random subsequences ─────────────────────

def train_seed(
    config_name: str,
    seed: int,
    frames: list[np.ndarray],
    actions: list[np.ndarray],
    sample_order: list[int],
) -> dict:
    """Train one model at one seed. Steps = optimizer steps (one per subsequence)."""
    device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))

    core_config = STABLE_CORE_CONFIG
    flags = CONFIGS[config_name]
    model = CoreV2Model(config=core_config, flags=flags)
    param_count = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()

    losses: list[float] = []
    step = 0
    t0 = time.perf_counter()
    s_idx = 0

    # Precompute subsequence indices
    subseq_indices = build_subsequence_indices(frames, SEQ_LEN)
    n_subseq = len(subseq_indices)
    print(f"  [{config_name}] {n_subseq} valid subsequences", flush=True)

    while step < TOTAL_STEPS:
        # Pick subsequence from shuffled order
        si = sample_order[s_idx % len(sample_order)]
        s_idx += 1
        e, start, end = subseq_indices[si]
        fr = frames[e]  # [T, 16, 16, 3]
        act = actions[e]  # [T, 2]

        # Convert subsequence frames to tensor
        sub_frames = fr[start:end]  # [SEQ_LEN, 16, 16, 3]
        obs_tensor = frames_to_tensor(sub_frames, device)  # [1, SEQ_LEN, 3, 32, 32]

        state = model.init_state(1, device)
        losses_t: list[torch.Tensor] = []

        for t in range(SEQ_LEN):
            frame_t = obs_tensor[:, t]  # [1, 3, 32, 32]
            abs_t = start + t
            prev_a = torch.tensor([int(act[abs_t, 0])], dtype=torch.long, device=device)

            output, state = model(frame_t, state, prev_action=prev_a)

            # Loss after burn-in (first BURN_IN ticks of subsequence)
            if t >= BURN_IN:
                target = torch.tensor([int(act[abs_t, 1])], dtype=torch.long, device=device)
                log_probs = F.log_softmax(output.decision.action_values, dim=-1)
                loss_t = F.nll_loss(log_probs, target)
                losses_t.append(loss_t)

        if losses_t:
            avg_loss = torch.stack(losses_t).mean()
            avg_loss.backward()
            gnorm = float(nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM).item())
            if not np.isfinite(gnorm):
                raise FloatingPointError(f"non-finite grad norm at step {step} seed {seed}")
            opt.step()
            opt.zero_grad()

            loss_val = float(avg_loss.item())
            losses.append(loss_val)
            step += 1

            if step % 500 == 0:
                print(
                    f"  [{config_name}] seed {seed} step {step}/{TOTAL_STEPS} "
                    f"loss={loss_val:.5f} ({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )

    ckpt_path = RESULT_DIR / config_name / f"seed_{seed}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config_name": config_name,
            "seed": seed,
            "param_sha256": param_fingerprint(model),
            "param_count": param_count,
        },
        ckpt_path,
    )

    return {
        "config": config_name,
        "seed": seed,
        "param_count": param_count,
        "param_sha256": param_fingerprint(model),
        "checkpoint": str(ckpt_path.relative_to(BRAIN_ROOT.parent)),
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "initial_loss": round(losses[0], 5) if losses else 0.0,
        "final_loss": round(losses[-1], 5) if losses else 0.0,
        "mean_first100": round(float(np.mean(losses[:100])), 5) if losses else 0.0,
        "mean_last100": round(float(np.mean(losses[-100:])), 5) if losses else 0.0,
        "loss_curve": [round(x, 6) for x in losses],
    }


# ── Corpus recall (gate U) ───────────────────────────────────────────────

def corpus_recall(
    config_name: str,
    seed: int,
    frames: list[np.ndarray],
    actions: list[np.ndarray],
) -> dict:
    device = torch.device("cpu")
    core_config = STABLE_CORE_CONFIG
    flags = CONFIGS[config_name]
    model = CoreV2Model(config=core_config, flags=flags)

    ckpt_path = RESULT_DIR / config_name / f"seed_{seed}.pt"
    sd = torch.load(ckpt_path, map_location=device)["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()

    correct = np.zeros(ACTION_CLASSES, dtype=np.int64)
    total = np.zeros(ACTION_CLASSES, dtype=np.int64)

    with torch.no_grad():
        for e, (fr, act) in enumerate(zip(frames, actions)):
            T = fr.shape[0]
            if T < BURN_IN + 2:
                continue
            obs_tensor = frames_to_tensor(fr, device)  # [1, T, 3, 32, 32]
            state = model.init_state(1, device)
            for t in range(1, T):
                frame_t = obs_tensor[:, t]
                prev_a = torch.tensor([int(act[t - 1, 0])], dtype=torch.long, device=device)
                output, state = model(frame_t, state, prev_action=prev_a)
                if t >= BURN_IN:
                    pred = output.decision.action_values.argmax(dim=-1).item()
                    label = int(act[t, 1])
                    total[label] += 1
                    correct[label] += int(pred == label)

    per_class = {
        str(c): {"n": int(total[c]), "recall": round(float(correct[c] / max(total[c], 1)), 6)}
        for c in range(ACTION_CLASSES)
    }
    overall = round(float(sum(correct) / max(sum(total), 1)), 6)
    return {"config": config_name, "seed": seed, "per_class": per_class, "overall": overall}


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> int:
    started = time.time()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    frames, actions = load_corpus()
    n_trans = sum(fr.shape[0] - 1 for fr in frames)
    print(f"corpus: {len(frames)} episodes, {n_trans} samples (t>=1)", flush=True)

    corpus_sha = provenance().get("corpus_manifest_sha256", "unknown")
    if not str(corpus_sha).startswith(CORPUS_SHA256_PREFIX):
        print(f"WARNING: corpus SHA prefix mismatch: {corpus_sha}", file=sys.stderr)

    all_records: list[dict] = []
    for config_name in CONFIGS:
        print(f"\n{'='*60}", flush=True)
        print(f"[train] {config_name} ...", flush=True)
        for seed in SEEDS:
            ckpt_path = RESULT_DIR / config_name / f"seed_{seed}.pt"
            if ckpt_path.exists():
                print(f"  seed {seed}: SKIP (checkpoint exists)", flush=True)
                continue
            print(f"  seed {seed} ...", flush=True)
            order_rng = torch.Generator()
            order_rng.manual_seed(stable_seed(f"t2_1.subseq_order.{config_name}.{seed}"))
            subseq_indices = build_subsequence_indices(frames, SEQ_LEN)
            sample_order = torch.randperm(len(subseq_indices), generator=order_rng).tolist()

            rec = train_seed(config_name, seed, frames, actions, sample_order)
            all_records.append(rec)
            print(f"  seed {seed}: final={rec['final_loss']:.5f} "
                  f"({rec['wall_seconds']:.0f}s)", flush=True)

    # Gate R: determinism — SKIP (verify separately after training completes)
    print(f"\n{'='*60}", flush=True)
    print("[gate R] skipped (verify separately)", flush=True)
    gate_r_results = {"note": "skipped — run gate_r_verify.py after training"}

    # Gate U: corpus recall — SKIP if --skip-recall flag present
    print(f"\n{'='*60}", flush=True)
    if "--skip-recall" in sys.argv:
        print("[gate U] skipped (--skip-recall)", flush=True)
        recall_records = []
    else:
        print("[gate U] corpus recall ...", flush=True)
        recall_records: list[dict] = []
        for config_name in CONFIGS:
            for seed in SEEDS:
                ckpt_path = RESULT_DIR / config_name / f"seed_{seed}.pt"
                if not ckpt_path.exists():
                    print(f"  {config_name} seed {seed}: SKIP (no checkpoint)", flush=True)
                    continue
                rec = corpus_recall(config_name, seed, frames, actions)
                recall_records.append(rec)
                print(f"  {config_name} seed {seed}: overall={rec['overall']:.4f}", flush=True)

    # Report
    report = {
        "experiment": "t2-1-core-v2-embodied-v1",
        "preregistration": "brain/docs/preregistrations/2026-08-31-t2-1-core-v2-embodied-v1.md",
        "corpus_sha256": corpus_sha,
        "total_steps": TOTAL_STEPS,
        "seq_len": SEQ_LEN,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "seeds": list(SEEDS),
        "burn_in": BURN_IN,
        "wall_seconds": round(time.time() - started, 1),
        "gate_r": gate_r_results,
        "training": all_records,
        "recall": recall_records,
    }

    report_path = RESULT_DIR / "2026-08-31-t2-1-core-v2-embodied-v1.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nreport: {report_path}", flush=True)
    print(f"wall time: {report['wall_seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
