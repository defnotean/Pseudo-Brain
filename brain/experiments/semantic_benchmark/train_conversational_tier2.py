"""Tier 2 DirectML Conversational Training & Checkpointing Engine.

Trains a Tier 2 (~10M-51M parameter) Pseudo-Brain cognitive core on AMD Radeon RX 9070 XT
using real-world Hugging Face conversational streams (UltraChat + Alpaca) transformed
into multi-threaded cognitive streams with synthetic preemption.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.device import (
    get_directml_device,
    get_directml_device_name,
    get_best_directml_device_index,
    is_directml_available,
)
from irene_brain.semantic.native_semantic_model import (
    NativeSemanticPseudoBrain,
    make_semantic_model,
)
from irene_brain.semantic.streaming_engine import StreamingCognitiveSession
from irene_brain.semantic.tokenizer import SemanticTokenizer
from semantic_benchmark.hf_dataset_loader import HFConversationalLoader
from semantic_benchmark.llm_data_transform import LLMDataTransformer, RawDialogue

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_tier2")


def build_and_cache_hf_corpus(
    loader: HFConversationalLoader,
    transformer: LLMDataTransformer,
    num_ultrachat: int = 25,
    num_alpaca: int = 25,
    num_episodes: int = 40,
    output_path: Optional[Path] = None,
) -> Path:
    """Download HF dialogues and transform them into multi-threaded cognitive episodes."""
    corpus_file = output_path or (_REPO_ROOT / "data" / "transformed_hf_conversational_corpus.jsonl")
    corpus_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching real conversational datasets from Hugging Face (UltraChat + Alpaca)...")
    raw_dicts = loader.fetch_combined_corpus(num_ultrachat=num_ultrachat, num_alpaca=num_alpaca, use_cache=True)
    raw_dialogues = transformer.parse_standard_dialogue_json(raw_dicts)
    logger.info(f"Successfully loaded and parsed {len(raw_dialogues)} valid multi-turn dialogues.")

    logger.info(f"Transforming into {num_episodes} multi-threaded cognitive streams with synthetic preemption...")
    episodes = transformer.transform_corpus(
        raw_dialogues=raw_dialogues,
        num_episodes=num_episodes,
        threads_per_episode=4,
        seed=42,
        paradigm="model_d",
    )

    # Extract individual focused turns from raw dialogues for strong conversational grounding
    focused_turns: List[Dict[str, Any]] = []
    for d in raw_dialogues:
        for turn_idx in range(len(d.turns) - 1):
            u = d.turns[turn_idx]
            a = d.turns[turn_idx + 1]
            if u.role == "user" and a.role == "assistant":
                u_text = u.content[:100].strip()
                a_text = a.content[:100].strip()
                turn_str = f"[THREAD:0]{u_text} [RESP]{a_text}[EOS]"
                toks = transformer.tokenizer.encode(turn_str)
                resp_id = transformer.tokenizer.resp_id
                targets = [-100] * len(toks)
                if resp_id in toks:
                    r_idx = toks.index(resp_id)
                    for ti in range(r_idx, len(toks) - 1):
                        targets[ti] = toks[ti + 1]
                focused_turns.append({
                    "episode_id": f"turn_{d.dialogue_id}_{turn_idx}",
                    "text": turn_str,
                    "tokens": toks,
                    "targets": targets,
                    "threads": [0] * len(toks),
                    "num_threads": 1,
                    "has_preemption": False,
                    "has_dependency": False,
                })

    # High-priority conversational templates for factual and dialogue recall
    templates = [
        ("Hello! What is your name?", "I am Pseudo-Brain, an autonomous recurrent cognitive agent."),
        ("Remember that project Alpha is due on Friday.", "I have recorded that project Alpha is due on Friday."),
        ("When is project Alpha due?", "Project Alpha is due on Friday."),
        ("How does photosynthesis work?", "Photosynthesis converts sunlight, water, and carbon dioxide into oxygen and energy."),
        ("Continue discussing plants.", "Plants use green chlorophyll pigments in their leaves to capture solar energy."),
        ("What can you do?", "I can reason across concurrent cognitive threads, retain long-term facts, and plan lookahead paths."),
        ("Remember Alice likes coffee.", "Noted: Alice prefers coffee."),
        ("What does Alice like?", "Alice likes coffee."),
        ("Remember Bob likes tea.", "Noted: Bob prefers tea."),
        ("What does Bob like?", "Bob likes tea."),
    ]
    for idx, (q, ans) in enumerate(templates):
        for tid in [0, 1]:
            turn_str = f"[THREAD:{tid}]{q} [RESP]{ans}[EOS]"
            toks = transformer.tokenizer.encode(turn_str)
            resp_id = transformer.tokenizer.resp_id
            targets = [-100] * len(toks)
            if resp_id in toks:
                r_idx = toks.index(resp_id)
                for ti in range(r_idx, len(toks) - 1):
                    targets[ti] = toks[ti + 1]
            focused_turns.append({
                "episode_id": f"template_{idx}_t{tid}",
                "text": turn_str,
                "tokens": toks,
                "targets": targets,
                "threads": [tid] * len(toks),
                "num_threads": 2,
                "has_preemption": False,
                "has_dependency": True,
            })

    with open(corpus_file, "w", encoding="utf-8") as f:
        for ep in episodes:
            item = {
                "episode_id": ep.episode_id,
                "text": ep.interleaved_text,
                "tokens": ep.tokens,
                "targets": ep.targets,
                "threads": ep.threads,
                "num_threads": ep.num_threads,
                "has_preemption": ep.has_preemption,
                "has_dependency": ep.has_dependency,
            }
            f.write(json.dumps(item) + "\n")
        for ft in focused_turns:
            f.write(json.dumps(ft) + "\n")

    logger.info(f"Transformed corpus saved to {corpus_file} ({len(episodes)} episodes + {len(focused_turns)} focused turns).")
    return corpus_file


def train_tier2_directml(
    corpus_path: Path,
    num_steps: int = 60,
    batch_size: int = 2,
    lr: float = 1e-4,
    device_override: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    proj_dim: int = 2048,
    rank: int = 32,
    num_deep_layers: int = 2,
    threads: int = 16,
) -> Dict[str, Any]:
    """Execute Tier 2 model training on DirectML GPU or CPU fallback."""
    # 1. Device Resolution
    if device_override:
        device = torch.device(device_override)
        device_name = f"Custom ({device_override})"
    else:
        dml_dev = get_directml_device()
        if dml_dev is not None:
            device = dml_dev
            best_idx = get_best_directml_device_index()
            device_name = get_directml_device_name(best_idx)
        else:
            device = torch.device("cpu")
            device_name = "Host CPU (MKL)"

    logger.info(f"Selected Compute Hardware: {device_name} ({device})")

    # 2. Tokenizer & Model Initialization
    tokenizer = SemanticTokenizer(max_threads=threads)
    logger.info(f"Initializing Tier 2 Conversational Model (vocab_size={tokenizer.vocab_size}, K={threads}, proj_dim={proj_dim})...")
    
    model = make_semantic_model(
        model_type="pseudo_brain_tier2",
        vocab_size=tokenizer.vocab_size,
        K=threads,
        proj_dim=proj_dim,
        rank=rank,
        num_deep_layers=num_deep_layers,
        conditional_recurrence=True,
    ).to(device)

    counts = model.count_parameters()
    state_kb = model.state_bytes() / 1024.0
    logger.info(f"Tier 2 Model Architecture: {counts['total']:,} parameters | State Memory: {state_kb:.1f} KB")

    # 3. Load Dataset
    episodes: List[Dict[str, Any]] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    logger.info(f"Loaded {len(episodes)} episodes from {corpus_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-5)

    model.train()
    history: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    total_tokens_trained = 0

    logger.info(f"Beginning Tier 2 training loop ({num_steps} steps, batch_size={batch_size})...")

    for step in range(1, num_steps + 1):
        step_t0 = time.perf_counter()
        batch_indices = np.random.choice(len(episodes), size=batch_size, replace=True)
        batch_eps = [episodes[i] for i in batch_indices]

        # Pad tokens, targets, threads
        max_len = min(128, max(len(ep["tokens"]) for ep in batch_eps))
        b_toks = np.full((batch_size, max_len), tokenizer.pad_id, dtype=np.int64)
        b_targ = np.full((batch_size, max_len), -100, dtype=np.int64)
        b_thrd = np.zeros((batch_size, max_len), dtype=np.int64)

        for b_idx, ep in enumerate(batch_eps):
            ep_tokens = ep["tokens"]
            ep_targets = ep["targets"]
            ep_threads = ep["threads"]
            n_tokens = len(ep_tokens)
            if n_tokens <= max_len:
                start = 0
                end = n_tokens
            else:
                valid_targ_idxs = [i for i, t in enumerate(ep_targets) if t != -100]
                if valid_targ_idxs:
                    target_focus = int(np.random.choice(valid_targ_idxs))
                    start = max(0, min(target_focus - max_len // 2, n_tokens - max_len))
                else:
                    start = int(np.random.randint(0, n_tokens - max_len))
                end = start + max_len

            chunk_len = end - start
            b_toks[b_idx, :chunk_len] = ep_tokens[start:end]
            b_targ[b_idx, :chunk_len] = ep_targets[start:end]
            b_thrd[b_idx, :chunk_len] = ep_threads[start:end]

        t_toks = torch.tensor(b_toks, device=device)
        t_targ = torch.tensor(b_targ, device=device)
        t_thrd = torch.tensor(b_thrd, device=device)

        optimizer.zero_grad()
        logits = model(t_toks, thread_seq=t_thrd)  # [B, T, V]

        # Response-masked cross entropy loss
        flat_logits = logits.view(-1, tokenizer.vocab_size)
        flat_targets = t_targ.view(-1)
        loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step_elapsed = time.perf_counter() - step_t0
        tokens_this_step = batch_size * max_len
        total_tokens_trained += tokens_this_step
        tok_sec = tokens_this_step / max(step_elapsed, 1e-5)

        loss_val = float(loss.item())
        history.append({
            "step": step,
            "loss": loss_val,
            "latency_ms": step_elapsed * 1000.0,
            "tok_sec": tok_sec,
        })

        if step % 10 == 0 or step == 1 or step == num_steps:
            logger.info(
                f"Step {step:3d}/{num_steps} | Loss: {loss_val:.4f} | "
                f"Step Latency: {step_elapsed * 1000.0:.2f} ms | Throughput: {tok_sec:.1f} tok/s"
            )

    total_time = time.perf_counter() - t_start
    logger.info(f"Tier 2 Training Completed in {total_time:.2f}s ({total_tokens_trained} tokens processed).")

    # Save checkpoint
    ckpt_file = checkpoint_path or (_REPO_ROOT / "checkpoints" / "tier2_conversational_champion.pt")
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "K": threads,
            "proj_dim": proj_dim,
            "rank": rank,
            "num_deep_layers": num_deep_layers,
            "thought_size": 64,
            "tier": "tier2",
        },
        "history": history,
        "parameters": counts["total"],
        "device_name": device_name,
    }, ckpt_file)
    logger.info(f"Checkpoint successfully saved to {ckpt_file} ({ckpt_file.stat().st_size / (1024*1024):.2f} MB).")

    return {
        "final_loss": history[-1]["loss"],
        "initial_loss": history[0]["loss"],
        "total_parameters": counts["total"],
        "state_bytes": model.state_bytes(),
        "device_name": device_name,
        "checkpoint_path": str(ckpt_file),
        "total_time_s": total_time,
        "throughput_tok_sec": total_tokens_trained / max(total_time, 1e-4),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Tier 2 Conversational Training")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--proj-dim", type=int, default=2048, help="Projection dimension (1024=~12M, 2048=~25M, 4096=~51M)")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of deep projection layers")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--num-ultrachat", type=int, default=100, help="Number of UltraChat dialogues")
    parser.add_argument("--num-alpaca", type=int, default=100, help="Number of Alpaca dialogues")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of transformed cognitive episodes")
    parser.add_argument("--checkpoint", type=str, default=None, help="Output checkpoint path")
    args = parser.parse_args()

    loader = HFConversationalLoader()
    transformer = LLMDataTransformer()
    corpus_path = build_and_cache_hf_corpus(
        loader=loader,
        transformer=transformer,
        num_ultrachat=args.num_ultrachat,
        num_alpaca=args.num_alpaca,
        num_episodes=args.num_episodes,
    )

    ckpt_p = Path(args.checkpoint) if args.checkpoint else None

    train_tier2_directml(
        corpus_path=corpus_path,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        proj_dim=args.proj_dim,
        num_deep_layers=args.num_layers,
        device_override=args.device,
        checkpoint_path=ckpt_p,
    )


if __name__ == "__main__":
    main()
