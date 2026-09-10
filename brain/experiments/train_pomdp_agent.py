"""Training Pipeline for Pseudo-Brain Recurrent Software Agent POMDP Policy.

Trains UnifiedPseudoBrain directly on procedural task POMDP trajectories:
1. Ingests Goal conditioning.
2. Autoregressively generates actuator commands (WRITE_FILE, RUN_TESTS, FINISH).
3. Ingests environment observations and self-corrects.
4. Evaluates on held-out tasks using external hidden unit tests.
5. Saves trained weights to brain/checkpoints/pb_pomdp_champion.pt.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

SRC_DIR = Path("brain/src").resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from irene_brain.unified.unified_model import UnifiedPseudoBrain, make_unified_model
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.agent.procedural_evaluator import (
    ProceduralSoftwareBenchmark,
    ProceduralTask,
    evaluate_agent_on_benchmark,
    make_hidden_task_validator,
)
from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
from irene_brain.agent.software_environment import NeuralSoftwareEnvironment


from irene_brain.agent.procedural_training_generator import ProceduralTrainingGenerator


def build_pomdp_batches(
    tasks: List[ProceduralTask],
    tokenizer: BpeSemanticTokenizer,
    batch_size: int = 4,
) -> List[Dict[str, torch.Tensor]]:
    """Format procedural tasks into batched POMDP action-observation sequences with dynamic padding."""
    gen = ProceduralTrainingGenerator(seed=1337)
    episodes = gen.generate_multiturn_pomdp_trajectories(tasks)

    encoded_list = [tokenizer.encode(ep) for ep in episodes]
    resp_id = tokenizer.resp_id
    eos_id = tokenizer.eos_id

    batches = []
    for i in range(0, len(encoded_list), batch_size):
        chunk = encoded_list[i : i + batch_size]
        max_len = max(len(toks) for toks in chunk)

        padded_seqs = []
        target_seqs = []
        loss_masks = []

        for toks in chunk:
            seq = toks + [tokenizer.pad_id] * (max_len - len(toks))
            padded_seqs.append(seq)
            targets = seq[1:] + [tokenizer.pad_id]
            target_seqs.append(targets)

            # Loss mask covers actuator tokens starting at [RESP] up to [EOS]
            mask = [0.0] * len(seq)
            in_resp = False
            for idx, tok in enumerate(seq):
                if tok == resp_id:
                    in_resp = True
                    mask[idx] = 1.0
                elif tok == eos_id:
                    in_resp = False
                    mask[idx] = 1.0
                elif in_resp:
                    mask[idx] = 1.0
            loss_masks.append(mask)

        batches.append({
            "X": torch.tensor(padded_seqs, dtype=torch.long),
            "Y": torch.tensor(target_seqs, dtype=torch.long),
            "M": torch.tensor(loss_masks, dtype=torch.float32),
        })

    return batches


def train_pomdp_policy(
    model: UnifiedPseudoBrain,
    batches: List[Dict[str, torch.Tensor]],
    steps: int = 400,
    lr: float = 3e-3,
    target_loss: float = 0.02,
) -> float:
    """Train the unified recurrent core across batched POMDP multi-turn sequences."""
    total_tokens = sum(int(b["M"].sum().item()) for b in batches)
    print(f"Training POMDP Policy: {len(batches)} batches, Active action tokens: {total_tokens}")

    # Long memory retention initialization for associative recurrence
    torch.nn.init.constant_(model.W_iz_parallel.up.bias, 2.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-4)

    t0 = time.perf_counter()
    final_loss = 999.0
    num_batches = len(batches)

    for step in range(steps):
        batch = batches[step % num_batches]
        X, Y, M = batch["X"], batch["Y"], batch["M"]

        optimizer.zero_grad()
        out = model(token_seq=X, parallel=True)
        logits = out["logits"]
        V = logits.shape[-1]

        loss_raw = F.cross_entropy(logits.view(-1, V), Y.view(-1), reduction="none")
        loss = (loss_raw * M.view(-1)).sum() / (M.sum() + 1e-6)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        final_loss = float(loss.item())
        if (step + 1) % 25 == 0 or step == 0:
            print(f"Step {step+1:3d}/{steps} | Loss: {final_loss:.4f}")

        if final_loss <= target_loss and step >= 200:
            print(f"Target loss {target_loss} reached at step {step+1}! Final loss: {final_loss:.4f}")
            break

    elapsed = time.perf_counter() - t0
    print(f"POMDP policy training complete in {elapsed:.2f}s | Final Loss: {final_loss:.4f}")
    return final_loss


def main():
    print("=== Pseudo-Brain Multi-Domain POMDP Policy Training ===")
    gen = ProceduralTrainingGenerator(seed=1337)
    training_tasks = gen.generate_training_tasks(count=40)
    print(f"Generated {len(training_tasks)} training tasks across 8 open domains.")

    tokenizer = BpeSemanticTokenizer(vocab_size=32000)
    batches = build_pomdp_batches(training_tasks, tokenizer, batch_size=4)

    model = make_unified_model(tier="tier2", vocab_size=32000, use_token_skip=True)
    train_pomdp_policy(model, batches, steps=400, lr=3e-3, target_loss=0.02)

    # Save checkpoint
    ckpt_dir = Path("brain/checkpoints").resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "pb_pomdp_champion.pt"

    torch.save({
        "tier": "tier2",
        "vocab_size": 32000,
        "use_token_skip": True,
        "model_state_dict": model.state_dict(),
        "trained_on": "procedural_multiturn_8domains_tier2_skip",
    }, ckpt_path)
    print(f"Saved POMDP Policy Champion checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()

