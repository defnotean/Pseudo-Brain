"""Target-Pointing Calibration & Multi-Objective POMDP Training Runner.

Executes Variant D:
1. Contextually Gated Token-Skip (beta_t = sigmoid(W_beta [H_t; e_t]))
2. Retention timescale regularization (L_ret = ReLU(0.92 - mean(A_gates))^2)
3. Target module loss weighting (target_weight = 15.0)
4. Cross-task latent state separation at [RESP] (L_sep)
5. Zero-shot 3-tier capability evaluation
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from irene_brain.unified.unified_model import UnifiedPseudoBrain, make_unified_model
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.agent.procedural_evaluator import (
    ProceduralSoftwareBenchmark,
    ProceduralTask,
    evaluate_three_tier_benchmark,
    BenchmarkEvaluationReport,
)
from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
from irene_brain.agent.procedural_training_generator import ProceduralTrainingGenerator


def build_pomdp_batches_calibrated(
    tasks: List[ProceduralTask],
    tokenizer: BpeSemanticTokenizer,
    batch_size: int = 4,
) -> List[Dict[str, torch.Tensor]]:
    gen = ProceduralTrainingGenerator(seed=1337)
    episodes = gen.generate_multiturn_pomdp_trajectories(tasks)

    encoded_list = [tokenizer.encode(ep) for ep in episodes]
    resp_id = tokenizer.resp_id
    eos_id = tokenizer.eos_id
    file_tok = 4036  # token id for 'FILE'
    nl_id = 273     # token id for '\n'

    batches = []
    for i in range(0, len(encoded_list), batch_size):
        chunk = encoded_list[i : i + batch_size]
        max_len = max(len(toks) for toks in chunk)

        padded_seqs = []
        target_seqs = []
        loss_masks = []
        target_masks = []
        resp_indices = []

        for toks in chunk:
            seq = toks + [tokenizer.pad_id] * (max_len - len(toks))
            padded_seqs.append(seq)
            targets = seq[1:] + [tokenizer.pad_id]
            target_seqs.append(targets)

            # Record first [RESP] position
            first_resp = toks.index(resp_id) if resp_id in toks else 0
            resp_indices.append(first_resp)

            mask = [0.0] * len(seq)
            tmask = [0.0] * len(seq)
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

            # Target module tokens in response (immediately following FILE up to \n)
            curr = 0
            while True:
                try:
                    f_pos = seq.index(file_tok, curr)
                except ValueError:
                    break
                try:
                    nl_pos = seq.index(nl_id, f_pos)
                except ValueError:
                    nl_pos = len(seq) - 1

                for p in range(f_pos, min(nl_pos, len(seq))):
                    if p < len(tmask):
                        tmask[p] = 1.0
                curr = nl_pos + 1

            loss_masks.append(mask)
            target_masks.append(tmask)

        batches.append({
            "X": torch.tensor(padded_seqs, dtype=torch.long),
            "Y": torch.tensor(target_seqs, dtype=torch.long),
            "M": torch.tensor(loss_masks, dtype=torch.float32),
            "target_masks": torch.tensor(target_masks, dtype=torch.float32),
            "resp_indices": torch.tensor(resp_indices, dtype=torch.long),
        })

    return batches


def train_calibrated_pomdp_policy(
    model: UnifiedPseudoBrain,
    batches: List[Dict[str, torch.Tensor]],
    steps: int = 400,
    lr: float = 3e-3,
    target_loss: float = 0.025,
    target_weight: float = 15.0,
    lambda_ret: float = 5.0,
    lambda_sep: float = 1.0,
) -> float:
    total_tokens = sum(int(b["M"].sum().item()) for b in batches)
    target_tokens = sum(int(b["target_masks"].sum().item()) for b in batches)
    print(f"Training POMDP Policy: {len(batches)} batches | Action tokens: {total_tokens} | Target tokens: {target_tokens}")

    # Set retention bias prior (mean a_t >= 0.92)
    torch.nn.init.constant_(model.W_iz_parallel.up.bias, 3.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-4)

    t0 = time.perf_counter()
    final_loss = 999.0
    num_batches = len(batches)

    for step in range(steps):
        batch = batches[step % num_batches]
        X, Y, M = batch["X"], batch["Y"], batch["M"]
        target_masks = batch["target_masks"]
        resp_indices = batch["resp_indices"]

        optimizer.zero_grad()
        out = model(token_seq=X, parallel=True)
        logits = out["logits"]
        A_gates = out["A_gates"]
        H_scanned = out["H_scanned"]
        B, T, V = logits.shape

        # 1. Weighted Token Loss
        weights = M + (target_weight - 1.0) * target_masks
        loss_raw = F.cross_entropy(logits.view(-1, V), Y.view(-1), reduction="none")
        L_token = (loss_raw * weights.view(-1)).sum() / (weights.sum() + 1e-6)

        # 2. Retention Loss: Mean gate >= 0.92
        mean_ret = A_gates.mean()
        L_ret = F.relu(0.92 - mean_ret).pow(2)

        # 3. Latent State Separation Loss at [RESP]
        if B >= 2:
            b_idx = torch.arange(B, device=X.device)
            H_resp = H_scanned[b_idx, resp_indices]
            H_norm = F.normalize(H_resp, p=2, dim=-1)
            sim_matrix = torch.matmul(H_norm, H_norm.T)
            mask_off_diag = ~torch.eye(B, dtype=torch.bool, device=X.device)
            off_diag_sim = sim_matrix[mask_off_diag]
            L_sep = F.relu(off_diag_sim - 0.70).pow(2).mean()
            mean_cos = off_diag_sim.mean().item()
        else:
            L_sep = torch.tensor(0.0, device=X.device)
            mean_cos = 1.0

        # 4. Skip Gate Suppression on Target Module Tokens (force recurrent state to drive target filename)
        skip_beta = out.get("skip_beta")
        if skip_beta is not None:
            L_gate = (skip_beta.squeeze(-1) * target_masks).sum() / (target_masks.sum() + 1e-6)
        else:
            L_gate = torch.tensor(0.0, device=X.device)

        loss = L_token + lambda_ret * L_ret + lambda_sep * L_sep + 1.0 * L_gate

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        final_loss = float(loss.item())
        if (step + 1) % 25 == 0 or step == 0:
            b_val = L_gate.item()
            print(f"Step {step+1:3d}/{steps} | Loss: {final_loss:.4f} (Tok: {L_token.item():.4f}, Ret: {L_ret.item():.4f}, Sep: {L_sep.item():.4f}, Gate: {b_val:.4f}) | A_gate: {mean_ret.item():.3f} | Cos: {mean_cos:.3f}")

        if final_loss <= target_loss and step >= 200:
            print(f"Target loss {target_loss} reached at step {step+1}! Final loss: {final_loss:.4f}")
            break

    elapsed = time.perf_counter() - t0
    print(f"Training complete in {elapsed:.2f}s | Final Loss: {final_loss:.4f}")
    return final_loss


def report_to_dict(r: BenchmarkEvaluationReport) -> Dict[str, Any]:
    return {
        "total_tasks": r.total_tasks,
        "tasks_completed": r.tasks_completed,
        "completion_rate": r.completion_rate,
        "total_actions": r.total_actions,
        "level_1_verb_grammar_count": r.verb_grammar_count,
        "level_1_verb_grammar_rate": r.verb_grammar_rate,
        "level_2_parsable_action_count": r.parsable_action_count,
        "level_2_parsable_action_rate": r.parsable_action_rate,
        "level_3_executable_action_count": r.executable_action_count,
        "level_3_executable_action_rate": r.executable_action_rate,
        "level_4_task_relevant_count": r.task_relevant_count,
        "level_4_task_relevant_rate": r.task_relevant_rate,
        "level_5_task_progressing_count": r.task_progressing_count,
        "level_5_task_progressing_rate": r.task_progressing_rate,
        "useful_first_action_count": r.useful_first_action_count,
        "useful_first_action_rate": r.useful_first_action_rate,
        "error_recoveries": r.error_recoveries,
        "mean_cycles": r.mean_cycles,
        "mode": r.mode,
        "task_results": r.task_results,
    }


def main():
    print("=== Pseudo-Brain POMDP Policy: Target-Pointing Calibration ===")
    gen = ProceduralTrainingGenerator(seed=1337)
    training_tasks = gen.generate_training_tasks(count=40)
    print(f"Generated {len(training_tasks)} training tasks across 8 open domains.")

    tokenizer = BpeSemanticTokenizer(vocab_size=32000)
    batches = build_pomdp_batches_calibrated(training_tasks, tokenizer, batch_size=4)

    model = make_unified_model(
        tier="tier2",
        vocab_size=32000,
        use_token_skip=True,
        use_gated_token_skip=True,
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,} (~{param_count/1e6:.2f}M)")

    train_calibrated_pomdp_policy(
        model,
        batches,
        steps=350,
        lr=3e-3,
        target_loss=0.025,
        target_weight=15.0,
        lambda_ret=5.0,
        lambda_sep=1.0,
    )

    # Save to champion checkpoint location
    ckpt_dir = Path("brain/checkpoints").resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "pb_pomdp_champion.pt"

    torch.save({
        "tier": "tier2",
        "vocab_size": 32000,
        "use_token_skip": True,
        "use_gated_token_skip": True,
        "model_state_dict": model.state_dict(),
        "trained_on": "procedural_multiturn_8domains_tier2_gated_skip_calibrated",
    }, ckpt_path)
    print(f"Saved Champion Checkpoint to: {ckpt_path}")

    # Evaluate on full 3-tier benchmark
    agent = RecurrentSoftwareAgent(checkpoint_path=ckpt_path)
    bench = ProceduralSoftwareBenchmark(seed=42)

    print("\n=======================================================")
    print("  STARTING 3-TIER BENCHMARK EVALUATION (ZERO-SHOT ISOLATION)")
    print("=======================================================\n")

    t0 = time.perf_counter()
    reports = evaluate_three_tier_benchmark(
        agent=agent,
        benchmark=bench,
        count_per_tier=10,
        max_cycles_per_task=4,
        mode="zero_shot",
    )
    elapsed = time.perf_counter() - t0

    rep_a = reports["level_a_lexical"]
    rep_b = reports["level_b_domain_transfer"]
    rep_c = reports["level_c_sealed_ood"]

    print("\n" + "=" * 65)
    print("  THREE-TIER EVALUATION RECEIPT SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<30} | {'Level A':<10} | {'Level B':<10} | {'Level C (Sealed)':<10}")
    print("-" * 65)
    print(f"{'Tasks Completed':<30} | {rep_a.tasks_completed:>2}/{rep_a.total_tasks:<7} | {rep_b.tasks_completed:>2}/{rep_b.total_tasks:<7} | {rep_c.tasks_completed:>2}/{rep_c.total_tasks:<7}")
    print(f"{'Completion Rate':<30} | {rep_a.completion_rate*100:>5.1f}%     | {rep_b.completion_rate*100:>5.1f}%     | {rep_c.completion_rate*100:>5.1f}%")
    print(f"{'Level 1: Verb Grammar':<30} | {rep_a.verb_grammar_rate*100:>5.1f}%     | {rep_b.verb_grammar_rate*100:>5.1f}%     | {rep_c.verb_grammar_rate*100:>5.1f}%")
    print(f"{'Level 2: Parsable Action':<30} | {rep_a.parsable_action_rate*100:>5.1f}%     | {rep_b.parsable_action_rate*100:>5.1f}%     | {rep_c.parsable_action_rate*100:>5.1f}%")
    print(f"{'Level 3: Executable Action':<30} | {rep_a.executable_action_rate*100:>5.1f}%     | {rep_b.executable_action_rate*100:>5.1f}%     | {rep_c.executable_action_rate*100:>5.1f}%")
    print(f"{'Level 4: Task Relevant':<30} | {rep_a.task_relevant_rate*100:>5.1f}%     | {rep_b.task_relevant_rate*100:>5.1f}%     | {rep_c.task_relevant_rate*100:>5.1f}%")
    print(f"{'Level 5: Task Progressing':<30} | {rep_a.task_progressing_rate*100:>5.1f}%     | {rep_b.task_progressing_rate*100:>5.1f}%     | {rep_c.task_progressing_rate*100:>5.1f}%")
    print(f"{'Useful First Action':<30} | {rep_a.useful_first_action_rate*100:>5.1f}%     | {rep_b.useful_first_action_rate*100:>5.1f}%     | {rep_c.useful_first_action_rate*100:>5.1f}%")
    print("=" * 65)

    receipt = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tier": "tier2",
        "parameters": param_count,
        "vocab_size": 32000,
        "use_token_skip": True,
        "use_gated_token_skip": True,
        "checkpoint": str(ckpt_path),
        "trained_on": "procedural_multiturn_8domains_tier2_gated_skip_calibrated",
        "evaluation_mode": "zero_shot",
        "state_isolation_per_task": True,
        "elapsed_seconds": elapsed,
        "tiers": {
            "level_a_lexical": report_to_dict(rep_a),
            "level_b_domain_transfer": report_to_dict(rep_b),
            "level_c_sealed_ood": report_to_dict(rep_c),
        },
    }

    receipt_path = Path("brain/experiments/three_tier_benchmark_receipt.json")
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"\nSaved 3-Tier Benchmark Receipt to: {receipt_path}")


if __name__ == "__main__":
    main()
