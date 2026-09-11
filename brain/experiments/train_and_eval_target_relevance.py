"""Target-Pointing Calibration & Multi-Objective POMDP Training Runner.

Executes Variant D:
1. Contextually Gated Token-Skip (beta_t = sigmoid(W_beta [H_t; e_t]))
2. Retention timescale regularization (L_ret = ReLU(0.92 - mean(A_gates))^2)
3. Exact target-span loss weighting and failure-context masking
4. Cross-task latent state separation at [RESP] (L_sep)
5. Zero-shot 3-tier capability evaluation
"""

from __future__ import annotations

import argparse
import hashlib
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

sys.path.insert(0, str(SRC_DIR.parent / "scripts"))
from run_provenance import provenance, apply_deterministic_mode

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
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    gen = ProceduralTrainingGenerator(seed=1337)
    episodes = gen.generate_multiturn_pomdp_trajectories(tasks)

    encoded_list = [tokenizer.encode(ep) for ep in episodes]
    batches = []
    for start in range(0, len(tasks), batch_size):
        chunk = encoded_list[start:start + batch_size]
        max_len = max(map(len, chunk))
        max_prompt = max(t.index(tokenizer.resp_id) for t in chunk)
        rows = {k: [] for k in ("X", "Y", "M", "target_masks", "resp_indices",
                                "prompt_tokens", "ptr_targets", "pointer_mask")}
        for j, toks in enumerate(chunk):
            task = tasks[start + j]
            first_resp = toks.index(tokenizer.resp_id)
            prompt = toks[:first_resp]
            seq = toks + [tokenizer.pad_id] * (max_len - len(toks))
            mask = [0.0] * max_len
            pmask = [0.0] * max_len
            tmask = [0.0] * max_len
            ptr = [-1] * max_len
            responses = [i for i, tok in enumerate(toks) if tok == tokenizer.resp_id]
            for turn, r in enumerate(responses):
                end = toks.index(tokenizer.eos_id, r)
                pmask[r:end] = [1.0] * (end - r)
                # Deliberate failures are teacher-forced context, not desired actions.
                if turn == 0 and (start + j) % 6 in (1, 2, 3, 4):
                    continue
                # X[t] predicts Y[t] = X[t+1]: include EOS, exclude following observation.
                mask[r:end] = [1.0] * (end - r)
                for name in (task.target_module, task.target_function):
                    span = tokenizer.encode(" " + name)
                    plen = len(span)
                    sources = [k for k in range(len(prompt) - plen + 1)
                               if prompt[k:k + plen] == span]
                    if not sources:
                        continue
                    for k in range(r + 1, end - plen + 1):
                        if toks[k:k + plen] != span:
                            continue
                        for offset in range(plen):
                            tmask[k - 1 + offset] = 1.0
                            ptr[k - 1 + offset] = sources[-1] + offset
            rows["X"].append(seq)
            rows["Y"].append(seq[1:] + [tokenizer.pad_id])
            rows["M"].append(mask)
            rows["pointer_mask"].append(pmask)
            rows["target_masks"].append(tmask)
            rows["ptr_targets"].append(ptr)
            rows["resp_indices"].append(first_resp)
            rows["prompt_tokens"].append(prompt + [tokenizer.pad_id] * (max_prompt - len(prompt)))
        batches.append({key: torch.tensor(value, dtype=(torch.float32 if key in
                        ("M", "target_masks", "pointer_mask") else torch.long))
                        for key, value in rows.items()})
    return batches


def pointer_attention_supervision(attention, targets, prompt_tokens, mode="position"):
    """Supervise copied output tokens without penalizing equivalent positions."""
    if mode not in ("position", "token_mass"):
        raise ValueError(f"Unknown pointer loss: {mode}")
    valid = targets >= 0
    safe_targets = targets.clamp(min=0)
    position_probability = attention.gather(2, safe_targets.unsqueeze(-1)).squeeze(-1)
    target_tokens = prompt_tokens.gather(1, safe_targets)
    matching_tokens = prompt_tokens.unsqueeze(1) == target_tokens.unsqueeze(-1)
    token_probability = (attention * matching_tokens).sum(-1)
    probability = token_probability if mode == "token_mass" else position_probability
    denominator = valid.sum() + 1e-6
    loss = -(torch.log(probability + 1e-6) * valid).sum() / denominator
    position_accuracy = ((position_probability > .5) * valid).sum() / denominator
    token_accuracy = ((token_probability > .5) * valid).sum() / denominator
    return loss, position_accuracy, token_accuracy


def train_calibrated_pomdp_policy(
    model: UnifiedPseudoBrain,
    batches: List[Dict[str, torch.Tensor]],
    steps: int = 250,
    lr: float = 1.5e-3,
    target_loss: float = 0.010,
    target_weight: float = 8.0,
    lambda_ret: float = 5.0,
    lambda_sep: float = 1.0,
    pointer_loss: str = "position",
) -> float:
    if not batches or steps < 1:
        raise ValueError("Training requires nonempty batches and positive steps")
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
        batch = {key: value.to(next(model.parameters()).device)
                 for key, value in batches[step % num_batches].items()}
        X, Y, M = batch["X"], batch["Y"], batch["M"]
        target_masks = batch["target_masks"]
        resp_indices = batch["resp_indices"]
        prompt_tokens = batch.get("prompt_tokens")
        ptr_targets = batch.get("ptr_targets")

        optimizer.zero_grad()
        language_mask = M.bool()
        out = model(token_seq=X, prompt_tokens=prompt_tokens, parallel=True,
                    pointer_mask=batch["pointer_mask"], language_mask=language_mask)
        logits = out["logits"]
        A_gates = out["A_gates"]
        H_scanned = out["H_scanned"]
        ptr_gamma = out.get("ptr_gamma")
        ptr_attn = out.get("ptr_attn")
        B, T = X.shape
        V = model.vocab_size

        # 1. Weighted Token Loss
        weights = 1.5 * M + (target_weight - 1.5) * target_masks
        loss_raw = F.cross_entropy(logits.reshape(-1, V), Y[language_mask], reduction="none")
        L_token = (loss_raw * weights[language_mask]).sum() / (weights.sum() + 1e-6)

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
            L_gate = (skip_beta.squeeze(-1) * target_masks[language_mask]).sum() / (target_masks.sum() + 1e-6)
        else:
            L_gate = torch.tensor(0.0, device=X.device)

        # 5. In-Context Pointer-Copy Loss (Exact Symbolic Target Binding)
        if ptr_gamma is not None and ptr_attn is not None and ptr_targets is not None:
            gamma_s = ptr_gamma.squeeze(-1)
            L_pgate_tgt = F.binary_cross_entropy(gamma_s, torch.ones_like(gamma_s), weight=target_masks, reduction="sum") / (target_masks.sum() + 1e-6)
            bg_mask = (M - target_masks).clamp(min=0.0)
            L_pgate_bg = F.binary_cross_entropy(gamma_s, torch.zeros_like(gamma_s), weight=bg_mask, reduction="sum") / (bg_mask.sum() + 1e-6)
            L_pgate = L_pgate_tgt + 0.3 * L_pgate_bg

            L_pattn, ptr_match_acc, ptr_token_acc = pointer_attention_supervision(
                ptr_attn, ptr_targets, prompt_tokens, mode=pointer_loss)
            L_ptr = L_pattn + L_pgate
        else:
            L_ptr = torch.tensor(0.0, device=X.device)
            ptr_match_acc = torch.tensor(0.0, device=X.device)
            ptr_token_acc = torch.tensor(0.0, device=X.device)

        loss = L_token + lambda_ret * L_ret + lambda_sep * L_sep + 1.0 * L_gate + 1.0 * L_ptr

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        final_loss = float(loss.item())
        if (step + 1) % 25 == 0 or step == 0:
            b_val = L_gate.item()
            acc_val = ptr_match_acc.item()
            print(f"Step {step+1:3d}/{steps} | Loss: {final_loss:.4f} (Tok: {L_token.item():.4f}, Ret: {L_ret.item():.4f}, Sep: {L_sep.item():.4f}, Gate: {b_val:.4f}, Ptr: {L_ptr.item():.4f}) | A_gate: {mean_ret.item():.3f} | Cos: {mean_cos:.3f} | PtrPosAcc: {acc_val*100:.1f}% | PtrTokenMassAcc: {ptr_token_acc.item()*100:.1f}%")

        if final_loss <= target_loss and step >= 150:
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
        "mean_target_span_token_accuracy": r.mean_target_span_token_accuracy,
        "mean_target_span_char_similarity": r.mean_target_span_char_similarity,
        "useful_first_action_count": r.useful_first_action_count,
        "useful_first_action_rate": r.useful_first_action_rate,
        "error_recoveries": r.error_recoveries,
        "exact_module_binding_rate": r.exact_module_binding_rate,
        "mean_cycles": r.mean_cycles,
        "mode": r.mode,
        "task_results": r.task_results,
    }


def audit_invariants(model, tokenizer):
    """Measure the actual checkpoint, including pointer/skip and turn boundaries."""
    text = ("[GOAL: Implement f in a.py]\n[TARGET_FUNCTION: f]\n"
            "[PHASE: WRITE_CODE]\n[TARGET_MODULE: a.py]\n"
            "[RESP]ACTION: WRITE_FILE a.py\ndef f():\n    return None[EOS]\n"
            "[OBSERVATION: Task incomplete: AssertionError]\n[PHASE: REPAIR_LOGIC]\n"
            "[TARGET_MODULE: a.py]\n[RESP]ACTION: FINISH Verified implementation of f[EOS]")
    device = next(model.parameters()).device
    tokens = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    boundary = int((tokens[0] == tokenizer.resp_id).nonzero()[0])
    prompt = tokens[:, :boundary]
    mask = torch.zeros_like(tokens, dtype=torch.float32)
    active = False
    for i, token in enumerate(tokens[0].tolist()):
        if token == tokenizer.resp_id:
            active = True
        elif token == tokenizer.eos_id:
            active = False
        mask[0, i] = float(active)
    model.eval()
    with torch.no_grad():
        parallel = model(tokens, parallel=True, prompt_tokens=prompt, pointer_mask=mask)
        streaming = model(tokens, parallel=False, prompt_tokens=prompt, pointer_mask=mask)
    delta = float((parallel["logits"] - streaming["logits"]).abs().max())
    state = model.init_state(1, device)
    working = state.hierarchical_state.working_thoughts
    fast_bytes = working[0].numel() * working.element_size()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable_parameters": parameters, "parameter_ceiling": 36_000_000,
            "fast_working_bytes": fast_bytes, "fast_dtype": str(working.dtype),
            "parity_max_abs_logits": delta, "parity_tolerance": 1e-6,
            "parity_tokens": tokens.shape[1], "device": str(device),
            "passed": parameters < 36_000_000 and fast_bytes == 4096 and delta < 1e-6}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--compensated-state", action="store_true")
    parser.add_argument("--pointer-mode", choices=("sequential", "parallel_predecessor"), default=None)
    parser.add_argument("--pointer-loss", choices=("position", "token_mass"), default="position")
    parser.add_argument("--retention-profile", choices=("legacy", "multiscale"), default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=SRC_DIR.parent / "checkpoints/pb_pomdp_champion.pt")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 350:
        parser.error("--steps must be in [1, 350]; longer runs require a new registered experiment")
    if args.device == "cuda":
        if sys.platform == "win32" or not Path("/content").is_dir():
            parser.error("CUDA training is restricted to the user-authorized Colab runtime")
        if os.environ.get("PSEUDO_BRAIN_CPU_ONLY") == "1":
            parser.error("CPU-only verification cannot use CUDA")
        if not torch.cuda.is_available():
            parser.error("Requested Colab CUDA device is unavailable")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["PSEUDO_BRAIN_CPU_ONLY"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    apply_deterministic_mode()
    run_dir = args.run_dir or SRC_DIR.parent / "runs" / ("pomdp-repair-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    run_dir.mkdir(parents=True, exist_ok=False)
    ckpt_path = args.checkpoint.resolve()
    agent = RecurrentSoftwareAgent(checkpoint_path=ckpt_path, device=torch.device(args.device)) if ckpt_path.exists() else None
    if agent is None:
        raise FileNotFoundError(f"Warm-start checkpoint does not exist: {ckpt_path}")
    model, tokenizer = agent.model, agent.tokenizer
    if args.retention_profile is not None:
        model.retention_profile = args.retention_profile
    if args.pointer_mode is not None:
        model.pointer_mode = args.pointer_mode
        agent.reset()
    if args.compensated_state:
        model.enable_compensated_state()
        agent.reset()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    invariants = audit_invariants(model, tokenizer)
    run_metadata = {
        "source_checkpoint": str(ckpt_path), "source_checkpoint_sha256": checkpoint_hash,
        "requested_steps": args.steps, "training_executed": False,
        "mode": "audit_only" if args.audit_only else "eval_only" if args.eval_only else "train_and_eval",
        "compensated_state": model.compensated_state,
        "pointer_mode": model.pointer_mode,
        "pointer_loss": args.pointer_loss,
        "retention_profile": model.retention_profile,
        "training_device": args.device,
        "evaluation_device": "cpu",
        "logical_slots": model.logical_slots,
        "language_readout": "supervised_positions_only",
        "invariants_before": invariants,
        "acceptance": {"level_a_completion_gt": 0.50, "level_b_completion_gt": 0.25,
                       "level_c_relevance_gt": 0.20, "executable_rate_gte": 0.90,
                       "level_a_exact_module_binding_gte": 0.90, "verified_repairs_gte": 1},
        "provenance": provenance(model=model, train_seed=args.seed, eval_seed=42, deterministic=True),
        "source_manifest": {str(path.relative_to(SRC_DIR.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in [*sorted((SRC_DIR / "irene_brain").rglob("*.py")), Path(__file__).resolve()]},
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print("Invariant audit:", json.dumps(invariants), flush=True)
    if args.audit_only:
        return 0 if invariants["passed"] else 2
    if not args.eval_only:
        if not invariants["passed"]:
            print("Training blocked by invariant audit; evidence saved to", metadata_path, flush=True)
            return 2
        gen = ProceduralTrainingGenerator(seed=args.seed)
        tasks = gen.generate_training_tasks(count=120, diversify_indices=True)
        batches = build_pomdp_batches_calibrated(tasks, tokenizer, batch_size=4)
        digest = hashlib.sha256()
        for batch in batches:
            for key in sorted(batch):
                digest.update(key.encode())
                digest.update(batch[key].numpy().tobytes())
        run_metadata["training_bank_sha256"] = digest.hexdigest()
        run_metadata["provenance"]["bank_digest"] = digest.hexdigest()
        metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
        model.train()
        loss = train_calibrated_pomdp_policy(model, batches, steps=args.steps, pointer_loss=args.pointer_loss)
        model.eval()
        ckpt_path = run_dir / "candidate.pt"
        torch.save({"tier": "tier2", "vocab_size": 32000, "use_token_skip": True,
                    "use_gated_token_skip": True, "use_pointer_copy": True,
                    "compensated_state": model.compensated_state,
                    "pointer_mode": model.pointer_mode,
                    "pointer_loss": args.pointer_loss,
                    "retention_profile": model.retention_profile,
                    "tokenizer_json": tokenizer.to_str(),
                    "model_state_dict": model.state_dict(),
                    "trained_on": "procedural_multiturn_8domains_repair_v2"}, ckpt_path)
        run_metadata.update(training_executed=True, final_loss=loss,
                            invariants_after=audit_invariants(model, tokenizer))
        metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
        if not run_metadata["invariants_after"]["passed"]:
            print("Candidate failed invariant audit; preserved without promotion.", flush=True)
            return 2

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
    print(f"{'Target Span Token Acc':<30} | {rep_a.mean_target_span_token_accuracy*100:>5.1f}%     | {rep_b.mean_target_span_token_accuracy*100:>5.1f}%     | {rep_c.mean_target_span_token_accuracy*100:>5.1f}%")
    print(f"{'Target Span Char Sim':<30} | {rep_a.mean_target_span_char_similarity*100:>5.1f}%     | {rep_b.mean_target_span_char_similarity*100:>5.1f}%     | {rep_c.mean_target_span_char_similarity*100:>5.1f}%")
    print(f"{'Useful First Action':<30} | {rep_a.useful_first_action_rate*100:>5.1f}%     | {rep_b.useful_first_action_rate*100:>5.1f}%     | {rep_c.useful_first_action_rate*100:>5.1f}%")
    print(f"{'Error Recoveries':<30} | {rep_a.error_recoveries:>5d}      | {rep_b.error_recoveries:>5d}      | {rep_c.error_recoveries:>5d}     ")
    print("=" * 65)

    receipt = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tier": "tier2",
        "parameters": param_count,
        "vocab_size": 32000,
        "use_token_skip": True,
        "use_gated_token_skip": True,
        "use_pointer_copy": True,
        "checkpoint": str(ckpt_path),
        "trained_on": "unchanged source checkpoint" if args.eval_only else "procedural_multiturn_8domains_repair_v2",
        "run_metadata": run_metadata,
        "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "evaluation_mode": "zero_shot",
        "state_isolation_per_task": True,
        "elapsed_seconds": elapsed,
        "tiers": {
            "level_a_lexical": report_to_dict(rep_a),
            "level_b_domain_transfer": report_to_dict(rep_b),
            "level_c_sealed_ood": report_to_dict(rep_c),
        },
    }

    acceptance = {
        "level_a_completion": rep_a.completion_rate > 0.50,
        "level_b_completion": rep_b.completion_rate > 0.25,
        "level_c_relevance": rep_c.task_relevant_rate > 0.20,
        "executable_actions": all(r.executable_action_rate >= 0.90 for r in reports.values()),
        "level_a_exact_binding": rep_a.exact_module_binding_rate >= 0.90,
        "autonomous_repair": sum(r.error_recoveries for r in reports.values()) >= 1,
        "invariants": run_metadata.get("invariants_after", invariants)["passed"],
    }
    receipt["acceptance"] = acceptance
    receipt["milestone_passed"] = all(acceptance.values())
    receipt_path = run_dir / "three_tier_benchmark_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\nSaved 3-Tier Benchmark Receipt to: {receipt_path}")
    print("Milestone acceptance:", json.dumps(acceptance))
    return 0 if receipt["milestone_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
