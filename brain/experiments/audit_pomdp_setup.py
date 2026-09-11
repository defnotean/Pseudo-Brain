"""Bounded, CPU-only audit of the actual POMDP checkpoint and evaluation setup.

No training, checkpoint promotion, external services, or benchmark solution injection.
The generated evidence distinguishes active pathways from advertised components.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from collections import Counter

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["PSEUDO_BRAIN_CPU_ONLY"] = "1"
    for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from run_provenance import provenance, apply_deterministic_mode
from train_and_eval_target_relevance import audit_invariants
from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
from irene_brain.agent.procedural_evaluator import ProceduralSoftwareBenchmark
from irene_brain.agent.procedural_training_generator import ProceduralTrainingGenerator


def solution_fingerprint(task):
    """Compare algorithm templates after removing only the task's renamed symbol."""
    code = re.sub(r"\b" + re.escape(task.target_function) + r"\b", "TARGET_SYMBOL", task.reference_solution)
    return hashlib.sha256(ast.dump(ast.parse(code), include_attributes=False).encode()).hexdigest()


def source_manifest():
    paths = sorted((ROOT / "src/irene_brain").rglob("*.py"))
    paths += [Path(__file__), ROOT / "experiments/train_and_eval_target_relevance.py"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def audit_model(model, tokenizer):
    torch.manual_seed(20260911)
    tokens = torch.tensor([tokenizer.encode("[RESP]ACTION: WRITE_FILE example.py\ndef example(x):\n    return x + 1[EOS]")])
    prompt = tokens[:, 1:12]
    model.zero_grad(set_to_none=True)
    out = model(tokens, prompt_tokens=prompt, parallel=True)
    torch.nn.functional.cross_entropy(out["logits"][:, :-1].reshape(-1, model.vocab_size),
                                     tokens[:, 1:].reshape(-1)).backward()
    parameters = {}
    for name, p in model.named_parameters():
        group = name.split(".")[0]
        item = parameters.setdefault(group, dict(total=0, gradient_reached=0, gradient_nonzero=0))
        item["total"] += p.numel()
        if p.grad is not None:
            item["gradient_reached"] += p.numel()
            if torch.count_nonzero(p.grad).item():
                item["gradient_nonzero"] += p.numel()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        state = model.init_state(1, torch.device("cpu"))
        tok = torch.tensor([tokenizer.resp_id])
        sensory = model.encode_sensory(token_ids=tok)
        reference, _ = model.step(sensory, copy.deepcopy(state), token_id=tok, prompt_tokens=prompt)
        perturbations = {}
        for name in ("active_slot", "other_working_slots", "episodic_memory", "previous_working_state"):
            changed = copy.deepcopy(state)
            hs = changed.hierarchical_state
            if name == "active_slot":
                hs.working_thoughts[:, 0] += torch.randn_like(hs.working_thoughts[:, 0])
            elif name == "other_working_slots":
                hs.working_thoughts[:, 1:] += torch.randn_like(hs.working_thoughts[:, 1:])
            elif name == "episodic_memory":
                hs.episodic_thoughts += torch.randn_like(hs.episodic_thoughts)
            else:
                hs.working_prev += torch.randn_like(hs.working_prev)
            altered, _ = model.step(sensory, changed, token_id=tok, prompt_tokens=prompt)
            perturbations[name] = float((altered["logits"] - reference["logits"]).abs().max())
        _, state = model.step(sensory, state, token_id=tok, prompt_tokens=prompt)
    state_bytes = {}
    for name, value in vars(state).items():
        if isinstance(value, torch.Tensor):
            state_bytes[name] = value.numel() * value.element_size()
    for name, value in vars(state.hierarchical_state).items():
        if isinstance(value, torch.Tensor):
            state_bytes["hierarchical_state." + name] = value.numel() * value.element_size()
    return {"parameter_groups": parameters, "state_perturbation_max_logit_change": perturbations,
            "persistent_tensor_bytes": state_bytes, "total_persistent_tensor_bytes": sum(state_bytes.values()),
            "immutable_prompt_token_tensor_bytes": prompt.numel() * prompt.element_size(),
            "notes": ["Fast working slots are a subset of persistent memory, not total inference memory.",
                      "Zero influence measurements concern the current default unrouted software policy.",
                      "The language-gradient probe does not claim multimodal heads should receive language loss."]}


def audit_data():
    train = ProceduralTrainingGenerator(seed=1337).generate_training_tasks(120)
    bench = ProceduralSoftwareBenchmark(seed=42)
    tiers = {"level_a": bench.generate_level_a_tasks(10), "level_b": bench.generate_level_b_tasks(10),
             "level_c": bench.generate_level_c_tasks(10)}
    fingerprints = {solution_fingerprint(t) for t in train}
    return {"training_tasks": len(train), "unique_training_algorithm_templates": len(fingerprints),
            "duplicate_training_task_ids": len(train) - len({t.task_id for t in train}),
            "training_domains": dict(Counter(t.domain for t in train)),
            "tiers": {name: {"tasks": len(tasks),
                      "algorithm_template_overlap_with_training": sum(solution_fingerprint(t) in fingerprints for t in tasks),
                      "exact_goal_overlap_with_training": sum(t.goal in {x.goal for x in train} for t in tasks)}
                      for name, tasks in tiers.items()},
            "interpretation": "Template overlap is expected for lexical holdout; it does not support unseen-algorithm claims."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/pb_pomdp_champion.pt")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Audit output already exists; choose a fresh path")
    torch.set_num_threads(1)
    torch.manual_seed(20260911)
    apply_deterministic_mode()
    agent = RecurrentSoftwareAgent(checkpoint_path=args.checkpoint)
    report = {"scope": "POMDP checkpoint, inference/training paths, memory and procedural evaluation",
              "training_executed": False, "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "provenance": provenance(model=agent.model, train_seed=None, eval_seed=20260911, deterministic=True),
              "invariants": audit_invariants(agent.model, agent.tokenizer),
              "model": audit_model(agent.model, agent.tokenizer), "data": audit_data(),
              "source_manifest": source_manifest(),
              "frontier_comparison": {"measured": False,
                  "reason": "Existing transformer comparison uses untrained models and lacks matched capability tests."}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("invariants", "data", "frontier_comparison")}, indent=2))
    print("State perturbations:", report["model"]["state_perturbation_max_logit_change"])
    print("Total persistent tensor bytes:", report["model"]["total_persistent_tensor_bytes"])
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
