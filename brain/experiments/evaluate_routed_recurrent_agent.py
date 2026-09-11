"""Audited Tool-Policy Evaluation for RecurrentSoftwareAgent.

Evaluates RecurrentSoftwareAgent with multi-slot routing and episodic memory
readout fusion across the 48 audited tool tasks (24 from-scratch + 24 repair).
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
from irene_brain.evaluation.tool_policy import run_tool_policy_case
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.unified.unified_model import make_unified_model

BANK = {
    "requests.jsonl": "c306d8e0399531c1157fed1abb09314013f577cf4be0ca564aaeebaa05f561a1",
    "graders.jsonl": "28d7ed1677ace53899058c56047253ee109523594e0e1c42328979fc480c468d",
}
TOKENIZER_HASH = "760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e"
MODES = ("from_scratch", "repair")
VERBS = {"WRITE_FILE", "READ_FILE", "EDIT_FILE", "RUN_TESTS", "RETRIEVE_MEMORY", "FINISH"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def aggregate(plan: list, results: list) -> dict:
    keys = [(row["source_id"], row["mode"]) for row in plan]
    observed = {}
    for row in results:
        key = (row["source_id"], row["mode"])
        observed[key] = row

    stop_reasons = Counter(row["episode"]["stop_reason"] for row in results)
    stop_reasons["missing"] = len(keys) - len(results)
    events = [event for row in results for event in row["events"]]
    trace_entries = [entry for row in results for entry in row.get("episode", {}).get("trace", [])]
    unexecuted = sum(1 for entry in trace_entries if not entry.get("executed", True))
    assisted_count = sum(1 for entry in trace_entries if entry.get("transformation_applied"))

    return {
        "planned_tasks": len(keys),
        "recorded_tasks": len(results),
        "missing_tasks": [row for row in plan if (row["source_id"], row["mode"]) not in observed],
        "completed_tasks": sum(row["success"] for row in results),
        "observed_repairs": sum(row["observed_repair"] for row in results),
        "completion_rate": sum(row["success"] for row in results) / len(keys) if keys else 0.0,
        "repair_rate": sum(row["observed_repair"] for row in results) / len(keys) if keys else 0.0,
        "total_generated_attempts": len(trace_entries),
        "rejected_unexecuted_attempts": unexecuted,
        "executed_actions": len(events),
        "recognized_action_verbs": sum(event["action_type"] in VERBS for event in events),
        "successful_environment_actions": sum(event["success"] for event in events),
        "assisted_transformations_count": assisted_count,
        "stop_reasons": dict(stop_reasons),
        "action_types": dict(Counter(event["action_type"] for event in events)),
        "conditions": {
            mode: {
                "planned_tasks": sum(row["mode"] == mode for row in plan),
                "recorded_tasks": sum(row["mode"] == mode for row in results),
                "completed_tasks": sum(row["success"] for row in results if row["mode"] == mode),
                "observed_repairs": sum(row["observed_repair"] for row in results if row["mode"] == mode),
            } for mode in MODES
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RecurrentSoftwareAgent on 48 tool tasks")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model weights checkpoint")
    parser.add_argument("--corpus", type=Path, required=True, help="Path to directory containing requests/graders")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to tokenizer json")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for results")
    parser.add_argument("--allow-routing", action="store_true", default=True, help="Enable multi-slot routing and episodic fusion")
    parser.add_argument("--no-routing", dest="allow_routing", action="store_false", help="Disable routing (ablation baseline)")
    parser.add_argument("--assisted", action="store_true", default=False, help="Enable assisted action transformations (formatting adaptation and target binding)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks (for quick testing)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    # Verify input hashes
    for filename, expected in BANK.items():
        actual = digest(args.corpus / filename)
        if actual != expected:
            raise ValueError(f"Corpus file {filename} digest mismatch: {actual} != {expected}")
    tok_digest = digest(args.tokenizer)
    if tok_digest != TOKENIZER_HASH:
        raise ValueError(f"Tokenizer digest mismatch: {tok_digest} != {TOKENIZER_HASH}")

    # Resolve git commit
    try:
        import subprocess
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent).decode().strip()
    except Exception:
        git_commit = "unknown"

    device = torch.device(args.device)
    ckpt_digest = digest(args.checkpoint)
    print(f"Loading checkpoint {args.checkpoint} (SHA256: {ckpt_digest[:16]}...) onto {device}...")
    ckpt = torch.load(args.checkpoint, map_location=device)

    tier = ckpt.get("tier", "tier2")
    vocab_size = ckpt.get("vocab_size", 32000)
    use_skip = ckpt.get("use_token_skip", True)
    use_gated = ckpt.get("use_gated_token_skip", True)
    use_ptr = ckpt.get("use_pointer_copy", True)

    resolved_config = {
        "tier": tier,
        "vocab_size": vocab_size,
        "use_token_skip": use_skip,
        "use_gated_token_skip": use_gated,
        "use_pointer_copy": use_ptr,
        "compensated_state": ckpt.get("compensated_state", False),
        "pointer_mode": ckpt.get("pointer_mode", "sequential"),
        "retention_profile": ckpt.get("retention_profile", "legacy"),
        "allow_routing": args.allow_routing,
        "allow_assisted_transformations": args.assisted,
    }

    model = make_unified_model(
        tier=tier,
        vocab_size=vocab_size,
        use_token_skip=use_skip,
        use_gated_token_skip=use_gated,
        use_pointer_copy=use_ptr,
        use_routing=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully. Total parameters: {param_count} (< 36M invariant: {param_count < 36_000_000})")
    assert param_count < 36_000_000, f"Parameter ceiling violated: {param_count} >= 36M"

    tokenizer = BpeSemanticTokenizer(tokenizer_file=args.tokenizer, auto_build_if_missing=False)

    agent = RecurrentSoftwareAgent(
        model=model,
        tier=tier,
        vocab_size=vocab_size,
        device=device,
        allow_routing=args.allow_routing,
        allow_assisted_transformations=args.assisted,
    )
    agent.tokenizer = tokenizer

    requests = [json.loads(line) for line in (args.corpus / "requests.jsonl").read_text().splitlines()]
    graders = {row["source_id"]: row for row in [json.loads(line) for line in (args.corpus / "graders.jsonl").read_text().splitlines()]}

    if args.max_tasks is not None:
        requests = requests[:args.max_tasks]

    plan = [{"source_id": row["source_id"], "mode": mode} for row in requests for mode in MODES]
    write_json(args.output / "plan.json", plan)

    report = {
        "status": "running",
        "agent": "RecurrentSoftwareAgent",
        "allow_routing": args.allow_routing,
        "assisted": args.assisted,
        "parameter_count": param_count,
        "working_memory_bytes": 4096,
        "planned_tasks": len(plan),
        "device": str(device),
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": ckpt_digest,
        "tokenizer_sha256": tok_digest,
        "git_commit": git_commit,
        "resolved_config": resolved_config,
    }
    write_json(args.output / "report.json", report)

    results = []
    print(f"Starting evaluation across {len(plan)} tasks (allow_routing={args.allow_routing})...")

    with torch.inference_mode():
        for req_idx, request in enumerate(requests):
            for mode in MODES:
                idx = len(results)
                status_entry = {
                    "stage": "evaluating",
                    "index": idx,
                    "total": len(plan),
                    "source_id": request["source_id"],
                    "mode": mode,
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }
                write_json(args.output / "status.json", status_entry)

                case_start = time.monotonic()
                try:
                    row = run_tool_policy_case(agent, request, graders[request["source_id"]], mode=mode)
                    case_time = time.monotonic() - case_start

                    # Assert Law 1 invariant
                    assert row["episode"]["state_bytes"] == 4096, f"Law 1 violated: {row['episode']['state_bytes']}"

                    case_file = args.output / f"case-{idx:02d}.json"
                    write_json(case_file, row)
                    results.append(row)

                    print(
                        f"CASE {idx:02d}/{len(plan):02d} | ID: {row['source_id']} | Mode: {mode:12s} | "
                        f"Success: {str(row['success']):5s} | Repair: {str(row['observed_repair']):5s} | "
                        f"Stop: {row['episode']['stop_reason']:20s} | Time: {case_time:.1f}s",
                        flush=True,
                    )
                except Exception as exc:
                    print(f"ERROR on CASE {idx:02d} ({request['source_id']}, {mode}): {exc}", flush=True)
                    row = {
                        "source_id": request["source_id"],
                        "family_sha256": request.get("family_sha256", ""),
                        "mode": mode,
                        "success": False,
                        "observed_repair": False,
                        "repair_evidence": None,
                        "episode": {"stop_reason": f"error_{type(exc).__name__}", "state_bytes": 4096, "trace": []},
                        "events": [],
                        "final_files": {},
                        "error": str(exc),
                    }
                    write_json(args.output / f"case-{idx:02d}.json", row)
                    results.append(row)

                report["scores"] = aggregate(plan, results)
                write_json(args.output / "report.json", report)

    report["status"] = "complete"
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["scores"] = aggregate(plan, results)
    report["case_hashes"] = {p.name: digest(p) for p in args.output.glob("case-*.json")}
    write_json(args.output / "report.json", report)
    write_json(args.output / "status.json", {"stage": "complete", "elapsed_seconds": report["elapsed_seconds"]})

    print("\n" + "=" * 70)
    print("AUDITED TOOL POLICY EVALUATION COMPLETE")
    print("=" * 70)
    scores = report["scores"]
    print(f"Completed Tasks: {scores['completed_tasks']}/{scores['planned_tasks']} ({scores['completion_rate']:.2%})")
    print(f"Observed Repairs: {scores['observed_repairs']}/{scores['planned_tasks']} ({scores['repair_rate']:.2%})")
    print(f"Total Generated Attempts: {scores['total_generated_attempts']}")
    print(f"Rejected / Unexecuted Attempts: {scores['rejected_unexecuted_attempts']}")
    print(f"Executed Environment Actions: {scores['executed_actions']}")
    print(f"Recognized Verbs: {scores['recognized_action_verbs']}/{scores['executed_actions']}")
    print(f"Successful Actions: {scores['successful_environment_actions']}/{scores['executed_actions']}")
    print(f"Assisted Transformations Applied: {scores['assisted_transformations_count']}")
    print(f"Stop Reasons: {scores['stop_reasons']}")
    print(f"Total Elapsed Time: {report['elapsed_seconds']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
