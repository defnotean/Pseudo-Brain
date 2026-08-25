"""Stage V2.0c: direct set policy with bounded recurrent state updates."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from irene_brain.evaluation.torture_suite import TASKS
from irene_brain.v2 import CONFIG_A_DECISION_ONLY, CONFIG_C_FULL, CoreV2Config
from run_provenance import apply_deterministic_mode, bank_digest, build_bank_pinned
from stage_v20_core_v2_deployed_loss_baseline import (
    REFERENCE_R,
    SEEDS,
    classify,
    run_v2_arm,
)


BANK_DIGEST = "b3bb5fc33fd5f605"
SMOKE_STEPS = 400
FULL_STEPS = 6000


def _summary(records: list[dict]) -> dict:
    completed = [record for record in records if record.get("status") == "completed"]
    if len(completed) != len(records):
        return {"runs": records, "status": "training_failure"}
    lifts = [record["final_eval"]["mean_lift"] for record in completed]
    mean_lift = float(np.mean(lifts))
    sigma = float(np.std(lifts))
    classification, delta = classify(mean_lift, sigma)
    return {
        "runs": records,
        "status": "completed",
        "mean_lift": round(mean_lift, 4),
        "sigma": round(sigma, 4),
        "delta_vs_V1": delta,
        "classification": classification,
    }


def _smoke_passed(arms: dict) -> bool:
    for arm in arms.values():
        if arm.get("status") != "completed" or len(arm["runs"]) != 1:
            return False
        record = arm["runs"][0]
        initial = record["initial_probe_loss"]
        final = record["final_probe_loss"]
        if not (
            np.isfinite(initial)
            and np.isfinite(final)
            and final < 1.45
            and final <= 0.90 * initial
            and record["final_eval"]["unique_actions"] >= 2
            and record["max_abs_belief"] <= 1.00001
            and record["max_abs_thought"] <= 11.0
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    apply_deterministic_mode()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    digest = bank_digest(build_bank_pinned(42, TASKS))
    if digest != BANK_DIGEST:
        raise RuntimeError(f"bank digest drift: {digest}")

    config = CoreV2Config(
        decision_aggregation="direct_mean_logits_v1",
        braincell_dynamics="normalized_mixture_v1",
        belief_dynamics="convex_gated_v1",
    )
    seeds = [42] if args.smoke else SEEDS
    steps = SMOKE_STEPS if args.smoke else FULL_STEPS
    result = {
        "schema_version": 1,
        "preregistration": "2026-08-24-core-v2-normalized-recurrence-prereg",
        "mode": "smoke" if args.smoke else "confirmatory",
        "decision_aggregation": config.decision_aggregation,
        "braincell_dynamics": config.braincell_dynamics,
        "belief_dynamics": config.belief_dynamics,
        "reference_R": REFERENCE_R,
        "bank_digest": digest,
        "steps": steps,
        "seeds": seeds,
        "arms": {},
    }

    for flags, name in (
        (CONFIG_A_DECISION_ONLY, "V2A_normalized"),
        (CONFIG_C_FULL, "V2C_normalized"),
    ):
        records = []
        for seed in seeds:
            print(f"[{name}] seed {seed} steps={steps} ...", flush=True)
            try:
                records.append(run_v2_arm(
                    flags, name, seed, device, digest,
                    config=config, total_steps=steps,
                ))
            except FloatingPointError as error:
                records.append({
                    "arm": name,
                    "seed": seed,
                    "status": "nonfinite",
                    "error": str(error),
                })
                break
        result["arms"][name] = _summary(records)
        summary = result["arms"][name]
        if summary["status"] == "completed":
            print(
                f"[{name}] mean={summary['mean_lift']:.4f} "
                f"sigma={summary['sigma']:.4f} "
                f"delta={summary['delta_vs_V1']:+.4f} "
                f"{summary['classification']}",
                flush=True,
            )
        else:
            print(f"[{name}] TRAINING FAILURE", flush=True)

    if args.smoke:
        result["smoke_gate"] = {
            "maximum_final_loss": 1.45,
            "minimum_relative_loss_reduction": 0.10,
            "minimum_unique_deployed_actions": 2,
            "maximum_absolute_belief": 1.00001,
            "maximum_absolute_thought": 11.0,
            "passed": _smoke_passed(result["arms"]),
        }

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(f"DONE -> {output}", flush=True)

    failed = any(arm["status"] != "completed" for arm in result["arms"].values())
    if failed or (args.smoke and not result["smoke_gate"]["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
