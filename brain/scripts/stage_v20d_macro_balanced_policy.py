"""Stage V2.0d: normalized recurrent policy with macro-balanced deployed loss."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os

import numpy as np
import torch

from irene_brain.evaluation.torture_suite import TASKS
from irene_brain.v2 import CONFIG_A_DECISION_ONLY, CONFIG_C_FULL, CoreV2Config
from run_provenance import apply_deterministic_mode, bank_digest, build_bank_pinned
from stage_v20_core_v2_deployed_loss_baseline import (
    REFERENCE_R, SEEDS, build_order, classify, run_v2_arm,
)


BANK_DIGEST = "b3bb5fc33fd5f605"
EXPOSURE_COUNTS = (24, 236, 268, 111, 113)
SMOKE_STEPS = 400
FULL_STEPS = 6000


def macro_weights() -> tuple[float, ...]:
    total = float(sum(EXPOSURE_COUNTS))
    return tuple(total / (len(EXPOSURE_COUNTS) * count)
                 for count in EXPOSURE_COUNTS)


def summarize(records: list[dict]) -> dict:
    completed = [record for record in records if record.get("status") == "completed"]
    if len(completed) != len(records):
        return {"runs": records, "status": "training_failure"}
    lifts = [record["final_eval"]["mean_lift"] for record in records]
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


def smoke_passed(arms: dict) -> bool:
    for arm in arms.values():
        if arm.get("status") != "completed" or len(arm["runs"]) != 1:
            return False
        record = arm["runs"][0]
        initial = record["initial_probe_loss"]
        final = record["final_probe_loss"]
        evaluation = record["final_eval"]
        if not (
            np.isfinite(initial)
            and np.isfinite(final)
            and final < 1.45
            and final <= 0.90 * initial
            and evaluation["unique_actions"] >= 3
            and evaluation["balanced_accuracy"] >= 0.25
            and record["max_abs_belief"] <= 1.00001
            and record["max_abs_thought"] <= 16.0
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
    bank = build_bank_pinned(42, TASKS)
    digest = bank_digest(bank)
    if digest != BANK_DIGEST:
        raise RuntimeError(f"bank digest drift: {digest}")
    exposure = Counter(bank[index].label for _, group in build_order(bank)
                       for index in group)
    observed_counts = tuple(exposure[action] for action in range(5))
    if observed_counts != EXPOSURE_COUNTS:
        raise RuntimeError(f"exposure-count drift: {observed_counts}")

    weights = macro_weights()
    config = CoreV2Config(
        decision_aggregation="direct_mean_logits_v1",
        braincell_dynamics="normalized_mixture_v1",
        belief_dynamics="convex_gated_v1",
    )
    seeds = [42] if args.smoke else SEEDS
    steps = SMOKE_STEPS if args.smoke else FULL_STEPS
    result = {
        "schema_version": 1,
        "preregistration": "2026-08-24-core-v2-macro-balanced-policy-prereg",
        "mode": "smoke" if args.smoke else "confirmatory",
        "decision_loss": "inverse_padded_exposure_macro_ce_v1",
        "class_exposure_counts": EXPOSURE_COUNTS,
        "class_weights": weights,
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
        (CONFIG_A_DECISION_ONLY, "V2A_macro_balanced"),
        (CONFIG_C_FULL, "V2C_macro_balanced"),
    ):
        records = []
        for seed in seeds:
            print(f"[{name}] seed {seed} steps={steps} ...", flush=True)
            try:
                records.append(run_v2_arm(
                    flags, name, seed, device, digest,
                    config=config,
                    total_steps=steps,
                    class_weights=weights,
                    full_bank_probe=True,
                ))
            except FloatingPointError as error:
                records.append({
                    "arm": name,
                    "seed": seed,
                    "status": "nonfinite",
                    "error": str(error),
                })
                break
        result["arms"][name] = summarize(records)
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
            "maximum_final_macro_probe_loss": 1.45,
            "minimum_relative_macro_probe_reduction": 0.10,
            "minimum_unique_deployed_actions": 3,
            "minimum_balanced_accuracy": 0.25,
            "maximum_absolute_belief": 1.00001,
            "maximum_absolute_thought": 16.0,
            "passed": smoke_passed(result["arms"]),
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
