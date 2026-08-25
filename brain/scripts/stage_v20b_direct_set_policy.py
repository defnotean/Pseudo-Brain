"""Stage V2.0b: direct set-policy repair for the failed V2.0 contract.

The default 6,000-step/four-seed execution is confirmatory.  ``--smoke`` is
the preregistered one-seed, 400-step infrastructure and learning gate.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from run_provenance import apply_deterministic_mode, bank_digest, build_bank_pinned
from irene_brain.evaluation.torture_suite import TASKS
from irene_brain.v2 import CONFIG_A_DECISION_ONLY, CONFIG_C_FULL, CoreV2Config
from stage_v20_core_v2_deployed_loss_baseline import (
    REFERENCE_R,
    SEEDS,
    classify,
    run_v2_arm,
)

BANK_DIGEST = "b3bb5fc33fd5f605"
SMOKE_STEPS = 400
FULL_STEPS = 6000


def _arm_summary(records: list[dict]) -> dict:
    lifts = [record["final_eval"]["mean_lift"] for record in records]
    mean_lift = float(np.mean(lifts))
    sigma = float(np.std(lifts))
    classification, delta = classify(mean_lift, sigma)
    return {
        "runs": records,
        "mean_lift": round(mean_lift, 4),
        "sigma": round(sigma, 4),
        "delta_vs_V1": delta,
        "classification": classification,
    }


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

    config = CoreV2Config(decision_aggregation="direct_mean_logits_v1")
    seeds = [42] if args.smoke else SEEDS
    steps = SMOKE_STEPS if args.smoke else FULL_STEPS
    result = {
        "schema_version": 1,
        "preregistration": "2026-08-24-core-v2-direct-set-policy-prereg",
        "mode": "smoke" if args.smoke else "confirmatory",
        "decision_aggregation": config.decision_aggregation,
        "reference_R": REFERENCE_R,
        "bank_digest": digest,
        "steps": steps,
        "seeds": seeds,
        "arms": {},
    }

    for flags, name in (
        (CONFIG_A_DECISION_ONLY, "V2A_direct_mean"),
        (CONFIG_C_FULL, "V2C_direct_mean"),
    ):
        records = []
        for seed in seeds:
            print(f"[{name}] seed {seed} steps={steps} ...", flush=True)
            records.append(run_v2_arm(
                flags, name, seed, device, digest,
                config=config, total_steps=steps))
        result["arms"][name] = _arm_summary(records)
        summary = result["arms"][name]
        print(
            f"[{name}] mean={summary['mean_lift']:.4f} "
            f"sigma={summary['sigma']:.4f} "
            f"delta={summary['delta_vs_V1']:+.4f} "
            f"{summary['classification']}",
            flush=True,
        )

    if args.smoke:
        smoke_records = [arm["runs"][0] for arm in result["arms"].values()]
        losses = [record["final_loss"] for record in smoke_records]
        initial_losses = [record["initial_loss"] for record in smoke_records]
        unique_actions = [
            record["final_eval"]["unique_actions"] for record in smoke_records
        ]
        learned = [
            initial is not None and final is not None
            and np.isfinite(initial) and np.isfinite(final)
            and final < 1.45 and final <= 0.90 * initial
            for initial, final in zip(initial_losses, losses)
        ]
        result["smoke_gate"] = {
            "maximum_final_loss": 1.45,
            "minimum_relative_loss_reduction": 0.10,
            "minimum_unique_deployed_actions": 2,
            "observed_initial_losses": initial_losses,
            "observed_final_losses": losses,
            "observed_unique_deployed_actions": unique_actions,
            "passed": all(learned) and all(count >= 2 for count in unique_actions),
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

    if args.smoke and not result["smoke_gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
