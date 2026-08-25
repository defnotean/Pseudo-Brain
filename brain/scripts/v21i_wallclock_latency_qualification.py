"""Run the frozen V2.1i CPU wall-clock gate for one exact checkpoint."""
from __future__ import annotations

import os

# These must be set before importing torch through the qualification module.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path

from irene_brain.evaluation.v21_wallclock_latency import (
    DEVELOPMENT_ENVELOPE,
    POST_DGX_ENVELOPE,
    WallClockQualificationFailed,
    qualify_v21i_checkpoint_wallclock_create_only,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure one exact V2.1i outcome-aware checkpoint on one CPU thread "
            "for 100 warmups plus 5,000 ticks in each of three repetitions."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-source-bundle-sha256", required=True)
    parser.add_argument("--expected-model-seed", required=True, type=int)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument(
        "--checkpoint-envelope",
        choices=(DEVELOPMENT_ENVELOPE, POST_DGX_ENVELOPE),
        default=DEVELOPMENT_ENVELOPE,
    )
    parser.add_argument("--post-dgx-release-receipt")
    parser.add_argument("--expected-post-dgx-release-receipt-sha256")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = qualify_v21i_checkpoint_wallclock_create_only(
            checkpoint_path=Path(args.checkpoint),
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_source_bundle_sha256=args.expected_source_bundle_sha256,
            expected_model_seed=args.expected_model_seed,
            preregistration_sha256=args.preregistration_sha256,
            output_path=Path(args.output),
            checkpoint_envelope=args.checkpoint_envelope,
            post_dgx_release_receipt_path=args.post_dgx_release_receipt,
            expected_post_dgx_release_receipt_sha256=(
                args.expected_post_dgx_release_receipt_sha256
            ),
        )
    except WallClockQualificationFailed as error:
        print(json.dumps(error.report.receipt(), indent=2, sort_keys=True))
        raise SystemExit(1) from error
    print(json.dumps(report.receipt(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
