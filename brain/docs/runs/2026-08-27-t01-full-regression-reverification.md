# T0-1: Full Regression Suite Re-verification (2026-08-27)

**Date:** 2026-08-27
**Status:** CLOSED — measured, recorded; suite NOT all-green at HEAD (expected structural, not a behavioral regression)
**Mode:** local CPU only, deterministic flags, single-thread, CUDA hidden
**Base HEAD:** `80f8a83`
**Runner:** `OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 py -3.11 -m unittest discover -s tests -p "test_*.py"`
**Wall:** 132.5 s
**Log:** `brain/scratch/t01_full_regression_20260827.log`

## Headline

The prior session's log claimed "1051 pass / 2 expected skips" for the full
suite. That claim was **UNVERIFIED on disk** (WORK_QUEUE.md T0-1 honesty note).
This re-run is the first measured full-suite result on the current tree:

| Metric | Value |
|---|---|
| Tests run | 1151 |
| Passed | 1143 |
| Skipped | 2 (expected: torch-optional + one torch-optional) |
| Failed | 3 |
| Errors | 3 |
| **All-green?** | **No — 6 non-green** |

None of the 6 is a behavioral regression of any model, baseline, or gate.
Every one is either (a) an environmental Windows failure, or (b) a
frozen-identity provenance canary that is *correctly* signaling the tree
advanced past a historical pin.

## Failure classification [MEASURED]

### Class A — Environmental Windows failures (2) — EXPECTED, documented

Both in `brain/tests/test_dgx_launch_contract.py`. They exercise POSIX
`realpath -m` / `/tmp` bash launch-contract functions that only resolve on the
Linux DGX host; on this Windows CPU host they hit
`FileNotFoundError: '\tmp\dgx-function-contract-…'`.

- `test_release_sync_requires_one_canonical_fixed_registration`
- `test_resume_preflight_refuses_terminal_failed_stage_gates`

These are **already documented as expected** in
`brain/docs/decisions/2026-08-26-local-cpu-canonical-compute.md` §4. They are
not regressions and must not be "fixed" by re-adding Linux-only paths.

### Class B — Stale frozen-identity provenance canaries (4) — CORRECT, NOT to be weakened

All four fail because the working tree has **advanced past a historical source
pin** that a sealed registration froze. The tests are behaving exactly as
designed (fail-closed on source drift). No gate, threshold, or baseline is
weakened here; these pins are the *old* tree state, not a scientific claim.

**B1 — V2.1i exact source-bundle pin (3 tests).**
`brain/scripts/v21i_strict_live_representation_probe_v1.py` and its two
dependents pin
`EXACT_V3_SOURCE_BUNDLE_SHA256 = 681f3f45422316ec40c299b6c449a90a4790f544ea21311597416cfa7fbb3862`
(the full `irene_brain` package + 2 pipeline scripts as of the V2.1i
diagnostic freeze, commit `38fa08f`).

Measured drift between the freeze and HEAD:

| File | Status |
|---|---|
| `brain/src/irene_brain/v2/embodied_interface.py` | **ADDED** (WIP commit `b94c920`, the embodied-interface program that started the current Pac-Man work) |
| all other bundle files | byte-identical to the freeze |

So the *only* file that moved the production bundle off the frozen sha is
`embodied_interface.py`. The canary fires by design because the embodied
program (Tier 1 of the owner's work queue) introduced a new module after the
V2.1i diagnostic era sealed its pin.

- `test_current_diagnostic_source_bundle_is_bound_to_v3_production`
  (`test_v21i_frozen_representation_continuation_diagnostic_v1.py`)
- `test_complete_strict_live_source_bundle_is_recomputed`
  (`test_v21i_strict_live_component_localization_v1.py`)
- `test_real_parent_and_registration_payload_validate_before_registration`
  (`test_v21j_fresh_bz_production_form_diagnostic_v1.py`)

**B2 — Baseline-architecture-manifest pin (1 test).**
`brain/tests/test_matched_baselines.py` re-hashes the 8 source files recorded
in `brain/configs/baseline-architecture-manifest.json` (frozen 2026-08-18).
Two have advanced past that manifest:

| File | Manifest sha (prefix) | Now | Last changed |
|---|---|---|---|
| `src/irene_brain/model/actuator.py` | `f6c7f9e5031a…` | `d2dad5e74a6f…` | `7af6c81` (RCQ-v3 recipe options) |
| `src/irene_brain/model/torch_model.py` | `5e4eceb721da…` | `8fa20a7eddf9…` | `3232f87` (Topological Goal Routing) |

- `test_checked_in_manifest_and_recipes_are_strictly_identified`
  (`test_matched_baselines.py`)

## Decision recorded [DECISION]

- **Do NOT re-freeze / weaken any of these pins.** Re-freezing the V2.1i
  source-bundle pin or the baseline-architecture manifest to the *current*
  tree would silently re-baseline sealed historical registrations — that is a
  scientific-identity / claim decision, not an autonomous maintenance task.
  The owner's contract forbids weakening a baseline and requires a fresh
  preregistration + create-only registration to change any sealed identity.
- **Do NOT add skip/lifecycle guards to these tests autonomously.** That would
  mask the drift signal. The correct path (if the pins are to be superseded)
  is a new decision record + a new registration that explicitly retires the
  old pin — out of scope for T0-1.
- **Preserve the negative result verbatim.** The suite is not all-green at
  HEAD; the 6 failures are structural and expected on this tree. This record
  is the authoritative T0-1 result and supersedes the prior session's
  unverified "1051 pass / 2 skips" claim.

## Consequence for the work queue

T0-1 is **closed**: the full regression suite was run to completion and the
result recorded with full classification. The 4 Class-B canaries are logged
as a known, correct, pre-existing condition. Tier 0 is now fully executed.
The next active item is **T1-1** (Pac-Man-like harness preregistration, v3 §4
contract). No gate was relaxed; no sealed identity was re-baselined.
