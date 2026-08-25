# PB21L NZ upmix consumed feasibility — sealed infrastructure failure

**Date:** 2026-08-25  
**Mode:** `pb21l_nz_upmix_consumed_feasibility_v1`  
**Classification:** `exploratory_consumed_data_failed_nonqualifying`  
**Verdict:** terminal infrastructure failure; no scientific PB21L result

PB21L is sealed. Its create-only attempt exists and `retry_allowed` is false.
The process consumed the registered PB21K DEV endpoint, then failed while
constructing authoritative evidence. No evidence NPZ exists, no authoritative
metrics were computed or published, and no checkpoint, candidate, nomination,
qualification, or fresh-data architecture license was emitted. The absence of
published evidence does not make DEV fresh again.

## Immutable artifacts and hashes

| Artifact | Canonical path | SHA-256 |
|---|---|---|
| Registration | `brain/runs/v21l-diagnostics/2026-08-25-pb21l-nz-upmix-consumed-v1.registration.json` | `069ff428f84b772a7b853637248d45deb225738794f918a5e8887bce9f8937f0` |
| Attempt | `brain/runs/v21l-diagnostics/2026-08-25-pb21l-nz-upmix-consumed-v1.attempt.json` | `7a361cb03cb25c3699ae3b6fe2dcea26e1dd992d3217ea7026dc7b7de8935156` |
| Failure result | `brain/runs/v21l-diagnostics/2026-08-25-pb21l-nz-upmix-consumed-v1.json` | `eaf7bc1565666e8ddc371f02643b7e79a7b2c7a010798d35731fb251ffa2ad1c` |
| Registered PB21L source bundle | embedded in registration | `21ba2a450fcf2106b60dd596ebc25cf1f3ccbb6cd7f1d2ff935281601f243122` |
| Registered PB21L runner | `brain/scripts/v21l_nz_upmix_consumed_feasibility_v1.py` | `5d63309dbde1213d78da423f356e6a0879777a98fbd936ec012b70c403586699` |
| Registered PB21L focused tests | `brain/tests/test_v21l_nz_upmix_consumed_feasibility_v1.py` | `20bdfde82fa7b03d8c5341d17b04fa7997a0766fd6c230a70e1924bf684a2a0f` |
| Registered PB21L preregistration | `brain/docs/preregistrations/2026-08-25-pb21l-nz-upmix-consumed-v1.md` | `8dc1e186fdb57f0e982b0f8b361a589f2e9cb57ad56d8bf2b9f0e0a8bdb92934` |

There is deliberately no evidence-artifact hash: the canonical evidence NPZ
was never published. The failure result records `authoritative_evidence: null`,
`candidate_publication_allowed: false`, `qualification_claimed: false`, and
`retry_allowed: false`.

The frozen PB21K parent identities retained by the attempt are:

| Parent artifact | SHA-256 |
|---|---|
| Registration | `f6bc3e25412ea6c3058bc38ace7622e30fdec72c6732388cb7a7d7dfa38f7058` |
| Attempt | `bd2ae53311733900a421b9928f2db73f32e5aa82d14a523f27f7bcf367b2a094` |
| Evidence | `9d57bf4a6bd60121e8f11e2b9405c1a6d64a9346381503054d6ad9cd4ece385c` |
| Result | `083ac9c44d3327f4f806b1e4842ea3991ec54e5bab65992e7c5446ad09ca02d4` |
| Source bundle | `b13b764c2ff55dd15649945fec7c05211a3c59fc04c7ecfb667d1b711590e03a` |

## What executed and what did not

The runner completed the registered FIT training, pruning, and TRAIN-CAL AA
fitting path. It then reconstructed the consumed DEV tape and computed the raw
and calibrated DEV output arrays in memory. The next stage called
`build_evidence_arrays`; that stage failed before evidence publication. The
authoritative reload and `evaluate_authoritative_evidence` stages therefore
never ran.

Consequently:

- DEV is consumed and permanently unavailable to a retry or later decision;
- there is no authoritative PB21L evidence or metric table;
- no BAL-UPMIX versus POOL-UPMIX decision exists;
- no observed in-memory value may be reconstructed, reported, or used to tune
  PB21M; and
- PB21L remains nonqualifying and cannot be rerun under another filename.

## Root cause

The failure is exactly:

```text
AttributeError: 'PerActionAffineHazardCalibrator' object has no attribute 'provenance'
```

`build_evidence_arrays` treated each real
`PerActionAffineHazardCalibrator` as if it retained a nested `.provenance`
object. The real component instead copies provenance into flat immutable fields
such as `source_namespace`, `calibration_group_digest`,
`upstream_checkpoint_sha256`, and `source_bundle_sha256`. The runtime created a
valid separate `TrainCalibrationProvenance`, passed it into AA fitting, then
stored only the returned calibrator before evidence construction.

The 22 focused tests did not execute the real producer-to-consumer boundary.
They tested loss formulas, schema, tamper rejection, and provenance validators
against synthetic arrays, but never called `build_evidence_arrays`, `_run_impl`,
or `run` with a real AA calibrator. The evidence-builder parameter and runtime
matrix were typed as `object`, so static attribute checking could not reject the
invalid `.provenance` access.

## Mandatory regression before another registration

Before PB21M or any successor is registered, a no-lifecycle integration smoke
must pass using real components and synthetic fixture data only. It must:

1. create a real `TrainCalibrationProvenance` with valid disjoint groups and
   hashes;
2. call the production AA fitter and obtain a real
   `PerActionAffineHazardCalibrator`, with no mock, `SimpleNamespace`, or
   fabricated attribute;
3. pass real calibrators through the exact production evidence-building path
   for every scorer/cell slot;
4. publish only to a temporary directory, reopen with `allow_pickle=False`,
   validate every flat provenance field, and deterministically refit AA with
   exact scales, biases, and acceptance;
5. reach the production numerical-decision entry point; and
6. prove that no canonical registration, attempt, evidence, result, checkpoint,
   or reserved scientific partition was opened or written.

The evidence path must use the concrete calibrator/provenance types rather than
`object`, and the applicable static type check plus this integration smoke must
be green before source-bundle hashing and registration. Passing isolated unit
tests is not sufficient.

## Next experiment: fresh PB21M CAL-futility

The next scientific action is a separately preregistered PB21M fresh-data
CAL-futility screen, not a PB21L retry. PB21M must disclose this infrastructure
failure and the absence of authoritative PB21L metrics. It must use mutually
disjoint fresh FIT and CAL data and reserve an untouched fresh DEV endpoint;
the PB21K/PB21L DEV endpoint is forbidden.

PB21L supplies no arm selection. PB21M must either freeze one arm before data
access or carry both arms under fixed precedence. Its CAL rule is a one-way,
preregistered futility stop with immutable support and threshold rules. A CAL
failure means only that the arm failed that screen; a CAL pass is not scientific
success and may only unlock the already-reserved, one-shot fresh DEV stage.
There is no adaptive threshold, arm change, or retry.
