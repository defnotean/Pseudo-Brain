# Stage V2.0b direct set policy smoke — terminal partial failure

**Date:** 2026-08-24  
**Release:** `r20260824t201824z-50f1ceb6c031`  
**Archive SHA-256:** `50f1ceb6c031dd01513e3779c9e9c25665b1cdeba5d2233877d8aecb591a3b4f`  
**Container:** `v20b-smoke-20260824` / `d4287a916cfe...`  
**Verdict:** smoke FAIL. The 6,000-step confirmatory run was not launched.

## Frozen smoke result

The immutable release matched all five preregistered source SHA-256 values.
The container used the pinned image `177a406d7cb2`, no network, 12 CPUs,
64 GiB RAM, PID limit 512, and UID/GID 1000. GPU temperature remained at or
below 43°C in observed snapshots.

V2-A passed every behavioral smoke condition:

- initial/final training loss `1.65076 -> 1.26842`,
- held-out mean lift `-0.0100`, delta `+0.2030` vs frozen V1 R,
- three deployed actions with histogram `[17, 245, 488, 0, 0]`, and
- 56.8 seconds wall time.

V2-C failed:

- held-out mean lift remained the constant-policy `-0.3567`,
- action histogram was `[750, 0, 0, 0, 0]`,
- final training loss was NaN, and
- strict `allow_nan=False` JSON serialization stopped at that field.

The output directory was correctly owned by `defnotean:defnotean`, and the
partial `.tmp` artifact is preserved. Thus the V2.0 permission defect was
fixed; this non-zero exit is a model/numerics failure. The writer should have
persisted a structured non-finite record instead of encountering NaN during
serialization; Stage V2.0c adds fail-fast handling.

## Causal diagnosis

The failure reproduced on CPU: V2-C started at `1.61169` and was NaN before
update 100. A new fail-fast diagnostic isolated the mechanism:

| Arm | First failure | Evidence |
|---|---|---|
| consequence-only, legacy recurrence | backward at update 9 | thoughts `2.589e10`, logits `1.260e9`, loss `4.206e8` |
| world + prediction-error feedback, legacy recurrence | backward at update 15 | thoughts `2.023e8`, logits `1.092e7`, loss `2.615e6` |

Prediction error is not necessary for the failure. The historical BrainCell
adds five independent sigmoid-gated branches. One cycle can approach
`2*thought + 2*candidate`; three cycles over a long sequence are expansive.

Normalizing only the thought update kept thoughts finite for 120 updates, but
then exposed a second unbounded recurrence: belief magnitude reached
`299229.44` under `old + gate*delta`. With both corrections enabled, the full
V2-C chain completed 120 CPU updates with finite gradients and parameters:

- maximum observed belief magnitude `0.41733`,
- maximum observed thought magnitude `7.98662`,
- maximum observed prediction-error magnitude `1.12263`, and
- final diagnostic loss `0.93998`.

The corrected dynamics are a distinct Stage V2.0c experiment. No V2.0b gate
or result was rewritten.

