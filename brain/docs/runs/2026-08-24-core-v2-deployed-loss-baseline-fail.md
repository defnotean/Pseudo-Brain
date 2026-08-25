# Stage V2.0 Core V2 deployed-loss baseline — terminal failure

**Date:** 2026-08-24  
**Preregistration:** `brain/docs/preregistrations/2026-08-24-core-v2-deployed-loss-baseline-prereg.md`  
**Verdict:** OUTCOME GAMMA for both arms. Do not rerun this design.

## Spark execution evidence

The bounded DGX Spark container completed all eight registered 6,000-step
runs, then exited non-zero because the unprivileged container user could not
write the final JSON into a root-owned host output directory. The aggregate
results remain in the preserved container logs; per-seed curves and the final
JSON were not recovered and must not be reconstructed.

| Field | Observed value |
|---|---|
| Container | `v20run` / `37f6ded79b11...` |
| Image SHA | `177a406d7cb2...` |
| Start | `2026-08-24T14:18:51Z` |
| Finish | `2026-08-24T16:23:32Z` |
| Exit | `1` after result computation |
| V2-A mean lift / sigma | `-0.3567 / 0.0000` |
| V2-A delta vs V1 R=-0.2130 | `-0.1437` — GAMMA regression |
| V2-C mean lift / sigma | `-0.3567 / 0.0000` |
| V2-C delta vs V1 R=-0.2130 | `-0.1437` — GAMMA regression |
| Artifact error | `PermissionError: /workspace/run/v20_results.json` |

The artifact-write failure is an infrastructure defect, not a reason to
discard or rerun the scientific failure. Future output directories must be
created by the same UID that runs the container, and the smoke stage must
prove result persistence before a full run is allowed.

## Root cause

The registered aggregator did not provide the learning contract claimed by
the preregistration. It computed one scalar consequence utility per slot,

`U_k = reward_k - 3 * hazard_k`,

then formed each action value from that utility weighted by slot action
probabilities. Both registered configurations initialize reward and hazard at
exactly zero. Therefore every deployed action value is exactly zero, the
deployed policy is uniform, and deployed cross entropy has no gradient path
through the action probabilities.

A deterministic local autograd reproduction on the frozen legacy path found:

- loss `1.60943794` (`log(5)`),
- all five deployed action values exactly `0`, and
- `0/56` parameters with a non-zero gradient.

The identical V2-A and V2-C held-out results are consistent with this
mechanism: both deployed policies remained the same constant action policy.

## Corrective hypothesis

Each exchangeable thoughtlet must emit **action-specific decision evidence**
directly. The deployed policy should be a smooth permutation-invariant
aggregate of those action logits. Unsupervised reward, hazard, confidence,
existence, and branch heads must remain diagnostic until grounded targets
make their semantics identifiable.

This is tested as a new experiment, Stage V2.0b. The historical scalar-utility
path remains selectable only to reproduce this failure.

