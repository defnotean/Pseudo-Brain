# Stage V2.0j accumulated macro smoke — PASS

**Date:** 2026-08-24  
**Release:** `r20260824t214353z-211184c0cebe`  
**Archive SHA-256:** `211184c0cebe408cf49c74a4936212a9d92ad04e7ec9bde62e001c1b9e2ae70b`  
**Container:** `v20j-smoke-20260824` / `e91dcdef0c75...`  
**Artifact:** `runs/v20j-smoke-20260824/v20j_smoke_results.json`  
**Verdict:** all frozen smoke gates PASS; confirmatory run authorized.

The strict JSON parsed successfully, the container exited 0 without OOM, and
both 32-cycle arms completed finite with an unprivileged, atomically persisted
artifact.

| Arm | Macro probe | Train actions | Held-out balanced acc. | Held-out actions | Lift | Delta vs V1 |
|---|---:|---:|---:|---:|---:|---:|
| V2-A accumulated | `1.87822 -> 1.20306` | 5 | `0.4924` | 5 | `-0.0660` | `+0.1470` |
| V2-C accumulated | `1.61324 -> 1.39621` | 5 | `0.5005` | 5 | `-0.0873` | `+0.1257` |

V2-A held-out recall was `[1,.3308,.2795,.1538,.6977]`; V2-C was
`[1,.0722,.3737,.3590,.6977]`. Both one-seed lifts classify ALPHA against the
frozen Core V1 reference, but confirmation is required before a replicated
claim.

State safety margins were large:

- V2-A belief/thought maxima: `0.20663` / `10.47984`;
- V2-C belief/thought maxima: `0.18578` / `10.28505`.

The smoke establishes a working supervised optimizer contract: exact macro
weights, complete-cycle gradient accumulation, one global clip, deterministic
reshuffling, and lr `1e-4`. It does not establish useful consequence learning,
memory advantage, or live game competence.

The unchanged immutable release was launched for the preregistered 128-cycle,
four-seed confirmatory run as `v20j-confirmatory-20260824`.

