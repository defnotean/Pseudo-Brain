# Phase 2.6: torture-suite baseline — first results (2026-08-22)

**Core frozen at tag `phase26-baseline-core` (84009fd).** All 25 task builders
verified (deterministic, chance-calibrated). Runner: `run_torture_suite.py`;
data: Spark `runs/torture-baseline/*.json`.

## Baseline runs [MEASURED]

| Configuration | Mean lift over chance | Tasks >+2% lift | Notes |
|---|---|---|---|
| PB K=32, fresh init | −0.246 | 1/24 | wait-rate ~40–60% |
| GRU, fresh init | −0.333 | 2/24 | never waits |
| PB K=32, surviving escalation checkpoint (seed 742) | −0.380 | 0/24 | waits 70–100% (!) |

## What this tells us [INFERRED]

1. **The untrained/fresh baselines are below chance everywhere**, as expected for
   argmax-decision scoring with untrained heads: the models have no pressure yet
   to use frames {1,2,3} as answer tokens. This is fine — these are *reference
   zeros*, not judgments of architecture quality.

2. **The trained escalation checkpoint transfers nothing to any torture task and
   waits almost always** (70–100%). Its competence is entirely bound to the
   escalation task's frame statistics; outside that distribution it emits WAIT.
   [MEASURED] Zero-shot generalization from the Phase 2 task to the torture
   suite is nonexistent — which is exactly what the roadmap's Phase 3 question
   will probe after Core V1 hardening.

3. **Suite validity checks pass:** fresh models sit near/below chance rather than
   accidentally solving tasks (no leakage); per-task chance floors behave; wait
   rates differentiate architectures (PB idles by default, GRU commits).

## Decision taken

Fresh-init numbers are the correct **architecture-native baseline** for
before/after comparison of Core changes (they measure information extraction
without training confounds), but they cannot detect improvements in *learning*,
only in architecture. Therefore the locked baseline must ALSO include
trained-on-suite numbers. Adding a canonical training recipe on a composite of
the suite tasks (multi-task training) as baseline part 2 before locking.
