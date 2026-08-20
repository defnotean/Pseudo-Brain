# Phase 2.5 CPU diagnostics: belief collapse, VoI, multi-horizon arena (2026-08-20)

Status: instrumentation. Local CPU-only verification; no Spark train, no
RCQ-v2 resume, no TEST range, no rcq-v3 registration.

These diagnostics sit on the thought-mediated Phase 2.5 stack. They measure
whether parallel hypotheses actually collapse, persist, and change action
under delayed, nested, and non-stationary uncertainty — not whether a GPU
campaign has already been won.

## What landed

- **Belief collapse / VoI** (`evaluation/belief_collapse_diagnostics.py`):
  pre/post entropy, posterior surge vs suppression, WAIT vs detour vs
  gamble breakdown, and a closed-form WAIT-beats-RIGHT example
  (`Q(WAIT)=+1.8` vs `Q(RIGHT)=-5.5`).
- **Multi-horizon ambiguity arena** (`environments/multi_horizon_ambiguity_arena.py`):
  staggered H=4 hazards, delayed resolution (10–30 ticks), nested 4→2→1
  factorized uncertainty, and an 80/20→20/80 regime shift.
- **Sequential probe resolution** on the stochastic occluded suite: WAIT at
  t=0, then resolve after the ghost turn is revealed.
- **Stochastic branch targets** now treat `_maze` as walkable cells (not
  walls) and emit 10 branches (2 WAIT + 4 actions × 2 ghost outcomes) with
  action-conditional expected utilities.
- **Multiplicity-proof aggregator** (`evaluation/multiplicity_aggregator.py`):
  Q(a) is a probability-weighted mean over slots that propose action `a`,
  so duplicate RIGHT slots cannot outvote a single higher-utility LEFT.
- Campaign harness `dgx_run_thought_mediated_campaign.py` gained a
  `--cpu-smoke` path for these evals. The full K-sweep training job stays
  Spark-only. Do not launch it from this workstation.

## Local verification

Play-safe discovers every `tests/test_*.py` module. New coverage:

- `test_analytic_voi.py`
- `test_belief_collapse.py`
- `test_multi_horizon_arena.py`
- `test_multiplicity_aggregator.py`
- `test_sequential_training.py`
- sequential probe case in `test_stochastic_occluded_benchmark.py`

Local play-safe on 2026-08-20: **66 modules, 707 tests, 2 expected skips** (CUDA unavailable; Linux dirfd/flock), runner exit 0.

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\brain\scripts\run_play_safe_tests.ps1
```

Optional CPU-only campaign smoke (no training):

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Resolve-Path .\brain\src).Path
C:\Users\Demon\AppData\Local\Programs\Python\Python311\python.exe brain\scripts\dgx_run_thought_mediated_campaign.py --cpu-smoke --output-json $env:TEMP\phase25_cpu_smoke.json
```
