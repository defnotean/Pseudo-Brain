# Phase 2.6: corrected PB baseline (per-slot CE) — baseline re-locked (2026-08-22)

**Reason:** the first locked baseline trained PB through a slot-averaged decision head
(`action_logits.mean(dim=1)`), which differs from the Phase-2-style per-slot CE used
by every escalation campaign. That disadvantaged PB in a way the constitution
(C10 resource/protocol honesty) does not allow in a reference baseline.

**Change:** training loss now replicates the label across all K slots and applies
per-slot CE (`action_logits.view(-1,5)`), exactly like the escalation trainer.
Evaluation protocol unchanged (argmax of `action_dist` at the decision frame —
which is the consequence aggregator's output, not the training head).

## Corrected canonical baseline [MEASURED]

| Configuration | Mean lift | Tasks >+2% |
|---|---|---|
| PB K=32 fresh | −0.246 | 1/24 |
| GRU fresh | −0.333 | 2/24 |
| PB K=32 multitask-trained, **per-slot CE (corrected)** | **−0.215** | **1/25** |
| GRU multitask-trained (4000 steps, unchanged protocol) | −0.042 | 8/25 |

## Reading [LABELS]

[MEASURED] The per-slot CE head improves PB only marginally (−0.245 → −0.215).
The gap to GRU (−0.042) is therefore **not primarily a training-head artifact**:
under the fair, Phase-2-faithful recipe, PB still learns the torture tasks far
more slowly than the GRU at this scale/recipe.

[INFERRED] This is itself a significant Phase 2.6 audit finding: PB substantially
underperforms GRU in light-budget multitask sample efficiency on the locked
torture-suite baseline. Candidate explanations to test in the BrainCell/lifecycle
workstreams: (a) 32 slots dilute gradient signal per task; (b) the consequence
aggregator needs task-set conditioning it doesn't have here; (c) 4000 steps is
far too few for K=32 (escalation used 3000 steps for ONE task).

## Old result preserved

Slot-averaged-head run: `runs/torture-baseline/torture_pb_trained.json`
[DIAGNOSTIC / SUPERSEDED BASELINE] — kept for the record, not the reference.

## Baseline re-locked

Reference numbers for all Core V1 before/after comparisons:
- PB K=32: **−0.215 mean lift, 1/25 tasks** (per-slot CE, 4000 steps, seed 42)
- GRU: **−0.042 mean lift, 8/25 tasks** (unchanged)

Headroom confirmed: both far from ceiling; suite discriminates trained-vs-fresh.
Proceeding to Episodic Memory v0 on this baseline.
