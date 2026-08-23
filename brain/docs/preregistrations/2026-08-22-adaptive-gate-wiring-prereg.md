# Preregistration: Adaptive Thought-Update Gate — minimal wiring experiment

**Frozen:** 2026-08-22, after the wire-or-descope audit
(`PHASE2_6_GATE_HALTING_AUDIT.md`), BEFORE any core modification.
**Justification:** the audit found the existing gate orphaned from the canonical
core but aimed exactly at the one replicated measured weakness (evidence-intake
deficit). This experiment tests whether THAT mechanism, wired minimally into the
canonical VectorizedPseudoBrain, changes intake behavior.

## Design

Two arms only, everything else frozen:
- **Control:** canonical VectorizedPseudoBrain K=32 W=120 C=3 (untouched).
- **Gate-wired:** identical core + `AdaptiveThoughtUpdateGate` inserted as the
  thought-refresh blend between cycles (replacing nothing; the lifecycle
  keep/expiry blend stays as-is; gate output alpha blends refreshed-vs-current
  thoughts: `thoughts = alpha*refreshed + (1-alpha)*thoughts`).

Shared weights across slots preserved (gate MLP is slot-shared); permutation
invariance exact (slot-symmetric ops); no belief→action bypass added; same
actuator/head/K/optimizer/steps/seeds {42,142,242}.

## Measurements (frozen)

1. Ideal-evidence probe (identical protocol to BrainCell screening):
   correct/zero/wrong injected evidence → decision accuracy. PRIMARY.
2. Torture-suite mean lift + tasks-above at 6000 steps.
3. Gate telemetry: alpha distribution on no-evidence vs strong-evidence frames
   (sensible gating = lower alpha without surprise, higher with).
4. Latency p50 vs control on the same DGX run.
5. Escalation collapse rate at 1200 steps (protect Phase 2 strength).

## Acceptance gates (numeric, frozen now)

Screening pass requires ALL:
1. Memory-causal pattern present in ≥ 3/3 seeds
   (correct > zero AND wrong < correct)
2. Torture lift ≥ control mean − 1σ (no regression beyond noise)
3. Collapse rate ≤ control + 0.05
4. Permutation test exact on all arms
5. Latency p50 ≤ 1.25× control

On PASS → 5-seed confirmation before any Core V1 candidacy claim.
On FAIL → record, archive the gate alongside halting, no re-tuning loop.

## Explicit non-goals

No redesign of the gate internals. No new loss terms (the legacy
surprise-supervision loss is NOT revived; the gate trains only through the
standard task gradient). If untrained-through-task-gradient gating cannot learn
causal intake, that is itself the recorded answer.
