# Phase 2.6 Stage C: adaptive-gate wiring experiment — FAIL per preregistered gates (2026-08-22)

**Prereg:** `2026-08-22-adaptive-gate-wiring-prereg.md` (frozen before run).
**Data:** Spark `runs/adaptive_gate_results.json`.

## Results [MEASURED]

| Metric | control | gate_wired |
|---|---|---|
| Torture mean lift (3 seeds) | −0.1811 ± 0.0376 | −0.2109 ± 0.0383 |
| Escalation collapse rate | 0.792 | 0.775 |
| Memory causal signature | 0/3 | **0/3** |
| Permutation test | exact ×3 | **test harness error** (see below) |
| Latency p50 | 1.09 ms | 1.25 ms (within 1.25× gate) |

## Gate verdicts

1. G1 memory causal 3/3: **FAIL** (0/3 — the gate did NOT produce causal
   evidence intake under task-gradient training alone)
2. G2 lift no-regression: PASS (−0.211 ≥ −0.181 − 0.038, borderline)
3. G3 collapse no-worse: PASS (0.775 ≤ 0.842)
4. G4 permutation exact: **FAIL — HARNESS DEFECT, not architecture verdict**
   (`GateWiredCore` lacks `initial_state`; the perm test errored and recorded
   False for all three seeds). The underlying property was never measured.
5. G5 latency: PASS

**SCREENING_PASS = false → adaptive-gate wiring REJECTED per prereg.**

## Honest caveats recorded

- The escalation collapse rates here (~0.78) are far higher than the
  confirmation-round runs (~0.005–0.015) because THIS experiment's escalation
  eval ran models trained on the torture suite only, not escalation-trained.
  Cross-arm comparison remains valid (both arms identical protocol); absolute
  values are NOT comparable to earlier docs.
- The permutation contract is a real open question: GateWiredCore adds a
  slot-symmetric gate (should preserve invariance), but it was never verified
  due to the harness defect. Recorded as UNMEASURED, not as violation.

## Consequence

Per prereg failure handling: no re-tuning loop, no second attempt without a new
prereg. The adaptive-gate joins halting in DESCOPED status. The audit verdict
(PROMOTE-TO-EXPERIMENT) has been discharged: the experiment ran clean and
failed its primary gate — evidence intake through task gradient alone does not
emerge from this mechanism, consistent with the BrainCell campaign pattern
(three gate/decomposition variants now show the same outcome).

## Updated Phase 2.6 ledger

| Mechanism | Verdict |
|---|---|
| Episodic Memory v0/v1 | REJECTED |
| Dynamic-K v0/v1 | REJECTED |
| BrainCell gated-GRU | REJECTED (screening) |
| BrainCell evidence-residual | REJECTED (confirmation) |
| Adaptive-halting | DESCOPED (audit) |
| **Adaptive-gate (wired)** | **REJECTED (preregistered experiment)** |

Next: Core V1 freeze review with the smaller core; intake deficit enters as an
explicit KNOWN LIMITATION with four failed candidate fixes documented.
