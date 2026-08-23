# CURRENT_WORK — ACTIVE FRONTIER

**Last updated:** 2026-08-22 (overnight autonomous session)
**HEAD:** `ac1bebb` · **Tag:** `core-v1` @ `890e4d0` (frozen foundation)

## PHASE 2.6: CLOSED — Core V1 frozen

Six mechanism candidates tested under preregistration; **zero promoted**:

| Mechanism | Verdict | Evidence |
|---|---|---|
| Episodic Memory v0/v1 | REJECTED | causal direction right, magnitude failed prereg |
| Dynamic-K v0/v1 | REJECTED | v0 gain failed replication; no true closure |
| BrainCell gated-GRU | REJECTED (screening) | worse lift, no causal intake |
| BrainCell evidence-residual | REJECTED (confirmation) | 1/5 causal seeds; small lift regression; σ-stabilization finding preserved |
| Adaptive halting | DESCOPED (audit) | telemetry-only, never skips cycles, `.item()` sync defect |
| Adaptive gate (wired) | REJECTED (prereg experiment) | memory-causal 0/3 under task gradient |

**Constitution CI added:** 15 contracts A–O, all green
(`brain/tests/test_constitution_ci.py`, runner `run_constitution_ci.py`).
Torture regression confirmed code-state changes caused zero behavioral drift.
GRU baseline note updated: its locked −0.042 was seed-optimistic (n=3 σ=0.14).

**Core V1 known limitations (explicit):**
1. evidence-intake deficit [MEASURED, 4 failed fixes]
2. weak multitask sample efficiency vs GRU-on-good-seeds
3. no adaptive compute (fixed C=3)
4. keep/accept decomposition stabilizes variance but doesn't fix intake

## PHASE 2.7: OPEN — throughput + scaling prep

| Item | Result |
|---|---|
| Training profile | GPU-bound; 80 ms/step; forward+backward 97.7%; recurrent serialization dominates |
| Episode-batched engine | equivalence PASS per prereg (3/3 seeds ≤0.03); **~12× per-episode throughput** |
| W-scaling sweep | W120 −0.233±0.004 → W240 −0.218±0.026 → W480 −0.172±0.134; latency flat ~1 ms |

## Next recommended work

1. **W=480 variance disambiguation**: 5-seed × {6000, 12000 steps} to separate
   "wide needs more data" from "wide is unstable" — the batched engine makes
   this ~1 h instead of a day.
2. Multi-model batching (stacked-state ensembling) on top of episode batching.
3. K×W joint scaling cell (e.g. K64×W240) once variance question resolved.
4. Post-V1 memory program gated on an update rule that passes ideal-evidence probe.

## Session totals (2026-08-22 full day → overnight)

~40 commits pushed. All experiments preregistered or audit-only; every number
traceable to Spark run artifacts under `runs/`.
