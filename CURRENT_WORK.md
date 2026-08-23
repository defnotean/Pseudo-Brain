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
| W-scaling sweep (v1) | mean lift ↑ with width but variance ↑↑ — **later found CONFOUNDED** |
| Salted-hash root cause | `hash()` episode banks differ per process; within-process comparisons stand, cross-process absolute numbers confounded [MEASURED] |
| Determinism audit B/C | **Outcome A**: normal-mode same-seed runs diverge from step 50 (1 ULP → amplified); deterministic flags make 3/3 reps bitwise identical (lift −0.2273 ×3) |

## Determinism protocol (now mandatory for all scaling runs)

```python
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
# banks: zlib.crc32(fn.__name__.encode()) % 99991  (NEVER hash())
```

## Next recommended work

1. ~~W=480 variance disambiguation~~ superseded by root-cause + audit work.
2. **Stage E (running): corrected seed-variance curve** — W{120,240,480} × 4
   seeds × 6k steps under pinned banks + deterministic mode = first honest
   optimization-stability read across width.
3. If Stage E shows clean reproducibility and manageable seed variance →
   rerun the W-scaling curve properly, then K×W cells.
4. Multi-model ensemble batching (deferred prereg) — re-evaluate after E.
5. Post-V1 memory program gated on an update rule that passes ideal-evidence probe.

## Session totals (2026-08-22 full day → overnight)

~40 commits pushed. All experiments preregistered or audit-only; every number
traceable to Spark run artifacts under `runs/`.
