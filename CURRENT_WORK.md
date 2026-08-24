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
2. ~~Stage E corrected W-curve~~ **DONE 2026-08-23: width does NOT buy
   capability** (W120 −0.227±0.013 / W240 −0.232±0.039 / W480 −0.257±0.057).
   v1 sweep improvement was artifact; Core V1 stays W=120 on clean evidence.
3. ~~Stage 3a training-budget curve~~ **DONE 2026-08-23: MIXED by strict
   gate (Outcome C requires |Δ|≤0.03 in both arms; SCHEDULED |Δ|=0.039
   fails). Scientific read: strong train/eval decoupling / generalization
   plateau.** FIXED Δ(24k-6k)=−0.008, SCHEDULED Δ=−0.039. SCHEDULED
   overfits loss→0.0 by step 7k but generalizes worse (−0.233 vs −0.221);
   reduces σ(24k) 0.102→0.046 without improving mean. Core V1 not
   undertrained; cannot convert optimization on current data/objective into
   capability. All three capacity axes (W, K, budget) closed.
4. ~~Stage 3b Data / Generalization Audit~~ **DONE 2026-08-23: AMBIGUOUS by
   frozen gate (strongly MECHANISM-leaning).** DATA rejected at 6k (Δ=−0.008,
   not ≥+0.03). MECHANISM gate fails second limb: ARM B loss ≈1.85 did NOT
   collapse (ARM A ≈1.01). Two explanations alive: (A) learning
   mechanism/objective wrong; (B) fresh stream undertrained at 6k. ARM B 16×
   higher seed variance (σ 0.007→0.112), 13.6× wall cost (9976s vs 732s).
   Seed 142 (−0.046) is strong relative outlier, still negative, undecided
   noise-vs-basin. **Disambiguation:** train ARM B to matched training-loss
   criterion, then compare — requires Training Engine V2 first.
   b. **Stage 3c0 — Learning Contract Audit (NEXT, highest priority):** No arch
      changes. Diagnose whether the training objective supervises what the eval
      path uses. Gradient audit (which heads get grad?), train-vs-eval output
      audit (per-slot accuracy vs aggregated action_dist), thought diversity,
      consequence calibration, aggregator ablation (replace with mean of slot
      logits → does held-out jump?), oracle-consequence test. Potential defect:
      replicated per-slot CE on action_logits while deployed decision uses
      consequence utilities/branch_prob/confidence that are NOT supervised.
   c. **Intake-mechanism program** (only path that ever moved memory causality:
      ideal-evidence probe in Phase 2) — pending 3c0 outcome.
   d. Optimizer stability workstream deferred — σ(24k)=0.102 at FIXED but no
      mean improvement; SCHEDULED reduces σ to 0.046 without mean gain.
5. Provenance helper (`run_provenance.py`) mandatory in all new scripts:
   bank digest · init param digest · seeds · det flag · torch/CUDA versions.
6. Multi-model ensemble batching (deferred prereg) — motivation weakened;
   re-evaluate only if a future axis shows budget-bound instability.
7. Post-V1 memory program gated on an update rule that passes ideal-evidence probe.

## Session totals (2026-08-22 full day → 2026-08-23)

~58 commits pushed. All experiments preregistered or audit-only; every number
traceable to Spark run artifacts under `runs/`.
