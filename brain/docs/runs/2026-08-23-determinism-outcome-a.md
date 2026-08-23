# Phase 2.7 Determinism Audit Stages B/C: Outcome A confirmed (2026-08-23)

**Design:** `brain/scripts/determinism_audit.py` (frozen inline prereg).
W480 s142 ×3 reps ×2 modes, 2000 steps, pinned banks (Stage A PASS:
`b3bb5fc33fd5f605` across independent processes).

## Results [MEASURED]

### Stage B — normal fast GPU execution
- init digest: IDENTICAL across reps (65.568355)
- first-batch data digest: IDENTICAL (`5bdd442f1fdc7843`)
- loss @ step 10: IDENTICAL to 6 decimals
- FIRST DIVERGENCE @ step 50: L = 1.321728 vs 1.321729 (one ULP)
- by step 2000: final param L2 66.659 / 66.548 / 66.554; lifts
  −0.183 / −0.215 / −0.286

### Stage C — torch.use_deterministic_algorithms(True),
### CUBLAS_WORKSPACE_CONFIG=:4096:8, TF32 off
- ALL reps bitwise identical: same final digest (66.467416), same losses,
  SAME lift −0.2273 three times.

## Verdict

**Outcome A:** deterministic execution fully fixes reproducibility at W480.
The Stage-L/L2 "instability" decomposes into two identified causes:
1. per-process episode banks (salted hash) — fixed with crc32 offsets;
2. GPU reduction-kernel nondeterminism entering via backward atomics,
   amplified by the wide model's landscape — fixed with deterministic mode.

No evidence of genuine optimization instability yet; that question is now
answerable only after Stage E (corrected seed-variance curve).

## Operational consequences

- All future scaling runs run with deterministic flags ON.
- Cost: deterministic kernels are slower (~1.4–2× typical); acceptable at
  current scale, and the batched engine's headroom absorbs it.
- The single-number lift of a deterministic run is a *reproducible*
  measurement; cross-run spread from here is real seed/optimization signal.
