# Pseudo-Brain Current Work & Roadmap Status

**Last updated:** 2026-08-28 (T1-4 in progress; canonical compute = local CPU)
**Primary Compute Platform:** Local Windows CPU (single-thread, CUDA hidden) — per `docs/decisions/2026-08-26-local-cpu-canonical-compute.md` (DGX-Spark-primary record is SUPERSEDED; the Spark is reserved for owner-hosted Qwen).
**Forward roadmap:** `docs/MASTER_ROADMAP.md` · **Execution ledger:** `brain/scratch/WORK_QUEUE.md` (LIFO; the ledger + HEAD are authoritative over any stale note).

---

## 1. Roadmap Phase Progression

```text
================================================================================
PHASE 0: EXPERIMENTAL FOUNDATION
[████████████████████] COMPLETE & LOCKED
PHASE 1: REAL-TIME RUNTIME
[████████████████████] COMPLETE (~1.1 ms GB10 forward at K=32 vs 16.67 ms
                            budget; vectorized slot-recurrent core)
PHASE 2 / 2.5: ARCHITECTURE VALIDATION
[████████████████████] CLOSED (V2-C four-seed confirmation AMBIGUOUS,
                            sigma 0.0653 > 0.06 frozen tol; no retry licensed;
                            closure recorded, negatives preserved)
PHASE 2.6: FOUNDATION HARDENING / CORE V1
[████████████████████] CLOSED — Core V1 frozen at tag `core-v1`
                            (6 mechanism candidates tested, zero promoted;
                            constitution CI green; known limitations explicit)
PHASE 2.7: SCALING-LAW / THROUGHPUT PREP
[████████░░░░░░░░░░░░] OPEN (episode-batched engine ~12x; width/budget
                            axes closed on clean evidence; Core V2 workstream
                            active below)
CORE V2 / EMBODIED QUALIFICATION (Tier-0..3 workstream)
[████████████░░░░░░░░] ACTIVE — Tier-1 in progress (T1-4)
================================================================================
```

## 2. Where the execution stands (Tier ledger, `scratch/WORK_QUEUE.md`)

- **[DONE] Tier 0** — T0-1 full regression (1151 run / 1143 pass / 2 skip /
  6 structural non-green, none a behavioral regression); T0-2 V2-C
  confirmation closed AMBIGUOUS; T0-3 local-CPU-canonical-compute decision
  (DGX-Spark-primary SUPERSEDED).
- **[DONE] T1-1** — Pac-Man-like harness prereg v1 frozen (no-pause 60 Hz,
  5 families, P1–P6, seed partitions, H1–H7 gates).
- **[DONE] T1-2** — deterministic sim + harness; H1–H7 PASS at frozen scale
  (re-verified 2026-08-28); P1–P6 test module reconstructed 25/25; dual-stack
  adjudication closed (`runs/2026-08-28-t12-...adjudication.md`).
- **[DONE] T1-3** — expert BC corpus on TRAIN: 250 episodes, 507,768
  transitions, all 5 gates PASS (C determinism byte-match, D causal boundary
  C8, E action validity, F scale, G competence floor: mean pellet-fraction
  0.77–1.00, mean M_P 0.52–0.94 — a competent teacher, not a strawman).
  `corpus_sha256=59c41b6dc71f…` (artifact dir `datasets/embodied-corpus-v1/`,
  gitignored).
- **[ACTIVE] T1-4** — reactive baseline (Q2/Q3 comparison arm, never
  weakened): prereg frozen 2026-08-28
  (`docs/preregistrations/2026-08-28-pacman-reactive-baseline-v1.md`);
  model `src/irene_brain/v2/reactive_baseline.py` (ReactiveBaselineV1,
  133,773 params, 4-frame + prev-action, non-recurrent); runner
  `scripts/run_reactive_baseline_v1.py` (6000 steps, batch 16,
  AdamW 5e-4/1e-4, grad-clip 1.0, seeds [42,142,242,342], gates R/S/T/U,
  create-only result `runs/embodied-reactive-baseline-v1/
  2026-08-28-reactive-baseline-v1.json`). Runs local CPU, single-thread,
  below-normal OS priority (gaming-polite).

## 3. Next (queue)

1. **T1-4 closeout** — record gate R/S/T/U results; commit + push.
2. **Fruit-Fly Brain Reference** — fly-inspired architecture variant (preregistered 2026-08-29)
   - [DONE] Preregistration document
   - [DONE] Literature review & connectome mapping
   - [DONE] Architecture sketch (4-module topology, sparse coding, DAN gating)
   - [DONE] Probe script & comparison script in `brain/scratch/`
   - [DONE] Fly-inspired model module: `brain/src/irene_brain/model/fly_inspired.py`
   - [DONE] RCQ evaluation wrapper: `brain/src/irene_brain/evaluation/fly_rcq.py`
   - [DONE] Training script: `brain/scripts/train_fly_inspired.py`
   - [ ] Run comparison experiment (fly-inspired vs canonical)
   - [ ] Run training script
   - [ ] Analyze results and issue go/no-go recommendation
3. **T2-1** — train Core V2 (W120/K32/C3) + EmbodiedInterfaceV1 heads
2. **T2-1** — train Core V2 (W120/K32/C3) + EmbodiedInterfaceV1 heads jointly
   on the same corpus under the deployed decision loss; multi-seed.
3. **T2-2** — CPU-QUAL preregistration (seal before opening; TEST untouched).
4. **Tier 3 battery** — Q1 live endurance (2x 2h), Q2 candidate-vs-baseline
   paired 97.5% LCB, Q3 perturbation retention ≥0.75, Q4 causal
   recurrence/thoughtlet gates.

## 4. Standing rules for this workstream

- Local verification is CPU-only, CUDA-hidden, single-threaded, deterministic
  (`zlib.crc32` seed banks, never builtin `hash()`); no network services, no
  screen capture, no background jobs beyond the explicit runner.
- Frozen gates are never re-tuned; a failed gate is recorded verbatim and the
  next iteration is a new preregistered probe.
- Baselines are never weakened; negative/ambiguous results are preserved.
- No DEV/CAL/TEST seeds are touched by Tier-0/1 work; TEST opens only under
  the Tier-2/3 qualification preregistrations.
- Runs are executed at below-normal OS priority so interactive desktop work
  (gaming) is unaffected.
