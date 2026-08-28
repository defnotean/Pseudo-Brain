# WORK_QUEUE — persistent ledger (owner contract §3)

LIFO execution. Loop: pop → prereg (freeze gates) → run local → record →
commit+push → push dependent items. Sealed/closed branches listed in
`brain/docs/ARCHITECTURE_IDEAS_LEDGER.md` are never reopened.

**Honesty note (2026-08-26):** a prior session's log claimed this ledger and the
Tier 0 items were committed; they never landed on disk (HEAD was b94c920 with a
clean tree and no such files). Those artifacts were never created. This ledger is
the real one; all Tier 0 items below are executed for real and verified before
any completion claim.

---

## Tier 0 — immediate

- [DONE] T0-1 — full regression suite re-run to completion, 2026-08-27.
  **MEASURED: 1151 run / 1143 pass / 2 skip / 3 fail / 3 error = 6 non-green.**
  NONE is a behavioral regression: 2 are the documented Windows-`/tmp`
  environmental failures (`test_dgx_launch_contract`, expected per the
  2026-08-26 decision record); 4 are **stale frozen-identity provenance
  canaries** correctly signaling the tree advanced past historical pins
  (3× the V2.1i `EXACT_V3_SOURCE_BUNDLE_SHA256` pin, fired solely by
  `embodied_interface.py` being added in `b94c920`; 1× the
  baseline-architecture-manifest pin, fired by `actuator.py`/`torch_model.py`
  advancing past the 08-18 manifest). **Decision: NOT re-frozen, NOT
  weakened, NOT skip-gated** — re-baselining a sealed identity is a
  scientific-claim decision needing a fresh registration, not autonomous
  maintenance. The prior session's "1051 pass / 2 skips" was unverified and
  is superseded. Full record:
  `brain/docs/runs/2026-08-27-t01-full-regression-reverification.md`.
- [DONE] T0-2 — V2-C four-seed confirmation closed out: AMBIGUOUS (mean -0.037,
  delta +0.176, sigma 0.0653 > 0.06 tol), per frozen classes; no retry/relaxation.
  Report: `brain/docs/runs/2026-08-26-v20j-confirmatory-v2c-arm-closeout.md`.
- [DONE] T0-3 — decision record superseding DGX-Spark-primary:
  `brain/docs/decisions/2026-08-26-local-cpu-canonical-compute.md` + old record
  status line amended to SUPERSEDED. (Written for real, this session.)

## Tier 1 — Pac-Man-like harness

- [DONE] T1-1 — harness preregistration frozen: `2026-08-27-pacman-harness-v1.md`
  (no-pause 60Hz contract, 5 families, 6 perturbations P1–P6, seed partitions,
  H1–H7 gates). P2/P4/P5 amended post-commit to channel-level rules
  (working-tree text is the operative frozen contract; see the 2026-08-28
  adjudication record for why the original P4/P5 rounding rule was degenerate
  and why a divergent "construction" reconstruction was retired).
- [DONE] T1-2 — deterministic sim + harness built:
  `environments/pacman_harness.py` + `scripts/pacman_harness_gates.py`.
  H1–H7 all PASS at frozen scale on 2026-08-27 (2.387 s) AND re-verified
  2026-08-28 (3.419 s, `runs/pacman-harness-gates/2026-08-28-full-rerun.json`).
  The P1–P6 test module `tests/test_pacman_harness.py` (cited as 21/21 in the
  T1-2 report but never committed) was reconstructed 2026-08-28: 25/25 OK.
  Frozen §4.3 timing gates live in the harness layer (`TIMING_GATES`).
  Adjudication record: `brain/docs/runs/2026-08-28-t12-pacman-harness-stack-
  adjudication.md`.
- [DONE] T1-3 — expert/scripted policy → behavior-cloning corpus on TRAIN.
  `scripts/run_pacman_corpus_v1.py` ran the frozen pixel-only expert
  (family-matched knobs) on `partition_seeds(f,"TRAIN")[:50]` × 5 families.
  **All 5 gates PASS** (250 episodes, 507,768 transitions, wall 692 s):
  C determinism (byte-match), D causal boundary (C8), E action-class
  validity, F scale (exact seeds), G expert-competence floor (mean
  pellet-fraction 0.77–1.00, mean M_P 0.52–0.94 — a competent teacher, not a
  strawman). `corpus_sha256=59c41b6dc71f…`. Report:
  `brain/docs/runs/2026-08-28-t13-pacman-corpus-generated.md`.
- [ ] T1-4 — reactive baseline (never weakened), candidate's budget class.

## Tier 2 — integrated training

- [T2-1] Train Core V2 (W120/K32/C3) + EmbodiedInterfaceV1 heads jointly on
  corpus under deployed decision loss; multi-seed.
- [T2-2] CPU-QUAL preregistration (seal before opening; TEST never touched).

## Tier 3 — battery (wall-clock)

- [T3-1] Q1 live endurance (two independent 2h runs, v3 no-pause contract).
- [T3-2] Q2 competence (random + reactive baseline, positive paired 97.5% LCBs,
  legal-action rate ≥99%).
- [T3-3] Q3 perturbation retention ≥0.75 paired ratio, six registered
  perturbations.
- [T3-4] §8.1/8.2 causal recurrence/thoughtlet gates.

## Tier 4+ — past WORKING-PRACTICAL

- [T4-1] Scaffold non-human-blocked Minecraft-like side: local voxel/2D-world
  sim, six Q4 task families, adapters, clock contract; local server interface
  if Minetest/Offline-Minecraft needed; measure endpoint uncertainty vs 1.00 ms
  budget and record, never proceed blind.
- [T4-2] Deliverable organization on Desktop (owner preference), categorized.
- [BLOCKED-HUMAN] Participant study (Q6, 24-person) — never fake.
- [BLOCKED-HUMAN] Any irreversible claim-grade decisions — OWNER_REPORT.md.
