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

## Tier 1 — Pac-Man-like harness (env does NOT exist yet)

- [T1-1] Preregistration for harness construction: v3 §4 contract
  (cycle_id/frame_id/action_id, no-pause 60Hz, 5 difficulty families, six
  registered perturbations, TRAIN/DEV/CAL/TEST seeds by layout family, no
  privileged state reaching the model — C8).
- [T1-2] Build deterministic sim; determinism protocol mandatory (zlib.crc32
  banks, never hash()).
- [T1-3] Expert/scripted policy → behavior-cloning corpus.
- [T1-4] Reactive baseline (never weakened).

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
