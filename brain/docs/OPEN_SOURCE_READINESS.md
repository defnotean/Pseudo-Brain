# OPEN_SOURCE_READINESS — INTERNAL ONLY

**Status:** Repository stays PRIVATE. This document is release preparation,
not a release. Do not publish this file, do not change repo visibility, do not
apply a license without explicit owner instruction.

**Last updated:** 2026-08-24 (Stage V2.0 in flight)

---

## A. Scientific readiness — 0/1 flagship phenomenon

| Item | State |
|---|---|
| Deploy-aligned learning signal | ✅ MEASURED locally (smoke: deployed CE 1.609→0.0003) |
| Core V2 vs V1 held-out lift | ⏳ Stage V2.0 running (frozen gates α/β/γ vs R=−0.2130) |
| Independent confirmation | ❌ not attempted yet |
| Causal mechanism ablation | ❌ interfaces exist (interventions), tests not run |
| Valid corrected benchmark (Torture Suite V2) | ❌ not built |

Gate path: GATE A → B → C → D before any release conversation.

## B. Documentation readiness — draft stage

- [x] CURRENT_WORK.md accurate and current
- [ ] Public README draft (`release_drafts/README_PUBLIC_DRAFT.md`) — skeleton only
- [ ] Architecture diagram (public-safe)
- [ ] Reproduction quickstart

## C. Reproducibility readiness — good foundation

- [x] Deterministic mode mandatory + provenance helper (`run_provenance.py`)
- [x] Frozen preregs with immutable gates per experiment
- [x] Bank digests pinned (b3bb5fc33fd5f605), eval-seed protocol fixed
- [ ] Clean-checkout reproduction script (documented command → artifact)
- [ ] Checkpoint save/hash policy decided

## D. Security / privacy readiness — audit done 2026-08-24

Secret-pattern scan (tree + full git history, 260 commits):
- API tokens / private keys / cloud keys: **0 hits** ✅
- `.venv` untracked (one regex false positive inside torch distinfo RECORD, untracked) ✅
- `.gitignore` correctly covers .env, venvs, checkpoints, runs ✅

Identity/environment leaks PRESENT in history (1628 blob hits across ~all history):
- SSH alias `defnotean`, hostname `gx10-db18.local`, LAN IP `192.168.0.176`,
  home paths `/home/defnotean`, `C:\Users\Demon`
- Concentrated in: `brain/docs/DGX_SPARK_TRAINING.md`, run logs under
  `brain/docs/runs/`, prereg docs, hardware manifest, AGENTS.md, configs.

Assessment: low credential risk (no secrets, auth is key-based via SSH alias),
but real privacy exposure of infra topology + usernames. **Any public release
requires either (a) history rewrite/filter, or (b) a fresh public repo with
curated history only.** Do NOT flip visibility on this repo as-is.

Recommended eventual procedure (not executed): create fresh public repo,
push curated subset (see §F inventory), keep this repo as private archive.

## E. Licensing / IP readiness — undecided (by design)

See `brain/docs/LICENSE_OPTIONS_INTERNAL.md`. No license file exists; owner
must choose explicitly. Flagged: public disclosure has IP consequences;
recommend professional IP advice if patents/commercial protection matter.

## F. Release packaging readiness — inventory only (nothing split/moved)

| Component | Disposition candidate |
|---|---|
| Torture Suite (V1) + harness + prereg discipline + provenance tooling | PUBLIC |
| Core V1 frozen (@ core-v1 tag) + negative-results corpus | PUBLIC |
| Core V2 architecture skeleton (v2/*.py, smoke tests) | PUBLIC (if results justify) |
| Training Engine V2 batching profile (te_v2_profiler) | PUBLIC |
| DGX launch/provisioning scripts, docker invocations | PRIVATE (infra-specific) |
| Session/meta-learning trainer (V2.3+, when built) | POSSIBLY PRIVATE until published results exist |
| Consolidation / fast-plasticity (dormant hooks) | POSSIBLY PRIVATE |
| Checkpoints | decide per-release; hash + provenance required |

---

## Scorecard (per §17 of the strategic directive)

| Item | Score |
|---|---|
| Scientific phenomenon demonstrated | 0/1 |
| Independent confirmation | 0/1 |
| Causal ablation | 0/1 |
| Valid V2 benchmark | 0/1 |
| Clean reproduction | 0/1 |
| README release-ready | 0/1 |
| Secret/privacy audit | 1/1 (done; exposure documented, remediation planned) |
| License/IP decision | 0/1 |
| Release artifact/checkpoint | 0/1 |
| Clean install/test path | 0/1 |

**Recommendation:** remain private. Critical blockers: no confirmed
phenomenon, no causal test, no valid benchmark, unresolved IP decision,
identity-laden history.
