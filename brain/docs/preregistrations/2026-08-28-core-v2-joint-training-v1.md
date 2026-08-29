# Preregistration — Core V2 + EmbodiedInterfaceV1 joint training v1 (T2-1)

**Status:** FROZEN 2026-08-28 before any training run. Gates below are
acceptance criteria fixed before execution; they are never re-tuned after
seeing results. Negative results are preserved verbatim.

**Compute:** local Windows CPU, `py -3.11`, single-thread
(`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`), CUDA hidden, no network
services. Process priority class below-normal (gaming-polite, 0x4000).
Per `docs/decisions/2026-08-26-local-cpu-canonical-compute.md`.

## 1. Candidate

Core V2 (`irene_brain.v2.core.CoreV2Model`) with the **registered
W120/K32/C3** configuration: `width=120`, `thoughtlets=32`, and the
CONFIG_C-class feature flags (full world model `all_action_table_v1`,
predictive-coding PE, memory, session manager — per
`2026-08-27-pacman-harness-v1.md` §Candidate), plus
`EmbodiedInterfaceV1` heads trained **jointly** (single optimizer over core +
interface params). `world_model_action` is supplied from the logged expert
action so all five outcome-table rows supervise per step.

## 2. Data (frozen, read-only)

- Corpus: `datasets/embodied-corpus-v1`, `corpus_sha256=59c41b6dc71f…`
  (T1-3, 250 episodes / 507,768 transitions, TRAIN seeds only,
  `partition_seeds(family,"TRAIN")[:50]` × 5 families).
- Observation tensor per step (frozen, same contract as corpus manifest):
  64×64 ego-centered window → 4096 + 6116 = **10212-dim** flat vector:
  `[0:4096]` normalized pixels, `[4096]` pellets_remaining_norm,
  `[4097]` elapsed_norm, `[4098:4194]` last 96 actions one-hot (left-padded,
  zero before episode start), `[4194:4290]` last 96 rewards (raw),
  `[4290:5090]` last 80 steps ego position one-hot, `[5090:6114]` last 16
  steps relative enemy position one-hot, `[6114]` died_prev (0/1).
- Replay buffer iterated with the frozen episode-major + uniform-step sampler
  seeded by the run seed (same sampler scheme as
  `scripts/run_reactive_baseline_v1.py`).
- DEV seeds remain untouched by this task (they belong to T2-2/Q battery);
  TEST partition is never touched. This task is a TRAIN-only fit.

## 3. Loss (deployed decision loss, frozen)

Total = `L_dec + 0.5·L_outcome + 0.1·L_hazard + L_pe + 0.1·L_session`:
- `L_dec` = `deployed_decision_loss(output.decision.action_values,
  expert_action)`: sparse CE applied to the **deployed**
  `aggregated_action_values` (aggregation `direct_mean_logits_v1` — mean over
  hypothesis dimension), not to any non-deployed path.
- `L_outcome` = all-action table: per-action MSE of predicted next latent
  (cosine-normalized) + symlog two-hot CE reward, on all 5 rows.
- `L_hazard` = BCE on hazard head vs `died_prev`-window target (raw scorer;
  no post-hoc calibrator — the affine/isotonic family is closed per ledger
  L1/L4/L7).
- `L_pe`, `L_session` = the model's built-in prediction-error and session
  losses, unweighted beyond the constants above.

## 4. Optimization (budget class matched to T1-4, never weaker)

AdamW, lr 5e-4, weight decay 1e-4, grad-clip 1.0, batch 16, **6000 steps**
per seed, seeds `[42, 142, 242, 342]` (same budget class as
ReactiveBaselineV1; wall-clock estimate ~40–60 min/seed local CPU — recorded
honestly before the run, not after).

## 5. Acceptance gates (frozen; ALL must PASS per seed)

- **V — Determinism.** Seed-42 repeated run: parameter-set SHA-256 and full
  6000-step loss curve byte-identical.
- **W — Causal boundary.** Static audit of the runner: reads only corpus
  files + observation channels enumerated in §2 (C8 channel-boundary rule,
  same 3 tagged input-line audit as T1-4); no harness future/ground-truth
  access at decision time.
- **X — Deployed-path supervision.** Autograd receipt: gradient norm of
  `L_dec` w.r.t. `aggregator` params > 0 and w.r.t. core params > 0 (loss
  genuinely trains the deployed decision path jointly with the core);
  recorded per seed.
- **Y — Budget fidelity.** Exactly 6000 steps × 4 seeds; corpus manifest
  hash verified equal to `59c41b6dc71f…` before any step.
- **Z — Non-degeneracy + learning signal.** Every seed emits all 5 action
  classes on the corpus probe set (500 held-out-corpus steps), and corpus
  self-recall (argmax match on probe) > 0.25 (above 1/5 chance floor by a
  margin), and final training loss < initial training loss for every seed.
- **AA — Wall-clock honesty.** Measured wall time recorded per seed; no
  gate depends on it (it is data, not a criterion).

**Pass rule:** V–Z all PASS on all four seeds → T2-1 closes as trained
integration (still NOT a competence claim — competence is T3-2 Q2 against
random + reactive baseline). Any gate fails on any seed → close T2-1 as
FROZEN NEGATIVE with the failing evidence verbatim; no gate relaxation, no
re-run with tuned constants, no seed swap-out.

## 6. Artifacts

- Runner: `scripts/run_core_v2_joint_v1.py` (create-only JSON results).
- Output: `brain/runs/embodied-core-v2-joint-v1/2026-08-28-core-v2-joint-v1.json`
  + `seed_*.pt` checkpoints (gitignored).
- Report: `brain/docs/runs/2026-08-28-t21-core-v2-joint-training.md`.
- Ledger updates: `brain/scratch/WORK_QUEUE.md`, `CURRENT_WORK.md`.

## 7. Declared non-goals

No DEV/TEST seed contact; no qualification claim; no perturbation or
endurance claim; no comparison claim against the baseline (that is T3);
no reopening of sealed ledger branches (PB21M/N/O, V2-C ambiguity stays
as recorded).
