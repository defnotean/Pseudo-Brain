# GRU vs Thoughtlet — Final Report (2026-09-04)

## Question
Do persistent shared-weight thoughtlets (32×12-d = 384-d, shared BrainCell + attention)
outperform a conventional GRU (384-d) and a reactive baseline at matched state budgets?

## Answer (short)
**No — at 3k steps GRU wins clearly; at 15k steps they tie (~51 pel/1k) with the
thoughtlet using 28× fewer recurrent params (65K vs 1.8M). Neither approaches the
scripted expert (~460 pel/1k on dense families). BC + 4 rounds of DAgger plateaus at
~40/1k. "Fully working" (expert-level autonomous play) was NOT reached with BC.**

## Method fixes discovered en route (all verified by experiment)
1. **Sequence training is mandatory.** Single-frame training never exercises recurrence
   (`h=None` every step). Rewrote `train.py` around `CorpusSequenceDataset` (32-step
   windows) + per-timestep scheduled sampling (ramp 0→1 over first 30%).
2. **Eval families must match corpus mechanics.** Custom families (wrong periods/loops)
   gave 0.02–0.14 pellet fractions; registered `FAMILIES` (pacman_harness.py:197) give
   0.2–0.6 for the same checkpoints.
3. **Catch = respawn, not terminal** (maze_chase.py:398-406). Episodes run to clear/cap.
   Old break-on-catch eval measured the wrong thing.
4. **Pure argmax deadlocks all models** against walls (frame diff 0.00, proved via
   pixel-diff trace). Added `--anti-stuck` (2nd-best + sustained bans on static frames).
5. **Pellet-fraction saturates** (respawn + 4–8k caps → even random wandering clears).
   Honest metric = **efficiency (pellets/1k ticks) + catches/1k ticks**.
6. **Ablation "0.97 miracle" was the saturation artifact.** On efficiency, no_persist/k8
   are WORSE than the persistent base (45/51 vs 82 on F1).

## Results — efficiency (pellets/1k ticks | catches/1k, temp=1.0, registered families)

| model | pel/1k | cat/1k | params | frame acc |
|---|---|---|---|---|
| reactive-42 (3k) | 29.1 | 25.5 | 133,773 | 0.92 |
| gru-142 (3k) | 53.3 | 24.4 | 1,816,205 | 0.91 |
| gru-142 (15k) | 50.7 | 24.6 | 1,816,205 | 0.924–0.944 |
| thoughtlet-142 (3k) | 37.0 | 21.7 | 65,093 | 0.926 |
| thoughtlet-142 (15k) | **50.9** | 22.5 | 65,093 | 0.918 |
| thoughtlet + DAgger R1/R2/R3 | 38.4 / 40.3 / 40.5 | 23–30 | 65,093 | 0.926/0.926/0.919 |
| thoughtlet + DAgger R4 (noisy) | 35.7 | 28.0 | 65,093 | 0.874 (REGRESSION) |
| expert planner (matched knobs) | 71–463 | 0–15 | — | — |

## Results — ablations (F1 efficiency, temp=1.0, seed 42 only, suggestive)

k8 (8×48, no attn) 51.2 > no_persistence 44.7 > k2 42.1 > k1/k16/no_attn ~31 >
k32 24.2 > k4 23.5. independent_slots = NaN logits (broken). Base persistent
thoughtlet-142 = 81.7 on same probe — persistence + attention + sharing all help,
K=8 looks best but n=1 seed each.

## DAgger verdict (negative result)
4 rounds (planner expert, family-matched knobs, 40→60 eps/round, 75k→282k transitions):
+3%, +2%, +0%, −30% (noisy-student R4 poisoned the mix). Balanced 50/50 mixing was
required (naive pooling drowns dagger 10 000:1). Conclusion: BC+DAgger plateaus far
below expert; the gap is policy-class/optimization, not data distribution.

## Scale verdict (positive, partial)
3k→15k steps (0.1→0.5 epoch): thoughtlet 37→51 pel/1k (+38%). GRU flat (53→51).
Thoughtlet matches GRU at 15k with 28× fewer params. Frame acc saturates ~0.92 for all.

## Latency (CPU / DirectML RX 9070 XT)
reactive 0.34/0.57ms, gru 0.58/1.0ms, thoughtlet 0.66/2.0ms. All < 16.67ms budget.
(DML training is 8–10× slower/step than CPU for this tiny model — kernel launch
overhead; CPU 16-thread is the throughput king here. Models patched DML-compatible:
ManualGRUCell, _prev_onehot.)

## Checkpoints (brain/runs/gru_vs_thoughtlet/)
- seed_{42,142,242,342}_{reactive,gru,thoughtlet}.pt (3k, sequence+SS+labelsmoothing)
- long15k-thoughtlet-142.pt, long15k-gru-142.pt (BEST: 50.9 / 50.7 pel/1k)
- dagger-r{1,2,3,4}-thoughtlet-142.pt, dagger-r{1,3,4} datasets (brain/datasets/)
- ablations/seed_{42,142,242}_{k1,k2,k4,k8,k16,no_persistence,no_attention,independent_slots}.pt (unified recipe, supersedes old seed_42 set)
- eval: eval_results_as6.json (custom fams, argmax+as6), eval_results_t1.json (custom, temp1.0)

## RL phase verdict (NEGATIVE — 2026-09-05)
REINFORCE+baseline from BC inits (pellet +1, catch −5, tick −0.002, γ=0.99,
truncated BPTT-32, entropy bonus), 500 updates × 8 eps:
- Plain (256-tick eps): thoughtlet 50.9→33.7, gru 50.7→25.4 pel/1k. Both worse.
- KL-anchored to BC init (coef 0.1, 512-tick eps, 300 updates): thoughtlet 71.2→54.2,
  gru 91.0→47.0 (4-seed panel). Still worse. Catches flat ~30/1k throughout.
- DEV partition (clean generalization set): BC thoughtlet 63.7/23.0, BC gru 69.5/34.1;
  KL thoughtlet 46.4/22.9, KL gru 56.9/27.2. Same story. (DEV ≈ arbitrary seeds:
  no layout-memorization; models generalize uniformly.)
- Thoughtlet degraded LESS under RL and is SAFER on DEV (23.0 vs 34.1 catches/1k
  at 63.7 vs 69.5 pel/1k) — weak pro-thoughtlet-robustness signal, confounded by
  both regressing.
Diagnosis: short-horizon TRAIN-episode objective doesn't transfer to full episodes;
high-variance PG destroys good BC init faster than it teaches, even KL-anchored.
Checkpoints: rl-{thoughtlet,gru}-142.pt, kl-{thoughtlet,gru}-142.pt.

## Ablation verdict (NULL, n=3 — 2026-09-06)
Unified recipe (models.py factory + train_model duck-typing, seq-32/SS/label-smooth,
3000 steps, batch 16), 8 variants x 3 seeds = 24 ckpts (dGPU 9070 XT), eval 5 fams x
4 eps temp=1.0 anti-stuck=6. Pareto (mean over seeds, pel/1k +- std):
- k2 36.5+-6.7 | k32 35.3+-5.4 | k8 29.5+-2.3 | no_persist 29.3+-2.0 | k4 28.5+-1.2
  | no_attn 27.8+-1.6 | k1 27.7+-1.7 | ind_slots 27.2+-2.6 | k16 26.7+-0.3
- catches/1k ~31 flat (ind_slots 36.8, worse); recovery 4.4-8.0; latency 1.0-3.9ms.
FINDINGS: (1) K-sweep flat — slot count 1..32 changes nothing (k1 ~= k32 within
noise). (2) no_persistence ~= full persistence: recurrence is DECORATIVE, the
conv-encoder reactive path carries the policy. (3) no_attention ~= full: attention
adds nothing. (4) Only real effect is STABILITY: independent_slots diverges
(mean 3976 NaN ticks/ep, worst catches); shared weights prevent NaN divergence.
k4 shows mild instability (118 NaN ticks). NaN-guard added to run_episode
(nan_to_num + uniform fallback + nan_ticks metric); argmax unaffected.
CONCLUSION: at BC-3k scale no architectural knob matters; all variants 27-37
pel/1k within seed noise. Bottleneck is data/algorithm (BC ceiling), not arch.
Checkpoints: ablations/seed_{42,142,242}_{k1,k2,k4,k8,k16,no_persistence,
no_attention,independent_slots}.pt; results: ablations/ablation_pareto.json,
eval_{A,B,C,D}.json + eval_k32.json.

## Proper PPO verdict (NEGATIVE, rigorous — 2026-09-06)
Recurrent PPO from long15k-thoughtlet-142 (best BC): full-episode rollouts on
TRAIN seeds, zero-init value head (+142 params, BC-identical load), per-episode
GAE(0.99/0.95), batched truncated-BPTT (SEG=128, exact h_in recompute, masked
minibatches), clipped surrogate (0.2) + clipped value (0.5) + entropy (0.01),
EXACT KL-to-BC via parallel carried ref state (Schulman k3 — first attempt used
an invalid estimator that went negative, fixed), 2 arms x 100 updates x 5 eps
(identical hparams/seeds, kl-coef 0.05 vs 0.0). Full instrumentation: pg/vf/ent/
KL/approxKL/grad-norm/clipfrac/explained-var/pel-1k/cat-1k/NaN/h-stats.
- KL arm: DIVERGED TO NaN weights by upd 30 (h_std 0.6→6.4→NaN, grad norms 300+,
  vf loss 145). KL penalty (~0.1) never constrained PG steps (~0.5); k3 hit 7.1.
- No-KL arm: survived 100 updates but converged to RANDOM-policy performance.
- Same-protocol eval (5 fams x 4 eps, t=1.0): BC 54.9/28.6/rec-10.6, PPO-KL
  24.1/37.6/4.3, PPO-noKL 22.4/38.1/4.5 pel-1k/cat-1k/recovery.
- DEV held-out layouts: BC 78.0/27.3, PPO-KL 24.8/39.2, PPO-noKL 24.7/40.1.
  PPO-noKL after 100 updates == NaN-weight random policy. Total destruction.
DIAGNOSIS: unnormalized attention-residual recurrence is stable under tiny BC
steps but explodes under PG updates (h_absmax →16+ →NaN); critic learns nothing
(ev≈0 throughout); clipfrac 0.8+ means every update fights the trust region.
CONCLUSION: on this task, interaction with proper RL cannot improve the policy —
it can only destroy it. BC/DAgger/REINFORCE/PPO all exhausted. The reactive CNN
path carries the policy (ablation), so there is nothing for RL to shape in the
recurrence. Next: memory-required environment where recurrence is load-bearing.
Checkpoints: ppo-kl-142.pt (NaN), ppo-nokl-142.pt, logs ppo_kl/nokl.json;
evals: ppo_vs_bc_eval.json, ppo_dev_eval.json. Code: ppo.py, dev_eval.py,
eval_ppo.py, models.ThoughtletModel.actor_critic + value_head.

## What's next: MEMORY-REQUIRED ENVIRONMENT (not done)
Controlled partially-observable task where current frame is insufficient —
delayed-memory / hidden-state / delayed-consequence. Baselines: reactive vs GRU
vs thoughtlet at matched state budgets, no crippled baselines. Success = clear
interaction (reactive ≈ others without memory demand; reactive << others with
it). Only then: prediction → surprise → plasticity → scale. Do NOT scale
(1B→1T) or assume 16x/26x/28x ratios generalize until small-scale advantage
is demonstrated.
