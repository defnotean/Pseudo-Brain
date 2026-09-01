# T2-1 — Core V2 + EmbodiedInterfaceV1 under deployed decision loss (preregistration)

**Frozen:** 2026-08-31, before any Model B/C training or scenario evaluation.
**Branch:** `biogenesis/t2-1-core-v2-embodied-v1`. Experiment dir: `brain/experiments/t2_1_core_v2_embodied_v1/`.
**Scope:** TRAIN partition only (diagnostic + development discipline; no DEV/CAL/TEST seed touched; no qualification claim — this is an L-series-style controlled experiment whose conclusions are architectural, not publishable scores).

---

## 0. Core question

Does the **Core V2 one-brain architecture** (explicit timescales: ENCODE → ERROR → RETRIEVE → THINK × C → BELIEVE → HYPOTHESIZE → AGGREGATE → PREDICT) trained under the **deployed decision loss** on the shared TRAIN corpus, and extended with **EmbodiedInterfaceV1** record-emitting heads, measurably improve closed-loop decision-making on the embodied scenario battery **without** destroying the reactive baseline's ghost-avoidance capability, under matched parameter, training, and latency budgets?

This maps exactly onto the T1-4 failure: near-inert lockup on F0_calm (0 pellets) and F4_open_slow (1–8 pellets) is a *selection-state* failure — the reactive baseline has no persistent state to hold "keep foraging" when the teacher's manifold is left. Core V2's explicit recurrent belief/thought cycles + world model + deployed decision loss are the mechanism under test.

---

## 1. Biological / computational principles investigated

| # | Mechanism | Computational principle | Selected? |
|---|---|---|---|
| 1 | Explicit multi-timescale cognition (encode/error/think/believe/hypothesize/aggregate/predict) | Separate fast (latent/prediction error) and slow (belief/thought/session) state streams; each tick has a defined computational phase order | **YES (core)** |
| 2 | Deployed decision loss (CE on aggregated action_dist, not per-slot logits) | The loss trains the *actual deployed output path*; per-slot logits are proxies with zero gradient at init under legacy scalar-utility aggregation | **YES (core)** |
| 3 | All-action outcome table + factual gather | Vectorized 5-class outcome table (next_latent, reward, hazard, branch_logit, confidence) computed independently; factual supervision gathers the applied action — cannot alter proposal policy | **YES (CONFIG_B)** |
| 4 | Dedicated stop-gradient hazard path | Hazard head reads frozen belief + prediction error; cannot alter next-state, reward, decision, belief, or thought parameters | **YES (CONFIG_B)** |
| 5 | Probability-sigmoid hazards (not unbounded) | Hazards and utilities share 0..1 semantics; prediction error feedback on hazards is calibrated | **YES (CONFIG_B)** |
| 6 | Cosine next-latent targets | Directional latent comparison normalizes feedback deltas; encoder magnitude not part of target | **YES (CONFIG_B)** |
| 7 | Convex normalized thought updates (BrainCell) | Replaces legacy additive gates that amplified state; bounded per-cycle update | **YES (CONFIG_C)** |
| 8 | GRU-style convex belief update | Replaces legacy unbounded residual; bounded recurrent world state | **YES (CONFIG_C)** |
| 9 | Prediction error fusion (latent + outcome + surprise) | Fused feedback signal into thought field; stop-gradient target | **YES (CONFIG_C)** |
| 10 | EMA target encoder (world model) | Slow-moving target encoder for world model consistency | **YES (CONFIG_C)** |
| 11 | Turn-weighted causal maze trajectories | Training examples weighted by teacher direction changes, not corridor holds | **YES (CONFIG_C)** |
| 12 | EmbodiedInterfaceV1 heads (locomotion, look, cursor, buttons, hotbar) | Frozen Core V2 core + supervised heads emitting common record; core is stop-gradient, byte-preserved | **YES (heads only)** |
| 13 | Episodic memory (retrieval at tick 3) | Optional in CONFIG_C; gated by flag, not in initial T2-1 candidate | DEFER (CONFIG_C_FULL flag, not in B) |
| 14 | Session adaptation / meta-learning | Optional in CONFIG_C; not in initial T2-1 candidate | DEFER |

**Minimum viable combination selected: CONFIG_B (Core V2-A) + EmbodiedInterfaceV1 heads (Model C) vs CONFIG_B only (Model B) vs ReactiveBaselineV1 (Model A).**
- Model B isolates "Core V2 recurrent architecture + deployed loss" from "embodied heads".
- Model C adds the EmbodiedInterfaceV1 heads (frozen core + new head parameters).
- Model A is the frozen T1-4 reactive baseline (never retrained, never weakened).

---

## 2. Models (all pixel-only, constitution C8: rendered frames + previous applied action; no coordinates, reward, clock, or privileged state ever enters ANY model input)

### Model A — ReactiveBaselineV1 (frozen, untouched)
- The T1-4 checkpoint (133,773 params), last-4 frames + prev action → 5-class argmax.
- Frozen checkpoint from `brain/runs/embodied-reactive-baseline-v1/seed_42.pt`.

### Model B — Core V2-A (CONFIG_B, no embodied heads)
- CoreV2Model with `DEFAULT_CONFIG` (W=120, K=32, C=3) and `CONFIG_B_PREDICTIVE` flags:
  - `deploy_aligned_learning=True`
  - `consequence_learning=True` (all_action_table_v1 + dedicated_stopgrad_v1 hazard)
  - `world_model_learning=True` (EMA target encoder)
  - `prediction_error_feedback=True` (latent_outcome_surprise_v1)
  - `episodic_memory=False`, `session_adaptation=False`, `meta_learning=False`
- **Training objective**: `total_v2_loss` with `deployed_decision_loss` as primary (weight 1.0), auxiliary consequence losses (reward/hazard/next_latent) at config weights (0.25/0.25/0.5).
- **Input**: 32×32 RGB frame (single current frame, not frame stack — the Core V2 encoder handles temporal integration via recurrent belief/thoughts), previous applied action (0..4 one-hot).
- **Output**: `StepOutput.decision.action_values` (deployed action_dist logits [B, 5]).

### Model C — Core V2-C + EmbodiedInterfaceV1 (CONFIG_B core + record-emitting heads)
- Same CoreV2Model as Model B (byte-preserved core, stop-gradient).
- `EmbodiedInterfaceV1(core, heads, init_seed)` composes the core with supervised heads:
  - Heads map `cat([latent, belief])` [B, 2W=240] → per-field logits for 8 record fields (locomotion 5, look_yaw 7, look_pitch 7, cursor_dx 5, cursor_dy 5, buttons 6, hotbar 10).
  - Heads are the *only* new parameters; core is always `detach()`.
  - Heads initialized with `zlib_crc32(f"irene.brain.embodied_interface.v1:{seed}")` for process-stable determinism.
- **Training objective**: Same `total_v2_loss` on Core V2 deployed decision (action_dist) + **auxiliary head losses** on the EmbodiedInterfaceV1 record fields (cross-entropy per field, small weight e.g. 0.1).
- **Input**: Same as Model B (32×32 frame + prev_action).
- **Output**: `StepOutput.decision.action_values` (for decision loss) + `EmbodiedInterfaceRecord` per tick (for head supervision).

---

## 3. Training corpus & supervision (frozen)

**Corpus:** T1-3 corpus (same as T1-4), `brain/datasets/embodied-corpus-v1/`, SHA256 `59c41b6dc71fdda1ff051fa4a505668ecddaaccd9988d3d043d3183007858ff8`.

**Supervision per sample (tick t ≥ 1):**
- `frame_t`: Current rendered frame (16×16 RGB) → upsampled to 32×32 by Core V2 encoder.
- `prev_action_{t-1}`: Applied action at tick t-1 (0=idle, 1=W, 2=A, 3=S, 4=D).
- `target_action_t`: Teacher's applied action at tick t (0..4).
- `target_reward_t`: Scalar reward at tick t (for consequence heads).
- `target_hazard_t`: Binary hazard at tick t (1 if caught, 0 otherwise).
- `target_next_latent_t`: Encoded latent of frame t+1 (for world model; stop-grad target).
- `all_action_reward_t`: [5] rewards for each counterfactual action at tick t.
- `all_action_hazard_t`: [5,1] hazards for each counterfactual action.
- `all_action_next_latent_t`: [5,W] next latents for each counterfactual action.

**Note on all-action targets:** The embodied-corpus-v1 manifest includes counterfactual rollouts for all 5 actions at each tick (generated by the procedural maze_chase simulator). These are required for the `all_action_table_v1` outcome architecture.

**Training (frozen, matched to T1-4 budget class):**
- AdamW(lr=5e-4, weight_decay=1e-4), grad-clip max_norm=1.0, batch 16
- Exactly 6000 optimizer steps per seed
- Confirmatory seed cohort [42, 142, 242, 342]
- Deterministic mode: `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.backends.cudnn.deterministic=True`, `torch.backends.cuda.matmul.allow_tf32=False`
- Single-thread CPU, CUDA hidden, below-normal OS priority
- Model init seeds: `zlib.crc32(f"t21.{model_label}.{seed}")` (Model A uses its existing frozen checkpoint; Model B uses `t21.B.{seed}`, Model C uses `t21.C.{seed}`)
- Corpus sample order: `zlib.crc32(f"t21.sample_order.{seed}")` (shared across all models for fair comparison)

**Anti-cheat declarations:** No pellet coordinates in any input; scenario harness uses env internals ONLY to build scenarios and score ground truth — never to form an observation.

---

## 4. Scenario battery (built on the frozen harness env, deterministic)

Same 5 difficulty families as FlyMerge V1 (for direct comparability to T1-4 and FlyMerge):

- **S1 visible pellet:** standard episode; report acquisition (pellet fraction), time-to-first-pellet, trajectory efficiency.
- **S2 pellet disappears:** after agent's 20th pellet, ALL pellets removed; score: post-removal movement toward unexplored/last-known region vs lockup (idle fraction).
- **S3 distraction:** force-spawn direct-chase ghost during pursuit; score: survival, pellet loss during threat window (±30t), **resumption** (pellet rate in 100t after vs before threat).
- **S4 self-induced distribution shift:** run on F0_calm for 1500t then teleport to uneaten corridor; score: action entropy over 200t post-teleport, recovery pellet rate.
- **S5 multiple pellets / target preference:** standard episode; score = directional commitment (entropy of 60t-window displacement directions).

**Families (exact MazeChaseEnv parameters from pacman_harness.py):**
```
F0_calm:        ghost_count=1, ghost_period=3, ghost_rule="shy", ghost_elroy=False, extra_loops=24, player_period=1, max_ticks=4000
F1_standard:    ghost_count=3, ghost_period=2, ghost_rule="mixed", ghost_elroy=False, extra_loops=16, player_period=1, max_ticks=6000
F2_pressured:   ghost_count=4, ghost_period=2, ghost_rule="direct", ghost_elroy=False, extra_loops=12, player_period=1, max_ticks=6000
F3_fast:        ghost_count=4, ghost_period=1, ghost_rule="mixed", ghost_elroy=True,  extra_loops=16, player_period=1, max_ticks=8000
F4_open_slow:   ghost_count=2, ghost_period=3, ghost_rule="ambush", ghost_elroy=False, extra_loops=40, player_period=2, max_ticks=8000
```

---

## 5. State-utility tests (mandatory; Model C only)

1. **Temporal persistence:** Autocorrelation of belief/thoughts over 60t within episodes.
2. **Task correlation:** Linear probes from belief → (a) pellet-present flag, (b) distance-to-nearest-pellet (binned), (c) ghost distance, (d) lockup state. Report R² per target; compare vs Model B's belief.
3. **Interventions:** Freeze belief; zero belief each tick; random-walk noise into belief; reset belief at t=1500; disable heads (use core decision only); disable world model learning; disable prediction error feedback. Each re-run on S1–S5 battery; behavior delta = causal contribution.
4. **Specialization:** No pre-assigned labels; cluster belief dimensions/post-hoc correlate top-variance directions with task variables; lesion-by-zeroing each belief-dim, report which lesion hurts which metric.

---

## 6. Latency / cost gates

Forward-pass latency measured on the same machine, 1000 warm single-tick forwards, p50/p95/p99/max, compared to Model A. Budget: the harness TIMING_GATES target median ≤ 8 ms L_plus — both must stay far under; any model with p99 > 1 ms (CPU single-thread forward) is flagged.

**Parameter budget:** Model B/C must stay within 10% of Model A (133,773 ± 13,377). The Core V2 CONFIG_B is ~1.1M params; the preregistration *accepts* this overage because the architectural comparison is the primary question — but if Model C does not substantially beat Model A/B on the scenario battery, the param cost is a rejection criterion.

---

## 7. Frozen acceptance classes (decide BEFORE running; class the result, never re-tune thresholds after)

- **KEEP** if Model C vs Model A on the family battery: pellet-fraction improvement ≥ +0.10 mean AND no family drops below Model A by more than 0.02 AND survival never-dying property preserved AND latency gate passes AND state-utility probes show belief carries task-persistent information beyond Model B.
- **TEST FURTHER** if direction positive but any seed cohort member contradicts, or effect ≤ 0.10.
- **REJECT** if Model C ≤ Model A, or Model C ≈ Model B (embodied heads add no value over core), or latency/param budget violated without compensatory capability gain.
- **Model A is never retrained, re-tuned, or weakened; Model B is the honest architecture-matched control.**

---

## 8. Reproduction

```bash
# From brain/ directory:
cd brain && py -3.11 experiments/t2_1_core_v2_embodied_v1/train_t2_1.py          # trains B, C (4 seeds each)
cd brain && py -3.11 experiments/t2_1_core_v2_embodied_v1/eval_t2_1_battery.py   # families S1-S5 + ablations + state probes
```

CPU-only, CUDA hidden, single-thread, below-normal priority (gaming-polite), deterministic mode, zlib.crc32 seed banks (NEVER builtin hash). Outputs to `brain/runs/t2-1-core-v2-embodied-v1/` (create-only).