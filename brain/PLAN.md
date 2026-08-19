# Pseudo-Brain: Streaming Thought-Field Model Build Plan

Status: architecture and execution plan, version 0.1  
Date: 2026-08-16  
Deployment target: local NVIDIA RTX 5070 with 12,227 MiB VRAM  
Training target: local workstation for smoke tests; the DGX Spark for
larger experiments and offline consolidation

## 1. Executive decision

We will build a new kind of small, recurrent sensorimotor model before trying
to build a large general model.

Its central state will not be a sentence, a chain of thought, or a single
summary vector. It will maintain:

1. A persistent belief about the visible and hidden world.
2. A field of many small, incomplete latent predictions called thoughtlets.
3. Working and episodic memory that survive across frames and attempts.
4. A direct, anytime motor readout that can act after every internal cycle.

The first scientific question is deliberately narrow:

> At equal parameters, FLOPs, data, and physical deadline, do many persistent
> parallel thoughtlets improve prediction, adaptation, and live control over
> one monolithic recurrent state or one longer serial computation?

We will answer that question in small, branchable procedural worlds. We will
not begin with Minecraft, a language model, or billions of parameters. If the
mechanism does not produce a causal equal-compute advantage in the small
experiment, scaling it would only hide the failure under more data and compute.

The initial thesis model was conceived at roughly 35–60 million parameters
with $K=32$ thoughtlets. As documented below in the Current Empirical State,
the active research configuration has since progressed to a calibrated $\sim 29.7\text{M}$
setup with $K=4$ thoughtlets and adaptive cognitive mechanisms. A useful first
result is not human-level intelligence. It is one checkpoint that continuously
observes pixels, discovers unfamiliar controls, tracks several possible near
futures, acts without pausing, and becomes measurably better during a lifetime
while its weights remain frozen.

## Current Empirical State — 2026-08-19

> **Distinction of Record:** This document remains the canonical long-term
> architecture blueprint for Pseudo-Brain. The original thesis design specified
> a $K=32$, core width $d=384$, 3-cycle model ($\sim 35\text{--}60\text{M}$ parameters).
> The active experimental implementation has since progressed to a calibrated
> $\sim 29.7\text{M}$ parameter research configuration that incorporates adaptive
> gating, multi-horizon prediction, dynamic cognitive depth, counterfactual foresight,
> and topological routing.

### Current Research Model Configuration
- **Parameter scale:** $\sim 29.7\text{M}$ parameters ($d=384$).
- **Thoughtlet count:** $K=4$ active parallel thoughtlets.
- **Adaptive update gating:** Situation-dependent surprise gate $\alpha_t \in [0, 1]$ regulating recurrent state overwrites.
- **Multi-horizon prediction:** Action-conditioned rollouts predicting future rewards, hazards, and latent states $t+1 \dots t+H$.
- **Adaptive cognitive depth:** Dynamic halting mechanism allocating compute cycles ($C \in [1, 6]$) based on hazard intensity and state uncertainty.
- **Counterfactual action foresight:** Latent lookahead tree evaluation pruning branches with predicted collision hazards ($\hat{c}_{t+k} > 0.5$).
- **Topological goal routing:** Directional unit vector alignment bonuses guiding branch selection toward global pellet clusters.

### Best Balanced Verified Champion
- **Variant E (Full Adaptive System: Gate + Multi-Horizon Prediction + Adaptive Halting):**
  - **5.65 mean pellets collected** (+71.2% over baseline 3.30).
  - **83 total ghost catches** (-88.6% vs baseline 727) across 20 held-out evaluation seeds (2,400 decision steps).
  - **Dynamic compute allocation:** $\sim 1.35$ cycles in open corridors vs $\sim 3.80$ cycles in active ghost hazards.
  - *Record:* [docs/runs/2026-08-18-five-way-cognitive-ablation.md](docs/runs/2026-08-18-five-way-cognitive-ablation.md).

### Exploratory Candidates Characterized
- **Variant F (Counterfactual Foresight):** High-reward but reckless (**7.80 pellets / 654 catches**).
- **Variant G (Topological Goal Routing):** Ultra-safe but over-conservative (**4.90 pellets / 83 catches**).
- *Champion classification rule:* Pellets $\uparrow$, then catches $\downarrow$. Variant $E$ strictly beats Variant $G$ on task reward while matching safety.

### Current Active Open Questions
1. **Statistical Multi-Seed Replication:** Quantifying $\text{mean} \pm \text{std}$ across 5 training seeds $\times$ 20 validation seeds (100 episodes per variant) for $E$, $F$, and $G$.
2. **Wall Recovery Latency & Gate Dynamics:** Diagnosing why Gate-only ($B$) gives weak recovery latency improvement ($24.8$ ticks vs baseline $25.2$ ticks, $E$: $23.1$ ticks), tracing actuator momentum persistence across surprise events.
3. **Calibrated Risk Utility (The F/G Sweet Spot):** Recovering $F$'s $\sim 7.8$-pellet goal-seeking drive while maintaining $E$'s $\sim 83$ ghost catch safety floor.
4. **Decisive Matched-Baseline Suite:** Executing the make-or-break comparison of Pseudo-Brain against matched GRUs, recurrent Transformers, and conventional world-model actors under identical parameter, FLOP, latency, and experience budgets.

## 2. What success means

The project maintains a strict three-tiered research progression:

### 2.1 Current Work
- Validated sensorimotor and thought-field foundations.
- Multi-horizon predictive world modeling and counterfactual collision foresight.
- Situation-dependent dynamic cognitive depth ($C \in [1, 6]$).
- Statistically pinned multi-seed replication batteries on held-out procedural worlds.

### 2.2 Near-Term Research
- Closed-loop real-time arcade competence (Pac-Man family, $60\text{ Hz}$ frame skip one).
- Hardware latency and continuous capture/control harness (p99 $< 16.67\text{ ms}$).
- Spatial reasoning in rich 3D voxel worlds (first-person navigation, occlusion memory, control discovery).
- DAgger distillation with expert recovery supervision to eliminate policy collapse.

### 2.3 Long-Term North Star
The ultimate objective is a single persistent, general-purpose cognitive agent capable of:
- Understanding natural-language instructions and maintaining persistent task goals.
- Decomposing difficult tasks into dynamic subgoals and parallel thought latents.
- Using software tools, APIs, code editors, file operations, web searches, and physical actuators under a unified sensorimotor framework.
- Comparing potential actions counterfactually before execution.
- Detecting mismatches between predictions and real-world results, and revising internal beliefs.
- Operating computers and physical robots through the exact same persistent cognitive core.

This project will not use behavior as evidence of consciousness. Here,
understanding has operational meanings: calibrated prediction, system
identification, memory, transfer, causal action selection, and improvement with
experience.

## 3. Non-negotiable one-brain contract

A result counts only when all of the following are true:

- There is one deployed parameter set, theta.
- Thoughtlets are persistent states updated by shared weights. They are not
  agents, copied policies, or separately trained experts.
- The same checkpoint is used across all evaluation games.
- There is no game ID, per-game adapter, per-game policy head, handcrafted
  object detector, external planner, or model swap.
- Modality stems and motor heads are jointly trained parts of the one model.
  Different signal geometry makes image, audio, and text stems reasonable; it
  does not justify different brains.
- Evaluation inputs are generic human-facing observations and previous generic
  controls. Simulator state can label training examples but can never be an
  inference input.
- The environment continues running during inference.
- An action is available after every internal cycle.
- The fast action path is a readout from the same recurrent model, not a second
  reflex policy.
- Frozen-weight evaluation is the primary test of within-lifetime learning.
  Any later gradient-based consolidation is reported separately.
- Complete world families, rule combinations, visual systems, and control
  mappings are held out. Random frames from a known game do not count as an
  unseen game.

An automated compliance test will inspect model configuration, observation
fields, checkpoint hashes, adapters, action heads, and evaluator wiring before
accepting a result.

## 4. Explicit non-goals for the first prototype

The first prototype will not:

- Generate or supervise written chains of thought.
- Attempt unrestricted desktop control.
- Include a large language-model backbone.
- Train directly on the current Irene personality/chat corpus.
- Depend on remote network inference in the action loop.
- Claim universal game competence.
- Train from scratch at billion-parameter scale.
- Use Minecraft or Atari content before rights and dataset terms have been
  formally reviewed.
- Optimize only game score while ignoring latency, prediction, adaptation, and
  causal slot use.

## 5. Falsifiable hypotheses

These hypotheses must be registered in the experiment configuration before
training:

H1. At matched parameters and FLOPs, a wide persistent thought field reduces
multi-horizon prediction error compared with the strongest monolithic
recurrent baseline.

H2. At matched wall-clock deadline, it improves closed-loop control on tasks
with several simultaneous objects, hazards, or plausible futures.

H3. With weights frozen, persistent state implements rapid learning: control
discovery and return improve during an unfamiliar lifetime.

H4. The advantage survives strict real-time execution and is not caused by
pausing the environment or spending more test-time compute.

H5. At least a meaningful minority of thoughtlets are causally useful. Their
ablation, shuffling, or duplication has predictable effects.

H6. The same checkpoint and generic device interface transfer from procedural
worlds to open arcade-like and first-person voxel worlds.

If H1 through H3 fail under the stopping rules in this document, the
microthought thesis will not be scaled to Minecraft.

## 6. System overview

~~~text
pixels / audio / text / time / previous control
                         |
                         v
              causal modality stems
                         |
                         v
        spatial sensor tokens + persistent belief
                         |
            +------------+-------------+
            |                          |
            v                          v
   K persistent thoughtlets      working / episodic memory
   shared BrainCell weights      per-thought retrieval
   three recurrent cycles              |
            +------------+-------------+
                         |
                         v
       generic actuator queries read all state directly
                         |
                         v
     keyboard / mouse / gamepad state and short action plan
~~~

The system has no mandatory one-vector integration stage. Coordination exists,
but it is a narrow shared workspace plus direct attention from actuator queries
to the complete sensory, belief, memory, and thought state.

## 7. State and tensor contract

Let:

- B be the number of environments in a training batch.
- d be the core width.
- K be the number of thoughtlets.
- R be the number of registers per thoughtlet.
- C be recurrent cognitive cycles per observation.

Inputs at time t:

~~~text
RGB frames          Xv[t]      [B, F, 3, H, W]
audio samples       Xa[t]      [B, samples]
new text tokens     Xl[t]      [B, L]
previous controls   U[t-1]     [B, D_action]
elapsed real time   dt[t]      [B, 1]
~~~

Modality stems produce:

~~~text
visual tokens       V[t]       [B, Nv, d]
audio tokens        A[t]       [B, Na, d]
language tokens     L[t]       [B, Nl, d]
action/time tokens  U[t]       [B, Nu, d]
sensor tokens       O[t]       [B, No, d]
~~~

Persistent state:

~~~text
belief tokens       P[t]       [B, Nb, d]
working memory      W[t]       [B, Nw, d]
thought field       H[t]       [B, K, R, d]
goal/context        G[t]       [B, Ng, d]
episodic memory     M[t]       keys, values, age, provenance, confidence
~~~

Sensor tokens retain spatial location, modality, capture time, and age.
Pooling the whole frame into one vector is prohibited in the core experiment.

Each thoughtlet's registers are latent. They can jointly represent a focus,
condition, prediction, horizon, uncertainty, confidence, relevance, or
relationship. We will not permanently label slots as logic, enemy, route,
creative, or Minecraft slots. Dynamic binding is part of the hypothesis.

## 8. The recurrent update

The model receives real elapsed time rather than assuming perfectly regular
frames. A learned continuous-time gate gives different state elements
different update rates:

~~~text
g = 1 - exp(-softplus(rate) * dt)
state[t] = (1 - g) * state[t-1] + g * proposed_state[t]
~~~

This permits fast visual and danger states to update every frame while goals
and spatial beliefs change more slowly.

First, the persistent belief absorbs the latest sensors, the previous applied
control, and elapsed time:

~~~text
P[t,0] = GatedCrossAttention(P[t-1], O[t], U[t-1], dt[t])
~~~

Each thoughtlet then persists, refreshes, or respawns:

~~~text
H[i,t,0] =
    keep[i,t] * H[i,t-1,C]
  + (1 - keep[i,t]) * Seed(P[t,0], O[t], exchangeable_noise[i])
~~~

The same BrainCell weights are reused over all thoughtlets, frames, and cycles:

~~~text
for cycle c in 1..C:
    P[t,c], W[t,c], H[t,c] =
        BrainCell_theta(
            P[t,c-1],
            W[t,c-1],
            H[t,c-1],
            O[t],
            retrieved_memory[t],
            G[t],
            U[t-1],
            dt[t],
            cycle_id=c
        )
    action_distribution[t,c] = ActReadout_theta(all current state)
~~~

The default C is three. We expect, but do not hard-code, a progression like:

1. Notice and bind potentially important features.
2. Develop short predictions, alternatives, or remembered patterns.
3. Check confidence, conflicts, and usefulness.

## 9. BrainCell attention topology

Dense all-to-all attention would both homogenize thoughtlets and eventually
create a quadratic cost. The BrainCell therefore uses structured recurrent
attention.

Belief tokens can read:

- Current sensor tokens.
- Previous applied controls and elapsed time.
- Other belief tokens.
- Working memory.

Each thoughtlet can read:

- Its own R registers.
- Belief and relevant sensor tokens.
- Goal/context tokens.
- Its own retrieved episodic memories.
- Only its top two to four routed thoughtlet neighbors.

Working-memory tokens can read:

- Belief tokens.
- High-utility thoughtlet writes.
- Retrieved memories.

Actuator queries can read:

- Current sensor tokens for immediate correction.
- All belief tokens.
- Every thoughtlet register.
- Working memory and retrieved memories.

Cycle one starts with private thoughtlet updates. Sparse cross-thought messages
become available in cycles two and three. This protects early alternatives
from instantly becoming copies while still permitting coordination.

For the first K=32 prototype, exact pairwise routing scores are acceptable.
For larger K, focus buckets or approximate top-r routing must make communication
scale approximately linearly.

This design is related to Recurrent Independent Mechanisms and shared global
workspace research, but the persistent future-conditioned thought field,
anytime direct motor readout, and training against branch bundles are the
specific thesis being tested here.

## 10. Thoughtlet lifecycle and output contract

A thoughtlet follows:

~~~text
spawn -> bind -> predict -> persist -> observe evidence
      -> revise / pause -> resolve or expire
~~~

For supervision and diagnostics, every thoughtlet emits:

~~~text
focus distribution over sensor and belief tokens
horizon distribution: 1, 2, 4, 8, 16, 32 frames
optional candidate-action condition
predicted future or event embedding
occurrence probability
uncertainty
urgency and expected usefulness
keep / pause / expire gate
memory-write gate
~~~

These are compact latent and numeric outputs, not natural-language thoughts.
A slot can be incomplete: it may retain a possible collision, route, object,
control effect, or unexplained pattern until later evidence arrives.

Age, horizon, confidence, and provenance travel with the slot. New observations
can confirm, revise, or invalidate it. Stale high-confidence thoughtlets are
penalized through calibration and time-to-live targets.

## 11. Preventing thought collapse

The greatest architecture risk is K identical states that look parallel but
carry one hypothesis. We will use several defenses together:

- Randomly permute slot order during training.
- Use exchangeable seed noise rather than permanent semantic IDs.
- Keep cycle-one thought attention private.
- Limit later communication to a narrow workspace and sparse neighbors.
- Train against an unordered set of distinct branch outcomes.
- Match predictions to targets with Hungarian or Sinkhorn assignment.
- Apply random slot dropout at the action and prediction readouts.
- Penalize duplicates only when focus, condition, horizon, and prediction are
  all redundant. Hidden-vector orthogonality alone is easy to game.
- Teach a marginal-utility head using loss changes under slot masking.
- Randomize K, active-slot budgets, frame delay, and cognitive-cycle budget.
- Permit null or sleeping slots when one future is sufficient.

Required collapse measurements:

- Distinct verified event coverage at K.
- Pairwise duplicate rate.
- Slot utilization entropy and starvation rate.
- Effective rank of the thought field.
- Calibration by slot and horizon.
- Marginal prediction and control loss when each slot is removed.
- Targeted ablation around threats, occlusion, junctions, and surprise.
- Performance after slot shuffling, duplication, state reset, and delay.

A visual latent-space plot is not evidence that the thoughts differ. Only
prediction and intervention count.

## 12. Solving the integration bottleneck

The user's concern about a single integration state is correct. A rich thought
field can be destroyed if all information must pass through one small vector
before action.

The proposed solution is a bank of generic actuator queries:

~~~text
256 USB HID keyboard queries
8 mouse-button queries
mouse X and mouse Y queries
scroll query
standard gamepad button and axis queries
~~~

Every actuator query independently cross-attends the complete current state.
A shallow coordination layer among actuator queries represents chords such as
forward plus sprint plus jump.

Outputs:

- Keyboard, mouse-button, and gamepad-button states use multi-label Bernoulli
  distributions.
- Mouse motion uses a zero-inflated bivariate mixture distribution.
- Gamepad axes use bounded mixture or discretized logistic distributions.
- Every control includes confidence and intended hold duration.
- The model also predicts a rolling 50–100 ms control plan.

The final motor signal is low-dimensional because the physical device is
low-dimensional. That is not a harmful bottleneck. The harmful bottleneck
would be compressing cognition before the device queries, which this design
avoids.

The same action head is used everywhere. Environments map their allowed
controls to fixed HID codes; they do not introduce game-semantic action heads.

## 13. Anytime computation

The model emits a valid action at:

- Cycle 0: new sensors plus the previous persistent state.
- Cycle 1: one thought update.
- Cycle 2: two thought updates.
- Cycle 3: the full default update.

Training randomly truncates computation and supervises every exit. At runtime,
the deepest result completed before the motor deadline becomes active. A late
cycle cannot block the environment.

If no new result is ready, the model's most recent rolling action plan
continues. This plan comes from the same model. It prevents an accidental
neutral pause while preserving the one-brain rule.

Any result carries the source frame ID and state version. Results that are
stale relative to a newer world-state update are discarded.

## 14. Memory and two learning timescales

### Fast within-lifetime learning

During evaluation, theta remains frozen. Learning occurs through:

- Persistent belief state.
- Persistent thoughtlet states.
- Working-memory writes and reads.
- Per-thought episodic retrieval.
- Learned control and dynamics identification.
- Prediction-error-driven confidence updates.

Training data must therefore consist of lifetimes, not shuffled independent
frames. Rules and control mappings remain stable within a lifetime and vary
between lifetimes. The recurrent core is trained to implement an update rule.

The primary adaptation comparison uses identical continued and reset copies:

~~~text
continued state sees previous attempts
reset state receives the same current observations but no prior lifetime state
~~~

The continued copy should predict and act better.

### Slow weight consolidation

Between deployment sessions, the same weights can later be updated using
replay:

- Mix current and old environment families.
- Anchor older behavior through policy distillation.
- Preserve stable representations where useful.
- Re-encode episodic memory keys after major weight changes.
- Re-run the frozen cross-game regression suite before accepting a checkpoint.

Target encoders or older checkpoints are permitted during training. Deployed
inference still contains one current checkpoint.

## 15. Concrete thesis MVP

Starting configuration:

| Component | Value |
|---|---:|
| Input | 128 × 128 RGB; causal optional audio; cached text |
| Initial rate | 30 Hz training and debugging; benchmark 60 Hz |
| Core width d | 384 |
| Belief tokens Nb | 48 |
| Working-memory tokens Nw | 16 |
| Thoughtlets K | 32 |
| Registers per thoughtlet R | 3 |
| Cognitive cycles C | 3 |
| Routed neighbors | 2 |
| BrainCell blocks | 4, tied across cycles |
| Attention heads | 8 |
| Episodic memory | 512 entries |
| Retrieval | 2 entries per thoughtlet |
| Expected parameters | 35–60 million total |
| Runtime precision | BF16 or FP16 initially |

Approximate persistent BF16 state per environment:

~~~text
(48 + 16 + 32 * 3) * 384 * 2 bytes = about 123 KB
~~~

Core arithmetic is roughly 7–10 GFLOP per observation before
implementation-dependent sensory and actuator costs. Theoretical FLOPs are not
the acceptance metric. Batch-one kernel launch overhead, memory traffic, frame
capture, and input dispatch determine the real result.

The first scaling pilot, only after the thesis gate passes:

| Component | Value |
|---|---:|
| Core width | 512 |
| Belief tokens | 64 |
| Working memory | 32 |
| Thoughtlets | 64 |
| Registers | 4 |
| Cycles | 3 |
| BrainCell blocks | 6 |
| Episodic memory | 4,096 entries |
| Expected parameters | 60–120 million |

## 16. Canonical data unit: a lifetime with branch bundles

We will not build a dataset of prose that claims to contain small thoughts.
The fundamental training unit is:

> What the model saw, what control physically took effect, what happened next,
> and what would have happened under several other controls.

A lifetime spans multiple attempts in one hidden world. It must preserve:

- First exposure.
- Exploration and useless actions.
- Control discovery.
- Mistakes and deaths.
- Recoveries.
- Partial rule hypotheses.
- Later competence.
- Distraction, return, and recall.

Storage segments must not imply memory reset.

Six data families are required:

1. Continuous self-play lifetimes.
2. Exact timestamped next-state streams.
3. Counterfactual futures branched from identical states.
4. Human novice, intermediate, and expert demonstrations.
5. Perturbation and recovery episodes.
6. Privileged simulator labels used only as targets.

Expert-only imitation is insufficient because it does not show how controls and
rules are discovered.

## 17. Universal environment interface

The model never sees game-specific action IDs. It sees and emits:

~~~text
keys_down          fixed USB-HID multihot
mouse_dx, mouse_dy continuous counts
mouse_buttons      fixed multihot
mouse_wheel        signed delta
gamepad_axes       fixed continuous vector
gamepad_buttons    fixed multihot
~~~

The environment adapter has two modes:

~~~text
Accelerated training:
    reset -> step -> snapshot -> restore -> branch

Continuous evaluation:
    poll_latest_observation()
    submit_control(control_state, apply_deadline)
~~~

A normal synchronous environment step pauses the world while inference runs.
It can be used for accelerated training, but it cannot support the final
real-time claim.

## 18. Canonical trajectory format

The native format is a branch DAG:

- Parquet for lifetime, step, event, and branch metadata.
- MP4/AV1 video shards for training imagery.
- Lossless audit clips for deterministic diagnosis.
- Optional FLAC audio.
- Content-addressed compressed state snapshots.
- Immutable manifests and SHA-256 hashes.

Exporters may later target RLDS, Minari, and LeRobotDataset, but a linear
episode format cannot represent the branch structure faithfully.

### Lifetime manifest

~~~json
{
  "lifetime_id": "...",
  "world_lineage_id": "...",
  "environment_family": "...",
  "environment_build_sha256": "...",
  "container_digest": "...",
  "generator_config": {},
  "seed_vector": {},
  "control_mapping_hash": "...",
  "graphics_config": {},
  "source_type": "human|self_play|branch",
  "license_record_id": "...",
  "split": "train|validation|test"
}
~~~

### Per-step record

~~~json
{
  "lifetime_id": "...",
  "episode_id": "...",
  "step": 4821,
  "environment_tick": 4821,
  "frame_id": 9012,
  "capture_time_qpc": 1940039441,
  "observation_ready_qpc": 1940041127,
  "action_requested_qpc": 1940047310,
  "action_applied_qpc": 1940050122,
  "rgb_ref": "video-shard:frame",
  "audio_ref": null,
  "requested_control": {},
  "applied_control": {},
  "reward_raw": 0.0,
  "events": ["near_collision"],
  "terminated": false,
  "truncated": false,
  "dropped_frames": 0,
  "snapshot_hash": null,
  "privileged_state_ref": null
}
~~~

Requested and applied controls are both mandatory. Input buffering, action
repeat, and sticky actions otherwise corrupt action-effect supervision.

### Branch record

~~~json
{
  "branch_group_id": "...",
  "root_snapshot_hash": "...",
  "root_lifetime_id": "...",
  "root_step": 4821,
  "branch_id": 17,
  "rng_state_hash": "...",
  "forced_control_sequence": [],
  "continuation_policy_id": "...",
  "horizon_ticks": 16,
  "future_frames_ref": "...",
  "state_delta_ref": "...",
  "events": ["collision"],
  "return": -1.0,
  "replay_exact": true
}
~~~

## 19. Generating counterfactual supervision

At an informative root state:

1. Save complete simulator and random-number state.
2. Restore exactly before every branch.
3. Apply each primitive control for one to four ticks.
4. Sample short control sequences or policy continuations.
5. Record outcomes at 1, 2, 4, 8, 16, and 32 frames.
6. In stochastic worlds, repeat candidates with several random draws.
7. Convert the branch bundle to an unordered set of events and futures.

Automatic targets can include:

- Future visual features.
- Object displacement and continued existence.
- Collision, damage, or death probability.
- Visibility and occlusion.
- Time to a junction, hazard, reward, or interaction.
- Inventory change.
- Whether an attempted control causes a change.
- Reward or achievement delta.
- Prediction surprise and plan invalidation.

Branch only high-information states such as junctions, contact, uncertainty
spikes, new objects, threats, or disagreement between candidate controls.
Start with 1–5 percent of encountered states and 16–64 branches per root.

For one recorded human future, supervision is observational. Branchable
simulators are especially important because a single observed future encourages
all slots to predict the same thing.

## 20. Environment ladder

Use open procedural worlds as the training foundation:

1. A small in-repository moving-shapes world for deterministic CI and latency.
2. XLand-MiniGrid for randomized rules, goals, layouts, and control mappings.
3. Craftax for fast open-world survival, crafting, combat, and exploration.
4. Procgen where its code and cleared/generated assets fit the use case.
5. Craftium/Luanti with an original or cleared asset pack for first-person
   voxel interaction through keyboard and mouse.
6. An original open Pac-Man-like environment for 60 Hz arcade validation.
7. Native Minecraft only as a late external evaluation and possible training
   source after rights review.
8. ALE/Atari only after separate code, ROM, gameplay-recording, automation,
   and training-rights review.

Minecraft and Atari are validation targets, not the legal or computational
foundation of the project.

For every source, track four separate records:

- Environment or emulator code license.
- Game assets or ROM rights.
- Trajectory and human-annotation license.
- Model-training and redistribution terms.

A permissive code repository does not automatically grant rights to the
underlying game imagery. No public-data ingestion begins until its license
record is accepted.

## 21. Data splits

Never split random frames. Split by:

- World lineage.
- Generator family.
- Mechanics and rule family.
- Visual and asset pack.
- Control mapping.
- Source session and human participant.
- Game family.

All segments from one persistent world belong to one split. Validation and
test must include:

- Unseen layouts.
- Unseen combinations of known mechanics.
- Entire unseen mechanics.
- Remapped controls.
- Changed physics and input delay.
- Unseen visual systems.
- Entire unseen environment families.

This separates interpolation, composition, adaptation, and genuine transfer.

## 22. Training objectives

No written reasoning labels are required. The initial normalized objective is:

~~~text
L =
    1.00 * L_action
  + 0.50 * L_anytime
  + 1.00 * L_future_latent
  + 0.50 * L_event_set
  + 0.25 * L_multi_horizon_value
  + 0.10 * L_inverse_dynamics
  + 0.10 * L_calibration
  + 0.10 * L_memory_retrieval
  + 0.20 * L_adaptation
  + 0.02 * L_duplicate
  + 0.01 * L_routing_balance
  + small normalized L_active_slots
  + scheduled L_actor_critic
~~~

These are starting coefficients, not sacred constants. Every component is
normalized by a running scale and logged independently.

Definitions:

- L_action: likelihood of demonstrated or successful generic controls.
- L_anytime: action supervision at every cycle under randomized deadlines.
- L_future_latent: action-conditioned future sensory features at several
  horizons.
- L_event_set: permutation-invariant coverage of distinct branch events.
- L_multi_horizon_value: reward, damage, survival, goal progress, termination,
  and time-to-event.
- L_inverse_dynamics: which applied action caused the observed change.
- L_calibration: Brier or likelihood loss for occurrence and uncertainty.
- L_memory_retrieval: retrieve earlier situations that improve prediction now.
- L_adaptation: continued lifetime state must beat a reset counterfactual.
- L_duplicate: penalize redundant predictions when diverse verified targets
  exist.
- L_routing_balance: prevent early slot starvation.
- L_active_slots: charge for unnecessary computation.
- L_actor_critic: online control improvement after predictive training is
  stable.

For a branch bundle containing an unordered target event set E, prediction uses
permutation-invariant assignment:

~~~text
L_event_set =
    minimum assignment cost between predicted thought events and E
    plus a calibrated null-event loss for unused thoughts
~~~

Hungarian assignment is the simple first implementation. Sinkhorn matching can
be tested later if exact assignment becomes a bottleneck.

## 23. Training curriculum

### Stage A: sensory and dynamics warm-up

- Train causal visual features, temporal consistency, inverse dynamics, and
  one-step prediction.
- Mix multiple procedural environment families from the beginning.
- Do not yet optimize game score.

### Stage B: thought-field supervision

- Enable K=8, then K=32 persistent slots.
- Train multi-horizon and branch-event set prediction.
- Introduce persistence, age, confidence, expiry, slot permutation, and slot
  dropout.
- Compare immediately against monolithic and fixed-head baselines.

### Stage C: anytime imitation and control

- Train every action exit.
- Randomize cognitive budgets, frame ages, action delays, and dropped frames.
- Use novice, failure, recovery, and expert trajectories.
- Require the one-cycle result to remain safe when later cycles miss.

### Stage D: within-lifetime meta-learning

- Group samples into multi-attempt lifetimes.
- Randomize hidden rules, physics, goals, visuals, and controls between
  lifetimes.
- Preserve state across attempts.
- Train continued-versus-reset adaptation.
- Use curriculum sampling near the model's current frontier.

### Stage E: online reinforcement learning

- Add actor-critic or imagination-based policy improvement only after
  prediction, calibration, and replay are stable.
- Preserve branch and imitation losses during online learning.
- Include explicit compute and deadline costs.
- Guard against thought collapse and catastrophic forgetting.

### Stage F: slow consolidation

- Replay a balanced mixture of current and older worlds.
- Validate on the frozen cross-game suite.
- Reject a consolidated checkpoint if old capabilities regress beyond the
  registered tolerance.

Use truncated backpropagation through time with burn-in. An initial sequence
configuration is 32 burn-in steps plus 64–128 optimized steps. Snapshot
recurrent state for longer lifetimes rather than attempting gradients through
hours.

## 24. Real-time runtime architecture

The action-critical path runs locally:

~~~text
Windows Graphics Capture or DXGI Desktop Duplication
    -> GPU texture
    -> GPU resize and normalization
    -> one recurrent thought-field model
    -> current action and rolling control plan
    -> keyboard, mouse, or virtual gamepad
~~~

The current Irene live stack remains useful for Discord, recording, and
eventual integration, but it is not placed in this path. Its one-FPS,
JPEG-based, request-oriented screen flow is too slow and allocates in the wrong
places.

Thread/process roles:

- Capture thread writes a two- or three-slot texture ring.
- Inference thread owns the GPU context and always takes the newest complete
  frame.
- Action thread submits scheduled generic control state.
- Recorder/profiler is lower priority and droppable.
- Watchdog releases all controls if capture or inference becomes unhealthy.

There is no unbounded observation queue. If inference is behind, an old
unprocessed frame is overwritten. Fresh state matters more than processing
every frame.

Do not use HTTP, JPEG, base64, Pillow, remote inference, or Python object queues
in the production loop. Python is acceptable for the early research harness.
The final loop should use a compiled model runtime and native capture/control.

## 25. Timing and clock contract

Use one monotonic Windows QueryPerformanceCounter clock. Never use wall-clock
time for latency.

Every step records:

~~~text
frame ID and source presentation timestamp
capture acquired
preprocess start and end
encoder start and end
thought cycle 1 start and end
thought cycle 2 start and end
thought cycle 3 start and end
action ready
action submitted
first visible resulting effect
model-state version and source-frame ID
~~~

GPU events measure kernel time; QPC measures the complete path.

Derived metrics:

~~~text
observe-to-submit = action submitted - capture acquired
closed-loop latency = first visible effect - capture acquired
frame age = inference start - capture acquired
deadline slack = action deadline - action submitted
action-to-effect = first visible effect - action submitted
~~~

At 60 Hz, one frame is 16.67 ms. Minecraft's simulation is commonly 20 ticks
per second, but camera correction and visible action should still run at render
rate rather than only at the simulation tick.

## 26. Initial runtime budgets

Treat 12 GB VRAM as a hard physical limit and 9 GB as the safe steady-state
peak because Windows and display workloads need headroom.

- No steady-state allocation in the live loop.
- Keep at least 2 GB free.
- Check available memory at startup.
- Precompile fixed profiles for 1, 2, and 3 cycles and several active-slot
  counts.
- Start with BF16/FP16; test quantization only after correctness is stable.
- Use static buffers, fused kernels, and CUDA Graphs or an equivalent compiled
  plan.

Provisional 60 Hz budget:

| Stage | Target |
|---|---:|
| Capture and scheduling | 2.0 ms |
| GPU preprocessing | 0.5 ms |
| Visual/world encoder | 3.5 ms |
| Thought field, 1–3 cycles | 5.5 ms |
| Action readout and submission | 1.0 ms |
| Jitter reserve | 4.17 ms |

Acceptance after a 30–60 minute thermal soak:

- Model kernel p99 at or below 8 ms for the registered profile.
- Observation-to-submit p50 at or below 10 ms.
- Observation-to-submit p95 at or below 16.67 ms.
- Initial p99 at or below 25 ms; drive it below 16.67 ms before the final
  physical speed claim.
- Deadline misses below 1 percent initially and below 0.1 percent before the
  arcade proof.
- No unexplained input gap above 33.3 ms in the 60 Hz environment.
- Zero out-of-memory events and no unbounded queue.

Always report p50, p95, p99, p99.9, maximum, raw deadline misses, VRAM, power,
temperature, and observation age. Average FPS can hide an unusable controller.

## 27. Graceful degradation

Before a deadline is missed:

1. Retain the last valid current action and rolling plan.
2. Drop stale pending frames.
3. Use the deepest completed action exit.
4. Mask low-utility thoughtlets.
5. Defer thought-to-thought communication.
6. Reuse visual features when measured scene change is very small.
7. Switch to a precompiled lower-resolution profile.
8. Defer episodic writes, recording, retrieval, and telemetry.

All modes use the same weights and persistent state.

Safety conditions:

- Release all keys and zero pointer/controller deltas if capture becomes stale,
  the model emits non-finite values, or no valid action arrives before the
  watchdog timeout.
- Provide a physical keyboard kill switch.
- Never hold a hazardous one-shot control indefinitely after a stall.
- An out-of-memory condition forces neutral control and a clean runtime restart.
- Keep initial testing inside a game-only sandbox.

## 28. Baseline suite

Every phase uses the same data, evaluator, observation interface, and action
interface for:

1. Random and no-op policies.
2. A simple scripted diagnostic policy.
3. Reactive CNN or ViT policy with no memory.
4. Matched CNN-GRU or state-space recurrent controller.
5. Standard recurrent Transformer controller.
6. A wider monolithic state with the same total state and parameters.
7. A deeper serial model with matched FLOPs.
8. Fixed multi-horizon heads with no persistent thoughtlets.
9. Thoughtlets with persistence removed.
10. Thoughtlets with coverage and competition removed.
11. Thoughtlets with unrestricted dense communication.
12. A matched-cost independent ensemble as an experimental control.
13. A conventional recurrent latent world-model actor.
14. A task specialist as a capability ceiling.
15. A privileged-state oracle for diagnosis only.

Results must include equal-parameter, equal-FLOP, equal-latency, and
equal-experience comparisons. A larger or slower model winning is not evidence
for the architecture.

## 29. Required ablations

- K in 1, 8, 16, 32, 64, and later 128.
- C in 1, 2, 3, and 6 with width adjusted to match compute.
- Wide/short versus narrow/deep at equal wall time and FLOPs.
- Persistent versus reset thoughtlets.
- Private/sparse versus unrestricted thought communication.
- Event-set matching removed.
- Duplicate and slot-dropout losses removed.
- Observational versus action-conditioned prediction.
- Direct actuator queries versus one pooled integration token.
- Reflex sensor access present versus absent.
- Working memory only versus episodic retrieval.
- Per-thought retrieval versus one central query.
- Memory retained versus reset at episode boundaries.
- One-step versus multi-horizon targets.
- Final-cycle-only versus anytime supervision.
- Fixed K versus learned active-slot scheduling.
- Pixel-only targets versus privileged simulator targets.
- Unchanged generalist checkpoint versus game-specific fine-tuning.

## 30. Evaluation metrics

### Control and adaptation

- Interquartile mean and median return over registered seeds.
- Survival, task completion, recovery, and exploration efficiency.
- Performance as a function of minutes of exposure.
- Area under the within-lifetime adaptation curve.
- Control remapping and rule discovery time.
- Recall after an episode boundary, distractor, or ten-minute delay.
- Retention after returning to a previous environment.
- Performance regression after consolidation.

### Prediction and thought quality

- Future-feature likelihood at 1, 2, 4, 8, 16, and 32 frames.
- Brier score and reliability diagrams.
- Collision, damage, death, and time-to-event accuracy.
- Action-conditioned branch prediction.
- Object persistence through occlusion.
- Distinct correct future coverage at K.
- Duplicate thought rate.
- Active-slot count and routing entropy.
- Marginal effect of every slot on loss and return.
- Benefit near surprise, threats, junctions, occlusion, and recovery.

### Systems

- Model and end-to-end p50, p95, p99, and p99.9.
- Deadline miss rate.
- Observation age and longest action gap.
- Dropped-frame and delayed-input robustness.
- VRAM/RAM usage, power, clocks, and sustained thermals.
- Quality per FLOP, wall-clock millisecond, joule, and byte of state.

### Human comparison

Human novice curves use the same display, controls, instructions, practice
budget, and environment versions. Simulator scores alone cannot establish
physical reaction speed. A later physical comparison should measure both:

- Framebuffer-to-input latency, which is the practical deployed system.
- Display/camera-to-physical-actuator latency, which is more comparable to a
  human viewing a monitor.

## 31. Phase roadmap and gates

### Phase 0: research foundation

Duration: 2–4 weeks.

Build:

- Isolated Python package, dependency lock, tests, typed configs, and
  experiment registry.
- Safe ignore rules and a known repository baseline before large artifacts.
- Deterministic moving-shapes environment.
- Universal control types.
- Timestamped logger and branch-DAG storage.
- Snapshot, restore, replay, and branch tests.
- Continuous real-time wrapper and loopback latency test.
- Reactive and recurrent baselines.

Data: roughly one million generated smoke-test frames and 10,000 branch groups.

Go gate:

- Clock/timestamp error below 1 ms.
- Exact environments reproduce a 1,000-step replay and branch regardless of
  branch order.
- Privileged fields cannot reach model inputs.
- No-model capture-to-input p99 is below 4 ms.
- Two-hour soak has no queue growth, memory leak, or timing drift.
- Every baseline runs through the same evaluator.

No architecture training begins until replay and timing are trustworthy.

### Phase 1: continuous sensorimotor kernel

Duration: 4–8 weeks.

Build:

- Causal visual encoder.
- Persistent belief and working memory.
- K=16 and K=32 thought fields.
- Three tied BrainCell cycles.
- Age, confidence, horizon, expiry, and per-thought retrieval.
- Direct generic actuator queries.
- Cycle 0–3 anytime exits.

Worlds include moving objects, controllable bodies, collision, targets,
occlusion, and noisy observations.

Model: 5–30M parameters for engineering, then the 35–60M thesis MVP.

Go gate:

- K=32 and C=3 run at the registered 60 Hz profile.
- Kernel p99 at or below 8 ms.
- End-to-end p95 at or below 16.67 ms.
- Deadline miss rate below 0.1 percent for one hour.
- Stable behavior with 5 percent dropped frames and zero-to-two-frame delay.
- Cycle-one output is useful when deeper cycles are intentionally withheld.

Stop if K=16 still exceeds twice the deadline after one focused compilation and
kernel-optimization pass.

### Phase 2: prove or reject the microthought thesis

Duration: 8–12 weeks, with the first gate review targeted at day 90.

Task families:

- Multi-object tracking and collision avoidance.
- Pursuit and evasion.
- Junction and route choice.
- Keys, doors, and simple inventory dependencies.
- Partially observable mazes.
- Changed physics and controls.
- Ambiguous hidden causes.
- Situations with several plausible future paths.

Data:

- 50–200 million transitions from at least 10,000 randomized worlds.
- Counterfactual bundles at informative states.
- Three registered random seeds per design.

Go gate, all at matched compute:

- At least 10 percent relative reduction in multi-horizon prediction error.
- At least 10 percent gain in interquartile-mean return.
- Positive 95 percent bootstrap interval in at least four of five held-out task
  families.
- At least eight of 32 slots are causally useful across the suite.
- Slot shuffling/removal causes at least a 5 percent relevant degradation.
- The advantage remains inside the real-time deadline.

Hard stop:

After two substantially different thought-field designs and three seeds each,
if neither prediction nor control improves by at least 5 percent over the
strongest matched baseline, retire parallel microthoughts as the central
hypothesis. The project may continue as a conventional recurrent world model,
but it will not be described as validation of this architecture.

### Phase 3: unseen procedural-game adaptation

Duration: 3–6 months after Phase 2.

Game laboratory:

- Maze navigation.
- Collect/avoid.
- Timing and platforming.
- Pursuit/evasion.
- Manipulation and simple combat.
- Resource chains and inventory.
- Sparse or hidden rewards.
- Random controls, visuals, rules, physics, and goals.
- Other moving actors.

Data target:

- 0.5–2 billion frames.
- 50,000–200,000 lifetimes of 2–30 minutes.
- Optional 100–500 hours of novice and expert human play.
- Entire generator families held out.

Frozen-weight gate:

- Positive adaptation in at least 80 percent of held-out families.
- Adaptation AUC at least 10 percent above the strongest matched recurrent
  baseline.
- Final-quarter performance improves by at least 0.15 on a
  random-to-novice-human normalized scale relative to first-quarter
  performance.
- At least 80 percent of learned performance remains after a ten-minute
  distractor or episode boundary.
- No task ID, gradient, or external planner is used.

Adaptation that relies only on weight updates or memorized game identity fails.

### Phase 4: original 60 Hz arcade proof

Duration: 2–4 months.

Use an original, rights-cleared Pac-Man-like family with changed mazes, motion,
visuals, controls, enemy rules, and sticky/delayed actions.

Evaluate frame skip one. Measure score, survival, junction decisions, and
predictions at 100 ms, 250 ms, 500 ms, and one second.

Gate:

- Kernel p99 at or below 8 ms.
- Emulator end-to-end p99 at or below 16.67 ms.
- Physical loop initially at or below 33.3 ms and then driven toward 16.67 ms.
- Deadline misses below 0.1 percent.
- At least 10 percent score gain over the matched real-time recurrent/world
  model across at least 100 seeds.
- Positive frozen-weight learning curve in excluded arcade variants.

Any advantage that disappears under random starts, sticky actions, frame skip
one, or physical capture is a simulator artifact.

### Phase 5: first-person 3D bridge

Duration: 3–6 months.

Add:

- Continuous camera and mouse control.
- 3D navigation, occlusion, and collision.
- Moving hazards and simple interaction.
- Inventory and short resource chains.
- Five-to-ten-minute goals.
- Map memory and revisiting locations.
- Procedural textures, geometry, rules, controls, and physics.

Environment: Craftium/Luanti with cleared/original content.

Data: roughly 1–5 billion steps and 1,000–5,000 hours of procedural experience.

Gate:

- Fast action/camera p99 at or below 33 ms; cognitive refresh below 50 ms.
- Thirty-minute uninterrupted sessions.
- At least 10 percent success or adaptation-AUC gain over the matched
  recurrent baseline.
- Measurable spatial-memory benefit after five minutes and occlusion.
- Frozen adaptation to held-out controls and movement physics.
- Thoughtlet removal specifically harms anticipation and recovery.

Do not collect full Minecraft-scale data until the model can retain a goal and
spatial belief for 30 minutes here.

### Phase 6: Minecraft

Expected duration: 6–18 months for a small research team after all earlier
gates.

Subphase A, motor/perception:

- 100–500 hours of precisely synchronized screen, keyboard, and mouse data.
- 1,000–5,000 hours of raw video for representation and inverse dynamics.
- Look, move, jump, interact, inventory, and camera tracking.

Subphase B, behavior foundation:

- 1,000–10,000 hours of high-quality or pseudo-labeled action data.
- Novice, exploratory, failed, recovery, and expert play.
- New seeds, UI scales, textures, controls, and rule changes.
- 1–10 billion online steps for navigation, collection, avoidance, and basic
  crafting.

Subphase C, one-checkpoint live evaluation:

- Discover remapped controls.
- Navigate to a visible target.
- Return to a previously seen place.
- Collect a specified nearby resource.
- Avoid or escape a common hazard.
- Recover from a movement error.
- Use inventory for a short crafting sequence.
- Maintain a goal for 10–30 minutes.
- Follow a simple language instruction without loading a new policy.

Gate, registered after a pilot:

- Thirty-minute sessions with no inference pause.
- Cognitive refresh p99 at or below 50 ms.
- Fast action/camera p99 at or below 33 ms, with 16.67 ms as the stretch goal.
- Deadline misses below 0.5 percent and no unintended input gap above 100 ms.
- At least 80 percent success on basic motor/interface tasks.
- At least 60 percent on held-out navigation/collection tasks.
- At least 25 percent on a preregistered multi-step crafting task.
- At least 10 percent adaptation-AUC gain over the matched recurrent baseline.
- Positive frozen-weight adaptation to remapped controls or a small rule
  change.
- The same checkpoint retains at least 90 percent of its prior procedural and
  arcade performance.

### Phase 7: general artificial-agent evaluation

Freeze one checkpoint and test at least ten held-out game families with no
identity signal or game-specific code. Give the model and novice humans matched
exposure and interface.

A credible first generalist result requires:

- Above-random behavior in at least eight of ten families.
- Measurable frozen-weight improvement in at least six.
- Adaptation AUC at least 70 percent of novice-human AUC in a majority.
- Deadline-compliant continuous interaction in every environment.
- Retention after returning to an earlier game.
- Causal thoughtlet benefits around anticipation, uncertainty, and surprise.

That would support the claim that one persistent model can interact with and
learn unfamiliar games in real time. It would still not establish
consciousness or human-general intelligence.

## 32. First 90 days

### Weeks 1–2: make experiments trustworthy

- Create the isolated brain package and lock dependencies.
- Establish safe ignore rules and a known source baseline without modifying
  Irene's existing live behavior.
- Write universal observation/control/state types.
- Build deterministic moving-shapes and hidden-dynamics environments.
- Implement QPC timestamping, recorder, branch schema, replay, and state hash.
- Build no-model capture/control loopback benchmark.
- Register train, validation, and held-out generator splits.
- Write the one-brain compliance test.

Deliverable: replayable one-million-step smoke dataset and systems report.

### Weeks 3–4: establish honest baselines

- Reactive CNN.
- CNN-GRU.
- Standard recurrent Transformer or SSM.
- Equal-state wider monolith.
- Equal-FLOP deeper serial updater.
- Fixed multi-horizon prediction heads.
- Common trainer, evaluator, profiler, and experiment manifest.

Deliverable: baseline prediction, control, adaptation, latency, and VRAM curves.

### Weeks 5–7: implement the thought field

- K=8 then K=16/32.
- R=3 registers.
- Persistent/refresh/expire gates.
- Three tied BrainCell cycles.
- Cycle-one privacy and sparse later routing.
- Direct actuator queries.
- Anytime exits and randomized deadline training.
- Event-set matching, slot permutation, dropout, and marginal-utility probes.

Deliverable: unit-tested forward/backward pass, recurrent replay equivalence,
and 100,000-step latency sweep.

### Weeks 8–9: create decisive data

- Add junctions, pursuit/evasion, occlusion, hidden physics, and remapped
  controls.
- Generate 10,000+ branch groups, then scale.
- Verify branch order independence and RNG restoration.
- Start lifetimes that preserve exploration, mistakes, and later competence.

Deliverable: versioned branch-DAG dataset and data-quality report.

### Weeks 10–11: train and ablate

- Train three seeds of the thesis MVP and all matched baselines.
- Sweep K and C within the local runtime envelope.
- Run persistence, integration-token, set-loss, and direct-readout ablations.
- Measure causal slot utility and duplicate rate.
- Run dropped-frame, delayed-control, and thermal-soak tests.

Deliverable: preregistered result table with confidence intervals.

### Week 12: go/no-go review

Answer:

1. Does the thought field improve multi-horizon prediction?
2. Does it improve control at equal compute?
3. Does it remain useful inside the physical deadline?
4. Are at least eight slots causally useful?
5. Does frozen state produce genuine within-lifetime improvement?

Proceed, redesign once, or stop according to the Phase 2 rules.

## 33. Initial repository layout

~~~text
brain/
  README.md
  PLAN.md
  pyproject.toml
  uv.lock
  configs/
    model/
    environment/
    experiment/
  docs/
    architecture.md
    dataset-spec.md
    evaluation-protocol.md
    decisions/
  schemas/
    lifetime.schema.json
    step.schema.json
    branch.schema.json
    experiment.schema.json
  src/irene_brain/
    model/
      sensory.py
      belief.py
      brain_cell.py
      thought_field.py
      memory.py
      actuator.py
      full_model.py
    data/
      records.py
      writer.py
      reader.py
      branching.py
      validation.py
    environments/
      protocol.py
      moving_shapes.py
      realtime_wrapper.py
      adapters/
    runtime/
      scheduler.py
      clocks.py
      capture/
      controls/
      watchdog.py
    training/
      losses.py
      matcher.py
      trainer.py
      replay.py
    evaluation/
      baselines.py
      metrics.py
      ablations.py
      compliance.py
  tests/
    unit/
    replay/
    leakage/
    integration/
  benchmarks/
    kernel/
    loopback/
    soak/
  scripts/
    collect.py
    train.py
    evaluate.py
    benchmark.py
  artifacts/
  runs/
  data/
~~~

Artifacts, checkpoints, videos, raw trajectories, caches, and run directories
must be ignored and stored outside source control or under explicit local-only
paths.

## 34. Engineering standards

Before the first expensive run:

- Pin Python, PyTorch, CUDA/runtime, JAX, and environment versions.
- Use typed dataclasses or schemas for every recurrent state and trajectory.
- Make all configs immutable once a run starts.
- Record code hash, data-manifest hash, checkpoint hash, environment hashes,
  seeds, hardware, driver, precision, and compiled profile.
- Unit-test tensor shapes, masks, missing modalities, slot permutation, and
  every anytime exit.
- Property-test snapshot/restore and branch ordering.
- Test that privileged target fields cannot enter model inputs.
- Check numerical finiteness and state version on every debug build.
- Snapshot recurrent state every few seconds for binary-searching divergence.
- Compare hidden-state hashes during deterministic replay.
- Keep a strict deterministic CI mode and a faster distributional benchmark
  mode.
- Never accept a result from a single seed.

## 35. Hardware plan

### Current workstation

Observed:

- AMD Ryzen 7 9800X3D, 8 cores / 16 threads.
- NVIDIA RTX 5070, 12,227 MiB VRAM.
- Roughly 9.45 GiB was free during inspection.
- Windows Performance Recorder is available.

Use it for:

- Environment development.
- Data replay and visualization.
- Small model training and smoke tests.
- All action-critical deployment inference.
- End-to-end capture/control profiling.

### DGX Spark

Use the existing Spark path for:

- Larger offline training.
- Branch-dataset preprocessing.
- Slow consolidation.
- Teacher labels where justified.
- Experiments whose memory exceeds the local 12 GB card.

Do not put Spark or any network service in the real-time action loop. Do not
collide with Irene's existing live model and voice service ports or GPU
allocations. Training throughput must be measured with a one-million-step pilot
before predicting a calendar duration.

### Scale-up triggers

- No hardware purchase is justified before the Phase 2 equal-compute gate.
- Multi-GPU training is justified only after data loading and environment
  actors keep one accelerator saturated.
- Minecraft-scale compute is justified only after the first-person 3D gate.

## 36. Major risks and mitigation

| Risk | Early test or mitigation |
|---|---|
| Thoughtlets become copies | Branch-set matching, private first cycle, sparse workspace, slot dropout, causal ablation |
| Extra capacity explains gains | Equal-parameter, equal-state, equal-FLOP, and equal-latency baselines |
| Action head ignores thoughts | Direct readout, random slot masking, learned marginal utility, zero/shuffle tests |
| Thoughts become stale | Age, confidence decay, time-to-live, event-triggered invalidation |
| Integration destroys detail | Actuator queries read all state; compare against one pooled token |
| Width misses deadlines | Static shapes, tied weights, sparse routes, compiled profiles, active-slot budgets |
| Agent imitates but does not learn | Novice/failure data, hidden rules, remapped controls, frozen-weight lifetime tests |
| Memorized game identity looks adaptive | No game ID, visual randomization, held-out families/mechanics/controls |
| Privileged state leaks | Schema separation, input allowlist, leakage tests, pixel-only replay |
| Long-term state drifts | 30–120 minute soak, gated memory, snapshots, reset/return protocols |
| Online RL collapses thoughts | Preserve predictive/set losses and old replay during policy improvement |
| Slow learning forgets old games | Balanced consolidation, anchor distillation, frozen cross-game regression |
| Reward hacking | Withheld outcome metrics, branch validation, human review |
| Physical I/O dominates | Independently measure capture, model, dispatch, and visible effect |
| Remote inference adds delay | Local deployment only |
| Minecraft consumes the project | Hard phase gates and capped data/compute |
| Licensing blocks data | Open procedural foundation and per-source rights ledger |
| Interpretability becomes storytelling | Accept only calibrated prediction and causal intervention evidence |
| Unsafe control | Game sandbox, fixed HID allowlist, watchdog, neutral state, hardware kill switch |

## 37. Program-level stop conditions

Pause scaling if any condition holds:

- Two thought-field designs with three seeds each fail to improve either
  prediction or control by 5 percent at matched compute.
- Fewer than four effective thoughtlets remain from K=32 across three accepted
  checkpoints.
- K=16 with three cycles still exceeds twice the deadline after one focused
  runtime optimization cycle.
- No frozen-weight adaptation appears after the capped procedural lifetime
  dataset.
- Arcade gains disappear under frame skip one, random starts, sticky/delayed
  actions, or captured-screen input.
- Minecraft success requires task IDs, state APIs, adapters, or
  evaluation-time gradients.
- Consolidation causes more than 20 percent persistent regression after one
  replay mitigation cycle.
- A phase exceeds its preregistered compute/data budget by more than twofold
  without a larger measured effect.
- A result cannot be reproduced from code, data, environment, checkpoint, and
  seed hashes.

Stopping the thesis is a successful scientific outcome if the experiment was
decisive.

## 38. Decisions fixed for implementation

The following choices are fixed for the first build so implementation can
start without further architecture debate:

- New code lives under brain and does not alter the deployed live package.
- Python 3.11 and PyTorch are the initial research stack.
- The first environment is original and deterministic.
- The canonical data representation is a lifetime plus branch DAG.
- The model begins at K=32, R=3, C=3, d=384 after smaller smoke variants.
- BrainCell weights are tied across cycles and thoughtlets.
- Cycle one uses private thought updates; cycles two and three use sparse
  communication.
- Motor readout uses fixed generic HID queries and no pooled integration token.
- Evaluation uses frozen weights for fast adaptation.
- Remote inference is forbidden in deadline measurements.
- Every claim includes matched baselines, multiple seeds, tail latency, and
  causal slot ablations.
- Open procedural games precede proprietary commercial games.

## 39. Decisions deferred until measured

These choices require data rather than preference:

- CNN versus small ViT visual stem.
- GRU-style versus attention-style belief update.
- Exact versus approximate top-r routing beyond K=32.
- Learned features versus privileged event features in the branch matcher.
- BF16, FP16, INT8, or lower runtime precision.
- Best screen resolution for each latency tier.
- Number and capacity of episodic entries.
- Whether audio materially helps before first-person 3D.
- Whether text should enter as cached instructions or a jointly trained small
  language stem.
- Exact actor-critic/world-model control algorithm.
- Native Windows inference backend after the PyTorch prototype.

Each will be settled by a registered ablation or systems benchmark.

## 40. Implementation progress & empirical milestones achieved

The implementation has substantially advanced beyond the initial Phase 0 backlog:

1. **Phase 0A Deterministic Core:** Universal `ModelObservation`, `GenericControl`, `TimestampTrace`, snapshot/restore, state hashing, branch DAG generation, and resource guards — fully implemented and verified.
2. **Phase 1 Model & Training Engine:** Single shared-weight `BrainCell` thought field ($d=384, K=4$), structured 307-channel action projection, anytime exits, DAgger on-policy distillation, and mixed-precision optimization — verified across 59 play-safe test modules.
3. **Cognitive Gating & Dynamic Depth ($A\text{--}E$ Ablation):** Implemented adaptive thought-update gating ($\alpha$), multi-horizon world modeling, and situation-dependent dynamic compute halting ($C \in [1, 6]$). Verified on 20 held-out seeds (Variant $E$: 5.65 pellets / 83 catches vs baseline 3.30 pellets / 727 catches).
4. **Counterfactual Foresight & Topological Routing ($F\text{--}G$ Variants):** Implemented `LatentLookaheadPlanner` and `TopologicalGoalFieldHead` for latent action-tree pruning and pellet cluster gradient routing.
5. **Procedural Skill Ladder:** Implemented pursuit/evasion (`pursuit.py`), maze junction choice (`junction.py`), occlusion memory (`occlusion.py`), key/door sequencing (`keys_doors.py`), and a 13-world $\times$ 8-policy canonical diagnostic matrix.

## 41. Research foundations

This project is not identical to any one prior system, but several primary
sources establish useful pieces:

- Recurrent Independent Mechanisms: sparse, selectively updated recurrent
  modules. https://arxiv.org/abs/1909.10893
- Coordination Among Neural Modules Through a Shared Global Workspace:
  bandwidth-limited coordination among specialist states.
  https://arxiv.org/abs/2103.01197
- Horde: many real-time predictive questions learned from one sensorimotor
  stream. https://arxiv.org/abs/1206.6262
- Hybrid Reward Architecture: many decomposed action-value considerations in
  Ms. Pac-Man, while relying on significant engineered structure.
  https://arxiv.org/abs/1706.04208
- Gato: one parameter set across many modalities, tasks, and embodiments.
  https://arxiv.org/abs/2205.06175
- AdA: human-timescale in-context adaptation emerging from a rich task
  distribution, memory architecture, and curriculum.
  https://arxiv.org/abs/2301.07608
- DreamerV3: a recurrent latent world model and one robust configuration across
  many control domains, including Minecraft.
  https://www.nature.com/articles/s41586-025-08744-2
- Video PreTraining: large-scale Minecraft video and synchronized
  keyboard/mouse action learning.
  https://openai.com/index/vpt/
- XLand-MiniGrid:
  https://github.com/dunnolab/xland-minigrid
- Craftax:
  https://github.com/michaeltmatthews/craftax
- Procgen:
  https://github.com/openai/procgen
- Craftium:
  https://github.com/mikelma/craftium
- Minecraft's documented 20-tick-per-second game loop:
  https://learn.microsoft.com/en-us/minecraft/creator/documents/scripting/introduction

These works demonstrate pieces of the problem. None by itself demonstrates the
specific target: one persistent, human-facing, deadline-bound model with a
causally useful field of many incomplete latent thoughts that adapts across
unfamiliar games.

## 42. Definition of the active scientific gate: the decisive matched-baseline comparison

The early implementation and internal ablation phases have established an important engineering result:

$$\text{Pseudo-Brain Version 2 (Full Adaptive System $E$)} \gg \text{Pseudo-Brain Version 1 (Baseline $A$)}$$

However, proving internal improvement across design iterations does not yet prove architectural superiority over standard paradigms. The decisive, make-or-break scientific gate for this project is:

$$\mathbf{Pseudo\text{-}Brain} \quad \text{vs.} \quad \mathbf{GRU} \quad \text{vs.} \quad \mathbf{Recurrent\ Transformer} \quad \text{vs.} \quad \mathbf{World\text{-}Model\ Controller}$$

strictly evaluated at **matched parameter counts, matched FLOP budgets, matched latency deadlines, and matched environment experience**.

### Hard Stopping Rule & Scientific Integrity
In accordance with §37 and [BASELINE_PROTOCOL.md](docs/BASELINE_PROTOCOL.md):
- If multiple thought-field configurations across multiple random seeds fail to achieve a statistically significant improvement ($\ge 10\%$ relative multi-horizon prediction gain and $\ge 10\%$ IQM return gain) over the strongest matched conventional baseline at equal compute, the parallel microthought thesis will be formally retired rather than endlessly moving the goalposts.
- The project proceeds to larger compute and wider domains (voxel worlds, language grounding, robotics) only if the persistent thoughtlet core demonstrates a causal, compute-matched advantage on this decisive benchmark.

## 43. Long-term research direction: general-purpose language, tool-use & long-horizon agent

This section defines the major long-term expansion of the single persistent cognitive architecture. It does **not** replace, rewrite, or deprioritize the current sensorimotor, predictive cognition, robotics, thoughtlet, ablation, and architecture-validation work (Phases 0–7). Rather, it formalizes how the very same recurrent thoughtlet core scales to general artificial agency across digital and physical domains.

### 43.1 Long-term north star

The ultimate objective is to develop Pseudo-Brain into a **general-purpose cognitive agent** that can:

- Understand natural-language instructions and maintain persistent context.
- Maintain a persistent, evolving understanding of a user's goal.
- Decompose complex, ambiguous, or multi-step tasks into dynamic subgoals.
- Perform exploratory research, query databases, and search the web.
- Read, parse, and critically analyze unstructured documents and codebases.
- Use software tools, command-line interfaces, and web/system APIs.
- Inspect, debug, modify, and refactor software source code.
- Execute programs, compile projects, and run test suites.
- Detect environmental, logical, or verification failures.
- Revise incorrect hypotheses and internal beliefs upon encountering contrary evidence.
- Remember unfinished tasks, open subgoals, and working constraints across long sessions.
- Compare candidate actions counterfactually in latent space prior to committing.
- Sustain autonomous work over extended multi-step, multi-hour horizons.
- Produce structured, fluent natural-language explanations and answers.
- Eventually operate both digital computing environments and physical robots using the exact same underlying cognitive architecture.

The architecture strictly unifies all inputs and outputs:
- **Observations:** Language tokens, visual frames, tool output strings, file contents, sensor streams, and environment state feedback are all treated as diverse forms of *sensory observations*.
- **Actions:** Tool calls, code edits, shell commands, database queries, natural language responses, and physical actuator torques/keypresses are all treated as diverse forms of *actions*.

The central cognitive engine coordinating this loop is the single, persistent Pseudo-Brain.

### 43.2 Core principle: the persistent cognitive loop

The unified cognitive loop across physical and digital tasks follows a strict predictive, belief-updating cycle:

~~~text
OBSERVATION
    ↓
Language / Vision / Tool / Sensor Encoding
    ↓
Persistent World + Task Belief State
    ↓
Parallel Persistent Thoughtlets
    ├─ Multi-Horizon Future Prediction
    ├─ Counterfactual Action Evaluation
    ├─ Uncertainty Estimation
    ├─ Episodic & Task Memory Retrieval
    └─ Dynamic Subgoal Tracking
    ↓
Adaptive Cognitive Depth (Cycle Halting: C ∈ [1, C_max])
    ↓
ACTION
    ↓
Tool Call / Motor Action / Code Edit / Shell Command / Text Response
    ↓
Observe Execution Result & Environment Consequence
    ↓
Compare Reality Against Latent Prediction (Prediction Error)
    ↓
Surprise-Triggered Thought Revision & Reconsideration
    ↺
~~~

The goal is **not** to convert Pseudo-Brain into a standard autoregressive Large Language Model (LLM). Language is treated as an **interface to cognition**, not the cognition itself.

### 43.3 Language as an input/output interface

Pseudo-Brain does not need to learn human grammar or natural language tokens from scratch in its early stages. Initial implementations utilize frozen or pretrained language foundation components as sensory and motor interfaces:

- **Language Encoder (Ears):** Converts natural language prompts, system messages, and task instructions into dense semantic latent embeddings.
- **Language Decoder (Mouth):** Projects Pseudo-Brain's final task state and thought-field latents into fluent, human-readable natural language text.

~~~text
User Language Input
       ↓
Language Encoder (Ears)
       ↓
Pseudo-Brain Cognitive Core (Belief + Thoughtlets + Memory)
       ↓
Language Decoder (Mouth)
       ↓
Natural Language Response
~~~

The pretrained language modules function strictly as sensory transduction and motor generation interfaces. Persistent planning, memory retrieval, task control, tool selection, counterfactual foresight, reconsideration, and goal tracking remain exclusive responsibilities of Pseudo-Brain's recurrent core.

### 43.4 Tool use as sensorimotor control

Software tools and operating system APIs are treated under the exact same sensorimotor paradigm as continuous physical robot actuators or game controllers:

~~~text
Thought Field Latent State
       ↓
Structured Tool Action Head
       ↓
Select Tool (Discrete Action Index)
       ↓
Generate Tool Arguments / Command Parameters
       ↓
Execute Tool (OS Command, HTTP Request, File I/O)
       ↓
Receive Structured Result (Stdout, Stderr, JSON, Return Code)
       ↓
Encode Result into Observation Tokens
       ↓
Update World & Task Belief
       ↓
Continue Thinking / Trigger Next Action
~~~

The action space expands cleanly to include:
- `SEARCH_WEB`: Query internet search indices.
- `READ_PAGE`: Ingest markdown/HTML web content.
- `READ_FILE`: View local filesystem content with byte/line offsets.
- `WRITE_FILE`: Create or overwrite filesystem artifacts.
- `EDIT_CODE`: Apply targeted contiguous or multi-chunk diff replacements.
- `RUN_COMMAND`: Propose and execute shell commands with output capture.
- `RUN_TESTS`: Invoke test runners and parse test suite summaries.
- `QUERY_DATABASE`: Execute structured SQL or vector queries.
- `CALL_API`: Invoke external REST or RPC interfaces.
- `OPEN_DOCUMENT`: Render and inspect PDF, image, or video media.
- `EXECUTE_PROGRAM`: Launch runtime interpreters or debuggers.
- `RESPOND_TO_USER`: Emit user-facing dialogue or completed deliverables.

Tool results immediately become new observations. This forms the exact same loop as robotics:
$$\text{Act} \longrightarrow \text{Observe Consequence} \longrightarrow \text{Update Belief} \longrightarrow \text{Act Again}$$

### 43.5 Task belief & persistent goals

To support long-horizon problem solving without context degradation, Pseudo-Brain maintains a dedicated **Task Belief / Goal State** that tracks:
- The overarching user objective and success criteria.
- Completed sub-tasks and verified milestones.
- Unfinished work and pending obligations.
- Invariant operational constraints (e.g. CPU-only execution, safety rules).
- Gathered empirical evidence and diagnostic logs.
- Unresolved questions and ambiguous requirements.
- Failed attempts and discarded hypotheses.
- Calibrated confidence and empirical uncertainty.
- Formal verification status against test suites.

By maintaining this persistent internal task representation, the system avoids needing to reconstruct the full task context from scratch at every interaction step.

### 43.6 Dynamic subgoals & parallel thoughtlet decomposition

Rather than executing a brittle, hardcoded checklist, complex tasks are decomposed dynamically:

~~~text
Main Goal: Fix Failing Software Project
   │
   ├─ Subgoal 1: Ingest and map repository architecture
   ├─ Subgoal 2: Reproduce failure with deterministic minimal test
   ├─ Subgoal 3: Localize root cause via diagnostic execution
   ├─ Subgoal 4: Apply candidate code modifications
   ├─ Subgoal 5: Execute test suite and inspect regressions
   ├─ Subgoal 6: Revise hypothesis if test fails
   ├─ Subgoal 7: Verify all constraints and style requirements
   └─ Subgoal 8: Synthesize walkthrough and report deliverable
~~~

Subgoals emerge, adapt, pause, resume, or abort dynamically based on real-time observations. Different thoughtlets can simultaneously maintain different aspects of the task (e.g. one tracking edge-case regression risks, another tracking tool execution syntax, another tracking high-level milestone progress).

### 43.7 Counterfactual tool reasoning & latent foresight

Extending action-conditioned foresight beyond physical navigation, the agent evaluates tool actions in latent space before execution:

~~~text
Given State z_t and Candidate Tool Actions {a_A, a_B}:

Option A: Modify Parser Directly
   ├─ Predicted outcome: Fixes immediate syntax error
   ├─ Expected usefulness: High (0.85)
   ├─ Uncertainty: Moderate (0.35)
   └─ Regression risk: High (0.70 - potential downstream breakage)

Option B: Modify AST Validation Layer
   ├─ Predicted outcome: Validates AST nodes before parser ingest
   ├─ Expected usefulness: High (0.82)
   ├─ Uncertainty: Low (0.15)
   └─ Regression risk: Low (0.10 - strictly additive validation)

Decision: Latent Lookahead Planner selects Option B.
~~~

This counterfactual evaluation is conducted as a lightweight latent rollout, avoiding exponential action-tree explosions while preventing catastrophic tool blunders.

### 43.8 Prediction error & multi-domain self-correction

The adaptive thought-revision ("change my mind") mechanism generalizes seamlessly from sensorimotor navigation to digital tasks:

| Domain | Expected Prediction | Actual Observation | Triggered Cognitive Response |
|---|---|---|---|
| **Physical Navigation** | Corridor is clear | Collision with wall / hazard | Gate surprise $\alpha \to 1.0$, flush momentum, re-route |
| **Tool Execution** | Command returns exit 0 | Command returns non-zero error / exception | Surprise trigger, parse error output, re-plan command |
| **Software Engineering**| Code edit fixes bug | Unit test fails with regression | Reject candidate fix, update failure log, revise hypothesis |
| **Research & Analysis** | Source confirms hypothesis | Empirical data contradicts hypothesis | Lower belief confidence, seek primary sources, revise conclusion |

Prediction mismatch is the primary operational signal that drives internal belief revision across all domains.

### 43.9 Explicit uncertainty representation & adaptive compute

Pseudo-Brain explicitly computes epistemic and aleatoric uncertainty over its internal task belief. It distinguishes:
$$\text{"I have high confidence in the next action."} \quad \text{vs.} \quad \text{"My understanding is weak or data is missing."}$$

When uncertainty is high, the system automatically:
- Allocates additional internal cognitive cycles ($C \to C_{\max}$) before emitting an action.
- Dispatches exploratory information-gathering tools (`READ_FILE`, `SEARCH_WEB`).
- Retrieves relevant prior solutions from episodic memory.
- Performs multi-branch counterfactual candidate comparisons.
- Enforces stricter verification gates before concluding.

### 43.10 Long-term episodic & task memory

The cognitive architecture includes an episodic memory store for compressed task experiences:
- Successful debugging strategies and verified bug fixes.
- Documented failure modes and dead-end approaches.
- Environment-specific quirks and tool constraints.
- Key empirical findings, benchmark receipts, and citation traces.

Retrieval is driven by associative relevance queries generated directly by the thoughtlets, preventing memory bloat while making past experiences available when facing analogous challenges.

### 43.11 Staged development roadmap (Phases 1 to 8)

The transition from sensorimotor foundations to general artificial agency is structured across eight concrete, verifiable phases:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Current Sensorimotor Foundation (Active Priority)             │
│          Thoughtlets, Gating, Multi-Horizon Prediction, Depth, Memory   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Simple Language Grounding                                      │
│          Natural language modifies persistent goal state in 2D/3D worlds│
└────────────────────────────────────┬────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Basic Tool Interface                                           │
│          Tool action head (READ_FILE, SEARCH, RUN_COMMAND, WRITE_FILE)  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Multi-Step Tool Tasks                                          │
│          Goal retention across 5–10 sequential tool calls & verification│
└────────────────────────────────────┬────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 5: Self-Correcting Tasks                                          │
│          Error detection, belief revision, and recovery during tool play│
└────────────────────────────────────┬────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 6: Coding Agent                                                   │
│          Reproduce bug, apply fix, run test suites, explain rationale   │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 7: Research Agent                                                 │
│          Multi-source investigation, citation verification, uncertainty │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Phase 8: Open-Ended General Agent                                       │
│          Autonomous subgoal creation & long-horizon task completion     │
└─────────────────────────────────────────────────────────────────────────┘
```

1. **Phase 1 — Current Sensorimotor Foundation (Active Priority):**
   - Persistent thoughtlets, adaptive gating, multi-horizon world modeling, counterfactual foresight, dynamic cognitive depth, and continuous physical control.
2. **Phase 2 — Simple Language Grounding:**
   - Ground natural-language instructions (e.g. *"Move to the red key"*, *"Evade the orange hazard"*) directly into the goal context vector.
   - Measure zero-shot instruction following in unfamiliar environments.
3. **Phase 3 — Basic Tool Interface:**
   - Introduce a minimal discrete tool action vocabulary (`READ_FILE`, `SEARCH`, `RUN_COMMAND`, `WRITE_FILE`).
   - Supervise tool selection and argument generation.
4. **Phase 4 — Multi-Step Tool Tasks:**
   - Benchmark multi-hop tasks (e.g. *"Find config.json and report the server port"*).
   - Evaluate goal retention, state tracking, and answer extraction.
5. **Phase 5 — Self-Correcting Tasks:**
   - Tasks requiring active recovery (e.g. *"Modify this configuration, run the application, and fix any resulting syntax errors"*).
   - Measure error detection latency, thought revision, and recovery rate.
6. **Phase 6 — Autonomous Coding Agent:**
   - Software engineering benchmarks (bug localization, minimal reproduction, test suite execution, regression avoidance).
   - Measure task completion, patch cleanliness, and recovery from incorrect diagnoses.
7. **Phase 7 — Autonomous Research Agent:**
   - Open-ended investigation of technical claims with conflicting evidence.
   - Evaluate source evaluation, citation fidelity, uncertainty calibration, and synthesized findings.
8. **Phase 8 — Open-Ended General Cognitive Agent:**
   - Complex repository refactoring, novel prototype construction, and multi-domain problem solving.
   - Dynamic subgoal creation and long-horizon execution across hours of autonomous operation.

### 43.12 Unification with physical robotics

A central thesis of Pseudo-Brain is that physical robotic control and digital software manipulation share the exact same underlying cognitive loop:

~~~text
                          PSEUDO-BRAIN COGNITIVE CORE
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ↓                          ↓                          ↓
     Physical Robot              Computer Tools            Natural Language
      • Motor Torques             • Shell Commands          • User Dialogue
      • Joint Angles              • Code Edits              • Explanation
      • Gripper Actuation         • API Invocations         • Structured Reports
~~~

Whether navigating around a dynamic physical obstacle or recovering from a failing unit test, the agent executes the identical abstract cycle:
$$\text{Maintain Goal} \to \text{Predict Consequence} \to \text{Act} \to \text{Observe Reality} \to \text{Detect Mismatch} \to \text{Revise Belief} \to \text{Advance Goal}$$

### 43.13 Non-negotiable architectural rule

The addition of language and tool capabilities must **never** degrade Pseudo-Brain into a superficial wrapper around an external Large Language Model.

Pretrained encoders and decoders may be utilized as perceptual and vocal interfaces, but experiments must rigorously isolate and verify that Pseudo-Brain's internal recurrent mechanisms provide measurable, causal advantages for:
- Persistent goal retention over long sequences.
- Tool selection and argument precision.
- Latent counterfactual evaluation before execution.
- Adaptive compute allocation under uncertainty.
- Fast belief revision upon tool failure.
- Robustness against unexpected observation shifts.

The cognitive contribution of the thought field must remain independently and empirically verifiable.

### 43.14 Scientific validation protocol

When language and tool phases commence, Pseudo-Brain variants must be benchmarked against standard autoregressive and reactive agents under controlled, matched conditions:
- **Task Completion Rate:** Percentage of multi-step tasks fully solved.
- **Tool Efficiency:** Total tool calls required per completed objective.
- **Error Recovery Rate:** Successful resolution after encountering unexpected tool failures.
- **Goal Retention:** Stability of the primary objective over long horizons ($> 100$ steps).
- **Latency & Compute:** End-to-end wall-clock time and FLOP consumption.
- **Hallucination / Regression Rate:** Introduction of false claims or broken codebase invariants.
- **Cognitive Depth Scaling:** Measured performance gains as dynamic cycle count $C$ scales.
- **Thoughtlet Scaling:** Performance and robustness improvements as slot count $K$ scales.

Claims of architectural superiority will be accepted only with preregistered protocols, multi-seed statistical replication, and confidence intervals.

### 43.15 Ultimate program objective

The long-term mission of Pseudo-Brain is:

> **A single persistent cognitive architecture capable of receiving natural-language goals, reasoning continuously through parallel internal thought states, predicting possible futures, using physical or digital tools, recognizing when its beliefs are wrong, correcting itself, remembering unfinished work, and completing long-horizon tasks without requiring every intermediate thought to be represented as language.**
