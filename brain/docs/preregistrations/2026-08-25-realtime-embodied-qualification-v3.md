# Real-time embodied Pseudo-Brain qualification v3

**Date:** 2026-08-25
**Status:** documentation-only design; no run is authorized
**Scope:** Pac-Man-like and Minecraft-like environments
**Claim class:** operational embodied-agent qualification, not evidence of
consciousness or a human-equivalent artificial brain

## 1. Purpose and present-system boundary

This contract defines the evidence required before calling one recurrent
Pseudo-Brain checkpoint a working real-time embodied-agent prototype. It is
downstream of the current component-learning program. It cannot open TEST,
authorize DGX training, nominate a recipe, or supersede a failed scientific
gate. A separate create-only run registration must bind the implementation,
checkpoint, hardware, partitions, human protocol, seeds, and evidence schema
before execution.

Canonical Core V2 currently has `W=120`, `K=32`, `C=3`, and one five-class
`idle/W/A/S/D` action output. It has no environment token and no common
Minecraft-capable action schema. Therefore the current implementation is not a
Q0 candidate. A versioned `EmbodiedInterfaceV1` architecture and its fresh
component/integration qualification are prerequisites, not changes authorized
by this document.

All gates are conjunctive. A favorable mean cannot compensate for a failed
timing, task, robustness, recurrence, safety, or provenance gate.

## 2. Versioned interface prerequisite

`EmbodiedInterfaceV1` must expose one common model-output record on every tick:

| Field | Frozen values |
|---|---|
| `locomotion` | `NOOP`, `FORWARD`, `BACKWARD`, `LEFT`, `RIGHT` |
| `look_yaw_deg` | `-30`, `-15`, `-5`, `0`, `5`, `15`, `30` |
| `look_pitch_deg` | `-30`, `-15`, `-5`, `0`, `5`, `15`, `30` |
| `cursor_dx_px` | `-64`, `-16`, `0`, `16`, `64` |
| `cursor_dy_px` | `-64`, `-16`, `0`, `16`, `64` |
| `buttons` | independent booleans for `jump`, `crouch`, `sprint`, `attack`, `use`, `inventory` |
| `hotbar` | `HOLD`, `1` through `9` |

Every field is emitted directly by frozen model heads. A command lasts exactly
one native control tick; holding a key requires the model to emit it again.
There are no scripted action sequences, options, macros, planners, or
adapter-owned state.

Each environment may have a separately hashed adapter, but an adapter may only
resize/normalize pixels, serialize the common output, translate it to ordinary
key/mouse events, and ignore fields that the environment physically lacks. It
may not observe rewards, hazards, privileged state, maps, coordinates, task
labels, or history; choose an action; resolve a plan; persist state; or inject
an environment/task ID. The candidate receives no environment token or task
metadata. Any later environment token or richer action interface is a new
version requiring a new preregistration.

When a task requires an explicit goal, that goal appears only as an ordinary
in-world sign, book, or rendered GUI panel inside the captured pixel stream.
The pixels, presentation duration, and interaction needed to view it are
identical for agent and human participants. No goal ID, task index, text side
channel, adapter flag, API argument, or out-of-band instruction is supplied.

## 3. Frozen recurrent candidate

A candidate consists of one parameter set across both domains with:

- Core V2 width `W=120`;
- `K=32` thoughtlets represented as one `[B,32,120]` tensor;
- `C=3` serial recurrent thought cycles per environment tick;
- shared-weight thoughtlet updates and registered attention/aggregation paths;
- persistent online `CoreV2State`; and
- the frozen, already-qualified `EmbodiedInterfaceV1` heads and stateless
  format adapters.

Within each of the three serial cycles, K must be implemented as batched tensor
operations with no Python or host loop over thoughtlet index. This is called
vectorized/parallel thoughtlet processing; this contract makes no claim that
physical hardware executes all 32 elements at literally the same instant.

`W`, `K`, and `C` are part of checkpoint identity. The initial qualification
is W120/K32/C3. A changed value requires a separately versioned, fresh scaling
preregistration and cannot inherit this qualification.

During qualification there are no gradients, optimizer steps, replay search,
tree search, serial verbal chains, future frames, simulator internals,
privileged coordinates, checkpoint switching, or task-specific weight changes.

## 4. Clock, queue, and no-pause contract

### 4.1 Authoritative events

The harness assigns every native cycle a `cycle_id`, every rendered observation
a `frame_id`, and every output an `action_id`. Each event has a host-side
`time.perf_counter_ns` receipt from one process-wide monotonic clock and, when
available, its native source timestamp:

- `t_emit`: capture source declares `frame_id` complete and readable;
- `t_enqueue`: that frame enters the observation queue;
- `t_dequeue`: controller removes that exact frame;
- `t_infer_start` and `t_infer_end`;
- `t_dispatch`: the bound action is sent; and
- `t_accept`: a game-side input hook acknowledges that `action_id` was polled
  for the target native cycle.

A host-clock event has normalized timestamp `hat(t)=t` and uncertainty
`epsilon=0`. For an endpoint originating in another clock domain, an affine
mapping to the host clock is fitted only from CAL synchronization pulses and
frozen before TEST. Its endpoint uncertainty is the maximum absolute CAL
mapping residual plus both timestamp resolutions. Each endpoint uncertainty
must be at most 0.25 ms for Pac-Man-like and 1.00 ms for Minecraft-like;
otherwise Q1 is invalid.

The conservative authoritative latency is

```text
L_plus = (hat(t_accept) - hat(t_emit)) + epsilon_accept + epsilon_emit.
```

Every latency summary and threshold uses `L_plus`. Endpoint uncertainty is
summed, never averaged, canceled, or subtracted. This covers capture
availability, queueing, preprocessing, device transfer, inference, transport,
and game-side input acceptance. Model/device timers are diagnostic only.

### 4.2 Queue and correspondence

The observation queue has capacity one and latest-complete-frame semantics.
Overwrite, duplicate, stale-frame reuse, and empty-dequeue events are logged.
Pac-Man-like scheduled starts are exactly
`s_n=t_0+floor((n*1_000_000_000+30)/60)` ns; Minecraft-like starts are exactly
`s_n=t_0+n*50_000_000` ns. The deadline is `d_n=s_(n+1)` and the cycle's
integer-nanosecond period is `P_n=d_n-s_n`.

At `s_n`, the controller must dequeue the newest frame with
`t_enqueue<=s_n`. The action record carries its exact `frame_id`, `cycle_id`,
and `action_id` through acceptance. Any ambiguous correspondence or an action
whose conservative age
`t_dequeue-hat(t_emit)+epsilon_emit>P_n` is a deadline miss. No inference
backlog or hidden secondary queue is permitted.

A miss is `hat(t_accept)+epsilon_accept>d_n`. A late action is
retained in evidence and the preceding action is held. Per-cycle schedule jitter
is

```text
J_plus_n = abs((hat(t_accept_n) - hat(t_accept_n-1))
               - (s_n - s_n-1))
           + epsilon_accept_n + epsilon_accept_n-1.
```

The first action after process start has no jitter value. All later actions,
including actions across automatic episode reset during endurance, count.
Quantiles use the nearest-rank definition over all eligible ticks.

### 4.3 Frozen timing gates

| Gate | Pac-Man-like | Minecraft-like |
|---|---:|---:|
| Native control rate | 60 Hz | 20 Hz |
| Nominal period | 16.666667 ms | 50.000000 ms |
| Median `L_plus` | <= 8.00 ms | <= 25.00 ms |
| p95 `L_plus` | <= 13.00 ms | <= 40.00 ms |
| p99 `L_plus` | <= 16.00 ms | <= 48.00 ms |
| Deadline-miss rate | <= 0.1% | <= 0.1% |
| p99 `J_plus` | <= 2.00 ms | <= 5.00 ms |
| Consecutive misses | zero | zero |

The game clock may never pause, slow, frame-advance, or wait for inference.

Q1 requires two independent live two-hour runs per domain, using fresh
processes and disjoint registered environment seeds. Both runs use the complete
capture-to-game-input path and both must pass every gate; a replay is not the
second live run. Automatic resets occur without stopping the native schedule.

Separately, all active scored ticks from Q2--Q5 are pooled by domain and must
pass the same latency, miss, jitter, correspondence, and no-pause gates. Each
task/perturbation family must independently satisfy the miss-rate and
no-consecutive-miss gates. Endurance cannot substitute for scored timing, and
scored timing cannot substitute for endurance.

## 5. Partitions, sampling, and shared statistics

TRAIN, DEV, CAL, and TEST are disjoint by procedural family and map/world seed.
Architecture, weights, interface heads, adapters, thresholds, reset policy,
quantization, and runtime freeze after CAL. TEST is opened exactly once.

At least three preregistered training seeds are evaluated. There is no best-seed
selection: every resulting checkpoint uses one unchanged identity across both
domains and must independently pass point/support gates. Grand comparisons
average the fixed checkpoints; initialization is not resampled.

Every attempted episode counts. Crash, timeout, invalid output, resource
failure, state corruption, and watchdog termination count as failures.
Intervals use 50,000 preregistered whole-episode/world-seed cluster-bootstrap
draws. Frames are never resampled independently. Clean/perturbed and
candidate/baseline comparisons share seed-paired draws. Primary gates use
one-sided 97.5% bounds. All registered limbs must pass; no endpoint, task,
checkpoint, or perturbation may be selected after inspection.

For a retention ratio `R=X_perturbed/X_clean`, every bootstrap draw with
`X_clean<=0` is assigned `R=-infinity`, never dropped or repaired. The ratio is
defined only when the clean point estimate is positive and its one-sided 97.5%
lower bound is greater than zero; otherwise the ratio is missing and its gate
fails.

## 6. Human reference and human-speed estimands

Exactly 24 novice-to-intermediate participants receive a frozen familiarization
budget. Each then plays 20 Pac-Man-like TEST episodes balanced across difficulty
families and two Minecraft-like TEST worlds for each Q4 task. Assignment uses a
preregistered balanced incomplete block so agent, human, and random/reference
records share manifest IDs. Each participant must contribute at least 20
eligible preregistered surprise events per domain or the reaction-time support
gate fails.

The bounded human-comparison metric is:

- Pac-Man-like `M_P`: `0.5 * attained_score/max_attainable_score + 0.5 *
  survival_ticks/max_ticks`, clipped to `[0,1]` per episode; and
- Minecraft-like `M_M`: the equal-weighted mean success over the six Q4 tasks.

For each of the 50,000 draws, participants are sampled with replacement, then
whole episode/world blocks are sampled with replacement within participant.
The matching agent and random/reference result for each selected manifest ID is
carried in the same draw. This nested participant-plus-episode paired bootstrap
is used for all human contrasts, including medians and ratios.

Define `D_RT=median_human_RT-median_agent_RT` and

```text
N = (M_agent - M_random) / (M_human - M_random).
```

No ratio draw is discarded. If its denominator is nonpositive, that draw is
`-infinity`. `N` is reportable only if the human-minus-random point denominator
is at least 0.10 and its one-sided 97.5% lower bound is greater than zero;
otherwise `N` is recorded missing and the human-speed claim fails.

A human-speed claim requires Q1, a one-sided 97.5% lower bound for `D_RT` at
least zero in each domain, and a one-sided 97.5% lower bound for `N` at least
0.50 in each domain. It is not a human-level-performance claim.

## 7. Q0--Q6 qualification ladder

### Q0 — Interface and causal integrity

Require successful fresh prerequisite qualification of `EmbodiedInterfaceV1`;
exact source, schema, adapter, model, manifest, and environment hashes; TEST
secrecy before candidate freeze; deterministic replay agreement; causal
observation checks; and byte-identical checkpoint state before and after
evaluation. Prove that adapters are stateless and format-only and that no
future, privileged, environment-token, or task-label input enters the model.

**Permitted claim:** the benchmark harness and candidate identity are valid.

### Q1 — Real-time closed loop

Both independent live endurance runs and the separately pooled scored ticks
must pass every clock, queue, latency, jitter, deadline, correspondence, and
native-clock gate in Section 4.

**Permitted claim:** the model acts continuously at native game speed.

### Q2 — Pac-Man-like sensorimotor competence

TEST contains 200 episodes: five deterministic episode replicates on each of 40
unseen layouts, balanced over five preregistered difficulty families. Primary
outcomes are `M_P`, raw attained score, and survival fraction. Collision rate,
levels completed, legal-action rate, and hazard-reaction latency are mandatory
secondary reports.

For every primary outcome, the candidate-minus-random and
candidate-minus-reactive paired one-sided 97.5% lower bounds must exceed zero.
Every difficulty-family point estimate must exceed both baselines. Legal-action
rate must be at least 99%, and scored Q1 gates must pass.

**Permitted claim:** the recurrent controller plays a rapidly changing arcade
environment in real time.

### Q3 — Pac-Man-like generalization and robustness

Use the same 40 layouts and five registered stochastic replicates per layout,
paired to clean Q2, for each of six conditions: palette/texture shift; exactly
10% deterministic frame loss; exactly one-frame observation delay; game speed
at 90%; game speed at 110%; and altered enemy-policy parameters. Thus each
condition has 200 episodes per checkpoint.

Primary endpoints are `M_P` and survival fraction. For each endpoint and each
condition:

- perturbed candidate minus the registered reactive baseline must have a
  positive paired one-sided 97.5% lower bound;
- the point estimate must exceed the matched random baseline; and
- the paired retention ratio `perturbed/clean` must satisfy the Section 5
  denominator rule and have a one-sided 97.5% lower bound at least 0.75.

Every condition must retain the scored timing gates.

**Permitted claim:** competence generalizes beyond memorized layouts under the
six registered perturbations.

### Q4 — Minecraft-like embodied competence

The six frozen task families are:

1. navigate to a visible landmark;
2. traverse a registered obstacle course;
3. gather at least four wood logs;
4. craft a crafting table and one wooden pickaxe;
5. acquire food and survive the registered interval; and
6. build the registered minimal shelter before night.

Each task uses 30 unseen TEST world seeds per checkpoint. The goal is presented
only through the ordinary rendered pixels/UI seen identically by human players,
under the Section 2 presentation contract. Coordinates, inventory APIs,
command interfaces, world-state queries, macros, task labels, goal IDs, and
out-of-band instructions are forbidden.

For every task, success point estimate must be at least 60% and its one-sided
97.5% lower bound at least 40%. Candidate-minus-random and
candidate-minus-reactive seed-paired success differences must each have a
positive one-sided 97.5% lower bound. Grand invalid-action rate must be at most
1%, and scored Q1 gates must pass.

**Permitted claim:** the same checkpoint performs multiple real-time
Minecraft-like behaviors.

### Q5 — Minecraft-like robustness and long horizon

For each Q4 task, run the same 30 paired clean world seeds under each of six
frozen perturbations: held-out biome/generation parameters; texture-pack shift;
resource displacement; changed starting orientation; exactly 10% deterministic
frame loss; and registered hostile-mob changes. Each perturbation therefore has
180 task-world sessions per checkpoint.

The primary perturbation endpoint is equal-task-weighted success `S`. For every
perturbation:

- `S` point estimate must be at least 50%;
- perturbed candidate minus perturbed reactive baseline must have a positive
  seed-paired one-sided 97.5% lower bound; and
- paired `S_perturbed/S_clean` must satisfy the Section 5 denominator rule and
  have a one-sided 97.5% lower bound at least 0.75.

The long-horizon test uses 30 additional unseen world seeds per checkpoint and
one uninterrupted 30-minute live trial per seed. Success requires, in order:
acquire at least four logs; craft a table and wooden pickaxe; construct a
registered sealed shelter with a door; and remain alive inside it at the first
night checkpoint. Success point estimate must be at least 40%, its one-sided
97.5% lower bound at least 20%, and candidate-minus-reactive success difference
must have a positive one-sided 97.5% lower bound.

No recurrent reset, macro, safety violation, or failed applicable timing gate
is permitted.

**Permitted claim:** the controller preserves useful online state during
extended, perturbed interaction.

### Q6 — One-model cross-domain qualification

Q0--Q5 must pass with the exact same checkpoint, W120/K32/C3 recurrent core,
`EmbodiedInterfaceV1` heads, inference graph, quantization, routing/aggregation,
and runtime parameters. Only the frozen stateless format adapters may differ.

**Permitted claim:** one recurrent Pseudo-Brain checkpoint controls two
materially different real-time game domains.

## 8. Recurrent-state and thoughtlet causal gates

### 8.1 Exact recurrent state interventions

The recurrent snapshot is a deep, non-aliased copy of exactly:

- `fast.belief [B,120]` and `fast.thoughts [B,32,120]`;
- every optional `fast.prediction_error` field: `latent_error`,
  `cognitive_error`, `reward_error`, `hazard_error`, `confidence_error`, and
  `surprise`;
- `fast.retrieved_memory`;
- every optional `fast.pending_prediction` field: `predicted_next_latent`,
  `predicted_reward`, `predicted_reward_logits`, `predicted_hazard`,
  `predicted_confidence`, and `predicted_branch_logit`;
- `session.session_latent [B,120]` and `session.trial_index`;
- the complete `session.session_stats` mapping, including every key and value;
  and
- all `session.episodic_memory` internals: `capacity`, `retrieval_k`,
  `write_pointer`, and the ordered `entries` list. Every entry includes `key`,
  `value`, `action`, `predicted_consequence`, `observed_consequence`,
  `prediction_error`, `reward`, `hazard`, `confidence`, `trial_index`, and
  `timestamp`, including all nested tensor/dictionary values.

Snapshot serialization, restoration, and hashes must cover those fields. An
unlisted mutable field added by a later code version invalidates this v3
contract until the snapshot schema is versioned again.

Throughout qualification, harness-supplied `actual_reward` and `actual_hazard`
inputs to `CoreV2.forward` are `None`. A non-`None` value is allowed only if it
is produced inside the hashed candidate solely from the same ordinary rendered
pixel history available to the agent; its derivation must be frozen before TEST
and may not query an environment reward, event, or hazard API.

At every registered intervention tick after the first eight native ticks, and
immediately before `CoreV2.forward`, compare the frozen full candidate with:

- **zero state:** zero belief, thoughts, and session latent; set prediction
  error, retrieved memory, and pending prediction to `None`; set
  `session.trial_index=0` and `session.session_stats={}`; retain the frozen
  episodic-memory `capacity` and `retrieval_k` but set `entries=[]` and
  `write_pointer=0`; preserve only the current rendered observation and factual
  previous action, with reward/hazard inputs governed by the `None` rule above;
  and
- **within-trajectory donor state:** replace the complete snapshot with one from
  a different simultaneously evaluated episode at the same task, condition,
  and tick ordinal, using a preregistered no-fixed-point episode derangement.
  Restore the donor's trial index, session stats, memory capacity, retrieval K,
  ordered entries, and write pointer together with every tensor; do not merge
  recipient and donor fields.

Donors are never drawn across tasks, conditions, or tick ordinals. At least 20
live donor trajectories must exist at every intervention tick; otherwise the
ablation support gate fails. This intervention cannot become a no-op through an
episode-boundary reset.

The full candidate must beat both interventions on `M_P` and Minecraft
equal-task-weighted success in each domain with positive paired one-sided 97.5%
lower bounds.

### 8.2 Exact thoughtlet interventions

Without retraining, test:

- **all-zero:** zero the `[B,32,120]` output of `ThoughtFieldV2.forward`
  immediately before HYPOTHESIZE;
- **group drop:** at the same point, zero each frozen group `[0:8]`, `[8:16]`,
  `[16:24]`, and `[24:32]` separately;
- **routing knockout:** within each of the three serial cycles, replace
  `attn_out` with zero immediately before the residual addition while leaving
  `BrainCellV2` unchanged; and
- **permutation control:** apply each of 20 preregistered whole-K permutations
  immediately before `BrainCellV2` in every cycle and invert it immediately
  after attention/residual before the next cycle or output.

The full candidate must beat all-zero and routing-knockout on both domain
primary metrics with positive paired one-sided 97.5% lower bounds. At least
three of four group-drop point estimates must favor full, and the prespecified
equal-group pooled comparison must have a positive paired one-sided 97.5% lower
bound. Every permutation must preserve action logits within `1e-5` absolute
error and preserve discrete actions.

Source inspection plus profiler tensor/kernel shapes must prove one batched
K-dimension implementation with no host/Python loop over K. The profiler must
also show the intentionally serial three-cycle loop. This is evidence of
vectorization and causal utility, not physical simultaneity, interpretable
thoughts, subjective experience, or consciousness.

## 9. Safety contract

Minecraft runs only in disposable offline worlds or an isolated allow-listed
server. The candidate has no shell, filesystem-write, unrestricted network,
chat, command-block, purchase, account, or public-server capability. The
harness provides an action allow-list, rate limiter, external watchdog,
emergency stop, and registered RAM, VRAM, power, and temperature limits.
Invalid and rejected actions remain in evidence. Sandbox escape, unauthorized
action, public-world connection, disabled watchdog, or uncontrolled resource
use invalidates Q0--Q6.

## 10. Required evidence

A valid create-only result bundle contains:

- the timestamped run registration and TEST-manifest commitment;
- source, dependency, environment, interface, checkpoint, adapter, hardware,
  clock-map, queue, and benchmark hashes;
- fresh `EmbodiedInterfaceV1` prerequisite evidence;
- training/CAL provenance and proof of TEST non-use before freeze;
- every raw timestamp, clock-domain mapping receipt, cycle/frame/action ID,
  queue event, deadline decision, and correspondence check;
- both independent live Q1 endurance runs per domain, not replays;
- synchronized raw video/action records and all scored timing traces;
- all episodes and failures, fixed bootstrap indices, and nested human data;
- exact recurrent snapshots/donor maps and every state/thoughtlet ablation;
- profiler graph evidence for batched K and serial C;
- pre/post checkpoint hashes, safety/watchdog logs, and a machine-readable
  Q0--Q6 gate matrix; and
- deterministic Q0 replay evidence, which is separate from live Q1 evidence.

Any identity mismatch, overwrite, omitted failure, nonfinite statistic,
unexpected key, TEST access before freeze, ambiguous action correspondence, or
checkpoint mutation makes the attempt terminal-invalid. Failed scientific gates
are not rerun under the same registration.

## 11. Claim boundary and later breakthrough standard

Passing Q0--Q6 plus the human-speed, recurrence, and thoughtlet gates permits
only:

> A working real-time recurrent embodied-agent prototype, using one
> W120/K32/C3 checkpoint with a vectorized 32-thoughtlet field in each of three
> serial recurrent cycles, demonstrated across Pac-Man-like and Minecraft-like
> environments.

It does not establish consciousness, a human-equivalent artificial brain,
general intelligence, safe open-world autonomy, literal simultaneous thought,
or monotonic scaling.

A breakthrough claim requires a separate fresh preregistration showing all of:
broad transfer to multiple unseen game families; adaptation without
task-specific retraining; human noninferiority across a diverse battery;
replicated positive capability slopes over at least three model, fresh-data,
and compute scales with at least three seeds per scale; no component, timing,
robustness, or safety regression; and independent external reproduction.
