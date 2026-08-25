# V2.1f same-state all-action supervision contract

Status: design only; no training is authorized by this document.

Version identifiers:

- transition contract: `irene.maze_chase.all_action_transition.v1`
- outcome table: `irene.discrete_action_outcome_table_target.v1`
- generator: `irene.maze_chase.planner_labels.factual_rollout.all_action_branches.v1`
- manifest hash domain: `IRMCALLACTION\x01`

## Purpose and non-negotiable semantics

At factual state `s_t`, the simulator is branched from one exact pre-action
snapshot under all five discrete actions. These five one-step outcomes supervise
the model's all-action outcome table. Exactly one behavior action is then taken
from that same root to advance the real trajectory.

There is only one recurrent state transition per tick, and it is the factual
transition. Counterfactual observations are targets only: they are never fed
back as later observations, never update belief or thoughts, and never create
additional expert-label exposures.

The expert is queried exactly once on the factual root observation. Its action
is the single imitation target for that state. It is not re-queried inside
branches, and it is not replaced with a hindsight action selected from branch
rewards.

For v1, fail closed unless `input_delay_ticks == 0` and
`sticky_direction == False`. This makes each action-table row identify both the
requested action and the action physically applied during that one-step branch.
Delayed or sticky actuation requires a later contract with an explicit actuator
queue state and a longer intervention horizon.

## Canonical action vocabulary

The action IDs are the existing Core V2 classes:

| ID | Semantic control |
|---:|---|
| 0 | idle |
| 1 | W |
| 2 | A |
| 3 | S |
| 4 | D |

IDs are semantic labels, not positional assumptions. The generator emits rows
in ascending ID order, but every consumer must still join predictions and
targets by `action_id`. Tests must permute rows to prove order independence.

## Proposed immutable data types

The types below are a contract sketch, not an instruction to modify the shared
historical `MovingShapesTransition` schema.

```python
@dataclass(frozen=True, slots=True)
class ActionBranchOutcomeTargetV1:
    action_id: int
    requested_control: GenericControl
    applied_control: GenericControl
    next_observation_target: ModelObservation
    reward_target: float
    event_targets: tuple[str, ...]
    terminated_target: bool
    truncated_target: bool
    result_state_sha256: str


@dataclass(frozen=True, slots=True)
class AllActionOutcomeTableTargetV1:
    schema_id: str
    root_observation_content_hash: str
    root_state_sha256: str
    rows: tuple[ActionBranchOutcomeTargetV1, ...]

    def row_for_action(self, action_id: int) -> ActionBranchOutcomeTargetV1: ...


@dataclass(frozen=True, slots=True)
class AllActionSupervisedTransitionV1:
    # Existing object, unchanged. Contains observation, one applied factual
    # control, one expert action_target, and one factual outcome.
    factual: MazeChaseTransition
    outcomes: AllActionOutcomeTableTargetV1


@dataclass(frozen=True, slots=True)
class AllActionSupervisedSequenceV1:
    # Reuses the existing sequence and its continuity/value invariants.
    factual_sequence: MazeChaseSequence
    outcome_tables: tuple[AllActionOutcomeTableTargetV1, ...]


@dataclass(frozen=True, slots=True)
class AllActionTrajectoryBatchV1:
    split: str
    burn_in_steps: int
    sequences: tuple[AllActionSupervisedSequenceV1, ...]
```

Nesting the historical factual transition/sequence avoids changing old hashes
or silently widening `TrajectoryBatch`. A V2.1f objective must require
`AllActionTrajectoryBatchV1`; passing a legacy batch must fail rather than fall
back to factual-only supervision.

`result_state_sha256` records the complete post-step simulator state, including
hidden RNG state. The model never receives it. It exists only to prove branch
determinism and execution-order independence. Likewise, `root_state_sha256` is
provenance, not a model feature.

## Per-table invariants

Construction must reject a table unless all of these hold:

1. `schema_id == "irene.discrete_action_outcome_table_target.v1"`.
2. There are exactly five rows.
3. Row action IDs are unique and their sorted values are exactly
   `(0, 1, 2, 3, 4)`.
4. Each requested control is the canonical control for its action ID.
5. In v1, `applied_control == requested_control` for every row.
6. Every branch began from the identical `root_state_sha256` and identical root
   observation content hash.
7. Every next observation has `frame_id == root.frame_id + 1`, later elapsed
   time, and `previous_control == applied_control`.
8. Reward is finite; events are immutable non-empty strings; terminal and
   truncated flags are booleans.
9. The row selected by the factual applied-action ID exactly equals the factual
   transition's next observation, reward, events, and boundary flags. Float
   equality uses the existing exact hexadecimal representation, not tolerance.
10. The expert target exists only at `factual.action_target`. No branch row has
    an expert label.

The selected factual row equivalence is the central anti-mismatch invariant: it
proves that the recurrent trajectory and the corresponding table row describe
the same physical transition.

## Sequence and split invariants

- `len(outcome_tables) == len(factual_sequence.transitions)` and table `t`
  belongs to factual transition `t`.
- Adjacent recurrent observations follow only the factual transition:
  `factual[t + 1].observation == factual[t].next_observation_target`.
- No non-factual branch result may appear as a later recurrent observation.
- A factual terminal transition ends the sequence. A terminal non-factual row
  does not end or fork the factual sequence.
- Discounted value targets remain factual-policy returns only. V1 branch rows
  have one-step reward/hazard/next-observation targets and no value target.
- TRAIN, VALIDATION, and TEST retain the existing disjoint 62-bit episode-seed
  namespaces. Every branch inherits its factual root's split and seed; branches
  do not allocate new episode seeds.
- Dataset generation and tuning may open TRAIN and VALIDATION only. TEST remains
  unopened until a separately registered final evaluation.
- Tiled windows may share an episode as before, but each window remains an
  independent recurrent sample. Branching must not carry state between windows.

## Deterministic generation algorithm

For each factual tick:

1. Capture `root_snapshot = environment.snapshot()` and its state hash before
   any action is applied.
2. Convert the current observation to the factual model observation.
3. Call `teacher.act(raw_observation)` exactly once to obtain the expert CE
   target. For v1's zero-delay planner this does not create an unsynchronized
   action FIFO.
4. Select the one factual behavior action using the registered behavior policy.
5. For every canonical action ID, restore `root_snapshot`, verify the root hash,
   execute exactly one simulator step with that action, and record the branch
   target plus complete result-state hash.
6. Repeat branch evaluation in any execution order without changing table
   identity. Canonical row serialization is action-ID order.
7. Restore `root_snapshot` again and verify its hash.
8. Execute the factual behavior action once. Assert byte/exact equality with
   the matching branch row.
9. Append one factual transition and one attached table, then advance
   `raw_observation` only to the factual outcome.

Branch evaluation must use `try/finally` restoration equivalent to the existing
`evaluate_branches` contract. Simulator RNG consumed inside a branch must be
restored before every other branch and before the factual step.

## Required manifest fields

The new generator must use a new manifest identity; historical manifests and
hashes are untouched. In addition to the existing environment, split, sequence,
window, teacher, behavior-policy, and licensing fields, record:

```json
{
  "generator_id": "irene.maze_chase.planner_labels.factual_rollout.all_action_branches.v1",
  "transition_schema": "irene.maze_chase.all_action_transition.v1",
  "outcome_table_schema": "irene.discrete_action_outcome_table_target.v1",
  "branch_root": "exact_pre_action_snapshot",
  "branch_horizon_ticks": 1,
  "branch_action_ids": [0, 1, 2, 3, 4],
  "branch_action_semantics": ["idle", "W", "A", "S", "D"],
  "branch_row_serialization": "ascending_action_id",
  "branch_cardinality": 5,
  "branch_coverage": "exhaustive",
  "branch_restore_contract": "snapshot_restore_state_hash_v1",
  "factual_branch_equivalence_required": true,
  "recurrent_path": "factual_only",
  "expert_ce_exposure": "once_per_factual_state",
  "branch_value_target": "none_one_step_only",
  "actuation_contract": "zero_delay_nonsticky_requested_equals_applied_v1",
  "model_input_excludes_branch_targets": true
}
```

The table/result hashes, rewards, events, split identity, episode seed, teacher
target, and future images remain metadata or supervision. None are model input.

## Objective contract

At each optimized factual state, call Core V2 exactly once. The current factual
observation and recurrent state produce both the decision and the predicted
`ActionOutcomeTable [B, A, ...]`. The factual applied action is gathered from
that table only to create the pending prediction used at tick `t + 1`.

The five branch images are encoded only by the frozen/EMA target encoder. Flatten
`[B, A, C, H, W]` to `[B*A, C, H, W]` for one vectorized target-encoder call,
then restore `[B, A, W]`. They never pass through the online encoder or Core
state update.

Join predicted and target rows by semantic action ID. Never assume column index
equals action ID, even though the generator serializes canonical order.

For `N` optimized factual states and `A=5`, normalize losses as follows:

```text
L_decision = (1/N) sum_state weighted_CE(deployed_logits, expert_action)

L_next = (1/N) sum_state [(1/A) sum_action cosine_distance(
    predicted_next_latent[state, action],
    stopgrad(target_next_latent[state, action]))) ]

L_reward = (1/N) sum_state [(1/A) sum_action
    symlog_twohot_CE(predicted_reward_logits[state, action], reward_target))]

L_hazard = (1/N) sum_state [(1/A) sum_action
    BCE(predicted_hazard[state, action], hazard_target))]

L_total = decision_weight * L_decision
        + next_weight * L_next
        + reward_weight * L_reward
        + hazard_weight * L_hazard
```

Important weighting rules:

- Expert CE is computed once per factual state, never once per action row.
- Each state has total outcome weight one; five branches are averaged, not
  summed. Adding exhaustive supervision must not multiply the auxiliary scale
  by five.
- Each action receives exactly `1/5` of a state's outcome weight, independent of
  which action the behavior policy selected.
- The factual outcome row is already included in the all-action mean. Do not add
  a second factual reward/hazard/latent loss.
- Burn-in follows the existing objective: Core state advances, but neither CE
  nor branch losses are counted before `burn_in_steps`.
- Terminal/truncated branch rows still receive valid one-step losses. They never
  cause recurrent unrolling down that branch.
- Keep ordinary probability BCE for the first contract and gate with balanced,
  per-action metrics. If class reweighting is later needed, preregister fixed
  TRAIN-only weights; do not derive weights from validation or a current batch.

Metrics must be state/action normalized in the same way as losses and include
per-action next-latent cosine, reward error/distributional loss, hazard Brier,
positive/negative probability means, and positive/negative counts. Aggregate
metrics alone cannot establish that all five rows learned.

## Required tests before any training

### Dataclass and manifest tests

- Accept exactly one valid exhaustive table.
- Reject missing, duplicate, out-of-range, or mismatched action IDs.
- Reject a control whose semantic action does not match its ID.
- Reject requested/applied differences under the v1 actuation contract.
- Reject wrong root observation hash, frame increment, elapsed time, or
  `previous_control`.
- Reject any mismatch between the factual transition and its selected branch
  row, including events and termination flags.
- Verify all new manifest fields and hash sensitivity to every branch/actuation
  knob.
- Verify the historical teacher/intervention manifest hashes are unchanged.

### Generator and simulator tests

- Materializing the same index twice produces identical factual and branch
  content hashes.
- Forward, reverse, and shuffled branch execution orders produce identical
  action-ID-keyed rows and result-state hashes.
- The environment state hash is unchanged after branch enumeration.
- The factual transition stream is field-for-field identical to the existing
  generator under the same seed and behavior settings. Manifest-dependent
  content hashes may differ, but physical transitions may not.
- A direct one-step replay from each stored root reproduces every branch row.
- A mock teacher proves exactly one `act` call per factual tick, not five.
- A branch that terminates does not terminate the factual sequence unless it is
  the selected factual row.
- Branch RNG use cannot change the later factual trajectory.
- TRAIN/VALIDATION episode seeds and all root/result hashes remain split-local;
  diagnostics assert `test_split_opened == false`.

### Batch and objective tests

- Legacy `TrajectoryBatch` is rejected by the V2.1f all-action objective.
- Core is called once per factual tick; branch observations never become Core
  inputs.
- The online encoder sees only factual observations. The target encoder sees
  exactly `B*A` branch next observations per optimized tick.
- Permuting target rows and their action IDs leaves every loss unchanged.
- Changing a single action row changes only that action's auxiliary contribution.
- Every one of the five predicted rows receives a finite non-zero gradient from
  its corresponding target.
- Decision CE exactly equals the existing one-target deployed decision loss.
  Changing or duplicating branch targets does not change the CE component.
- All-action auxiliary losses equal the arithmetic mean of five independently
  computed row losses, not their sum.
- The gathered factual pending prediction exactly matches the predicted row
  selected by semantic action ID.
- Changing non-factual branch targets cannot change same-tick decision output,
  recurrent state, or next-tick pending prediction.
- Changing the factual row changes the supervised factual error but the actual
  next recurrent observation still comes only from the factual sequence.
- Target latents are detached and branch losses cannot backpropagate into target
  pixels/encoder.
- Burn-in advances one factual recurrent path and contributes zero CE and zero
  branch loss.

### Qualification gates after implementation, before DGX

No long run should start until the CPU suite proves all invariants above and a
bounded TRAIN/VALIDATION diagnostic shows, for every action independently:

- target coverage includes both hazard and safe outcomes where the environment
  actually provides them;
- held-out next-latent and reward losses improve from initialization;
- hazard discrimination improves over action-prior and constant-prior controls;
- factual gather metrics exactly match the corresponding all-action row;
- expert CE exposure count equals optimized factual-state count, not five times
  that count;
- test remains unopened.

Passing these checks qualifies only a newly preregistered bounded experiment. It
does not retroactively qualify any frozen V2.1 script whose source archive did
not contain this contract.
