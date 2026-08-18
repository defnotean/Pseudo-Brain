# Remaining §28 baseline controls: scope finding and preregistration proposal (2026-08-17)

Status: design proposal only. **No model code, objective code, config, or
manifest change is made by this document.** Each proposal below changes the
shared training objective or the campaign shape, so — like RCQ-v3 — each
needs a frozen decision set and an owner veto window before implementation.
The veto window closes when the first implementation commit lands.

## The blocking finding

The matched baseline suite's fairness contract fixes one objective for
every variant: identical dataset, optimizer, schedule, and loss, with only
the model factory changing (enforced by recipe normalization in
`tests/test_matched_baselines.py`). The current shared objective trains
exactly one world-prediction term: a next-step sensor-encoding regression
with a hard minimum over slots (`training/objective.py`, "any thoughtlet
may own this short-horizon prediction"). The per-slot `horizon_logits`
produced by `ThoughtPredictionHead` are consumed by **nothing** — not the
objective, not evaluation (verified by full-text search 2026-08-17).

Consequence: three of the four remaining PLAN §28 controls cannot be added
as drop-in variants the way the nine registered ones were. A "fixed
multi-horizon heads" variant whose horizon structure no loss shapes would
train identically to the reactive control and its manifest identity would
misrepresent what was tested. A recurrent latent world-model actor needs
latent-state losses the shared objective does not have. A task specialist
is a training-regime distinction and needs at least a second registered
task. Each is a campaign-level decision, not a factory entry.

## Proposed frozen decisions

| # | Control (§28 item) | Proposal |
|---|---|---|
| B1 | Fixed multi-horizon heads, no persistent thoughtlets (item 8) | Extend the shared objective with a **multi-horizon world loss**: within each length-8 training window, regress horizon-specific future embeddings at offsets {1, 2, 4} (the most the window supports) onto the corresponding future sensor encodings, weight `world_weight` split evenly across horizons. The control variant (`irene.thought_field.fixed_multi_horizon.v1`) statically partitions its 32 slots into three horizon groups, each group's head trained only on its own offset, no persistence (reseeding like `reset_slots`). The reference and other variants gain the same loss terms with their existing flexible min-over-slots assignment per offset, so the comparison stays one-recipe. Offsets {8, 16, 32} wait for longer registered windows. |
| B2 | Recurrent latent world-model actor (item 13) | A conventional RSSM-style control is out of scope for the slot trunk; instead register `irene.world_model_actor.gru_latent.v1`: the parameter-matched monolithic GRU trunk plus a latent transition model trained with the B1 multi-horizon loss rolled through its own predictions (latent rollout), which the current objective cannot express. This needs its own objective terms and therefore its own preregistered recipe family; it must not ride on the slot-suite manifest. Budget: same 2,048 updates, parameter-matched to the reference within 1%. |
| B3 | Task specialist (item 14) | Register after the ladder datasets exist: the reference architecture fine-tuned per world (moving shapes specialist, maze_chase specialist) versus the unchanged generalist checkpoint — the §29 "unchanged generalist versus game-specific fine-tuning" ablation made manifest. Requires generated, registered datasets for at least two worlds; dataset generation is DGX-scale and currently deferred. |
| B4 | Standard recurrent transformer controller (item 5) | Not yet in the manifest at all. Proposal: a per-step Transformer encoder over the sensor/belief/working-memory token set with a recurrent carry token, parameter-matched by width enumeration like the monolith and ensemble controls. No objective change needed; implementable as a drop-in variant under the existing contract once B1 lands (or independently). |

## Recommended order

1. **B4** — pure drop-in, no objective change, same discipline as the
   ensemble control.
2. **B1** — one preregistered objective extension, one new variant; keeps
   the single-recipe fairness contract intact.
3. **B2** — own recipe family; the largest design surface.
4. **B3** — blocked on ladder dataset generation regardless.

## Alternatives considered and rejected

- *Register item 8 with untrained horizon heads.* Rejected: the identity
  would claim a distinction no loss shapes; the run record would be
  misleading by construction.
- *Give only the new variants extra loss terms.* Rejected: breaks the
  one-objective fairness contract the suite exists to satisfy.
- *Fold item 13 into the slot manifest with a "close enough" recurrent
  cell.* Rejected: a world-model actor's defining feature is its latent
  rollout objective; without it the variant is the monolithic GRU control
  under a second name.

## Owner veto

Reply changes or vetoes any of B1–B4 before its implementation commit.
Silence is not consent here: unlike RCQ-v3's delegated window, these were
scoped after the delegation was given, so each proposal waits for an
explicit go or for the next planning pass.
