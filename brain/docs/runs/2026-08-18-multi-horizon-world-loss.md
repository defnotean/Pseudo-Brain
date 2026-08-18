# B1: multi-horizon world loss and the fixed-horizon control (2026-08-18)

Status: implementation. Local CPU-only verification; no training run, no
accelerator, no sealed data. Implements preregistered control **B1** from
[2026-08-17-remaining-baseline-controls-preregistration.md](2026-08-17-remaining-baseline-controls-preregistration.md)
under the owner's explicit 2026-08-18 go.

## What was added

**Multi-horizon world loss** (`training/objective.py`). The shared
objective's single short-horizon world term is extended exactly as
preregistered: within each training window, horizon-specific future
embeddings at power-of-two offsets are regressed onto the corresponding
future sensor encodings, with `world_weight` split evenly across horizons.

- `_multi_horizon_offsets(sequence_length, burn_in_steps)` derives the
  supported offsets from the window itself: offset 1 always (its target is
  the transition's own `next_observation_target`); offsets 2 and 4 join
  when the recorded window keeps them eligible
  (`k <= sequence_length - 1 - burn_in_steps`). The registered length-8,
  burn-in-2 campaign window yields exactly `{1, 2, 4}`; offsets beyond 4
  are capped pending a future longer-window preregistration.
- A length-2 smoke window yields `(1,)`, so short-window behavior is
  bit-identical to the previous single-horizon loss — no existing smoke
  recipe changes numerics.
- The reference and all existing variants keep the flexible
  min-over-slots assignment independently per horizon. New per-horizon
  metrics (`world_loss_h1`, `world_loss_h2`, `world_loss_h4`) are
  reported alongside the combined `world_loss`.

**`irene.thought_field.fixed_multi_horizon.v1`**
(`FixedMultiHorizonSlotBaseline`, `model/baselines.py`). The B1 control:
32 slots statically partitioned into even contiguous horizon groups
(11/11/10 on the campaign geometry), each group's future embedding
trained only on its assigned offset, persistence removed exactly like the
reset-slots ablation. Allocated parameters are identical to the reference
(29,674,318 trainable; 1,155 architecturally disconnected — the same
inert lifecycle head as reset_slots), so the variant sits inside the
verified parameter-matched fairness regime by construction. Registered in
the regenerated architecture manifest (digest `8d93eeca…`) with its own
normalization-identical recipe
(`configs/training/baseline-stagea-fixed-multi-horizon.toml`) and factory
(`build_thesis_fixed_multi_horizon_model`).

## Verification

`tests/test_matched_baselines.py` gains four tests (15 total): the
offset/window rule and even-partition helper; per-horizon metric presence
with the combined loss exactly equal to their even split; bit-exact
single-horizon behavior on a length-2 window; and the control itself —
parameter/state-dict identity with reset_slots, inherited no-persistence,
and the ordering invariant that the group-restricted minimum bounds the
flexible minimum from above on every horizon under identical weights.

The full play-safe suite exits 0 (547 tests).

## Notes for the campaign

- The objective extension applies to every variant through the same code
  path, so the one-recipe fairness contract is intact; the manifest's
  `fixed_controls` already pin `objective`.
- B1's recipe is a campaign candidate like the other baseline recipes:
  DGX-scale execution remains owner-run through the bounded workflow.
- The smoke distillation probes recorded before this change
  ([2026-08-18-solver-smoke-distillation.md](2026-08-18-solver-smoke-distillation.md))
  used the single-horizon objective on length-32 windows; rerunning them
  now trains horizons {1, 2, 4} and will produce different numbers. The
  recorded zero-points remain valid dated references, not reproducible
  current values.
