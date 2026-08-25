# Preregistration: Stage V2.0b — direct set policy

**Date frozen:** 2026-08-24, before any V2.0b GPU execution  
**Status:** FROZEN  
**Predecessor:** Stage V2.0 is terminal OUTCOME GAMMA; it is not continued.

## Frozen implementation identity

The implementation is based on Git commit `106c193`. The experiment uses the
following uncommitted, owner-reviewed working-tree overlay; SHA-256 values are
frozen before GPU execution:

| File | SHA-256 |
|---|---|
| `brain/src/irene_brain/v2/config.py` | `a9715a3be7b7a6b26b2e13877372c7e34f31b1aed994d41754d709d7657c0b6f` |
| `brain/src/irene_brain/v2/aggregator.py` | `7670325f0bba1e6e1c47e464bc22087ccf55da86352d62d8a2b314d3fb4d2717` |
| `brain/scripts/stage_v20_core_v2_deployed_loss_baseline.py` | `8aa8a99c84fc7a38c302177a9fe2fd8cef533988d4f3c7bc61ace67b82bf5892` |
| `brain/scripts/stage_v20b_direct_set_policy.py` | `eb852a603ea1952e5eefe4a836ac93d8b218c39edaacd997f5cf5d8ca8e2eec1` |
| `brain/tests/test_core_v2_decision_contract.py` | `d110190bf530a24de0c2a24d75f725125dc9e8349ffaafd69576b89e7bcc4104` |

Any change to these bytes requires a new preregistration identifier; it may
not be silently substituted into Stage V2.0b.

## Research basis and falsifiable hypothesis

Deep Sets establishes pooling over transformed set elements as a standard
permutation-invariant construction. Set Transformer and attention-based
multiple-instance learning provide richer invariant alternatives, but add
complexity that this first repair does not need. Dueling networks separate
state value from action-specific advantage, while DreamerV3 trains an actor
that directly emits an action distribution and keeps model/critic semantics
separate. HRA's Pac-Man result likewise combines action-value components,
rather than multiplying an ungrounded scalar into a policy.

Primary sources:

- Zaheer et al., *Deep Sets*: https://arxiv.org/abs/1703.06114
- Lee et al., *Set Transformer*: https://proceedings.mlr.press/v97/lee19d.html
- Ilse et al., *Attention-based Deep Multiple Instance Learning*:
  https://proceedings.mlr.press/v80/ilse18a.html
- Wang et al., *Dueling Network Architectures*: https://arxiv.org/abs/1511.06581
- Hafner et al., *Mastering Diverse Control Tasks through World Models*:
  https://www.nature.com/articles/s41586-025-08744-2
- van Seijen et al., *Hybrid Reward Architecture for Reinforcement Learning*:
  https://arxiv.org/abs/1706.04208

**Hypothesis:** replacing the zero-gradient scalar-utility quotient with

`Q(a) = mean_k action_logit(k, a)`

will (1) create a non-zero deployed-loss gradient to the encoder, shared
BrainCell, and action head; (2) preserve exact slot-permutation invariance;
(3) learn a non-constant deployed policy in a bounded smoke; and (4) improve
held-out lift relative to the frozen Core V1 reference at full scale.

The mean is K-bounded and permutation-invariant. It is not claimed to be
invariant to duplicating only a subset of a multiset; earlier documentation
that called an ordinary weighted mean “multiplicity-proof” was too strong.

## Stage 0: mandatory local contracts

Before Spark execution, CPU-only deterministic tests must show:

1. the legacy path reproduces zero deployed gradient at zero utility;
2. the direct path gives non-zero gradient to action head, encoder, and
   shared BrainCell;
3. fixed-batch loss decreases over 60 updates without non-finite values;
4. permuting all K slots changes the deployed distribution by at most `1e-6`;
5. constitution CI and the ordinary play-safe suite remain green.

## Frozen smoke gate

Both arms run; neither may be skipped after seeing the other.

| Item | Frozen value |
|---|---|
| Script | `brain/scripts/stage_v20b_direct_set_policy.py --smoke` |
| Arms | V2-A decision-only and V2-C full |
| Train seed | `42` |
| Steps | `400` per arm |
| Bank | seed `42`; digest `b3bb5fc33fd5f605` |
| Optimizer | AdamW, lr `5e-4`, weight decay `1e-4`, clip `1.0` |
| Batching | fixed length groups, B=16 |
| Eval | 30 episodes/task; base seed `20260822`; deployed argmax |
| Determinism | mandatory |

The smoke passes only if **each** arm has:

- finite initial and final loss,
- final loss `< 1.45`,
- final loss at most `90%` of its initial loss,
- at least two distinct deployed actions over the complete evaluation, and
- a successfully persisted, parseable JSON artifact owned by the run user.

If any condition fails, stop. Do not launch the full run. Infrastructure-only
failures may be corrected and rerun under the same scientific configuration,
with the defect documented.

## Frozen confirmatory run and outcome classes

Only after a passing smoke, run both arms for 6,000 steps at seeds
`{42, 142, 242, 342}`. All bank, optimizer, batching, evaluation, and
determinism fields remain identical to the smoke. Reference R is frozen Core
V1 mean lift `-0.2130`.

For each arm independently, let `delta = mean_lift - R`:

- **ALPHA:** `delta >= +0.05` and seed sigma `<= 0.06`.
- **BETA:** `abs(delta) <= 0.02`.
- **GAMMA:** `delta <= -0.03`; stop and diagnose locally.
- Otherwise **AMBIGUOUS**; report distributions without retuning gates.

V2-A versus V2-C is an exploratory contrast, not a replicate comparison.

## Scope limits

A pass establishes a trainable, invariant deployed decision path on the
Torture Suite V1 distribution. It does **not** establish human-like game
understanding, useful consequence semantics, memory, real-time control, or a
thoughtlet advantage over matched recurrent/ensemble baselines. Those require
separate grounded objectives and closed-loop preregistered experiments.
