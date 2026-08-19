# Structural Action Grouping: Architecture & Verification (2026-08-18)

Status: implementation and verified. Local CPU-only verification; single-threaded, CUDA-hidden, no background jobs, no network services.

## Overview

The **Structural Action Grouping** module formalizes modular, structured action spaces over the canonical 307-channel generic HID wire layout (`GenericControl`).

It replaces naive monolithic action projections with specialized, mutually exclusive, and calibrated action groups that reflect the physical mechanics of humanoid and game inputs:

1. **Group A: Movement Direction**
   - 5-way categorical Softmax over `{None, W, A, S, D}`.
   - Optional 9-way categorical Softmax over `{None, W, A, S, D, WA, WD, SA, SD}` for orthogonal diagonal support.
   - **Guaranteed Invariant**: Conflicting opposite-direction coactivations ($W+S$ and $A+D$) are mathematically impossible under any input or model parameter values.

2. **Group B: Stance & Modifier Keys**
   - Independent Bernoulli probabilities with calibrated Sigmoid activations for stance modifiers (Left Shift, Space, Left Ctrl, Left Alt, C, Tab).

3. **Group C: Continuous Aim / Look Coordinates**
   - Continuous relative deltas ($mouse\_dx, mouse\_dy$) and axes (scroll, gamepad) bounded via $\tanh$ squashing.
   - Configurable deadzone behaviors (`deadzone_tanh` for structural quiescence inside $[-0.046875, 0.046875]$, `hard`, `smooth`, or `tanh`).

4. **Group D: Discrete Action Triggers**
   - Independent Bernoulli probabilities for mouse clicks (LMB, RMB, MMB) and action key triggers (E, R, F, Q, G, 1, 2, 3).

## Module Components

- **`brain/src/irene_brain/model/action_groups.py`**:
  - `MovementMode`, `MovementDirection5`, `MovementDirection9`: Enums with bidirectional key mapping and validation.
  - `StructuredActionSpec`: Validated configuration enforcing non-overlapping key sets, valid HID indices, and squash policies.
  - `StructuredActionPrediction`: Strongly typed dataclass container holding group logits, probabilities, bounded continuous vectors, query attention, and the reconstructed 307-channel wire control vector.
  - `StructuredActuatorHead`: Unpooled multi-head neural readout with 4 specialized group queries (`movement`, `modifier`, `aim`, `trigger`) attending the full unpooled thought field, belief, and sensor state, with deterministic recombination into canonical 307-channel wire control vectors and `GenericControl` instances.
  - `StructuredActionLoss`: Multi-group objective combining categorical cross-entropy on movement, Bernoulli binary cross-entropy on modifiers and triggers, continuous MSE / Smooth L1 on coordinates, and optional deadzone quiescence hinge penalties.

- **`brain/src/irene_brain/model/__init__.py`**:
  - Clean lazy exports for all action grouping classes and specifications.

- **`brain/tests/test_action_groups.py`**:
  - 13 comprehensive unit tests covering:
    1. Enum values, mappings, and spec validation.
    2. Unpooled forward pass from multi-thought state components.
    3. 9-way diagonal forward pass.
    4. 2D feature projection.
    5. Full backpropagation and gradient flow through all groups.
    6. Exact 307-channel wire format reconstruction.
    7. High-level `GenericControl` roundtrip conversion.
    8. Adversarial 5-way mutual exclusion stress test ($10,000$ saturated random batches: $W+S=0$, $A+D=0$).
    9. Adversarial 9-way mutual exclusion stress test ($10,000$ saturated random batches: $W+S=0$, $A+D=0$).
    10. Gumbel-Softmax and Bernoulli action sampling.
    11. Continuous deadzone and squashing invariants.
    12. Multi-group loss computation and target extraction from 307-vectors and `GenericControl` objects.
    13. Deadzone hinge loss penalties.

## Verification Results

All 13 unit tests ran CPU-only, CUDA-hidden, and single-threaded:
```
Ran 13 tests in 0.038s
OK
```
Sibling model and objective tests (`test_intent_and_actuator`, `test_action_loss_v2`, `test_model_spec`, `test_trainable_model`) all pass without regressions (30 tests in 0.977s).
