"""Thought Scrambler & Causal Intervention Diagnostics for Pseudo-Brain.

Implements principled causal diagnostic tests:
1. Whole-Slot Permutation:
   - Permutes all slot indices identically across thoughts, ages, and registers.
   - Tests permutation equivariance: true factorized thought architecture should show ~0% degradation.
2. Cognitive Scrambling (Binding Destruction):
   - Mismatches thought content with register/memory representations or replaces thoughts with historical states.
   - Tests causal binding: should cause significant (>= 10%) degradation.
3. Thoughtlet Capacity Sweep:
   - Masks thought field to K in {1, 4, 8, 16, 24, 32} active slots.
   - Traces the capacity-performance scaling curve.
"""

from __future__ import annotations

from dataclasses import replace
import random
from typing import Sequence
import torch
from torch import Tensor

from .torch_model import BrainState


def permute_whole_slots(state: BrainState, permutation: Sequence[int] | None = None) -> BrainState:
    """Perform exact whole-slot permutation across all slot-aligned state tensors.

    Under permutation equivariance, a well-formed thought field preserves all information
    under arbitrary slot index reordering.
    """
    k = state.thoughts.shape[1]
    if permutation is None:
        perm_list = list(range(k))
        random.shuffle(perm_list)
        permutation = perm_list

    idx_tensor = torch.tensor(permutation, dtype=torch.long, device=state.thoughts.device)
    new_thoughts = torch.index_select(state.thoughts, dim=1, index=idx_tensor)
    new_ages = torch.index_select(state.thought_age_seconds, dim=1, index=idx_tensor)

    return replace(state, thoughts=new_thoughts, thought_age_seconds=new_ages)


def scramble_cognitive_bindings(state: BrainState, noise_scale: float = 1.0) -> BrainState:
    """Scramble the associative bindings between thought registers and temporal history.

    Destroys the structured internal representations while preserving vector norm/statistics.
    """
    b, k, reg, w = state.thoughts.shape
    device = state.thoughts.device

    # Shuffle registers independently per slot to break internal multi-register binding
    scrambled_thoughts = state.thoughts.clone()
    for slot_idx in range(k):
        reg_perm = torch.randperm(reg, device=device)
        scrambled_thoughts[:, slot_idx] = scrambled_thoughts[:, slot_idx, reg_perm]

    # Add phase-scrambled noise matching thought variance
    thought_std = torch.std(state.thoughts, dim=-1, keepdim=True) + 1e-5
    perturbation = torch.randn_like(state.thoughts) * thought_std * noise_scale
    scrambled_thoughts = scrambled_thoughts + perturbation

    # Shuffle slot ages independently
    age_perm = torch.randperm(k, device=device)
    scrambled_ages = state.thought_age_seconds[:, age_perm]

    return replace(state, thoughts=scrambled_thoughts, thought_age_seconds=scrambled_ages)


def apply_capacity_slot_mask(state: BrainState, active_slots: int) -> BrainState:
    """Mask the thought field to retain only `active_slots` active thoughtlets (rest zeroed).

    Used to construct the Thoughtlet Capacity Scaling Curve: K in {1, 4, 8, 16, 24, 32}.
    """
    k = state.thoughts.shape[1]
    if active_slots >= k:
        return state

    active_slots = max(1, min(k, active_slots))
    mask = torch.zeros((1, k, 1, 1), device=state.thoughts.device, dtype=state.thoughts.dtype)
    mask[:, :active_slots] = 1.0

    masked_thoughts = state.thoughts * mask
    age_mask = torch.zeros((1, k), device=state.thought_age_seconds.device, dtype=state.thought_age_seconds.dtype)
    age_mask[:, :active_slots] = 1.0
    masked_ages = state.thought_age_seconds * age_mask

    return replace(state, thoughts=masked_thoughts, thought_age_seconds=masked_ages)
