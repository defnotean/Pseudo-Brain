"""Counterfactual Thought Transplant Diagnostic Suite.

Tests whether the exact semantic information inside thoughts causally controls
specific downstream actions by transplanting thought states from state A (where action A* is optimal)
into state B (where action B* is optimal), measuring the causal shift P(A*).
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from ..environments.phase2_suite import Phase2TaskEnvironment, make_family_suite, TaskFamily
from ..training.staged_branch_curriculum import generate_multi_step_trajectory_tree
from ..types import GenericControl, HidKey


@dataclass(frozen=True, slots=True)
class ThoughtTransplantPair:
    """A pair of decision states with conflicting optimal actions."""
    recipient_seed: int
    donor_seed: int
    recipient_optimal_action: int  # 0: Wait, 1: W, 2: A, 3: S, 4: D
    donor_optimal_action: int      # in {0..4}, != recipient_optimal_action
    recipient_rgb: Tensor          # [1, 3, 32, 32]
    donor_rgb: Tensor              # [1, 3, 32, 32]


@dataclass(frozen=True, slots=True)
class CounterfactualTransplantReport:
    """Quantitative evaluation report for counterfactual thought transplantation."""
    num_pairs_evaluated: int
    donor_action_adoption_rate_pct: float     # % of pairs where transplant causes model to take donor action
    mean_donor_prob_increase: float           # E[P_transplant(donor_action) - P_baseline(donor_action)]
    recipient_action_suppression_pct: float   # % of pairs where recipient's original action is suppressed
    semantic_fidelity_score: float            # Combined metric in [0, 1]


def collect_conflicting_state_pairs(
    env: Phase2TaskEnvironment,
    num_pairs: int = 50,
    base_seed: int = 8000,
    device: torch.device = torch.device("cpu"),
) -> list[ThoughtTransplantPair]:
    """Collect pairs of states where ground-truth lookahead planner recommends conflicting actions."""
    pairs: list[ThoughtTransplantPair] = []
    state_pool: list[tuple[int, int, Tensor]] = []  # (seed, opt_action, rgb_tensor)
    families = [
        TaskFamily.FAMILY_B_PURSUIT_EVASION,
        TaskFamily.FAMILY_C_JUNCTIONS,
        TaskFamily.FAMILY_D_PARTIAL_OBSERVABILITY,
        TaskFamily.FAMILY_E_CHANGED_DYNAMICS,
    ]
    dir_actions = [int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]

    rng = random.Random(base_seed)
    seed_idx = base_seed
    while len(state_pool) < max(num_pairs * 4, 30) and seed_idx < base_seed + 1000:
        fam = families[seed_idx % len(families)]
        fam_env = Phase2TaskEnvironment(make_family_suite(fam)[0])
        obs = fam_env.reset(seed_idx)

        # Rollout 0 to 6 random steps to reach distinct positions in the maze
        num_steps = rng.randint(0, 6)
        for _ in range(num_steps):
            act_key = rng.choice(dir_actions)
            step_out = fam_env.step(GenericControl(mouse_dx=0.0, mouse_dy=0.0, keys_down=(act_key,)))
            if step_out.terminated or step_out.truncated:
                break
            obs = step_out.observation

        bundle = generate_multi_step_trajectory_tree(fam_env, obs, horizon=2, device=device)
        gt_opt_action = int(torch.argmax(bundle.expected_action_utilities).item())

        raw_rgb = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
        rgb_tensor = torch.from_numpy(raw_rgb.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        if rgb_tensor.shape[-1] != 32:
            rgb_tensor = F.interpolate(rgb_tensor, size=(32, 32), mode="nearest")

        state_pool.append((seed_idx, gt_opt_action, rgb_tensor))
        seed_idx += 1

    # Match pairs with conflicting optimal actions
    for i in range(len(state_pool)):
        rec_seed, rec_act, rec_rgb = state_pool[i]
        for j in range(i + 1, len(state_pool)):
            don_seed, don_act, don_rgb = state_pool[j]
            if rec_act != don_act:  # conflicting actions
                pairs.append(ThoughtTransplantPair(
                    recipient_seed=rec_seed,
                    donor_seed=don_seed,
                    recipient_optimal_action=rec_act,
                    donor_optimal_action=don_act,
                    recipient_rgb=rec_rgb,
                    donor_rgb=don_rgb,
                ))
                if len(pairs) >= num_pairs:
                    return pairs

    return pairs


def evaluate_counterfactual_thought_transplant(
    model: nn.Module,
    env: Phase2TaskEnvironment,
    *,
    num_pairs: int = 50,
    device: torch.device = torch.device("cpu"),
) -> CounterfactualTransplantReport:
    """Evaluate whether transplanting thoughts from donor state into recipient state causally steers action."""
    model.eval()
    pairs = collect_conflicting_state_pairs(env, num_pairs=num_pairs, device=device)
    if not pairs:
        return CounterfactualTransplantReport(
            num_pairs_evaluated=0,
            donor_action_adoption_rate_pct=0.0,
            mean_donor_prob_increase=0.0,
            recipient_action_suppression_pct=0.0,
            semantic_fidelity_score=0.0,
        )

    ctrl_tensor = torch.zeros((1, 307), device=device)
    dt_tensor = torch.tensor([0.016667], device=device)
    dir_keys = [0, int(HidKey.W), int(HidKey.A), int(HidKey.S), int(HidKey.D)]

    adoptions = []
    prob_increases = []
    suppressions = []

    with torch.no_grad():
        for pair in pairs:
            # 1. Unperturbed Forward Pass on Recipient State
            rec_state = model.initial_state(1)
            if isinstance(rec_state, Tensor):
                rec_state = rec_state.to(device)
            rec_out = model(pair.recipient_rgb, ctrl_tensor, dt_tensor, state=rec_state)
            rec_logits = rec_out.action.button_logits[0].cpu().numpy()
            rec_scores = [0.0 if idx == 0 else float(rec_logits[key]) for idx, key in enumerate(dir_keys)]
            rec_probs = F.softmax(torch.tensor(rec_scores), dim=-1).numpy()
            base_choice = int(np.argmax(rec_scores))

            # 2. Extract Donor Thoughts from Donor State
            don_state = model.initial_state(1)
            if isinstance(don_state, Tensor):
                don_state = don_state.to(device)
            don_out = model(pair.donor_rgb, ctrl_tensor, dt_tensor, state=don_state)
            donor_thoughts = don_out.next_state.thoughts if hasattr(don_out.next_state, "thoughts") else don_out.next_state

            # 3. Transplant Donor Thoughts into Recipient Sensory State
            trans_out = model(
                pair.recipient_rgb,
                ctrl_tensor,
                dt_tensor,
                state=rec_state,
                thought_intervention="donor",
                donor_thoughts=donor_thoughts,
            )
            trans_logits = trans_out.action.button_logits[0].cpu().numpy()
            trans_scores = [0.0 if idx == 0 else float(trans_logits[key]) for idx, key in enumerate(dir_keys)]
            trans_probs = F.softmax(torch.tensor(trans_scores), dim=-1).numpy()
            trans_choice = int(np.argmax(trans_scores))

            # Did the model adopt the donor action?
            adopted = (trans_choice == pair.donor_optimal_action)
            adoptions.append(1.0 if adopted else 0.0)

            # Did the donor action probability increase?
            p_base_donor = float(rec_probs[pair.donor_optimal_action])
            p_trans_donor = float(trans_probs[pair.donor_optimal_action])
            prob_increases.append(p_trans_donor - p_base_donor)

            # Was the original recipient action suppressed?
            p_base_rec = float(rec_probs[pair.recipient_optimal_action])
            p_trans_rec = float(trans_probs[pair.recipient_optimal_action])
            suppressed = (p_trans_rec < p_base_rec)
            suppressions.append(1.0 if suppressed else 0.0)

    adoption_rate = float(np.mean(adoptions) * 100.0)
    mean_increase = float(np.mean(prob_increases))
    suppression_rate = float(np.mean(suppressions) * 100.0)
    fidelity = float(np.clip((adoption_rate / 100.0 + max(0.0, mean_increase)) / 2.0, 0.0, 1.0))

    return CounterfactualTransplantReport(
        num_pairs_evaluated=len(pairs),
        donor_action_adoption_rate_pct=adoption_rate,
        mean_donor_prob_increase=mean_increase,
        recipient_action_suppression_pct=suppression_rate,
        semantic_fidelity_score=fidelity,
    )
