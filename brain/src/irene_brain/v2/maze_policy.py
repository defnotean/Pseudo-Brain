"""Closed-loop maze-chase adapter for Core V2's five-action policy."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch

from ..training.objective import _rgb_tensor
from ..types import GenericControl, HidKey, Observation, RgbFrame
from .core import CoreV2Model
from .outcome_model import ActionOutcomeTable
from .state import PendingPrediction
from .trajectory_objective import control_action_class


_CLASS_KEYS = {
    1: int(HidKey.W),
    2: int(HidKey.A),
    3: int(HidKey.S),
    4: int(HidKey.D),
}


@dataclass(frozen=True, slots=True)
class OutcomeAwareUtilityCoefficients:
    """Registered V1 weights for one-step outcome-aware deployment utility."""

    decision: float = 1.0
    reward: float = 1.0
    hazard: float = 3.0


OUTCOME_AWARE_UTILITY_V1 = OutcomeAwareUtilityCoefficients()
"""Immutable deployment rule: decision + reward - 3 * calibrated hazard."""

OUTCOME_AWARE_MODEL_SEED_V1 = 2_126_202_608
"""Fixed cognition seed; deliberately independent of an environment seed."""


class CoreV2MazePolicy:
    """Pixel-only recurrent policy compatible with closed-loop evaluation."""

    uses_privileged_state = False
    identity = "irene.core_v2.maze_policy.v1"

    def __init__(self, model: CoreV2Model) -> None:
        if not isinstance(model, CoreV2Model):
            raise ValueError("model must be a CoreV2Model")
        self.model = model
        self._state = None
        self._episode_seed = None

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise ValueError("episode_seed must be an integer")
        if episode_seed < 0:
            raise ValueError("episode_seed must be nonnegative")
        device = next(self.model.parameters()).device
        torch.manual_seed(episode_seed)
        torch.cuda.manual_seed_all(episode_seed)
        self._state = self.model.init_state(1, device)
        self._episode_seed = episode_seed

    @torch.no_grad()
    def act(self, observation: Observation) -> GenericControl:
        if self._state is None:
            raise RuntimeError("CoreV2MazePolicy.act() called before reset()")
        if not isinstance(observation, Observation):
            raise ValueError("observation must be an Observation")
        device = next(self.model.parameters()).device
        pixels = _rgb_tensor(
            (observation.rgb,),
            device=device,
            resolution=(32, 32),
        )
        previous_action = torch.tensor(
            [control_action_class(observation.previous_control)],
            dtype=torch.long,
            device=device,
        )
        output, self._state = self.model(
            pixels,
            self._state,
            prev_action=previous_action,
        )
        action = int(output.decision.action_values.argmax(dim=-1).item())
        if action == 0:
            return GenericControl.neutral()
        return GenericControl(keys_down=(_CLASS_KEYS[action],))


class OutcomeAwareCoreV2MazePolicy:
    """RGB-only recurrent policy using the learned exhaustive outcome table.

    This is an opt-in deployment path.  It performs one Core V2 forward pass
    and no search.  The action score is the registered combination

    ``decision_value + predicted_reward - 3 * calibrated_hazard``.

    Outcome-table columns are always resolved by their semantic ``action_ids``
    before scoring.  The policy also owns its previous chosen action; metadata
    in :class:`Observation`, including ``previous_control``, never enters the
    model.
    """

    uses_privileged_state = False
    identity = "irene.core_v2.outcome_aware_maze_policy.v1"

    def __init__(self, model: CoreV2Model) -> None:
        if not isinstance(model, CoreV2Model):
            raise ValueError("model must be a CoreV2Model")
        if model.config.outcome_architecture != "all_action_table_v1":
            raise ValueError(
                "OutcomeAwareCoreV2MazePolicy requires all_action_table_v1"
            )
        if model.config.actions != len(_CLASS_KEYS) + 1:
            raise ValueError(
                "OutcomeAwareCoreV2MazePolicy requires semantic actions "
                "idle/W/A/S/D"
            )
        if model.world_model is None:
            raise ValueError(
                "OutcomeAwareCoreV2MazePolicy requires world-model learning"
            )
        self.model = model
        self._state = None
        self._previous_action = 0
        self._needs_seeded_initialization = False

    @property
    def coefficients(self) -> OutcomeAwareUtilityCoefficients:
        """The immutable, registered V1 utility coefficients."""

        return OUTCOME_AWARE_UTILITY_V1

    @property
    def model_seed(self) -> int:
        """The fixed recurrent initialization seed used by every episode."""

        return OUTCOME_AWARE_MODEL_SEED_V1

    def reset(self, episode_seed: int) -> None:
        if isinstance(episode_seed, bool) or not isinstance(episode_seed, int):
            raise ValueError("episode_seed must be an integer")
        if episode_seed < 0:
            raise ValueError("episode_seed must be nonnegative")
        # The argument is part of the shared policy protocol only.  It is
        # intentionally never stored or used to initialize model cognition.
        device = next(self.model.parameters()).device
        self._state = self.model.init_state(1, device)
        self._previous_action = 0
        self._needs_seeded_initialization = True

    @torch.no_grad()
    def act(self, observation: Observation) -> GenericControl:
        if self._state is None:
            raise RuntimeError(
                "OutcomeAwareCoreV2MazePolicy.act() called before reset()"
            )
        if not isinstance(observation, Observation):
            raise ValueError("observation must be an Observation")
        action = self.act_rgb(observation.rgb, self._previous_action)
        return _control_for_action(action)

    @torch.no_grad()
    def act_rgb(self, rgb: RgbFrame, previous_action_id: int) -> int:
        """Choose from RGB while auditing the caller's prior-action mirror.

        ``previous_action_id`` is never trusted as recurrent input.  It must
        match the action already owned by this policy, which closes the live
        qualifier's RGB-controller protocol without creating a second source
        of action history.
        """

        if self._state is None:
            raise RuntimeError(
                "OutcomeAwareCoreV2MazePolicy.act_rgb() called before reset()"
            )
        if not isinstance(rgb, RgbFrame):
            raise ValueError("rgb must be an RgbFrame")
        if isinstance(previous_action_id, bool) or not isinstance(
            previous_action_id, int
        ):
            raise ValueError("previous_action_id must be an integer")
        if previous_action_id not in range(len(_CLASS_KEYS) + 1):
            raise ValueError("previous_action_id must be semantic idle/W/A/S/D")
        if previous_action_id != self._previous_action:
            raise RuntimeError(
                "previous_action_id does not match the policy's internal prior"
            )

        device = next(self.model.parameters()).device
        # This is the complete sensory boundary for this policy.  In
        # particular, previous_control, clocks, frame IDs, audio, and text are
        # not read.
        pixels = _rgb_tensor(
            (rgb,),
            device=device,
            resolution=(32, 32),
        )
        previous_action = torch.tensor(
            [previous_action_id],
            dtype=torch.long,
            device=device,
        )

        rng_context = self._initialization_rng_context(device)
        with rng_context:
            output, self._state = self.model(
                pixels,
                self._state,
                prev_action=previous_action,
            )
        self._needs_seeded_initialization = False

        action = self._select_semantic_action(
            decision_values=output.decision.action_values,
            outcome_table=output.outcome_table,
        )
        self._install_selected_pending_prediction(output.outcome_table, action)
        self._previous_action = action
        return action

    def _initialization_rng_context(self, device: torch.device):
        if not self._needs_seeded_initialization:
            return nullcontext()

        return self._seeded_initialization_rng(device)

    @contextmanager
    def _seeded_initialization_rng(self, device: torch.device):
        cuda_devices: list[int] = []
        if device.type == "cuda":
            cuda_devices.append(
                torch.cuda.current_device() if device.index is None else device.index
            )
        with torch.random.fork_rng(devices=cuda_devices):
            # Seed only the generators the model can actually consume.  Their
            # prior state is restored by fork_rng when the first tick ends.
            torch.random.default_generator.manual_seed(self.model_seed)
            if device.type == "cuda":
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(self.model_seed)
            yield

    def _select_semantic_action(
        self,
        *,
        decision_values: torch.Tensor,
        outcome_table: ActionOutcomeTable | None,
    ) -> int:
        if outcome_table is None:
            raise RuntimeError(
                "all_action_table_v1 did not return an outcome table"
            )
        action_count = self.model.config.actions
        if decision_values.shape != (1, action_count):
            raise RuntimeError(
                f"decision values must have shape [1, {action_count}]"
            )
        if not bool(torch.isfinite(decision_values).all()):
            raise RuntimeError("decision values must be finite")
        if outcome_table.action_ids.shape != (1, action_count):
            raise RuntimeError(
                f"outcome action IDs must have shape [1, {action_count}]"
            )
        if outcome_table.predicted_reward.shape != (1, action_count, 1):
            raise RuntimeError(
                f"predicted reward must have shape [1, {action_count}, 1]"
            )
        if outcome_table.predicted_hazard.shape != (1, action_count, 1):
            raise RuntimeError(
                f"predicted hazard must have shape [1, {action_count}, 1]"
            )
        if outcome_table.action_ids.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise RuntimeError("outcome action IDs must use an integer dtype")
        ids = outcome_table.action_ids.to(
            device=decision_values.device,
            dtype=torch.long,
        )
        canonical = torch.arange(action_count, device=ids.device).unsqueeze(0)
        if not bool(ids.sort(dim=1).values.eq(canonical).all()):
            raise RuntimeError(
                "outcome action IDs must be a permutation of idle/W/A/S/D"
            )
        reward = outcome_table.predicted_reward.squeeze(-1).to(
            device=decision_values.device,
            dtype=decision_values.dtype,
        )
        hazard = outcome_table.predicted_hazard.squeeze(-1).to(
            device=decision_values.device,
            dtype=decision_values.dtype,
        )
        if not bool(torch.isfinite(reward).all() and torch.isfinite(hazard).all()):
            raise RuntimeError("outcome utilities must be finite")
        if not bool(hazard.ge(0.0).all() and hazard.le(1.0).all()):
            raise RuntimeError("calibrated hazard probabilities must be in [0, 1]")

        # Scatter to canonical semantic action order before argmax.  Besides
        # aligning rewards and hazards with the decision head, this gives
        # deterministic semantic tie-breaking independent of table order.
        semantic_reward = torch.empty_like(reward).scatter(1, ids, reward)
        semantic_hazard = torch.empty_like(hazard).scatter(1, ids, hazard)
        coefficients = self.coefficients
        utility = (
            coefficients.decision * decision_values
            + coefficients.reward * semantic_reward
            - coefficients.hazard * semantic_hazard
        )
        if not bool(torch.isfinite(utility).all()):
            raise RuntimeError("combined outcome-aware utility must be finite")
        return int(utility.argmax(dim=1).item())

    def _install_selected_pending_prediction(
        self,
        outcome_table: ActionOutcomeTable | None,
        action: int,
    ) -> None:
        if outcome_table is None:  # guarded by _select_semantic_action
            raise RuntimeError("outcome table is required")
        device = outcome_table.action_ids.device
        selected = outcome_table.gather(
            torch.tensor([action], dtype=torch.long, device=device)
        )
        self._state.fast.pending_prediction = PendingPrediction(
            predicted_next_latent=selected.predicted_next_latent.detach(),
            predicted_reward=selected.predicted_reward.detach(),
            predicted_reward_logits=(
                None
                if selected.predicted_reward_logits is None
                else selected.predicted_reward_logits.detach()
            ),
            predicted_hazard=selected.predicted_hazard.detach(),
            predicted_confidence=(
                None
                if selected.predicted_confidence is None
                else selected.predicted_confidence.detach()
            ),
            predicted_branch_logit=(
                None
                if selected.predicted_branch_logit is None
                else selected.predicted_branch_logit.detach()
            ),
        )


def _control_for_action(action: int) -> GenericControl:
    if action == 0:
        return GenericControl.neutral()
    try:
        key = _CLASS_KEYS[action]
    except KeyError as error:
        raise RuntimeError("action must be semantic idle/W/A/S/D") from error
    return GenericControl(keys_down=(key,))


__all__ = [
    "CoreV2MazePolicy",
    "OutcomeAwareCoreV2MazePolicy",
    "OutcomeAwareUtilityCoefficients",
    "OUTCOME_AWARE_UTILITY_V1",
    "OUTCOME_AWARE_MODEL_SEED_V1",
]
