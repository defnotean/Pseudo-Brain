"""Core V2 Model — the one-brain recurrent sensorimotor system.

Explicit step order per tick:
  1. ENCODE      observation -> latent
  2. ERROR       pending prediction vs reality -> prediction error state
  3. RETRIEVE    episodic memory query
  4. THINK       K thoughtlets update (BrainCell x C cycles + attention)
  5. BELIEVE     belief update
  6. HYPOTHESIZE consequence model -> K hypotheses
  7. AGGREGATE   multiplicity-proof weighted mean -> deployed action_dist
  8. PREDICT     world model -> pending prediction for next step
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CoreV2Config, FeatureFlags, DEFAULT_CONFIG, DEFAULT_FLAGS
from .state import (
    CoreV2State, FastState, SessionState,
    PredictionErrorState, PendingPrediction,
)
from .sensor_encoder import SensorEncoderV2
from .prediction_error import PredictionErrorV2
from .belief_updater import BeliefUpdaterV2
from .episodic_memory import EpisodicMemoryV2 as EpisodicMemoryModule
from .session_state import SessionStateV2
from .thought_field import ThoughtFieldV2
from .consequence_model import ConsequenceModelV2, ConsequenceHypothesis
from .world_model import WorldModelV2
from .outcome_model import ActionOutcomeTable
from .aggregator import MultiplicityProofAggregator, AggregatedDecision


@dataclass
class StepOutput:
    """Everything one cognitive tick produced — for training and diagnostics."""
    latent: torch.Tensor                    # [B, W] encoded observation
    prediction_error: PredictionErrorState  # error computed BEFORE cognition
    retrieved_memory: Optional[torch.Tensor]
    thoughts: torch.Tensor                  # [B, K, W] post-update
    belief: torch.Tensor                    # [B, W] post-update
    hypotheses: ConsequenceHypothesis       # K hypotheses
    decision: AggregatedDecision            # DEPLOYED action distribution
    pending_prediction: Optional[PendingPrediction]
    outcome_table: Optional[ActionOutcomeTable] = None


class CoreV2Model(nn.Module):
    """One brain: encode → error → retrieve → think → believe → hypothesize → aggregate → predict."""

    def __init__(
        self,
        config: CoreV2Config = DEFAULT_CONFIG,
        flags: FeatureFlags = DEFAULT_FLAGS,
    ):
        super().__init__()
        self.config = config
        self.flags = flags
        W = config.width

        # --- Modules ---
        self.encoder = SensorEncoderV2(config)
        self.error_module = PredictionErrorV2(config) if flags.prediction_error_feedback else None
        self.thought_field = ThoughtFieldV2(config)
        self.belief_updater = BeliefUpdaterV2(config)
        self.session_manager = SessionStateV2(config)
        self.memory = self.session_manager.episodic_memory if flags.episodic_memory else None
        self.consequence_model = ConsequenceModelV2(config) if flags.consequence_learning else None
        if not flags.consequence_learning:
            # CONFIG_A degenerate path: single shared action head over mean thought
            self.degenerate_action_head = nn.Linear(W, config.actions)
        self.world_model = WorldModelV2(config) if flags.world_model_learning else None
        if self.world_model is not None:
            self.world_model.set_target_encoder(self.encoder)
        self.aggregator = MultiplicityProofAggregator(config)

        # Meta-learning session adapter (disabled by default)
        self.meta_adapter = nn.Linear(W, W, bias=False) if flags.session_adaptation else None

    def init_state(self, batch_size: int, device: torch.device) -> CoreV2State:
        """Fresh state at session start."""
        fast = FastState(
            belief=torch.zeros(batch_size, self.config.width, device=device),
            thoughts=torch.zeros(batch_size, self.config.thoughtlets, self.config.width, device=device),
        )
        session = self.session_manager.init_session(batch_size, device)
        return CoreV2State(fast=fast, session=session)

    def forward(
        self,
        observation: torch.Tensor,               # [B, C, H, W] in [0,1]
        state: CoreV2State,
        prev_action: Optional[torch.Tensor] = None,   # [B] long or [B, A] one-hot
        actual_reward: Optional[torch.Tensor] = None, # [B, 1] reality of last action
        actual_hazard: Optional[torch.Tensor] = None, # [B, 1]
        world_model_action: Optional[torch.Tensor] = None, # [B] factual action for t -> t+1
        intervention_pe: Optional[str] = None,   # "normal"|"zero"|"scrambled"|...
        intervention_mem: Optional[str] = None,  # "normal"|"disable_read"|"clear"|...
        return_components: bool = False,
    ) -> tuple[StepOutput, CoreV2State]:
        """One cognitive tick. Order matters and is explicit."""
        B = observation.shape[0]
        device = observation.device

        # 1. ENCODE
        latent = self.encoder(observation)  # [B, W]

        # 2. ERROR (before cognition — how wrong was my last prediction?)
        pe_state = None
        if self.error_module is not None:
            pending = state.fast.pending_prediction
            pe_state = self.error_module(
                pending=pending,
                actual_latent=latent.detach(),   # stop-grad target
                actual_reward=actual_reward,
                actual_hazard=actual_hazard,
                intervention=intervention_pe or "normal",
            )

        # 3. RETRIEVE
        retrieved = None
        if self.memory is not None and state.session.episodic_memory and len(state.session.episodic_memory) > 0:
            if intervention_mem != "disable_read":
                retrieved_list = self.memory.retrieve(state.session.episodic_memory, latent, k=None)
                retrieved = retrieved_list.mean(dim=1)  # [B, k, W] -> [B, W]

        # 4. BELIEVE (integrate current observation into belief BEFORE deliberation,
        #    so the deployed decision always sees the current frame)
        new_belief = self.belief_updater(
            old_belief=state.fast.belief,
            observation=latent,
            prediction_error=pe_state,
            retrieved_memory=retrieved,
            session_latent=state.session.session_latent,
            prev_action=prev_action,
        )
        state.fast.belief = new_belief

        # 5. THINK (init thoughts on first tick of a trial)
        if state.fast.thoughts.abs().sum() == 0:
            noise = torch.randn(B, self.config.thoughtlets, self.config.width, device=device)
            state.fast.thoughts = self.thought_field.init_thoughts(
                new_belief, state.session.session_latent, noise=noise)

        thoughts = self.thought_field(
            state.fast.thoughts,
            new_belief,
            prediction_error=pe_state,
            retrieved_memory=retrieved,
            session_latent=state.session.session_latent,
        )
        state.fast.thoughts = thoughts

        # 6. HYPOTHESIZE
        if self.consequence_model is not None:
            proposal_hypotheses = self.consequence_model(thoughts)
        else:
            # Degenerate mode (CONFIG_A): single hypothesis from mean thought
            proposal_hypotheses = self._degenerate_hypothesis(thoughts)

        # 7. AGGREGATE (deployed output)
        decision = self.aggregator(proposal_hypotheses)

        # 8. PREDICT.  The output keeps the live graph for the current tick's
        #    supervised world-model loss; only the recurrent copy is detached
        #    so gradients cannot chain across ticks.
        pending = None
        outcome_table = None
        if self.world_model is not None:
            factual_action = (
                decision.action_values.argmax(dim=-1)
                if world_model_action is None
                else world_model_action
            )
            if self.config.outcome_architecture == "all_action_table_v1":
                # The full table is computed independently of the applied
                # action. Factual supervision is an exact semantic gather, so
                # it cannot alter the decision or counterfactual rows.
                hypotheses = proposal_hypotheses
                outcome_table = self.world_model.forward_all(new_belief)
                pending = outcome_table.gather(factual_action)
            else:
                # Historical/transitional consequence-head path.
                hypotheses = (
                    self.consequence_model(thoughts, action=factual_action)
                    if self.consequence_model is not None
                    and self.config.outcome_action_conditioning
                    == "factual_or_proposal_v1"
                    else proposal_hypotheses
                )
                predicted_next = self.world_model(
                    hypotheses,
                    new_belief,
                    action=factual_action,
                )
                pending = PendingPrediction(
                    predicted_next_latent=predicted_next,
                    predicted_reward=hypotheses.predicted_reward,
                    predicted_hazard=hypotheses.predicted_hazard,
                    predicted_confidence=hypotheses.confidence,
                    predicted_branch_logit=hypotheses.branch_logit,
                )
            state.fast.pending_prediction = PendingPrediction(
                predicted_next_latent=(
                    None
                    if pending.predicted_next_latent is None
                    else pending.predicted_next_latent.detach()
                ),
                predicted_reward=(
                    None
                    if pending.predicted_reward is None
                    else pending.predicted_reward.detach()
                ),
                predicted_reward_logits=(
                    None
                    if pending.predicted_reward_logits is None
                    else pending.predicted_reward_logits.detach()
                ),
                predicted_hazard=(
                    None
                    if pending.predicted_hazard is None
                    else pending.predicted_hazard.detach()
                ),
                predicted_confidence=(
                    None
                    if pending.predicted_confidence is None
                    else pending.predicted_confidence.detach()
                ),
                predicted_branch_logit=(
                    None
                    if pending.predicted_branch_logit is None
                    else pending.predicted_branch_logit.detach()
                ),
            )
        else:
            hypotheses = proposal_hypotheses

        output = StepOutput(
            latent=latent,
            prediction_error=pe_state,
            retrieved_memory=retrieved,
            thoughts=thoughts,
            belief=new_belief,
            hypotheses=hypotheses,
            decision=decision,
            pending_prediction=pending,
            outcome_table=outcome_table,
        )
        return output, state

    def _degenerate_hypothesis(self, thoughts: torch.Tensor) -> ConsequenceHypothesis:
        """CONFIG_A fallback: single hypothesis replicated from mean thought."""
        B, K, W = thoughts.shape
        mean_thought = thoughts.mean(dim=1, keepdim=True).expand(-1, K, -1)
        A = self.config.actions
        device = thoughts.device
        return ConsequenceHypothesis(
            action_logits=self.degenerate_action_head(mean_thought.reshape(B * K, W)).reshape(B, K, A),
            predicted_next_latent=torch.zeros(B, K, W, device=device),
            predicted_reward=torch.zeros(B, K, 1, device=device),
            predicted_hazard=torch.zeros(B, K, 1, device=device),
            confidence=torch.full((B, K, 1), 1 / K, device=device),
            existence_logit=torch.zeros(B, K, 1, device=device),
            branch_logit=torch.zeros(B, K, 1, device=device),
        )

    def begin_trial(self, state: CoreV2State):
        """Trial boundary: reset fast state, keep session."""
        state.reset_fast_state()
        self.session_manager.end_trial(state.session)

    def end_trial(self, state: CoreV2State, trial_summary: Optional[dict] = None):
        """Update session state with trial outcome."""
        if trial_summary:
            state.session.session_stats[f"trial_{state.session.trial_index}"] = trial_summary
        self.session_manager.end_trial(state.session)

    @torch.no_grad()
    def act(self, observation: torch.Tensor, state: CoreV2State) -> int:
        """Greedy deployment action for one observation."""
        output, state = self.forward(observation, state)
        return int(output.decision.action_dist.argmax(dim=-1).item())
