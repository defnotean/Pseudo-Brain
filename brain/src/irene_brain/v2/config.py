"""Core V2 Configuration — immutable initial research configuration.

Do not modify W/K/C without explicit approval. First prove the learning system.
"""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CoreV2Config:
    """Immutable Core V2 configuration."""

    # Architecture (LOCKED for initial research)
    width: int = 120
    thoughtlets: int = 32
    cycles: int = 3
    actions: int = 5

    # Deployed decision aggregation.  The legacy scalar-utility quotient is
    # retained only so the failed Stage V2.0 result remains reproducible.
    # New experiments use a direct, permutation-invariant mean of per-slot
    # action logits so the supervised deployed loss has a non-zero gradient
    # at initialization.
    decision_aggregation: Literal[
        "legacy_scalar_utility_v0",
        "direct_mean_logits_v1",
    ] = "direct_mean_logits_v1"

    # The original five independently-added gates can amplify state on every
    # cycle and is preserved for failed-run reproduction.  New sequential
    # experiments must select the normalized convex mixture explicitly.
    braincell_dynamics: Literal[
        "legacy_additive_v0",
        "normalized_mixture_v1",
    ] = "legacy_additive_v0"

    # The historical belief residual is also unbounded across long sequences.
    # Keep it selectable for reproduction; use the GRU-style convex update in
    # all newly registered sequential experiments.
    belief_dynamics: Literal[
        "legacy_residual_v0",
        "convex_gated_v1",
    ] = "legacy_residual_v0"

    # Historical runs emitted an unconstrained hazard scalar. Predictive
    # training uses an actual probability so feedback and utility share the
    # same 0..1 semantics.
    hazard_parameterization: Literal[
        "legacy_unbounded_v0",
        "probability_sigmoid_v1",
    ] = "legacy_unbounded_v0"

    # Historical next-latent supervision used raw MSE, which makes arbitrary
    # encoder magnitude part of the target. Predictive experiments may instead
    # compare direction in the latent space and normalize feedback deltas.
    latent_comparison: Literal[
        "raw_mse_v0",
        "cosine_distance_v1",
    ] = "raw_mse_v0"

    # Reward outputs stay in environment units. The symlog option compresses
    # scale only while computing loss and feedback.
    reward_comparison: Literal[
        "raw_mse_v0",
        "symlog_mse_v1",
    ] = "raw_mse_v0"

    # Historical reward/hazard heads only saw the thought state. The factual
    # option conditions outcome heads on the applied action for supervision,
    # while proposal actions are used for pre-decision counterfactuals.
    outcome_action_conditioning: Literal[
        "thought_only_v0",
        "factual_or_proposal_v1",
    ] = "thought_only_v0"

    # The v1 outcome architecture evaluates every discrete action in one
    # vectorized table, then gathers the physically applied action. It keeps
    # proposal policy and factual supervision semantically separate.
    outcome_architecture: Literal[
        "legacy_hypothesis_world_v0",
        "all_action_table_v1",
    ] = "legacy_hypothesis_world_v0"
    # A dedicated hazard path prevents reward/next-state multi-task gradients
    # from monopolizing the imminent-collision representation. Its input is
    # stop-gradient because the frozen-belief probe already established that
    # the recurrent state contains the required signal.
    hazard_outcome_path: Literal[
        "shared_outcome_v0",
        "dedicated_stopgrad_v1",
    ] = "shared_outcome_v0"
    reward_prediction: Literal[
        "scalar_v0",
        "symlog_twohot_v1",
    ] = "scalar_v0"
    reward_bins: int = 65
    reward_symlog_min: float = -4.0
    reward_symlog_max: float = 4.0

    prediction_error_fusion: Literal[
        "latent_only_v0",
        "latent_outcome_surprise_v1",
    ] = "latent_only_v0"

    # Sensor encoder
    encoder_channels: int = 32
    encoder_kernel: int = 3
    encoder_stride: int = 2
    encoder_pool: bool = True

    # Memory
    episodic_capacity: int = 256
    retrieval_k: int = 8
    session_latent_dim: int = 120

    # Prediction error
    error_encoder_dim: int = 120
    surprise_dim: int = 32

    # Belief updater
    belief_hidden: int = 120

    # BrainCell
    braincell_hidden: int = 120
    braincell_gate_dim: int = 120

    # Thought attention
    attention_heads: int = 4
    attention_dim: int = 120

    # Consequence heads
    consequence_hidden: int = 120

    # World model
    world_model_hidden: int = 120

    # Learning
    decision_weight: float = 1.00
    set_weight: float = 0.50
    next_weight: float = 0.50
    reward_weight: float = 0.25
    hazard_weight: float = 0.25
    branch_weight: float = 0.25
    confidence_weight: float = 0.10
    memory_retrieval_weight: float = 0.25
    memory_write_weight: float = 0.10
    meta_weight: float = 1.00

    # Utility
    utility_reward_coeff: float = 1.0
    utility_hazard_coeff: float = 3.0

    # Temperature
    action_temperature: float = 1.0

    # Meta-learning
    trials_per_session: int = 4
    trial_loss_weights: tuple = (0.25, 0.75, 1.00, 1.00)

    # Fast plasticity (disabled by default)
    fast_plasticity_enabled: bool = False
    fast_plasticity_rank: int = 4

    # Consolidation (disabled by default)
    consolidation_enabled: bool = False

    # EMA normalization
    ema_decay: float = 0.99

    # Deterministic mode
    deterministic: bool = True

    def __post_init__(self):
        if self.width != 120 or self.thoughtlets != 32 or self.cycles != 3:
            raise ValueError(
                "Core V2 research config locked: W=120, K=32, C=3. "
                "Do not scale until learning system is proven."
            )
        if self.decision_aggregation not in {
            "legacy_scalar_utility_v0",
            "direct_mean_logits_v1",
        }:
            raise ValueError(
                "decision_aggregation must be legacy_scalar_utility_v0 or "
                "direct_mean_logits_v1"
            )
        if self.braincell_dynamics not in {
            "legacy_additive_v0",
            "normalized_mixture_v1",
        }:
            raise ValueError(
                "braincell_dynamics must be legacy_additive_v0 or "
                "normalized_mixture_v1"
            )
        if self.belief_dynamics not in {
            "legacy_residual_v0",
            "convex_gated_v1",
        }:
            raise ValueError(
                "belief_dynamics must be legacy_residual_v0 or "
                "convex_gated_v1"
            )
        if self.hazard_parameterization not in {
            "legacy_unbounded_v0",
            "probability_sigmoid_v1",
        }:
            raise ValueError(
                "hazard_parameterization must be legacy_unbounded_v0 or "
                "probability_sigmoid_v1"
            )
        if self.latent_comparison not in {
            "raw_mse_v0",
            "cosine_distance_v1",
        }:
            raise ValueError(
                "latent_comparison must be raw_mse_v0 or cosine_distance_v1"
            )
        if self.reward_comparison not in {
            "raw_mse_v0",
            "symlog_mse_v1",
        }:
            raise ValueError(
                "reward_comparison must be raw_mse_v0 or symlog_mse_v1"
            )
        if self.outcome_action_conditioning not in {
            "thought_only_v0",
            "factual_or_proposal_v1",
        }:
            raise ValueError(
                "outcome_action_conditioning must be thought_only_v0 or "
                "factual_or_proposal_v1"
            )
        if self.outcome_architecture not in {
            "legacy_hypothesis_world_v0",
            "all_action_table_v1",
        }:
            raise ValueError(
                "outcome_architecture must be legacy_hypothesis_world_v0 or "
                "all_action_table_v1"
            )
        if self.hazard_outcome_path not in {
            "shared_outcome_v0",
            "dedicated_stopgrad_v1",
        }:
            raise ValueError(
                "hazard_outcome_path must be shared_outcome_v0 or "
                "dedicated_stopgrad_v1"
            )
        if (
            self.hazard_outcome_path == "dedicated_stopgrad_v1"
            and self.outcome_architecture != "all_action_table_v1"
        ):
            raise ValueError(
                "dedicated_stopgrad_v1 requires all_action_table_v1"
            )
        if (
            self.outcome_architecture == "all_action_table_v1"
            and self.hazard_parameterization != "probability_sigmoid_v1"
        ):
            raise ValueError(
                "all_action_table_v1 requires probability_sigmoid_v1 hazards"
            )
        if self.reward_prediction not in {"scalar_v0", "symlog_twohot_v1"}:
            raise ValueError(
                "reward_prediction must be scalar_v0 or symlog_twohot_v1"
            )
        if isinstance(self.reward_bins, bool) or not isinstance(self.reward_bins, int):
            raise ValueError("reward_bins must be an integer")
        if self.reward_bins < 2:
            raise ValueError("reward_bins must be at least 2")
        if not self.reward_symlog_min < self.reward_symlog_max:
            raise ValueError(
                "reward_symlog_min must be smaller than reward_symlog_max"
            )
        if (
            self.outcome_architecture == "all_action_table_v1"
            and self.outcome_action_conditioning != "thought_only_v0"
        ):
            raise ValueError(
                "all_action_table_v1 replaces consequence-head action conditioning"
            )
        if self.prediction_error_fusion not in {
            "latent_only_v0",
            "latent_outcome_surprise_v1",
        }:
            raise ValueError(
                "prediction_error_fusion must be latent_only_v0 or "
                "latent_outcome_surprise_v1"
            )


@dataclass(frozen=True)
class FeatureFlags:
    """V2 feature flag configurations for staged integration."""

    deploy_aligned_learning: bool = True
    consequence_learning: bool = False
    world_model_learning: bool = False
    prediction_error_feedback: bool = False
    episodic_memory: bool = False
    session_adaptation: bool = False
    meta_learning: bool = False
    fast_plasticity: bool = False
    consolidation: bool = False


CONFIG_A_DECISION_ONLY = FeatureFlags(
    deploy_aligned_learning=True,
    consequence_learning=False,
    world_model_learning=False,
    prediction_error_feedback=False,
    episodic_memory=False,
    session_adaptation=False,
    meta_learning=False,
)

CONFIG_B_PREDICTIVE = FeatureFlags(
    deploy_aligned_learning=True,
    consequence_learning=True,
    world_model_learning=True,
    prediction_error_feedback=True,
    episodic_memory=False,
    session_adaptation=False,
    meta_learning=False,
)

CONFIG_C_FULL = FeatureFlags(
    deploy_aligned_learning=True,
    consequence_learning=True,
    world_model_learning=True,
    prediction_error_feedback=True,
    episodic_memory=True,
    session_adaptation=True,
    meta_learning=True,
    fast_plasticity=False,
    consolidation=False,
)


DEFAULT_CONFIG = CoreV2Config()
DEFAULT_FLAGS = CONFIG_C_FULL
