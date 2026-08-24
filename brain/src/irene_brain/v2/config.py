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