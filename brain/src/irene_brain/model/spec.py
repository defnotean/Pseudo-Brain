"""Validated tensor and connectivity contract with no ML dependency."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def _exact_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class TensorShape:
    name: str
    dimensions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tensor shape name must be non-empty")
        if not isinstance(self.dimensions, tuple) or not self.dimensions:
            raise ValueError("tensor dimensions must be a non-empty tuple")
        for index, dimension in enumerate(self.dimensions):
            _positive_int(dimension, name=f"{self.name}.dimensions[{index}]")

    @property
    def elements(self) -> int:
        result = 1
        for dimension in self.dimensions:
            result *= dimension
        return result


@dataclass(frozen=True, slots=True)
class ActuatorQuerySpec:
    keyboard_keys: int = 256
    mouse_buttons: int = 8
    mouse_axes: int = 2
    scroll_axes: int = 1
    gamepad_buttons: int = 32
    gamepad_axes: int = 8
    continuous_squash: str = "none"

    def __post_init__(self) -> None:
        fixed_dimensions = {
            "keyboard_keys": 256,
            "mouse_buttons": 8,
            "mouse_axes": 2,
            "scroll_axes": 1,
            "gamepad_buttons": 32,
            "gamepad_axes": 8,
        }
        for name, expected in fixed_dimensions.items():
            value = _positive_int(getattr(self, name), name=name)
            if value != expected:
                raise ValueError(f"{name} is fixed at {expected} actuator queries")
        if self.continuous_squash not in {"none", "deadzone_tanh"}:
            raise ValueError("continuous_squash must be none or deadzone_tanh")

    @property
    def total_queries(self) -> int:
        return (
            self.keyboard_keys
            + self.mouse_buttons
            + self.mouse_axes
            + self.scroll_axes
            + self.gamepad_buttons
            + self.gamepad_axes
        )


@dataclass(frozen=True, slots=True)
class StateLayout:
    sensor: TensorShape
    belief: TensorShape
    working_memory: TensorShape
    thought_field: TensorShape
    goal_context: TensorShape
    retrieved_memory: TensorShape
    actuator_queries: TensorShape

    @property
    def persistent_elements(self) -> int:
        return (
            self.belief.elements
            + self.working_memory.elements
            + self.thought_field.elements
            + self.goal_context.elements
        )


@dataclass(frozen=True, slots=True)
class AttentionContract:
    """Structural reads allowed by the candidate architecture."""

    reads: Mapping[str, tuple[str, ...]]
    cycle_one_cross_thought_neighbors: int
    later_cross_thought_neighbors: int
    uses_pooled_integration_token: bool
    weights_shared_across_thoughtlets: bool
    weights_shared_across_cycles: bool

    def __post_init__(self) -> None:
        if not isinstance(self.reads, Mapping):
            raise ValueError("reads must be a mapping")
        frozen: dict[str, tuple[str, ...]] = {}
        for destination, sources in self.reads.items():
            if not destination or not isinstance(destination, str):
                raise ValueError("attention destination names must be non-empty")
            if not isinstance(sources, tuple) or any(
                not isinstance(source, str) or not source for source in sources
            ):
                raise ValueError("attention source lists must be tuples of names")
            frozen[destination] = sources
        object.__setattr__(self, "reads", MappingProxyType(frozen))
        cycle_one_neighbors = _nonnegative_int(
            self.cycle_one_cross_thought_neighbors,
            name="cycle_one_cross_thought_neighbors",
        )
        if cycle_one_neighbors != 0:
            raise ValueError("cycle one must keep thoughtlet updates private")
        _nonnegative_int(
            self.later_cross_thought_neighbors,
            name="later_cross_thought_neighbors",
        )
        uses_pool = _exact_bool(
            self.uses_pooled_integration_token,
            name="uses_pooled_integration_token",
        )
        shared_thought_weights = _exact_bool(
            self.weights_shared_across_thoughtlets,
            name="weights_shared_across_thoughtlets",
        )
        shared_cycle_weights = _exact_bool(
            self.weights_shared_across_cycles,
            name="weights_shared_across_cycles",
        )
        if uses_pool:
            raise ValueError("the thesis architecture forbids one pooled integration token")
        if not shared_thought_weights:
            raise ValueError("thoughtlets must use one shared parameter set")
        if not shared_cycle_weights:
            raise ValueError("cognitive cycles must reuse the BrainCell weights")

    @classmethod
    def thesis_default(cls, *, routed_neighbors: int) -> AttentionContract:
        return cls(
            reads={
                "belief": ("sensor", "prior_control", "belief", "working_memory"),
                "thoughtlet": (
                    "own_registers",
                    "sensor",
                    "belief",
                    "goal_context",
                    "own_retrieved_memory",
                    "routed_thoughtlets",
                ),
                "working_memory": ("belief", "high_utility_thought_writes", "memory"),
                "actuator_query": (
                    "sensor",
                    "belief",
                    "all_thoughtlet_registers",
                    "working_memory",
                    "retrieved_memory",
                    "goal_context",
                ),
            },
            cycle_one_cross_thought_neighbors=0,
            later_cross_thought_neighbors=routed_neighbors,
            uses_pooled_integration_token=False,
            weights_shared_across_thoughtlets=True,
            weights_shared_across_cycles=True,
        )


@dataclass(frozen=True, slots=True)
class ThoughtFieldConfig:
    schema_version: int = 1
    core_width: int = 384
    sensor_tokens: int = 64
    belief_tokens: int = 48
    working_memory_tokens: int = 16
    thoughtlets: int = 32
    registers_per_thoughtlet: int = 3
    goal_context_tokens: int = 8
    cognitive_cycles: int = 3
    brain_cell_blocks: int = 4
    attention_heads: int = 8
    routed_neighbors: int = 2
    episodic_memory_entries: int = 512
    retrieved_entries_per_thoughtlet: int = 2
    actuator: ActuatorQuerySpec = ActuatorQuerySpec()
    dense_routing: bool = False
    use_cgp: bool = False
    plastic_decay: float = 0.999
    plastic_lr: float = 0.25

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(self.schema_version, int):
            raise ValueError("schema_version must be an integer")
        if self.schema_version != 1:
            raise ValueError("unsupported thought-field config schema")
        for name in (
            "core_width",
            "sensor_tokens",
            "belief_tokens",
            "working_memory_tokens",
            "thoughtlets",
            "registers_per_thoughtlet",
            "goal_context_tokens",
            "cognitive_cycles",
            "brain_cell_blocks",
            "attention_heads",
            "episodic_memory_entries",
            "retrieved_entries_per_thoughtlet",
        ):
            _positive_int(getattr(self, name), name=name)
        _nonnegative_int(self.routed_neighbors, name="routed_neighbors")
        if self.routed_neighbors == 0 and self.thoughtlets != 1:
            raise ValueError("zero routed neighbors is reserved for one-latent controls")
        if self.core_width % self.attention_heads != 0:
            raise ValueError("core_width must be divisible by attention_heads")
        if self.routed_neighbors >= self.thoughtlets:
            raise ValueError("routed_neighbors must be smaller than thoughtlets")
        if self.retrieved_entries_per_thoughtlet > self.episodic_memory_entries:
            raise ValueError("retrieval count cannot exceed episodic memory capacity")
        if not isinstance(self.actuator, ActuatorQuerySpec):
            raise ValueError("actuator must be an ActuatorQuerySpec")
        _exact_bool(self.dense_routing, name="dense_routing")
        _exact_bool(self.use_cgp, name="use_cgp")
        if self.plastic_decay <= 0.0 or self.plastic_decay > 1.0:
            raise ValueError("plastic_decay must be in (0, 1]")
        if self.plastic_lr < 0.0:
            raise ValueError("plastic_lr must be nonnegative")

    @classmethod
    def thesis_mvp(cls) -> ThoughtFieldConfig:
        return cls()

    @classmethod
    def smoke(cls) -> ThoughtFieldConfig:
        """Return a small, topology-faithful configuration for CPU smoke tests.

        The actuator wire contract, three-register thoughtlets, private first
        cycle, sparse later routing, and tied recurrent cycles are deliberately
        unchanged.  Only tensor widths/counts are reduced.
        """

        return cls(
            core_width=64,
            sensor_tokens=16,
            belief_tokens=8,
            working_memory_tokens=4,
            thoughtlets=8,
            registers_per_thoughtlet=3,
            goal_context_tokens=2,
            cognitive_cycles=3,
            brain_cell_blocks=2,
            attention_heads=4,
            routed_neighbors=2,
            episodic_memory_entries=32,
            retrieved_entries_per_thoughtlet=2,
        )

    @property
    def attention_contract(self) -> AttentionContract:
        return AttentionContract.thesis_default(
            routed_neighbors=self.routed_neighbors,
        )

    def state_layout(self, *, batch_size: int) -> StateLayout:
        batch = _positive_int(batch_size, name="batch_size")
        width = self.core_width
        return StateLayout(
            sensor=TensorShape("sensor", (batch, self.sensor_tokens, width)),
            belief=TensorShape("belief", (batch, self.belief_tokens, width)),
            working_memory=TensorShape(
                "working_memory",
                (batch, self.working_memory_tokens, width),
            ),
            thought_field=TensorShape(
                "thought_field",
                (
                    batch,
                    self.thoughtlets,
                    self.registers_per_thoughtlet,
                    width,
                ),
            ),
            goal_context=TensorShape(
                "goal_context",
                (batch, self.goal_context_tokens, width),
            ),
            retrieved_memory=TensorShape(
                "retrieved_memory",
                (
                    batch,
                    self.thoughtlets,
                    self.retrieved_entries_per_thoughtlet,
                    width,
                ),
            ),
            actuator_queries=TensorShape(
                "actuator_queries",
                (batch, self.actuator.total_queries, width),
            ),
        )

    def persistent_state_bytes(self, *, batch_size: int, bytes_per_element: int = 2) -> int:
        element_bytes = _positive_int(bytes_per_element, name="bytes_per_element")
        return self.state_layout(batch_size=batch_size).persistent_elements * element_bytes

    @property
    def tied_core_parameter_estimate(self) -> int:
        """Projection/attention/FFN estimate; sensory and actuator stems excluded."""

        return 12 * self.brain_cell_blocks * self.core_width * self.core_width
