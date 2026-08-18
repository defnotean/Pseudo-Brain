"""Matched architecture controls for testing the Irene thought-field claim.

These are deliberately separate model classes and variant identities. A run
cannot silently substitute one control for another while retaining the same
factory path, runtime model class, or manifest digest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Mapping

from torch import Tensor, nn

from .brain_cell import (
    BrainCell,
    EnsembleBrainCell,
    MonolithicRecurrentCell,
    TransformerCarryCell,
)
from .torch_model import BrainState, IreneBrainModel, ModelOutput


@dataclass(frozen=True, slots=True)
class ArchitectureVariantIdentity:
    schema_version: int
    variant_id: str
    latent_topology: str
    peer_routing: bool
    thought_workspace_writes: bool
    pooled_recurrent_input: bool
    matching_role: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported architecture variant identity schema")
        for name in ("variant_id", "latent_topology", "matching_role"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.limitations, tuple) or not self.limitations:
            raise ValueError("limitations must be a non-empty tuple")
        if any(not isinstance(item, str) or not item for item in self.limitations):
            raise ValueError("every limitation must be a non-empty string")

    @property
    def canonical_json(self) -> str:
        payload = asdict(self)
        payload["limitations"] = list(self.limitations)
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


REFERENCE_IDENTITY = ArchitectureVariantIdentity(
    schema_version=1,
    variant_id="irene.thought_field.routed.v1",
    latent_topology="32_factorized_slots_x_3_registers",
    peer_routing=True,
    thought_workspace_writes=True,
    pooled_recurrent_input=False,
    matching_role="reference",
    limitations=(
        "Slot geometry alone does not establish distinct or causal thoughts.",
        "The candidate must pass intervention and matched-control evaluations.",
    ),
)

NO_COMMUNICATION_IDENTITY = ArchitectureVariantIdentity(
    schema_version=1,
    variant_id="irene.thought_field.isolated_slots.v1",
    latent_topology="32_factorized_slots_x_3_registers",
    peer_routing=False,
    thought_workspace_writes=False,
    pooled_recurrent_input=False,
    matching_role="exact_allocated_parameter_ablation",
    limitations=(
        "Routing and utility parameters remain allocated but are disconnected.",
        "Actuator queries still read every slot because action readout is not peer communication.",
    ),
)

MONOLITHIC_IDENTITY = ArchitectureVariantIdentity(
    schema_version=1,
    variant_id="irene.monolithic_gru.same_width.v1",
    latent_topology="single_gru_latent",
    peer_routing=False,
    thought_workspace_writes=True,
    pooled_recurrent_input=True,
    matching_role="measured_latency_or_flop_control",
    limitations=(
        "This is a conventional pooled GRU control, not every possible monolithic recurrent design.",
        "Its result cannot isolate parameter count until a capacity-matched width is selected.",
    ),
)

PARAMETER_MATCHED_MONOLITHIC_IDENTITY = ArchitectureVariantIdentity(
    schema_version=1,
    variant_id="irene.monolithic_gru.parameter_matched.v1",
    latent_topology="single_widened_gru_latent",
    peer_routing=False,
    thought_workspace_writes=True,
    pooled_recurrent_input=True,
    matching_role="nearest_width_parameter_control",
    limitations=(
        "Width is selected before training by nearest allocated trainable parameter count.",
        "Equal parameter count does not imply equal FLOPs, memory traffic, or latency.",
    ),
)

RESET_STATE_IDENTITY = ArchitectureVariantIdentity(
    schema_version=1,
    variant_id="irene.thought_field.reset_slots.v1",
    latent_topology="32_factorized_slots_x_3_registers",
    peer_routing=True,
    thought_workspace_writes=True,
    pooled_recurrent_input=False,
    matching_role="exact_allocated_parameter_persistence_ablation",
    limitations=(
        "Incoming thought state is discarded every step; slots reseed from sensors and belief.",
        "The lifecycle head remains allocated but is disconnected from the computation.",
    ),
)

DENSE_COMMUNICATION_IDENTITY = ArchitectureVariantIdentity(
    schema_version=1,
    variant_id="irene.thought_field.dense_routing.v1",
    latent_topology="32_factorized_slots_x_3_registers",
    peer_routing=True,
    thought_workspace_writes=True,
    pooled_recurrent_input=False,
    matching_role="exact_allocated_parameter_dense_communication_ablation",
    limitations=(
        "Routing is unrestricted all-to-all softmax instead of sparse top-k.",
        "Dense communication costs more message bandwidth per cycle than the reference.",
    ),
)

REACTIVE_IDENTITY = ArchitectureVariantIdentity(
    schema_version=1,
    variant_id="irene.thought_field.reactive.v1",
    latent_topology="32_factorized_slots_x_3_registers",
    peer_routing=True,
    thought_workspace_writes=True,
    pooled_recurrent_input=False,
    matching_role="exact_allocated_parameter_reactive_control",
    limitations=(
        "No state crosses steps; belief, working memory, and thoughts reseed every forward.",
        "A reactive policy cannot express temporal credit assignment across observations.",
    ),
)

SERIAL_DEPTH_IDENTITY = ArchitectureVariantIdentity(
    schema_version=1,
    variant_id="irene.thought_field.serial_depth.v1",
    latent_topology="32_factorized_slots_x_3_registers",
    peer_routing=True,
    thought_workspace_writes=True,
    pooled_recurrent_input=False,
    matching_role="approximate_flop_matched_serial_depth_control",
    limitations=(
        "Twelve untied serial blocks replace four tied blocks over three cycles; block applications match but parameters do not.",
        "Single-cycle execution produces two anytime exits instead of four.",
    ),
)

MATCHED_ENSEMBLE_IDENTITY = ArchitectureVariantIdentity(
    schema_version=1,
    variant_id="irene.thought_field.independent_ensemble.v1",
    latent_topology="4_untied_members_x_8_slots_x_3_registers",
    peer_routing=False,
    thought_workspace_writes=False,
    pooled_recurrent_input=False,
    matching_role="nearest_width_parameter_matched_independent_ensemble_control",
    limitations=(
        "Members share perception, belief, actuator, and prediction heads; only thought-slot updates are independent.",
        "Member blocks carry no routing, belief-maintenance, or workspace-write parameters, so width is widened to match the reference budget.",
        "Belief and working memory are maintained by the shared ingest pathway only, not by per-block attention.",
    ),
)


RECURRENT_TRANSFORMER_IDENTITY = ArchitectureVariantIdentity(
    schema_version=1,
    variant_id="irene.recurrent_transformer.carry_token.v1",
    latent_topology="single_carry_token_transformer_encoder",
    peer_routing=False,
    thought_workspace_writes=True,
    pooled_recurrent_input=True,
    matching_role="nearest_width_parameter_control",
    limitations=(
        "This is a conventional carry-token Transformer control, not every possible recurrent Transformer design.",
        "The carry token is the only slot-shaped cross-step state; belief and working memory re-read their own token positions each cycle.",
    ),
)

FIXED_MULTI_HORIZON_IDENTITY = ArchitectureVariantIdentity(
    schema_version=1,
    variant_id="irene.thought_field.fixed_multi_horizon.v1",
    latent_topology="32_factorized_slots_x_3_registers",
    peer_routing=True,
    thought_workspace_writes=True,
    pooled_recurrent_input=False,
    matching_role="exact_allocated_parameter_fixed_horizon_partition_control",
    limitations=(
        "Slots are statically partitioned into even contiguous horizon groups; each group's future embedding is trained only on its assigned offset.",
        "Incoming thought state is discarded every step; slots reseed from sensors and belief.",
    ),
)


class NoCommunicationSlotBaseline(IreneBrainModel):
    """Full slot model with both current-cycle communication paths severed."""

    architecture_identity = NO_COMMUNICATION_IDENTITY
    architecture_variant_id = NO_COMMUNICATION_IDENTITY.variant_id

    def _communication_policy(self, cycle: int) -> tuple[bool, bool]:
        del cycle
        return False, False


class ResetStateSlotBaseline(IreneBrainModel):
    """Full slot model with persistence removed: slots reseed every step."""

    architecture_identity = RESET_STATE_IDENTITY
    architecture_variant_id = RESET_STATE_IDENTITY.variant_id

    def _refresh_thoughts(
        self,
        *,
        thoughts: Tensor,
        sensors: Tensor,
        belief: Tensor,
        elapsed_seconds: Tensor,
        thought_age_seconds: Tensor,
        thought_noise: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del thought_age_seconds
        batch, thoughtlets, _registers, _width = thoughts.shape
        seeds = self._seed_thoughts(
            batch_size=batch,
            sensors=sensors,
            belief=belief,
            thought_noise=thought_noise,
        )
        ages = elapsed_seconds.expand(batch, thoughtlets)
        expire_probability = thoughts.new_ones((batch, thoughtlets))
        return seeds, ages, expire_probability


class DenseCommunicationSlotBaseline(IreneBrainModel):
    """Full slot model with unrestricted all-to-all peer routing."""

    architecture_identity = DENSE_COMMUNICATION_IDENTITY
    architecture_variant_id = DENSE_COMMUNICATION_IDENTITY.variant_id

    def _build_brain_cell(self, *, width: int) -> nn.Module:
        return BrainCell(
            width=width,
            heads=self.config.attention_heads,
            routed_neighbors=self.config.routed_neighbors,
            blocks=self.config.brain_cell_blocks,
            dense_routing=True,
        )


class ReactiveSlotBaseline(IreneBrainModel):
    """Full slot model with no cross-step state: a pure reactive policy."""

    architecture_identity = REACTIVE_IDENTITY
    architecture_variant_id = REACTIVE_IDENTITY.variant_id

    def forward(
        self,
        pixels: Tensor,
        previous_control: Tensor,
        elapsed_seconds: Tensor,
        state: BrainState | None = None,
        **kwargs: object,
    ) -> ModelOutput:
        del state  # Reactive control: incoming state must not influence output.
        return super().forward(
            pixels,
            previous_control,
            elapsed_seconds,
            state=None,
            **kwargs,
        )


class SerialDepthSlotBaseline(IreneBrainModel):
    """Untied serial depth with the reference's block-application count.

    The reference ties four blocks across three cognitive cycles (twelve
    block applications). This control runs twelve untied blocks in one cycle,
    matching sequential block FLOPs while removing weight tying and anytime
    exits beyond the first. Because the serial stack replaces recurrence,
    routing is permitted from the first block onward.
    """

    architecture_identity = SERIAL_DEPTH_IDENTITY
    architecture_variant_id = SERIAL_DEPTH_IDENTITY.variant_id

    def _communication_policy(self, cycle: int) -> tuple[bool, bool]:
        del cycle
        return True, True


class MatchedEnsembleBaseline(IreneBrainModel):
    """Independent untied member stacks as a matched-cost ensemble control.

    Four members each own eight of the 32 thought slots with their own
    untied lean blocks; members never route messages or write the shared
    workspace. Per-slot depth and cycle tying match the reference, so the
    only differences are weight independence across members, the absence of
    communication, and the widened core that spends the freed routing and
    workspace parameters. Width is selected before training by nearest
    allocated trainable parameter count, like the parameter-matched
    monolithic control.
    """

    ENSEMBLE_MEMBERS = 4

    architecture_identity = MATCHED_ENSEMBLE_IDENTITY
    architecture_variant_id = MATCHED_ENSEMBLE_IDENTITY.variant_id

    def _build_brain_cell(self, *, width: int) -> nn.Module:
        return EnsembleBrainCell(
            width=width,
            heads=self.config.attention_heads,
            blocks=self.config.brain_cell_blocks,
            members=self.ENSEMBLE_MEMBERS,
            thoughtlets=self.config.thoughtlets,
        )

    def _communication_policy(self, cycle: int) -> tuple[bool, bool]:
        del cycle
        return False, False


class MonolithicRecurrentBaseline(IreneBrainModel):
    """One pooled GRU latent with the same perception/action interfaces."""

    architecture_identity = MONOLITHIC_IDENTITY
    architecture_variant_id = MONOLITHIC_IDENTITY.variant_id

    def _build_brain_cell(self, *, width: int) -> nn.Module:
        return MonolithicRecurrentCell(
            width=width,
            heads=self.config.attention_heads,
            blocks=self.config.brain_cell_blocks,
        )


class ParameterMatchedMonolithicBaseline(MonolithicRecurrentBaseline):
    """Distinct identity for the preregistered nearest-width GRU control."""

    architecture_identity = PARAMETER_MATCHED_MONOLITHIC_IDENTITY
    architecture_variant_id = PARAMETER_MATCHED_MONOLITHIC_IDENTITY.variant_id


class FixedMultiHorizonSlotBaseline(ResetStateSlotBaseline):
    """Static horizon-partitioned slots with persistence removed (B1 control).

    Slots are split into even contiguous groups, one per supported
    prediction horizon of the training window; the shared objective trains
    each group's future embedding only on its assigned offset, while the
    reference and other variants keep the flexible min-over-slots
    assignment per offset. Persistence is removed exactly like the
    reset-slots ablation, so the variant isolates "fixed horizon structure
    without persistent thoughtlets" under the identical parameter count.
    """

    architecture_identity = FIXED_MULTI_HORIZON_IDENTITY
    architecture_variant_id = FIXED_MULTI_HORIZON_IDENTITY.variant_id
    fixed_horizon_partition = True


class RecurrentTransformerBaseline(IreneBrainModel):
    """A standard carry-token Transformer encoder as the item-5 control.

    One Transformer encoder token set per cycle — belief, working memory, a
    single recurrent carry token, sensors, action/time, goal context, and a
    pooled retrieved-memory token — with belief, memory, and the carry
    re-read from their own output positions. Width is selected before
    training by nearest allocated trainable parameter count, like the other
    matched controls.
    """

    architecture_identity = RECURRENT_TRANSFORMER_IDENTITY
    architecture_variant_id = RECURRENT_TRANSFORMER_IDENTITY.variant_id

    def _build_brain_cell(self, *, width: int) -> nn.Module:
        return TransformerCarryCell(
            width=width,
            heads=self.config.attention_heads,
            blocks=self.config.brain_cell_blocks,
        )


def allocated_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return exact allocated counts; disconnected parameters remain visible."""

    parameters = tuple(model.parameters())
    disconnected = 0
    if isinstance(model, NoCommunicationSlotBaseline):
        disconnected_fragments = (
            ".route_query.",
            ".route_key.",
            ".route_value.",
            ".utility.",
        )
        disconnected = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and name.startswith("brain_cell.blocks.")
            and any(fragment in name for fragment in disconnected_fragments)
        )
    if isinstance(model, ResetStateSlotBaseline):
        # The lifecycle head only gates persistence; with slots reseeded every
        # step its parameters are allocated but receive no task gradient.
        disconnected += sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and name.startswith("thought_predictions.lifecycle.")
        )
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    return {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": trainable,
        "architecturally_disconnected_trainable": disconnected,
        "architecturally_connected_trainable": trainable - disconnected,
    }


def _identity_for(model: nn.Module) -> ArchitectureVariantIdentity:
    identity = getattr(model, "architecture_identity", None)
    if identity is None and type(model) is IreneBrainModel:
        return REFERENCE_IDENTITY
    if not isinstance(identity, ArchitectureVariantIdentity):
        raise ValueError("model must expose a registered architecture identity")
    return identity


def architecture_manifest_entry(
    model: IreneBrainModel,
    *,
    model_factory: str,
    reference_trainable_parameters: int,
    bytes_per_persistent_element: int = 2,
) -> dict[str, object]:
    if not isinstance(model_factory, str) or ":" not in model_factory:
        raise ValueError("model_factory must use module.path:callable syntax")
    if reference_trainable_parameters < 1:
        raise ValueError("reference_trainable_parameters must be positive")
    if bytes_per_persistent_element < 1:
        raise ValueError("bytes_per_persistent_element must be positive")
    identity = _identity_for(model)
    counts = allocated_parameter_counts(model)
    config = model.config
    persistent_bytes = config.persistent_state_bytes(
        batch_size=1,
        bytes_per_element=bytes_per_persistent_element,
    )
    delta = counts["trainable"] - reference_trainable_parameters
    return {
        "variant_id": identity.variant_id,
        "variant_identity_sha256": identity.sha256,
        "identity": json.loads(identity.canonical_json),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_factory": model_factory,
        "allocated_parameters": counts,
        "trainable_parameter_delta_from_reference": delta,
        "absolute_trainable_parameter_delta_fraction": abs(delta)
        / reference_trainable_parameters,
        "persistent_state": {
            "batch_size": 1,
            "bytes_per_element": bytes_per_persistent_element,
            "bytes": persistent_bytes,
            "elements": persistent_bytes // bytes_per_persistent_element,
            "thought_elements": (
                config.thoughtlets
                * config.registers_per_thoughtlet
                * config.core_width
            ),
        },
        "shape_controls": {
            "core_width": config.core_width,
            "thoughtlets": config.thoughtlets,
            "registers_per_thoughtlet": config.registers_per_thoughtlet,
            "cognitive_cycles": config.cognitive_cycles,
            "brain_cell_blocks": config.brain_cell_blocks,
            "attention_heads": config.attention_heads,
            "routed_neighbors": config.routed_neighbors,
            "retrieved_entries_per_thoughtlet": (
                config.retrieved_entries_per_thoughtlet
            ),
            "aggregate_retrieved_tokens": (
                config.thoughtlets * config.retrieved_entries_per_thoughtlet
            ),
        },
    }


def build_architecture_manifest(
    models: Mapping[str, tuple[IreneBrainModel, str]],
    *,
    parameter_tolerance_fraction: float = 0.01,
) -> dict[str, object]:
    """Build the strict, JSON-serializable matched-baseline manifest.

    ``models`` is keyed by registered variant id and maps to ``(model,
    factory_path)``. Latency/FLOP matching remains explicitly unverified until
    a hardware-specific calibration artifact is attached.
    """

    if not (0.0 <= parameter_tolerance_fraction < 1.0):
        raise ValueError("parameter_tolerance_fraction must be in [0, 1)")
    reference_pair = models.get(REFERENCE_IDENTITY.variant_id)
    if reference_pair is None:
        raise ValueError("manifest requires the routed thought-field reference")
    reference_model, _reference_factory = reference_pair
    reference_count = allocated_parameter_counts(reference_model)["trainable"]
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for supplied_id in sorted(models):
        model, factory = models[supplied_id]
        identity = _identity_for(model)
        if supplied_id != identity.variant_id:
            raise ValueError("manifest key does not match model architecture identity")
        if identity.variant_id in seen:
            raise ValueError("architecture variant ids must be unique")
        seen.add(identity.variant_id)
        entries.append(
            architecture_manifest_entry(
                model,
                model_factory=factory,
                reference_trainable_parameters=reference_count,
            )
        )

    by_id = {str(entry["variant_id"]): entry for entry in entries}
    parameter_controls = (
        NO_COMMUNICATION_IDENTITY.variant_id,
        PARAMETER_MATCHED_MONOLITHIC_IDENTITY.variant_id,
        MATCHED_ENSEMBLE_IDENTITY.variant_id,
        RECURRENT_TRANSFORMER_IDENTITY.variant_id,
    )
    missing = [variant_id for variant_id in parameter_controls if variant_id not in by_id]
    if missing:
        raise ValueError(f"manifest is missing parameter controls: {missing}")
    parameter_match_passes = all(
        float(by_id[variant_id]["absolute_trainable_parameter_delta_fraction"])
        <= parameter_tolerance_fraction
        for variant_id in parameter_controls
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "reference_variant_id": REFERENCE_IDENTITY.variant_id,
        "variants": entries,
        "fairness_regimes": {
            "parameter_matched": {
                "comparison_variant_ids": list(parameter_controls),
                "maximum_allocated_trainable_parameter_delta_fraction": (
                    parameter_tolerance_fraction
                ),
                "verified": parameter_match_passes,
                "fixed_controls": [
                    "dataset_bytes_and_order",
                    "optimizer_and_schedule",
                    "objective",
                    "optimizer_steps",
                    "precision",
                    "seed_set",
                ],
            },
            "measured_latency_or_flop_matched": {
                "comparison_variant_ids": [MONOLITHIC_IDENTITY.variant_id],
                "verified": False,
                "status": "hardware_calibration_required_before_training",
                "matching_variables": ["cognitive_cycles"],
                "required_measurements": [
                    "device_fingerprint",
                    "median_end_to_end_latency_ns",
                    "p95_end_to_end_latency_ns",
                    "measured_or_profiled_flops_per_decision",
                    "peak_accelerator_memory_bytes",
                ],
                "maximum_latency_delta_fraction": 0.05,
            },
        },
        "claim_boundary": (
            "This manifest records architecture and resource controls only; it does "
            "not establish independent thoughts, causal specialization, superiority, "
            "or human-like cognition."
        ),
    }
    canonical_without_digest = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload["manifest_sha256"] = sha256(
        canonical_without_digest.encode("utf-8")
    ).hexdigest()
    return payload


__all__ = [
    "ArchitectureVariantIdentity",
    "DENSE_COMMUNICATION_IDENTITY",
    "DenseCommunicationSlotBaseline",
    "FIXED_MULTI_HORIZON_IDENTITY",
    "FixedMultiHorizonSlotBaseline",
    "MATCHED_ENSEMBLE_IDENTITY",
    "MatchedEnsembleBaseline",
    "MONOLITHIC_IDENTITY",
    "MonolithicRecurrentBaseline",
    "NO_COMMUNICATION_IDENTITY",
    "NoCommunicationSlotBaseline",
    "PARAMETER_MATCHED_MONOLITHIC_IDENTITY",
    "ParameterMatchedMonolithicBaseline",
    "REACTIVE_IDENTITY",
    "REFERENCE_IDENTITY",
    "RECURRENT_TRANSFORMER_IDENTITY",
    "RESET_STATE_IDENTITY",
    "ReactiveSlotBaseline",
    "RecurrentTransformerBaseline",
    "ResetStateSlotBaseline",
    "SERIAL_DEPTH_IDENTITY",
    "SerialDepthSlotBaseline",
    "allocated_parameter_counts",
    "architecture_manifest_entry",
    "build_architecture_manifest",
]
