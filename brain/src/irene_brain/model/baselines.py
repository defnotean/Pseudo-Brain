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

from torch import nn

from .brain_cell import MonolithicRecurrentCell
from .torch_model import IreneBrainModel


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


class NoCommunicationSlotBaseline(IreneBrainModel):
    """Full slot model with both current-cycle communication paths severed."""

    architecture_identity = NO_COMMUNICATION_IDENTITY
    architecture_variant_id = NO_COMMUNICATION_IDENTITY.variant_id

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
    "MONOLITHIC_IDENTITY",
    "MonolithicRecurrentBaseline",
    "NO_COMMUNICATION_IDENTITY",
    "NoCommunicationSlotBaseline",
    "PARAMETER_MATCHED_MONOLITHIC_IDENTITY",
    "ParameterMatchedMonolithicBaseline",
    "REFERENCE_IDENTITY",
    "allocated_parameter_counts",
    "architecture_manifest_entry",
    "build_architecture_manifest",
]
