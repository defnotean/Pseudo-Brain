"""Preregistered paired multi-seed comparison for architecture controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import itertools
import json
from math import isfinite
import re
from statistics import mean, median
import tomllib
from typing import Iterable, Mapping


_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


_CONFIG_SECTIONS = frozenset(
    {
        "schema_version",
        "run",
        "dataset",
        "optimization",
        "objective",
        "precision",
        "determinism",
        "logging",
        "resources",
    }
)


def _registered_config_payload(
    *,
    canonical_json: object,
    raw_toml: object,
    canonical_sha256: str,
    raw_sha256: str,
    name: str,
) -> dict[str, object]:
    if not isinstance(canonical_json, str) or not canonical_json:
        raise ValueError(f"{name} canonical JSON must be a nonempty string")
    if not isinstance(raw_toml, str) or not raw_toml:
        raise ValueError(f"{name} raw TOML must be a nonempty UTF-8 string")
    try:
        payload = json.loads(canonical_json)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{name} canonical JSON is invalid") from error
    if not isinstance(payload, dict) or set(payload) != _CONFIG_SECTIONS:
        raise ValueError(f"{name} has the wrong training-config sections")
    if type(payload.get("schema_version")) is not int or payload.get(
        "schema_version"
    ) != 2:
        raise ValueError(f"{name} must use training-config schema 2")
    if _canonical_json(payload) != canonical_json:
        raise ValueError(f"{name} canonical JSON is not canonical")
    if sha256(canonical_json.encode("utf-8")).hexdigest() != canonical_sha256:
        raise ValueError(f"{name} canonical config digest mismatch")
    if sha256(raw_toml.encode("utf-8")).hexdigest() != raw_sha256:
        raise ValueError(f"{name} raw config digest mismatch")
    try:
        parsed_toml = tomllib.loads(raw_toml)
    except (tomllib.TOMLDecodeError, RecursionError) as error:
        raise ValueError(f"{name} raw TOML is invalid") from error
    if _canonical_json(parsed_toml) != canonical_json:
        raise ValueError(f"{name} raw and canonical configs differ")
    run = payload.get("run")
    if not isinstance(run, dict) or set(run) != {
        "name",
        "seed",
        "model_factory",
        "max_optimizer_steps",
    }:
        raise ValueError(f"{name}.run has the wrong fields")
    _text(run.get("name"), name=f"{name}.run.name")
    _nonnegative_int(run.get("seed"), name=f"{name}.run.seed")
    _text(run.get("model_factory"), name=f"{name}.run.model_factory")
    _positive_int(
        run.get("max_optimizer_steps"),
        name=f"{name}.run.max_optimizer_steps",
    )
    for section in _CONFIG_SECTIONS.difference({"schema_version", "run"}):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"{name}.{section} must be an object")
    return payload


def _normalized_config_payload(
    payload: Mapping[str, object], *, remove_model_factory: bool
) -> dict[str, object]:
    normalized = json.loads(_canonical_json(payload))
    run = normalized["run"]
    del run["name"]
    del run["seed"]
    if remove_model_factory:
        del run["model_factory"]
    return normalized


def _normalized_training_budget_sha256(payload: Mapping[str, object]) -> str:
    normalized = _normalized_config_payload(payload, remove_model_factory=True)
    return sha256(_canonical_json(normalized).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ComparisonCriterion:
    minimum_training_seeds: int
    minimum_suites: int
    minimum_paired_mean_delta: float
    minimum_per_suite_mean_delta: float
    familywise_alpha: float
    maximum_exact_permutation_seeds: int = 20

    def __post_init__(self) -> None:
        _positive_int(self.minimum_training_seeds, name="minimum_training_seeds")
        _positive_int(self.minimum_suites, name="minimum_suites")
        _positive_int(
            self.maximum_exact_permutation_seeds,
            name="maximum_exact_permutation_seeds",
        )
        if self.minimum_training_seeds > self.maximum_exact_permutation_seeds:
            raise ValueError("minimum seeds exceed the exact permutation limit")
        for name in ("minimum_paired_mean_delta", "minimum_per_suite_mean_delta"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not -1.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [-1, 1]")
        if (
            isinstance(self.familywise_alpha, bool)
            or not isinstance(self.familywise_alpha, (int, float))
            or not isfinite(self.familywise_alpha)
            or not 0.0 < self.familywise_alpha < 1.0
        ):
            raise ValueError("familywise_alpha must be in (0, 1)")

    @property
    def sha256(self) -> str:
        return sha256(_canonical_json(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RegisteredSuite:
    suite_id: str
    weight: float
    evaluation_sample_manifest_sha256: str
    evaluation_code_sha256: str
    evaluated_samples: int

    def __post_init__(self) -> None:
        _text(self.suite_id, name="suite_id")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not isfinite(self.weight)
            or self.weight <= 0.0
        ):
            raise ValueError("suite weight must be finite and positive")
        _digest(
            self.evaluation_sample_manifest_sha256,
            name="evaluation_sample_manifest_sha256",
        )
        _digest(self.evaluation_code_sha256, name="evaluation_code_sha256")
        _positive_int(self.evaluated_samples, name="evaluated_samples")


@dataclass(frozen=True, slots=True)
class RegisteredRun:
    variant_id: str
    variant_identity_sha256: str
    training_seed: int
    run_id: str
    template_canonical_config_sha256: str
    template_raw_config_sha256: str
    template_config_canonical_json: str
    template_config_raw_toml: str
    canonical_config_sha256: str
    raw_config_sha256: str
    effective_config_canonical_json: str
    effective_config_raw_toml: str
    training_data_sha256: str
    training_code_sha256: str
    training_budget_sha256: str
    expected_optimizer_step: int

    def __post_init__(self) -> None:
        _text(self.variant_id, name="variant_id")
        _digest(self.variant_identity_sha256, name="variant_identity_sha256")
        _nonnegative_int(self.training_seed, name="training_seed")
        _text(self.run_id, name="run_id")
        for name in (
            "template_canonical_config_sha256",
            "template_raw_config_sha256",
            "canonical_config_sha256",
            "raw_config_sha256",
            "training_data_sha256",
            "training_code_sha256",
            "training_budget_sha256",
        ):
            _digest(getattr(self, name), name=name)
        for name in (
            "template_config_canonical_json",
            "template_config_raw_toml",
            "effective_config_canonical_json",
            "effective_config_raw_toml",
        ):
            _text(getattr(self, name), name=name)
        _positive_int(self.expected_optimizer_step, name="expected_optimizer_step")


@dataclass(frozen=True, slots=True)
class ExecutedRun:
    variant_id: str
    training_seed: int
    run_id: str
    checkpoint_sha256: str
    optimizer_step: int

    def __post_init__(self) -> None:
        _text(self.variant_id, name="variant_id")
        _nonnegative_int(self.training_seed, name="training_seed")
        _text(self.run_id, name="run_id")
        _digest(self.checkpoint_sha256, name="checkpoint_sha256")
        _positive_int(self.optimizer_step, name="optimizer_step")


@dataclass(frozen=True, slots=True)
class CampaignExecution:
    schema_version: int
    campaign_sha256: str
    runs: tuple[ExecutedRun, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported campaign execution schema")
        _digest(self.campaign_sha256, name="campaign_sha256")
        if not isinstance(self.runs, tuple) or not self.runs:
            raise ValueError("execution runs must be a nonempty tuple")
        if any(not isinstance(run, ExecutedRun) for run in self.runs):
            raise ValueError("execution runs must contain ExecutedRun values")
        keys = tuple((run.variant_id, run.training_seed) for run in self.runs)
        if len(set(keys)) != len(keys):
            raise ValueError("executed variant/seed runs must be unique")
        run_ids = tuple(run.run_id for run in self.runs)
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("executed run ids must be unique")

    @property
    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CampaignRegistration:
    schema_version: int
    campaign_id: str
    scope: str
    architecture_manifest_sha256: str
    fairness_regime: str
    criterion_sha256: str
    training_seeds: tuple[int, ...]
    suites: tuple[RegisteredSuite, ...]
    runs: tuple[RegisteredRun, ...]
    checkpoint_selection_rule: str
    stopping_rule: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported campaign registration schema")
        _text(self.campaign_id, name="campaign_id")
        _text(self.scope, name="scope")
        _digest(
            self.architecture_manifest_sha256,
            name="architecture_manifest_sha256",
        )
        _text(self.fairness_regime, name="fairness_regime")
        _digest(self.criterion_sha256, name="criterion_sha256")
        if not isinstance(self.training_seeds, tuple) or not self.training_seeds:
            raise ValueError("training_seeds must be a nonempty tuple")
        for seed in self.training_seeds:
            _nonnegative_int(seed, name="training_seed")
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("training_seeds must be unique")
        if not isinstance(self.suites, tuple) or not self.suites:
            raise ValueError("suites must be a nonempty tuple")
        if any(not isinstance(suite, RegisteredSuite) for suite in self.suites):
            raise ValueError("suites must contain RegisteredSuite values")
        suite_ids = tuple(suite.suite_id for suite in self.suites)
        if len(set(suite_ids)) != len(suite_ids):
            raise ValueError("registered suite ids must be unique")
        if not isinstance(self.runs, tuple) or not self.runs:
            raise ValueError("runs must be a nonempty tuple")
        if any(not isinstance(run, RegisteredRun) for run in self.runs):
            raise ValueError("runs must contain RegisteredRun values")
        run_keys = tuple((run.variant_id, run.training_seed) for run in self.runs)
        if len(set(run_keys)) != len(run_keys):
            raise ValueError("registered variant/seed runs must be unique")
        run_ids = tuple(run.run_id for run in self.runs)
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("registered run ids must be unique")
        _text(self.checkpoint_selection_rule, name="checkpoint_selection_rule")
        _text(self.stopping_rule, name="stopping_rule")

    @property
    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BenchmarkObservation:
    variant_id: str
    variant_identity_sha256: str
    training_seed: int
    suite_id: str
    normalized_score: float
    evaluated_samples: int
    evaluation_sample_manifest_sha256: str
    evaluation_code_sha256: str
    run_id: str
    checkpoint_sha256: str
    optimizer_step: int
    canonical_config_sha256: str
    raw_config_sha256: str
    training_data_sha256: str
    training_code_sha256: str
    training_budget_sha256: str

    def __post_init__(self) -> None:
        _text(self.variant_id, name="variant_id")
        _digest(self.variant_identity_sha256, name="variant_identity_sha256")
        _nonnegative_int(self.training_seed, name="training_seed")
        _text(self.suite_id, name="suite_id")
        if (
            isinstance(self.normalized_score, bool)
            or not isinstance(self.normalized_score, (int, float))
            or not isfinite(self.normalized_score)
            or not 0.0 <= self.normalized_score <= 1.0
        ):
            raise ValueError("normalized_score must be finite and in [0, 1]")
        _positive_int(self.evaluated_samples, name="evaluated_samples")
        for name in (
            "evaluation_sample_manifest_sha256",
            "evaluation_code_sha256",
            "checkpoint_sha256",
            "canonical_config_sha256",
            "raw_config_sha256",
            "training_data_sha256",
            "training_code_sha256",
            "training_budget_sha256",
        ):
            _digest(getattr(self, name), name=name)
        _text(self.run_id, name="run_id")
        _positive_int(self.optimizer_step, name="optimizer_step")


def _validate_architecture_manifest(
    manifest: Mapping[str, object],
    *,
    expected_sha256: str,
) -> tuple[str, dict[str, Mapping[str, object]]]:
    if not isinstance(manifest, Mapping):
        raise ValueError("architecture manifest must be an object")
    expected = _digest(expected_sha256, name="expected_architecture_manifest_sha256")
    embedded = _digest(
        manifest.get("manifest_sha256"),
        name="architecture manifest digest",
    )
    payload = dict(manifest)
    del payload["manifest_sha256"]
    actual = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    if actual != embedded:
        raise ValueError("architecture manifest self-digest mismatch")
    if actual != expected:
        raise ValueError("architecture manifest differs from the externally pinned digest")
    if type(manifest.get("schema_version")) is not int or manifest.get(
        "schema_version"
    ) != 1:
        raise ValueError("unsupported architecture manifest schema")
    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("architecture manifest variants are missing")
    by_id: dict[str, Mapping[str, object]] = {}
    for entry in variants:
        if not isinstance(entry, Mapping):
            raise ValueError("architecture variant entries must be objects")
        variant_id = _text(entry.get("variant_id"), name="variant_id")
        _digest(
            entry.get("variant_identity_sha256"),
            name="variant_identity_sha256",
        )
        if variant_id in by_id:
            raise ValueError("architecture manifest variant ids must be unique")
        by_id[variant_id] = entry
    return actual, by_id


def _exact_two_sided_sign_flip_pvalue(differences: tuple[float, ...]) -> float:
    """Exact randomization p-value under exchangeable, sign-symmetric deltas."""

    if not differences:
        raise ValueError("paired differences cannot be empty")
    observed = abs(mean(differences))
    extreme = 0
    total = 1 << len(differences)
    tolerance = 1e-15
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        permuted = abs(mean(sign * value for sign, value in zip(signs, differences)))
        if permuted + tolerance >= observed:
            extreme += 1
    return extreme / total


def evaluate_multiseed_comparison(
    observations: Iterable[BenchmarkObservation],
    *,
    architecture_manifest: Mapping[str, object],
    expected_architecture_manifest_sha256: str,
    campaign: CampaignRegistration,
    expected_campaign_sha256: str,
    execution: CampaignExecution,
    expected_execution_sha256: str,
    criterion: ComparisonCriterion,
) -> dict[str, object]:
    """Evaluate exactly one externally pinned, complete paired campaign."""

    if not isinstance(campaign, CampaignRegistration):
        raise ValueError("campaign must be a CampaignRegistration")
    if not isinstance(criterion, ComparisonCriterion):
        raise ValueError("criterion must be a ComparisonCriterion")
    if not isinstance(execution, CampaignExecution):
        raise ValueError("execution must be a CampaignExecution")
    manifest_digest, manifest_variants = _validate_architecture_manifest(
        architecture_manifest,
        expected_sha256=expected_architecture_manifest_sha256,
    )
    pinned_campaign = _digest(expected_campaign_sha256, name="expected_campaign_sha256")
    if campaign.sha256 != pinned_campaign:
        raise ValueError("campaign registration differs from the externally pinned digest")
    pinned_execution = _digest(
        expected_execution_sha256,
        name="expected_execution_sha256",
    )
    if execution.sha256 != pinned_execution:
        raise ValueError("campaign execution differs from the externally pinned digest")
    if execution.campaign_sha256 != pinned_campaign:
        raise ValueError("campaign execution binds a different registration")
    if campaign.architecture_manifest_sha256 != manifest_digest:
        raise ValueError("campaign pins a different architecture manifest")
    if campaign.criterion_sha256 != criterion.sha256:
        raise ValueError("campaign pins a different comparison criterion")

    regimes = architecture_manifest.get("fairness_regimes")
    if not isinstance(regimes, Mapping) or campaign.fairness_regime not in regimes:
        raise ValueError("campaign fairness regime is not registered")
    regime = regimes[campaign.fairness_regime]
    if not isinstance(regime, Mapping) or regime.get("verified") is not True:
        raise ValueError("campaign fairness regime is not verified")
    comparison_ids = regime.get("comparison_variant_ids")
    if (
        not isinstance(comparison_ids, list)
        or not comparison_ids
        or any(not isinstance(value, str) or not value for value in comparison_ids)
    ):
        raise ValueError("fairness regime comparison ids are invalid")
    reference_id = _text(
        architecture_manifest.get("reference_variant_id"),
        name="reference_variant_id",
    )
    expected_variants = frozenset((reference_id, *comparison_ids))
    if len(expected_variants) != 1 + len(comparison_ids):
        raise ValueError("reference and comparison variant ids must be distinct")
    if not expected_variants.issubset(manifest_variants):
        raise ValueError("fairness regime references an unregistered architecture")
    if campaign.fairness_regime != "parameter_matched":
        raise ValueError("only the verified parameter-matched regime is supported")
    fixed_controls = regime.get("fixed_controls")
    required_controls = {
        "dataset_bytes_and_order",
        "optimizer_and_schedule",
        "objective",
        "optimizer_steps",
        "precision",
        "seed_set",
    }
    if not isinstance(fixed_controls, list) or not required_controls.issubset(
        fixed_controls
    ):
        raise ValueError("parameter-matched regime does not freeze all controls")
    maximum_delta = regime.get(
        "maximum_allocated_trainable_parameter_delta_fraction"
    )
    if (
        isinstance(maximum_delta, bool)
        or not isinstance(maximum_delta, (int, float))
        or not isfinite(maximum_delta)
        or not 0.0 <= maximum_delta <= 1.0
    ):
        raise ValueError("parameter-matched regime has an invalid parameter tolerance")
    reference_parameters = manifest_variants[reference_id].get("allocated_parameters")
    if not isinstance(reference_parameters, Mapping):
        raise ValueError("reference allocated-parameter evidence is missing")
    reference_trainable = _positive_int(
        reference_parameters.get("trainable"),
        name="reference allocated trainable parameters",
    )
    for comparison_id in comparison_ids:
        parameters = manifest_variants[comparison_id].get("allocated_parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("comparison allocated-parameter evidence is missing")
        comparison_trainable = _positive_int(
            parameters.get("trainable"),
            name="comparison allocated trainable parameters",
        )
        delta = abs(comparison_trainable - reference_trainable) / reference_trainable
        if delta > float(maximum_delta) + 1e-15:
            raise ValueError("comparison exceeds the registered parameter tolerance")

    seeds = campaign.training_seeds
    suites = campaign.suites
    suite_ids = tuple(suite.suite_id for suite in suites)
    if len(seeds) < criterion.minimum_training_seeds:
        raise ValueError("campaign registers fewer training seeds than required")
    if len(seeds) > criterion.maximum_exact_permutation_seeds:
        raise ValueError("campaign exceeds the exact permutation seed limit")
    if len(suites) < criterion.minimum_suites:
        raise ValueError("campaign registers fewer suites than required")
    registered_runs = {(run.variant_id, run.training_seed): run for run in campaign.runs}
    expected_run_keys = {
        (variant_id, seed) for variant_id in expected_variants for seed in seeds
    }
    if set(registered_runs) != expected_run_keys:
        raise ValueError("campaign run registration is not the exact variant/seed matrix")
    for run in campaign.runs:
        variant = manifest_variants[run.variant_id]
        expected_identity = variant.get("variant_identity_sha256")
        if run.variant_identity_sha256 != expected_identity:
            raise ValueError("campaign run has the wrong architecture identity")
        recipe = variant.get("training_recipe")
        if not isinstance(recipe, Mapping):
            raise ValueError("architecture training-recipe identity is missing")
        if run.template_canonical_config_sha256 != recipe.get(
            "canonical_config_sha256"
        ):
            raise ValueError("campaign run has the wrong canonical recipe identity")
        if run.template_raw_config_sha256 != recipe.get("raw_sha256"):
            raise ValueError("campaign run has the wrong raw recipe identity")
        template_payload = _registered_config_payload(
            canonical_json=run.template_config_canonical_json,
            raw_toml=run.template_config_raw_toml,
            canonical_sha256=run.template_canonical_config_sha256,
            raw_sha256=run.template_raw_config_sha256,
            name="template config",
        )
        effective_payload = _registered_config_payload(
            canonical_json=run.effective_config_canonical_json,
            raw_toml=run.effective_config_raw_toml,
            canonical_sha256=run.canonical_config_sha256,
            raw_sha256=run.raw_config_sha256,
            name="effective config",
        )
        normalized_template = _canonical_json(
            _normalized_config_payload(
                template_payload,
                remove_model_factory=False,
            )
        )
        normalized_effective = _canonical_json(
            _normalized_config_payload(
                effective_payload,
                remove_model_factory=False,
            )
        )
        if normalized_template != normalized_effective:
            raise ValueError(
                "effective config changes fields beyond registered run name and seed"
            )
        effective_run = effective_payload["run"]
        if effective_run["name"] != run.run_id:
            raise ValueError("effective config run name differs from registration")
        if effective_run["seed"] != run.training_seed:
            raise ValueError("effective config seed differs from registration")
        if effective_run["max_optimizer_steps"] != run.expected_optimizer_step:
            raise ValueError("effective config optimizer steps differ from registration")
        expected_factory = variant.get("model_factory")
        if not isinstance(expected_factory, str) or not expected_factory:
            raise ValueError("architecture model factory is missing")
        if effective_run["model_factory"] != expected_factory:
            raise ValueError("effective config uses the wrong architecture factory")
        computed_budget = _normalized_training_budget_sha256(effective_payload)
        if run.training_budget_sha256 != computed_budget:
            raise ValueError("registered training budget digest is not derived from config")
    if len({run.training_code_sha256 for run in campaign.runs}) != 1:
        raise ValueError("campaign runs must use one training source identity")
    if len({run.training_budget_sha256 for run in campaign.runs}) != 1:
        raise ValueError("campaign runs must use one normalized training budget")
    if len({run.expected_optimizer_step for run in campaign.runs}) != 1:
        raise ValueError("campaign runs must use one optimizer-step target")
    for seed in seeds:
        if len(
            {
                registered_runs[(variant_id, seed)].training_data_sha256
                for variant_id in expected_variants
            }
        ) != 1:
            raise ValueError("paired variants must use identical per-seed training data")

    executed_runs = {
        (run.variant_id, run.training_seed): run for run in execution.runs
    }
    if set(executed_runs) != expected_run_keys:
        raise ValueError("campaign execution is not the exact registered run matrix")
    for key, planned in registered_runs.items():
        executed = executed_runs[key]
        if executed.run_id != planned.run_id:
            raise ValueError("executed run id differs from campaign registration")
        if executed.optimizer_step != planned.expected_optimizer_step:
            raise ValueError("executed checkpoint cursor differs from campaign")

    records = tuple(observations)
    if not records or any(not isinstance(item, BenchmarkObservation) for item in records):
        raise ValueError("observations must be BenchmarkObservation values")
    by_key: dict[tuple[str, int, str], BenchmarkObservation] = {}
    for record in records:
        key = (record.variant_id, record.training_seed, record.suite_id)
        if key in by_key:
            raise ValueError(f"duplicate comparison observation: {key}")
        by_key[key] = record
    expected_keys = {
        (variant_id, seed, suite_id)
        for variant_id in expected_variants
        for seed in seeds
        for suite_id in suite_ids
    }
    if set(by_key) != expected_keys:
        raise ValueError("observations differ from the exact registered matrix")

    suites_by_id = {suite.suite_id: suite for suite in suites}
    for record in records:
        run = registered_runs[(record.variant_id, record.training_seed)]
        executed = executed_runs[(record.variant_id, record.training_seed)]
        suite = suites_by_id[record.suite_id]
        for name in (
            "variant_identity_sha256",
            "run_id",
            "canonical_config_sha256",
            "raw_config_sha256",
            "training_data_sha256",
            "training_code_sha256",
            "training_budget_sha256",
        ):
            if getattr(record, name) != getattr(run, name):
                raise ValueError(f"observation {name} differs from campaign registration")
        if record.checkpoint_sha256 != executed.checkpoint_sha256:
            raise ValueError("observation checkpoint identity differs from execution")
        if record.optimizer_step != executed.optimizer_step:
            raise ValueError("observation checkpoint cursor differs from execution")
        if record.evaluated_samples != suite.evaluated_samples:
            raise ValueError("observation sample count differs from registered suite")
        if (
            record.evaluation_sample_manifest_sha256
            != suite.evaluation_sample_manifest_sha256
        ):
            raise ValueError("observation sample identity differs from registered suite")
        if record.evaluation_code_sha256 != suite.evaluation_code_sha256:
            raise ValueError("observation evaluator identity differs from registered suite")

    raw_matrix = [
        asdict(by_key[key])
        for key in sorted(by_key, key=lambda item: (item[0], item[1], item[2]))
    ]
    raw_matrix_sha = sha256(_canonical_json(raw_matrix).encode("utf-8")).hexdigest()
    weight_total = sum(float(suite.weight) for suite in suites)

    raw_results: list[dict[str, object]] = []
    for comparison_id in comparison_ids:
        differences: list[float] = []
        for seed in seeds:
            reference_score = sum(
                float(suite.weight)
                * by_key[(reference_id, seed, suite.suite_id)].normalized_score
                for suite in suites
            ) / weight_total
            comparison_score = sum(
                float(suite.weight)
                * by_key[(comparison_id, seed, suite.suite_id)].normalized_score
                for suite in suites
            ) / weight_total
            differences.append(reference_score - comparison_score)
        suite_differences = {
            suite.suite_id: mean(
                by_key[(reference_id, seed, suite.suite_id)].normalized_score
                - by_key[(comparison_id, seed, suite.suite_id)].normalized_score
                for seed in seeds
            )
            for suite in suites
        }
        paired = tuple(differences)
        raw_results.append(
            {
                "comparison_variant_id": comparison_id,
                "paired_seed_mean_deltas": list(paired),
                "paired_mean_delta": mean(paired),
                "paired_median_delta": median(paired),
                "positive_seed_count": sum(value > 0.0 for value in paired),
                "suite_mean_deltas": suite_differences,
                "minimum_suite_mean_delta": min(suite_differences.values()),
                "raw_two_sided_exact_sign_flip_pvalue": (
                    _exact_two_sided_sign_flip_pvalue(paired)
                ),
            }
        )

    ordered = sorted(
        raw_results,
        key=lambda item: (
            float(item["raw_two_sided_exact_sign_flip_pvalue"]),
            str(item["comparison_variant_id"]),
        ),
    )
    holm: dict[str, tuple[bool, float]] = {}
    prior_rejected = True
    total_comparisons = len(ordered)
    for index, result in enumerate(ordered):
        threshold = criterion.familywise_alpha / (total_comparisons - index)
        raw_p = float(result["raw_two_sided_exact_sign_flip_pvalue"])
        rejected = prior_rejected and raw_p <= threshold
        holm[str(result["comparison_variant_id"])] = (rejected, threshold)
        prior_rejected = rejected

    comparisons: list[dict[str, object]] = []
    for result in raw_results:
        comparison_id = str(result["comparison_variant_id"])
        significance_passed, holm_threshold = holm[comparison_id]
        effect_passed = (
            float(result["paired_mean_delta"]) >= criterion.minimum_paired_mean_delta
        )
        suite_floor_passed = (
            float(result["minimum_suite_mean_delta"])
            >= criterion.minimum_per_suite_mean_delta
        )
        enriched = dict(result)
        enriched.update(
            {
                "holm_bonferroni_threshold": holm_threshold,
                "familywise_significance_passed": significance_passed,
                "mean_effect_threshold_passed": effect_passed,
                "per_suite_floor_passed": suite_floor_passed,
                "passed": significance_passed and effect_passed and suite_floor_passed,
            }
        )
        comparisons.append(enriched)

    report: dict[str, object] = {
        "schema_version": 2,
        "campaign_id": campaign.campaign_id,
        "campaign_registration_sha256": pinned_campaign,
        "campaign_execution_sha256": pinned_execution,
        "scope": campaign.scope,
        "architecture_manifest_sha256": manifest_digest,
        "fairness_regime": campaign.fairness_regime,
        "criterion": asdict(criterion),
        "criterion_sha256": criterion.sha256,
        "reference_variant_id": reference_id,
        "training_seeds": list(seeds),
        "suite_ids": list(suite_ids),
        "raw_observation_matrix_sha256": raw_matrix_sha,
        "raw_observation_count": len(raw_matrix),
        "comparisons": comparisons,
        "passed": all(bool(result["passed"]) for result in comparisons),
        "claim_boundary": (
            "A pass supports only the externally registered scope and fairness "
            "regime; it does not establish causal thoughts, real-time superiority, "
            "general game intelligence, or human-like cognition."
        ),
    }
    report["report_sha256"] = sha256(
        _canonical_json(report).encode("utf-8")
    ).hexdigest()
    return report


__all__ = [
    "BenchmarkObservation",
    "CampaignExecution",
    "CampaignRegistration",
    "ComparisonCriterion",
    "ExecutedRun",
    "RegisteredRun",
    "RegisteredSuite",
    "evaluate_multiseed_comparison",
]
