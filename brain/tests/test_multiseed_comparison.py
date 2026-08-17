from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import unittest

from irene_brain.evaluation.multiseed_comparison import (
    BenchmarkObservation,
    CampaignExecution,
    CampaignRegistration,
    ComparisonCriterion,
    ExecutedRun,
    RegisteredRun,
    RegisteredSuite,
    _exact_two_sided_sign_flip_pvalue,
    evaluate_multiseed_comparison,
)


REFERENCE = "reference"
BASELINE_A = "baseline-a"
BASELINE_B = "baseline-b"
VARIANTS = (REFERENCE, BASELINE_A, BASELINE_B)
SEEDS = tuple(range(1, 9))


def _h(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _factory(variant_id: str) -> str:
    return f"test_models:{variant_id.replace('-', '_')}"


def _config_artifacts(
    variant_id: str,
    *,
    name: str,
    seed: int,
    learning_rate: float = 0.0001,
    determinism_enabled: bool | int = True,
) -> tuple[dict[str, object], str, str, str, str]:
    payload: dict[str, object] = {
        "schema_version": 2,
        "run": {
            "name": name,
            "seed": seed,
            "model_factory": _factory(variant_id),
            "max_optimizer_steps": 500,
        },
        "dataset": {"kind": "moving_shapes", "train_sequences": 4096},
        "optimization": {
            "learning_rate": learning_rate,
            "warmup_steps": 20,
        },
        "objective": {"action_weight": 1.0, "value_weight": 0.1},
        "precision": {"device": "cuda", "mode": "bfloat16"},
        "determinism": {"enabled": determinism_enabled, "num_workers": 0},
        "logging": {"validation_batches": 16, "evaluate_every_steps": 100},
        "resources": {"allow_gpu": True, "allow_capture": False},
    }
    raw = (
        "schema_version = 2\n"
        "\n[run]\n"
        f"name = {json.dumps(name)}\n"
        f"seed = {seed}\n"
        f"model_factory = {json.dumps(_factory(variant_id))}\n"
        "max_optimizer_steps = 500\n"
        "\n[dataset]\n"
        'kind = "moving_shapes"\n'
        "train_sequences = 4096\n"
        "\n[optimization]\n"
        f"learning_rate = {learning_rate}\n"
        "warmup_steps = 20\n"
        "\n[objective]\n"
        "action_weight = 1.0\n"
        "value_weight = 0.1\n"
        "\n[precision]\n"
        'device = "cuda"\n'
        'mode = "bfloat16"\n'
        "\n[determinism]\n"
        f"enabled = {json.dumps(determinism_enabled)}\n"
        "num_workers = 0\n"
        "\n[logging]\n"
        "validation_batches = 16\n"
        "evaluate_every_steps = 100\n"
        "\n[resources]\n"
        "allow_gpu = true\n"
        "allow_capture = false\n"
    )
    canonical = _canonical(payload)
    return (
        payload,
        canonical,
        raw,
        sha256(canonical.encode("utf-8")).hexdigest(),
        sha256(raw.encode("utf-8")).hexdigest(),
    )


def _budget_sha(payload: dict[str, object]) -> str:
    normalized = json.loads(_canonical(payload))
    run = normalized["run"]
    del run["name"]
    del run["seed"]
    del run["model_factory"]
    return sha256(_canonical(normalized).encode("utf-8")).hexdigest()


def _manifest(*, verified: bool = True) -> dict[str, object]:
    parameter_counts = {
        REFERENCE: 10_000,
        BASELINE_A: 9_950,
        BASELINE_B: 10_050,
    }
    variants = []
    for variant_id in VARIANTS:
        _payload, canonical, raw, canonical_sha, raw_sha = _config_artifacts(
            variant_id,
            name=f"template-{variant_id}",
            seed=1701,
        )
        variants.append(
            {
                "variant_id": variant_id,
                "variant_identity_sha256": _h(f"identity:{variant_id}"),
                "model_factory": _factory(variant_id),
                "allocated_parameters": {
                    "trainable": parameter_counts[variant_id]
                },
                "training_recipe": {
                    "canonical_config_sha256": canonical_sha,
                    "raw_sha256": raw_sha,
                },
                "_test_template_canonical_json": canonical,
                "_test_template_raw_toml": raw,
            }
        )
    value: dict[str, object] = {
        "schema_version": 1,
        "reference_variant_id": REFERENCE,
        "variants": variants,
        "fairness_regimes": {
            "parameter_matched": {
                "comparison_variant_ids": [BASELINE_A, BASELINE_B],
                "fixed_controls": [
                    "dataset_bytes_and_order",
                    "optimizer_and_schedule",
                    "objective",
                    "optimizer_steps",
                    "precision",
                    "seed_set",
                ],
                "maximum_allocated_trainable_parameter_delta_fraction": 0.01,
                "verified": verified,
            }
        },
        "claim_boundary": "test manifest",
    }
    value["manifest_sha256"] = sha256(
        _canonical(value).encode("utf-8")
    ).hexdigest()
    return value


def _criterion() -> ComparisonCriterion:
    return ComparisonCriterion(
        minimum_training_seeds=8,
        minimum_suites=3,
        minimum_paired_mean_delta=0.02,
        minimum_per_suite_mean_delta=-0.01,
        familywise_alpha=0.05,
    )


def _campaign(
    manifest: dict[str, object], criterion: ComparisonCriterion
) -> CampaignRegistration:
    variants = {
        entry["variant_id"]: entry for entry in manifest["variants"]  # type: ignore[index]
    }
    suites = tuple(
        RegisteredSuite(
            suite_id=suite_id,
            weight=weight,
            evaluation_sample_manifest_sha256=_h(f"samples:{suite_id}"),
            evaluation_code_sha256=_h(f"evaluator:{suite_id}"),
            evaluated_samples=512,
        )
        for suite_id, weight in (
            ("suite-a", 1.0),
            ("suite-b", 2.0),
            ("suite-c", 1.0),
        )
    )
    runs = []
    for variant_id in VARIANTS:
        entry = variants[variant_id]
        recipe = entry["training_recipe"]
        for seed in SEEDS:
            effective_payload, effective_json, effective_raw, config_sha, raw_sha = (
                _config_artifacts(
                    variant_id,
                    name=f"run-{variant_id}-{seed}",
                    seed=seed,
                )
            )
            runs.append(
                RegisteredRun(
                    variant_id=variant_id,
                    variant_identity_sha256=entry[
                        "variant_identity_sha256"
                    ],
                    training_seed=seed,
                    run_id=f"run-{variant_id}-{seed}",
                    template_canonical_config_sha256=recipe[
                        "canonical_config_sha256"
                    ],
                    template_raw_config_sha256=recipe["raw_sha256"],
                    template_config_canonical_json=entry[
                        "_test_template_canonical_json"
                    ],
                    template_config_raw_toml=entry[
                        "_test_template_raw_toml"
                    ],
                    canonical_config_sha256=config_sha,
                    raw_config_sha256=raw_sha,
                    effective_config_canonical_json=effective_json,
                    effective_config_raw_toml=effective_raw,
                    training_data_sha256=_h(f"train-data:{seed}"),
                    training_code_sha256=_h("training-code"),
                    training_budget_sha256=_budget_sha(effective_payload),
                    expected_optimizer_step=500,
                )
            )
    return CampaignRegistration(
        schema_version=1,
        campaign_id="campaign-v1",
        scope="synthetic-test-only",
        architecture_manifest_sha256=manifest["manifest_sha256"],
        fairness_regime="parameter_matched",
        criterion_sha256=criterion.sha256,
        training_seeds=SEEDS,
        suites=suites,
        runs=tuple(runs),
        checkpoint_selection_rule="exact optimizer step 500; no best-of selection",
        stopping_rule="run exactly the registered matrix once",
    )


def _execution(campaign: CampaignRegistration) -> CampaignExecution:
    return CampaignExecution(
        schema_version=1,
        campaign_sha256=campaign.sha256,
        runs=tuple(
            ExecutedRun(
                variant_id=run.variant_id,
                training_seed=run.training_seed,
                run_id=run.run_id,
                checkpoint_sha256=_h(
                    f"checkpoint:{run.variant_id}:{run.training_seed}"
                ),
                optimizer_step=run.expected_optimizer_step,
            )
            for run in campaign.runs
        ),
    )


def _matrix(
    campaign: CampaignRegistration,
    execution: CampaignExecution,
    *,
    signed_b_deltas: tuple[float, ...] | None = None,
) -> list[BenchmarkObservation]:
    runs = {(run.variant_id, run.training_seed): run for run in campaign.runs}
    executed = {
        (run.variant_id, run.training_seed): run for run in execution.runs
    }
    records: list[BenchmarkObservation] = []
    for seed_index, seed in enumerate(campaign.training_seeds):
        for suite_index, suite in enumerate(campaign.suites):
            reference_score = 0.80 + suite_index * 0.01
            b_delta = (
                0.04 if signed_b_deltas is None else signed_b_deltas[seed_index]
            )
            scores = {
                REFERENCE: reference_score,
                BASELINE_A: reference_score - 0.05,
                BASELINE_B: reference_score - b_delta,
            }
            for variant_id, score in scores.items():
                run = runs[(variant_id, seed)]
                completed = executed[(variant_id, seed)]
                records.append(
                    BenchmarkObservation(
                        variant_id=variant_id,
                        variant_identity_sha256=run.variant_identity_sha256,
                        training_seed=seed,
                        suite_id=suite.suite_id,
                        normalized_score=score,
                        evaluated_samples=suite.evaluated_samples,
                        evaluation_sample_manifest_sha256=(
                            suite.evaluation_sample_manifest_sha256
                        ),
                        evaluation_code_sha256=suite.evaluation_code_sha256,
                        run_id=run.run_id,
                        checkpoint_sha256=completed.checkpoint_sha256,
                        optimizer_step=completed.optimizer_step,
                        canonical_config_sha256=run.canonical_config_sha256,
                        raw_config_sha256=run.raw_config_sha256,
                        training_data_sha256=run.training_data_sha256,
                        training_code_sha256=run.training_code_sha256,
                        training_budget_sha256=run.training_budget_sha256,
                    )
                )
    return records


def _evaluate(
    records: list[BenchmarkObservation],
    manifest: dict[str, object],
    campaign: CampaignRegistration,
    execution: CampaignExecution,
    criterion: ComparisonCriterion,
) -> dict[str, object]:
    return evaluate_multiseed_comparison(
        records,
        architecture_manifest=manifest,
        expected_architecture_manifest_sha256=manifest["manifest_sha256"],
        campaign=campaign,
        expected_campaign_sha256=campaign.sha256,
        execution=execution,
        expected_execution_sha256=execution.sha256,
        criterion=criterion,
    )


class MultiSeedComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest()
        self.criterion = _criterion()
        self.campaign = _campaign(self.manifest, self.criterion)
        self.execution = _execution(self.campaign)
        self.records = _matrix(self.campaign, self.execution)

    def test_complete_strong_registered_matrix_passes(self) -> None:
        report = _evaluate(
            self.records,
            self.manifest,
            self.campaign,
            self.execution,
            self.criterion,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["training_seeds"], list(SEEDS))
        self.assertEqual(report["raw_observation_count"], 72)
        self.assertRegex(
            report["raw_observation_matrix_sha256"], r"^[a-f0-9]{64}$"
        )
        self.assertRegex(report["report_sha256"], r"^[a-f0-9]{64}$")
        for comparison in report["comparisons"]:
            self.assertEqual(
                comparison["raw_two_sided_exact_sign_flip_pvalue"], 2 / 256
            )
            self.assertTrue(comparison["familywise_significance_passed"])

    def test_altered_and_rehashed_manifest_is_rejected_by_external_pin(self) -> None:
        original_digest = self.manifest["manifest_sha256"]
        altered = json.loads(json.dumps(self.manifest))
        altered["claim_boundary"] = "silently changed"
        payload = dict(altered)
        del payload["manifest_sha256"]
        altered["manifest_sha256"] = sha256(
            _canonical(payload).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "externally pinned"):
            evaluate_multiseed_comparison(
                self.records,
                architecture_manifest=altered,
                expected_architecture_manifest_sha256=original_digest,
                campaign=self.campaign,
                expected_campaign_sha256=self.campaign.sha256,
                execution=self.execution,
                expected_execution_sha256=self.execution.sha256,
                criterion=self.criterion,
            )

    def test_altered_campaign_or_execution_is_rejected_by_external_pin(self) -> None:
        altered = replace(self.campaign, scope="post-hoc scope")
        with self.assertRaisesRegex(ValueError, "registration.*externally pinned"):
            evaluate_multiseed_comparison(
                self.records,
                architecture_manifest=self.manifest,
                expected_architecture_manifest_sha256=self.manifest[
                    "manifest_sha256"
                ],
                campaign=altered,
                expected_campaign_sha256=self.campaign.sha256,
                execution=self.execution,
                expected_execution_sha256=self.execution.sha256,
                criterion=self.criterion,
            )
        altered_execution = replace(
            self.execution,
            runs=self.execution.runs[:-1]
            + (
                replace(
                    self.execution.runs[-1],
                    checkpoint_sha256=_h("different-checkpoint"),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "execution.*externally pinned"):
            evaluate_multiseed_comparison(
                self.records,
                architecture_manifest=self.manifest,
                expected_architecture_manifest_sha256=self.manifest[
                    "manifest_sha256"
                ],
                campaign=self.campaign,
                expected_campaign_sha256=self.campaign.sha256,
                execution=altered_execution,
                expected_execution_sha256=self.execution.sha256,
                criterion=self.criterion,
            )

    def test_rejects_missing_or_extra_observation_cells(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact registered matrix"):
            _evaluate(
                self.records[:-1],
                self.manifest,
                self.campaign,
                self.execution,
                self.criterion,
            )
        extra = list(self.records)
        extra.append(replace(self.records[-1], training_seed=999))
        with self.assertRaisesRegex(ValueError, "exact registered matrix"):
            _evaluate(
                extra,
                self.manifest,
                self.campaign,
                self.execution,
                self.criterion,
            )

    def test_rejects_sample_checkpoint_and_code_identity_mismatches(self) -> None:
        cases = (
            (
                replace(
                    self.records[0],
                    evaluation_sample_manifest_sha256=_h("different-samples"),
                ),
                "sample identity",
            ),
            (
                replace(
                    self.records[0], checkpoint_sha256=_h("wrong-checkpoint")
                ),
                "checkpoint identity",
            ),
            (
                replace(
                    self.records[0], training_code_sha256=_h("wrong-code")
                ),
                "training_code_sha256",
            ),
        )
        for changed, message in cases:
            records = list(self.records)
            records[0] = changed
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _evaluate(
                        records,
                        self.manifest,
                        self.campaign,
                        self.execution,
                        self.criterion,
                    )

    def test_rejects_posthoc_effective_optimizer_change_with_fresh_hashes(self) -> None:
        original = self.campaign.runs[0]
        payload, canonical, raw, canonical_sha, raw_sha = _config_artifacts(
            original.variant_id,
            name=original.run_id,
            seed=original.training_seed,
            learning_rate=0.002,
        )
        changed_run = replace(
            original,
            canonical_config_sha256=canonical_sha,
            raw_config_sha256=raw_sha,
            effective_config_canonical_json=canonical,
            effective_config_raw_toml=raw,
            training_budget_sha256=_budget_sha(payload),
        )
        changed_campaign = replace(
            self.campaign,
            runs=(changed_run,) + self.campaign.runs[1:],
        )
        changed_execution = replace(
            self.execution,
            campaign_sha256=changed_campaign.sha256,
        )
        with self.assertRaisesRegex(ValueError, "beyond registered"):
            evaluate_multiseed_comparison(
                self.records,
                architecture_manifest=self.manifest,
                expected_architecture_manifest_sha256=self.manifest[
                    "manifest_sha256"
                ],
                campaign=changed_campaign,
                expected_campaign_sha256=changed_campaign.sha256,
                execution=changed_execution,
                expected_execution_sha256=changed_execution.sha256,
                criterion=self.criterion,
            )

    def test_config_derivation_is_type_sensitive_for_bool_and_int(self) -> None:
        original = self.campaign.runs[0]
        payload, canonical, raw, canonical_sha, raw_sha = _config_artifacts(
            original.variant_id,
            name=original.run_id,
            seed=original.training_seed,
            determinism_enabled=1,
        )
        changed_run = replace(
            original,
            canonical_config_sha256=canonical_sha,
            raw_config_sha256=raw_sha,
            effective_config_canonical_json=canonical,
            effective_config_raw_toml=raw,
            training_budget_sha256=_budget_sha(payload),
        )
        changed_campaign = replace(
            self.campaign,
            runs=(changed_run,) + self.campaign.runs[1:],
        )
        changed_execution = replace(
            self.execution,
            campaign_sha256=changed_campaign.sha256,
        )
        with self.assertRaisesRegex(ValueError, "beyond registered"):
            evaluate_multiseed_comparison(
                self.records,
                architecture_manifest=self.manifest,
                expected_architecture_manifest_sha256=self.manifest[
                    "manifest_sha256"
                ],
                campaign=changed_campaign,
                expected_campaign_sha256=changed_campaign.sha256,
                execution=changed_execution,
                expected_execution_sha256=changed_execution.sha256,
                criterion=self.criterion,
            )

    def test_schema_versions_are_strict_integers(self) -> None:
        with self.assertRaisesRegex(ValueError, "campaign registration schema"):
            replace(self.campaign, schema_version=1.0)
        with self.assertRaisesRegex(ValueError, "campaign execution schema"):
            replace(self.execution, schema_version=1.0)

    def test_manifest_parameter_evidence_is_verified(self) -> None:
        altered = json.loads(json.dumps(self.manifest))
        altered["variants"][1]["allocated_parameters"]["trainable"] = 8_000
        payload = dict(altered)
        del payload["manifest_sha256"]
        altered["manifest_sha256"] = sha256(
            _canonical(payload).encode("utf-8")
        ).hexdigest()
        altered_campaign = replace(
            self.campaign,
            architecture_manifest_sha256=altered["manifest_sha256"],
        )
        altered_execution = replace(
            self.execution,
            campaign_sha256=altered_campaign.sha256,
        )
        with self.assertRaisesRegex(ValueError, "parameter tolerance"):
            evaluate_multiseed_comparison(
                self.records,
                architecture_manifest=altered,
                expected_architecture_manifest_sha256=altered[
                    "manifest_sha256"
                ],
                campaign=altered_campaign,
                expected_campaign_sha256=altered_campaign.sha256,
                execution=altered_execution,
                expected_execution_sha256=altered_execution.sha256,
                criterion=self.criterion,
            )

    def test_unverified_fairness_regime_is_rejected(self) -> None:
        manifest = _manifest(verified=False)
        campaign = _campaign(manifest, self.criterion)
        execution = _execution(campaign)
        with self.assertRaisesRegex(ValueError, "not verified"):
            _evaluate(
                _matrix(campaign, execution),
                manifest,
                campaign,
                execution,
                self.criterion,
            )

    def test_sign_flip_handles_mixed_signs_zeros_and_holm_stopping(self) -> None:
        self.assertEqual(
            _exact_two_sided_sign_flip_pvalue((1.0, -1.0, 0.0)), 1.0
        )
        self.assertEqual(
            _exact_two_sided_sign_flip_pvalue((1.0, 1.0, 0.0)), 0.5
        )
        records = _matrix(
            self.campaign,
            self.execution,
            signed_b_deltas=(
                0.04,
                0.04,
                0.04,
                0.04,
                0.04,
                0.04,
                -0.04,
                -0.04,
            ),
        )
        report = _evaluate(
            records,
            self.manifest,
            self.campaign,
            self.execution,
            self.criterion,
        )
        by_id = {
            result["comparison_variant_id"]: result
            for result in report["comparisons"]
        }
        self.assertTrue(
            by_id[BASELINE_A]["familywise_significance_passed"]
        )
        self.assertFalse(
            by_id[BASELINE_B]["familywise_significance_passed"]
        )
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
