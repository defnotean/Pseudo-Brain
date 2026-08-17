from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (str(SRC), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)
TORCH_WAS_IMPORTED = "torch" in sys.modules


def _load_script(name: str, filename: str) -> object:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_script(
    "first_matched_suite_contract", "first_matched_suite_contract.py"
)
builder = _load_script(
    "build_first_matched_multiseed_campaign",
    "build_first_matched_multiseed_campaign.py",
)
TOOLING_IMPORTED_TORCH = not TORCH_WAS_IMPORTED and "torch" in sys.modules


@dataclass(frozen=True)
class _MetricResult:
    samples: int
    metrics: dict[str, float]


class FirstMatchedCampaignPreregistrationTests(unittest.TestCase):
    def _suites(self) -> tuple[dict[str, object], ...]:
        suite_set_path = ROOT / builder.SUITE_ROOT_RELATIVE / "suite-set.json"
        if suite_set_path.exists():
            return tuple(builder.load_suite_set(ROOT).suites)
        artifacts = builder.build_suite_artifacts(ROOT)
        suite_paths = sorted(
            path
            for path in artifacts
            if path.parent.name == "suites" and path.suffix == ".json"
        )
        return tuple(
            contract.strict_json_bytes(artifacts[path], name=path.as_posix())
            for path in suite_paths
        )

    @staticmethod
    def _digest(label: str) -> str:
        return sha256(label.encode("utf-8")).hexdigest()

    def _campaign_and_runtime(self) -> tuple[object, object]:
        digest = self._digest
        run = builder.RegisteredRun(
            variant_id="test.variant",
            variant_identity_sha256=digest("variant"),
            training_seed=1701,
            run_id="test-run-1701",
            template_canonical_config_sha256=digest("template-canonical"),
            template_raw_config_sha256=digest("template-raw"),
            template_config_canonical_json="{}",
            template_config_raw_toml="schema_version = 2\n",
            canonical_config_sha256=digest("effective-canonical"),
            raw_config_sha256=digest("effective-raw"),
            effective_config_canonical_json="{}",
            effective_config_raw_toml="schema_version = 2\n",
            training_data_sha256=digest("data"),
            training_code_sha256=digest("code"),
            training_budget_sha256=digest("budget"),
            expected_optimizer_step=2048,
        )
        suite = builder.RegisteredSuite(
            suite_id="test-suite",
            weight=1.0,
            evaluation_sample_manifest_sha256=digest("samples"),
            evaluation_code_sha256=digest("evaluation-code"),
            evaluated_samples=1,
        )
        campaign = builder.CampaignRegistration(
            schema_version=1,
            campaign_id="test-campaign",
            scope="binder adversarial test only",
            architecture_manifest_sha256=digest("architecture"),
            fairness_regime="parameter_matched",
            criterion_sha256=digest("criterion"),
            training_seeds=(1701,),
            suites=(suite,),
            runs=(run,),
            checkpoint_selection_rule="exact final checkpoint",
            stopping_rule="exactly 2048 optimizer steps",
        )
        runtime = builder.RuntimeRegistration(
            schema_version=1,
            campaign_sha256=campaign.sha256,
            runtime_fingerprint={"device": "cpu", "runtime_build": "test"},
        )
        return campaign, runtime

    def _binding_spec_path(
        self,
        campaign: object,
        runtime: object,
        checkpoint_path: str,
        *,
        schema_version: object = 1,
    ) -> Path:
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = directory / "binding.json"
        payload = {
            "schema_version": schema_version,
            "campaign_sha256": campaign.sha256,
            "runtime_registration_sha256": runtime.sha256,
            "runs": [
                {
                    "variant_id": "test.variant",
                    "training_seed": 1701,
                    "run_id": "test-run-1701",
                    "checkpoint_path": checkpoint_path,
                }
            ],
        }
        path.write_bytes((contract.canonical_json(payload) + "\n").encode("utf-8"))
        return path

    def test_historical_manifest_pin_is_exact_and_campaign_is_fail_closed(self) -> None:
        manifest_path = ROOT / "configs" / "baseline-architecture-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = manifest.pop("manifest_sha256")
        actual = sha256(contract.canonical_json(manifest).encode("utf-8")).hexdigest()
        self.assertEqual(actual, recorded)
        self.assertEqual(
            recorded,
            "52bba6a9b723dda51da18d2842c5eaf360453db011e7f97bbb066e4034905421",
        )
        with self.assertRaisesRegex(builder.CampaignBlockedError, "2048-step"):
            builder.build_campaign_artifacts()
        blocker = builder.campaign_blocker()
        self.assertEqual(blocker["status"], "blocked_non_launchable")
        self.assertEqual(blocker["historical_recipe_optimizer_steps"], 500)
        self.assertEqual(blocker["required_recipe_optimizer_steps"], 2048)
        for relative in builder.FORBIDDEN_WHILE_BLOCKED:
            self.assertFalse((ROOT / relative).exists(), relative.as_posix())

    def test_checked_in_static_artifacts_are_exact_generator_output(self) -> None:
        expected = builder.build_static_artifacts(ROOT)
        present = [relative for relative in expected if (ROOT / relative).exists()]
        if not present:
            return
        self.assertEqual(set(present), set(expected), "partial static artifact set")
        builder.check_static_artifacts(ROOT)
        for relative, content in expected.items():
            self.assertEqual((ROOT / relative).read_bytes(), content)

    def test_three_real_suites_are_disjoint_and_executable_without_torch(self) -> None:
        suites = self._suites()
        self.assertEqual(len(suites), 3)
        self.assertFalse(TOOLING_IMPORTED_TORCH)
        ranges: list[set[int]] = []
        hazard_counts: list[int] = []
        seed_offsets: list[int] = []
        for suite in suites:
            config = contract.dataset_config_from_suite(suite)
            self.assertEqual(config.split.value, "test")
            self.assertEqual(config.sequence_count, 256)
            self.assertEqual(config.sequence_length, 8)
            self.assertEqual(suite["evaluated_samples"], 1_536)
            hazard_counts.append(config.hazard_count)
            seed_offsets.append(config.seed_offset)
            ranges.append(set(range(config.seed_offset, config.seed_offset + 256)))
            first_batch = next(contract.iter_suite_batches(suite, batch_size=2))
            self.assertEqual(first_batch.split, "test")
            self.assertEqual(first_batch.burn_in_steps, 2)
            self.assertEqual(first_batch.sample_count, 12)
            self.assertEqual(len(first_batch.sequences), 2)
            self.assertEqual(len(first_batch.sequences[0].transitions), 8)
        self.assertEqual(hazard_counts, [1, 3, 5])
        self.assertEqual(seed_offsets, [2_097_152, 2_097_408, 2_097_664])
        for left_index, left in enumerate(ranges):
            for right in ranges[left_index + 1 :]:
                self.assertTrue(left.isdisjoint(right))
        suite_set_path = ROOT / builder.SUITE_ROOT_RELATIVE / "suite-set.json"
        if suite_set_path.exists():
            loaded = builder.load_suite_set(ROOT)
            self.assertEqual(len(loaded.registrations), 3)
            self.assertEqual(
                [suite.suite_id for suite in loaded.registrations],
                [suite["suite_id"] for suite in suites],
            )

    def test_suite_metric_is_the_existing_final_exit_exact_set_score(self) -> None:
        for suite in self._suites():
            metric = suite["metric"]
            self.assertEqual(metric["metric_id"], "movement_exact_match")
            self.assertEqual(metric["model_exit"], "final_anytime_exit")
            self.assertEqual(metric["prediction_rule"], "button_logit > 0.0")
            self.assertEqual(
                metric["movement_keys"],
                [
                    {"name": "w", "hid_usage_id": 26},
                    {"name": "a", "hid_usage_id": 4},
                    {"name": "s", "hid_usage_id": 22},
                    {"name": "d", "hid_usage_id": 7},
                ],
            )
        score = contract.aggregate_movement_exact_match(
            (
                _MetricResult(2, {"movement_exact_match": 0.5}),
                _MetricResult(6, {"movement_exact_match": 1.0}),
            )
        )
        self.assertEqual(score, 0.875)
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            contract.aggregate_movement_exact_match(
                (_MetricResult(1, {"movement_exact_match": 1.1}),)
            )

    def test_effective_config_derivation_changes_only_name_and_seed(self) -> None:
        paths = (
            ROOT / "configs/training/dgx-stagea-continuation-gate-b.toml",
            ROOT / "configs/training/baseline-stagea-no-communication.toml",
            ROOT / "configs/training/baseline-stagea-monolithic-parameter-matched.toml",
        )
        budget_hashes: set[str] = set()
        for index, path in enumerate(paths):
            template_bytes = path.read_bytes()
            template = tomllib.loads(template_bytes.decode("utf-8"))
            effective = builder.derive_effective_config(
                template_bytes,
                run_id=f"mechanical-test-{index}",
                training_seed=1708,
            )
            parsed = tomllib.loads(effective.raw_toml)
            self.assertEqual(parsed["run"]["name"], f"mechanical-test-{index}")
            self.assertEqual(parsed["run"]["seed"], 1708)
            template["run"]["name"] = f"mechanical-test-{index}"
            template["run"]["seed"] = 1708
            self.assertEqual(parsed, template)
            self.assertEqual(
                sha256(effective.raw_toml.encode("utf-8")).hexdigest(),
                effective.raw_sha256,
            )
            self.assertEqual(
                sha256(effective.canonical_json.encode("utf-8")).hexdigest(),
                effective.canonical_sha256,
            )
            budget_hashes.add(effective.training_budget_sha256)
        self.assertEqual(len(budget_hashes), 1)

        staged_template = paths[0].read_bytes().replace(
            b"schema_version = 2", b"schema_version = 3", 1
        )
        staged = builder.derive_effective_config(
            staged_template,
            run_id="future-staged-mechanical-test",
            training_seed=1702,
        )
        staged_payload = tomllib.loads(staged.raw_toml)
        self.assertEqual(staged_payload["schema_version"], 3)
        self.assertEqual(staged_payload["run"]["name"], "future-staged-mechanical-test")
        self.assertEqual(staged_payload["run"]["seed"], 1702)

    def test_suite_loader_rejects_duplicate_keys_and_noncanonical_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            contract.strict_json_bytes(b'{"x":1,"x":2}', name="duplicate")
        payload = self._suites()[0]
        with self.assertRaisesRegex(ValueError, "canonical JSON"):
            contract.load_suite_manifest(
                _TemporarySuiteFile(self, payload).path,
                brain_root=ROOT,
            )

    def test_suite_validator_rejects_bool_and_float_type_substitution(self) -> None:
        suite = self._suites()[0]
        mutations = []
        top_schema = copy.deepcopy(suite)
        top_schema["schema_version"] = True
        mutations.append(top_schema)
        integer_weight = copy.deepcopy(suite)
        integer_weight["weight"] = 1
        mutations.append(integer_weight)
        float_samples = copy.deepcopy(suite)
        float_samples["evaluated_samples"] = 1536.0
        mutations.append(float_samples)
        dataset_schema = copy.deepcopy(suite)
        dataset_schema["sample_manifest"]["dataset_manifest"]["schema_version"] = True
        mutations.append(dataset_schema)
        bool_range = copy.deepcopy(suite)
        bool_range["metric"]["normalized_range"] = [False, True]
        mutations.append(bool_range)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    contract.validate_suite_manifest(mutation)

    def test_binder_rejects_noninteger_schema_and_unsafe_checkpoint_paths(self) -> None:
        campaign, runtime = self._campaign_and_runtime()
        checkpoint_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        cases = (
            (True, "checkpoint.pt", "schema_version"),
            (1.0, "checkpoint.pt", "schema_version"),
            (1, "../outside.pt", "beneath checkpoint_root"),
            (1, str((checkpoint_root / "absolute.pt").resolve()), "POSIX relative"),
        )
        for schema_version, checkpoint_path, message in cases:
            spec = self._binding_spec_path(
                campaign,
                runtime,
                checkpoint_path,
                schema_version=schema_version,
            )
            with self.subTest(schema_version=schema_version, path=checkpoint_path):
                with self.assertRaisesRegex(ValueError, message):
                    builder.bind_campaign_execution(
                        campaign,
                        spec,
                        expected_campaign_sha256=campaign.sha256,
                        checkpoint_root=checkpoint_root,
                        runtime_registration=runtime,
                        expected_runtime_registration_sha256=runtime.sha256,
                    )

    def test_checkpoint_path_rejects_reparse_and_binder_detects_byte_change(self) -> None:
        campaign, runtime = self._campaign_and_runtime()
        run = campaign.runs[0]
        checkpoint_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        checkpoint = checkpoint_root / "checkpoint.pt"
        checkpoint.write_bytes(b"not loaded in this adversarial test")
        original_is_reparse = builder._is_reparse

        with mock.patch.object(builder, "_is_reparse", return_value=True):
            with self.assertRaisesRegex(ValueError, "checkpoint_root"):
                builder._resolve_checkpoint_path(checkpoint_root, "checkpoint.pt")

        def marked_reparse(path: Path) -> bool:
            return path == checkpoint or original_is_reparse(path)

        with mock.patch.object(builder, "_is_reparse", side_effect=marked_reparse):
            with self.assertRaisesRegex(ValueError, "symlink or reparse"):
                builder._resolve_checkpoint_path(checkpoint_root, "checkpoint.pt")

        first = self._digest("checkpoint-before")
        second = self._digest("checkpoint-after")
        loaded = SimpleNamespace(
            checkpoint_sha256=first,
            cursor=SimpleNamespace(optimizer_step=2048),
        )
        with mock.patch(
            "irene_brain.training.checkpoint.file_sha256",
            side_effect=(first, second),
        ), mock.patch(
            "irene_brain.training.checkpoint.load_checkpoint",
            return_value=loaded,
        ):
            with self.assertRaisesRegex(ValueError, "changed while binding"):
                builder._load_verified_checkpoint(
                    checkpoint,
                    run,
                    runtime.runtime_fingerprint,
                )

    def test_runtime_registration_is_exact_typed_and_externally_pinned(self) -> None:
        campaign, runtime = self._campaign_and_runtime()
        with self.assertRaisesRegex(ValueError, "schema_version"):
            builder.RuntimeRegistration(
                schema_version=True,
                campaign_sha256=campaign.sha256,
                runtime_fingerprint={"device": "cpu"},
            )
        with self.assertRaisesRegex(ValueError, "strings, integers, or booleans"):
            builder.RuntimeRegistration(
                schema_version=1,
                campaign_sha256=campaign.sha256,
                runtime_fingerprint={"threads": 1.0},
            )
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = directory / "runtime.json"
        path.write_bytes((runtime.canonical_json + "\n").encode("utf-8"))
        loaded = builder.load_runtime_registration(
            path,
            expected_runtime_registration_sha256=runtime.sha256,
        )
        self.assertEqual(loaded.sha256, runtime.sha256)
        with self.assertRaisesRegex(ValueError, "externally pinned"):
            builder.load_runtime_registration(
                path,
                expected_runtime_registration_sha256=self._digest("wrong-runtime"),
            )


class _TemporarySuiteFile:
    def __init__(self, case: unittest.TestCase, payload: object) -> None:
        self._directory = case.enterContext(tempfile.TemporaryDirectory())
        self.path = Path(self._directory) / "suite.json"
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
