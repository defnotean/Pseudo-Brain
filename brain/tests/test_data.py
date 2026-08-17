from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from irene_brain.data.jsonl import (
    JsonlRecordError,
    append_step,
    canonical_json,
    read_step_observations,
    read_steps,
)
from irene_brain.data.records import (
    MODEL_OBSERVATION_FIELDS,
    PRIVILEGED_STEP_FIELDS,
    BranchRecord,
    LifetimeRecord,
    StepRecord,
)
from irene_brain.data.splits import SplitIntegrityError, require_split_integrity
from irene_brain.runtime.policy import ResourceDenied, ResourcePolicy
from irene_brain.types import GenericControl, HidKey, ModelObservation, RgbFrame

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def make_lifetime() -> LifetimeRecord:
    return LifetimeRecord(
        lifetime_id="life-1",
        world_lineage_id="world-1",
        environment_family="moving_shapes",
        environment_build_sha256=HASH_A,
        container_digest="none:stdlib",
        generator_config={"hazards": 3},
        seed_vector={"world": 7},
        control_mapping_hash=HASH_B,
        qpc_frequency_hz=10_000_000,
        graphics_config={"width": 16, "height": 16},
        source_type="self_play",
        source_session_id="collector-session-1",
        license_record_id="original-project-content",
        split="train",
    )


def make_step(
    *,
    step: int,
    requested: GenericControl,
    applied: GenericControl,
    poison: str = "",
) -> StepRecord:
    base = 100 + step * 20
    return StepRecord(
        lifetime_id=f"life-1{poison}",
        episode_id=f"episode-1{poison}",
        step=step,
        environment_tick=step,
        frame_id=step,
        capture_time_qpc=base,
        observation_ready_qpc=base + 5,
        action_requested_qpc=base + 10,
        action_applied_qpc=base + 12,
        rgb_ref=f"rgb:{step}",
        audio_ref=None,
        requested_control=requested,
        applied_control=applied,
        reward_raw=17.0,
        events=(f"event{poison}",),
        terminated=False,
        truncated=False,
        dropped_frames=0,
        snapshot_hash=HASH_C,
        privileged_state_ref=f"privileged:{poison}",
        metadata={"future": f"future{poison}", "game_id": f"game{poison}"},
    )


class RecordTests(unittest.TestCase):
    def test_lifetime_round_trip_is_canonical_and_immutable(self) -> None:
        lifetime = make_lifetime()
        restored = LifetimeRecord.from_dict(lifetime.to_dict())

        self.assertEqual(restored, lifetime)
        self.assertEqual(
            json.loads(canonical_json(restored)),
            restored.to_dict(),
        )
        with self.assertRaises(TypeError):
            restored.generator_config["hazards"] = 99  # type: ignore[index]

    def test_invalid_hash_or_unknown_field_is_rejected(self) -> None:
        lifetime = make_lifetime().to_dict()
        lifetime["environment_build_sha256"] = "not-a-hash"
        with self.assertRaises(ValueError):
            LifetimeRecord.from_dict(lifetime)

        lifetime = make_lifetime().to_dict()
        lifetime["game_id"] = "forbidden"
        with self.assertRaises(ValueError):
            LifetimeRecord.from_dict(lifetime)

    def test_model_materialization_is_allowlisted_resolved_and_poison_free(self) -> None:
        poison = "__PRIVILEGED_POISON__"
        frame = RgbFrame(width=1, height=1, pixels=b"\x01\x02\x03")
        record = replace(
            make_step(
                step=0,
                requested=GenericControl(keys_down=(int(HidKey.W),)),
                applied=GenericControl(keys_down=(int(HidKey.A),)),
                poison=poison,
            ),
            rgb_ref=f"rgb:{poison}",
            audio_ref=f"audio:{poison}",
        )
        observation = record.materialize_model_observation(
            rgb_loader=lambda ref: frame,
            audio_loader=lambda ref: b"resolved-audio",
            previous_control=GenericControl.neutral(),
            lifetime_origin_qpc=90,
            qpc_frequency_hz=10,
        )
        encoded = repr(observation)

        self.assertIsInstance(observation, ModelObservation)
        self.assertEqual(set(observation.__dataclass_fields__), MODEL_OBSERVATION_FIELDS)
        self.assertTrue(PRIVILEGED_STEP_FIELDS.isdisjoint(MODEL_OBSERVATION_FIELDS))
        self.assertNotIn(poison, encoded)
        self.assertIs(observation.rgb, frame)
        self.assertEqual(observation.audio_pcm_s16le, b"resolved-audio")
        self.assertEqual(observation.elapsed_ns, 1_000_000_000)
        self.assertEqual(observation.observation_age_ns, 500_000_000)
        self.assertEqual(record.sensor_storage_locator.rgb_ref, record.rgb_ref)

    def test_jsonl_requires_explicit_write_capability_and_carries_applied_control(self) -> None:
        first = make_step(
            step=0,
            requested=GenericControl(keys_down=(int(HidKey.W),)),
            applied=GenericControl(keys_down=(int(HidKey.A),)),
        )
        second = make_step(
            step=1,
            requested=GenericControl(keys_down=(int(HidKey.S),)),
            applied=GenericControl(keys_down=(int(HidKey.D),)),
        )
        lifetime = replace(make_lifetime(), qpc_frequency_hz=10)
        frame = RgbFrame(width=1, height=1, pixels=b"\x00" * 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steps.jsonl"
            with self.assertRaises(ResourceDenied):
                append_step(path, first, policy=ResourcePolicy.play_safe())
            self.assertFalse(path.exists())

            writable = ResourcePolicy(allow_artifact_write=True)
            append_step(path, first, policy=writable)
            append_step(path, second, policy=writable)
            self.assertEqual(list(read_steps(path)), [first, second])
            observations = list(
                read_step_observations(
                    path,
                    lifetime_manifests={lifetime.lifetime_id: lifetime},
                    rgb_loader=lambda ref: frame,
                    audio_loader=lambda ref: b"",
                )
            )

        self.assertEqual(
            observations[0].previous_control,
            GenericControl.neutral(),
        )
        self.assertEqual(
            observations[1].previous_control,
            first.applied_control,
        )
        self.assertNotEqual(
            observations[1].previous_control,
            first.requested_control,
        )

    def test_observation_reader_fails_closed_on_incomplete_or_invalid_streams(self) -> None:
        writable = ResourcePolicy(allow_artifact_write=True)
        lifetime = make_lifetime()
        frame = RgbFrame(width=1, height=1, pixels=b"\x00" * 3)

        def read(path: Path, manifests: dict[str, LifetimeRecord]) -> list[ModelObservation]:
            return list(
                read_step_observations(
                    path,
                    lifetime_manifests=manifests,
                    rgb_loader=lambda ref: frame,
                    audio_loader=lambda ref: b"",
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = make_step(
                step=0,
                requested=GenericControl.neutral(),
                applied=GenericControl.neutral(),
            )

            unknown = root / "unknown.jsonl"
            append_step(unknown, first, policy=writable)
            with self.assertRaises(JsonlRecordError):
                read(unknown, {})

            partial = root / "partial.jsonl"
            append_step(partial, replace(first, step=1), policy=writable)
            with self.assertRaises(JsonlRecordError):
                read(partial, {lifetime.lifetime_id: lifetime})

            nonmonotonic = root / "nonmonotonic.jsonl"
            append_step(nonmonotonic, first, policy=writable)
            append_step(
                nonmonotonic,
                replace(make_step(
                    step=1,
                    requested=GenericControl.neutral(),
                    applied=GenericControl.neutral(),
                ), frame_id=0),
                policy=writable,
            )
            with self.assertRaises(JsonlRecordError):
                read(nonmonotonic, {lifetime.lifetime_id: lifetime})

            after_terminal = root / "after-terminal.jsonl"
            append_step(after_terminal, replace(first, terminated=True), policy=writable)
            append_step(
                after_terminal,
                make_step(
                    step=1,
                    requested=GenericControl.neutral(),
                    applied=GenericControl.neutral(),
                ),
                policy=writable,
            )
            with self.assertRaises(JsonlRecordError):
                read(after_terminal, {lifetime.lifetime_id: lifetime})

    def test_human_lifetime_requires_player_provenance(self) -> None:
        with self.assertRaises(ValueError):
            replace(make_lifetime(), source_type="human")
        human = replace(
            make_lifetime(),
            source_type="human",
            player_pseudonym="player-7",
        )
        self.assertEqual(human.player_pseudonym, "player-7")

    def test_branch_record_uses_return_wire_name(self) -> None:
        branch = BranchRecord(
            branch_group_id="group-1",
            root_snapshot_hash=HASH_A,
            root_lifetime_id="life-1",
            root_step=5,
            branch_id=2,
            rng_state_hash=HASH_B,
            forced_control_sequence=(GenericControl.neutral(),),
            continuation_policy_id="primitive",
            horizon_ticks=4,
            future_frames_ref="frames:1",
            state_delta_ref="delta:1",
            events=("safe",),
            return_value=1.25,
            replay_exact=True,
        )
        encoded = branch.to_dict()

        self.assertEqual(encoded["return"], 1.25)
        self.assertNotIn("return_value", encoded)
        self.assertEqual(BranchRecord.from_dict(encoded), branch)

        negative_zero_branch = replace(branch, return_value=-0.0)
        self.assertEqual(math.copysign(1.0, negative_zero_branch.return_value), 1.0)

        negative_zero_step = replace(
            make_step(
                step=0,
                requested=GenericControl.neutral(),
                applied=GenericControl.neutral(),
            ),
            reward_raw=-0.0,
        )
        self.assertEqual(math.copysign(1.0, negative_zero_step.reward_raw), 1.0)

    def test_control_schemas_match_runtime_dimensions_without_jsonschema(self) -> None:
        schema_root = Path(__file__).parents[1] / "schemas"
        for filename in ("step.schema.json", "branch.schema.json"):
            schema = json.loads((schema_root / filename).read_text(encoding="utf-8"))
            control_schema = schema["$defs"]["control"]
            control = control_schema["properties"]
            self.assertEqual(control["keys_down"]["maxItems"], 256)
            self.assertEqual(control["mouse_buttons"]["maxItems"], 8)
            self.assertEqual(control["mouse_buttons"]["items"]["maximum"], 7)
            self.assertEqual(control["gamepad_buttons"]["maxItems"], 32)
            self.assertEqual(control["gamepad_buttons"]["items"]["maximum"], 31)
            self.assertEqual(control["gamepad_axes"]["maxItems"], 8)
            self.assertEqual(
                control["gamepad_axes"]["oneOf"],
                [{"maxItems": 0}, {"minItems": 8}],
            )
            impulse_rule = control_schema["allOf"][0]
            self.assertEqual(
                impulse_rule["then"]["properties"]["impulse_sequence"]["minimum"],
                1,
            )

        lifetime_schema = json.loads(
            (schema_root / "lifetime.schema.json").read_text(encoding="utf-8")
        )
        human_rule = lifetime_schema["allOf"][0]
        self.assertEqual(human_rule["if"]["properties"]["source_type"]["const"], "human")
        self.assertIn("player_pseudonym", human_rule["then"]["required"])

    def test_split_integrity_rejects_world_session_and_ancestry_leakage(self) -> None:
        training = make_lifetime()
        leaked = replace(
            training,
            lifetime_id="life-2",
            split="test",
            ancestor_lifetime_ids=("life-1",),
        )

        with self.assertRaises(SplitIntegrityError) as raised:
            require_split_integrity((training, leaked))
        groupings = {leak.grouping for leak in raised.exception.leakages}
        self.assertTrue(
            {
                "world_lineage",
                "source_session",
                "family_control_mapping",
                "snapshot_ancestry",
            }.issubset(groupings)
        )


if __name__ == "__main__":
    unittest.main()
