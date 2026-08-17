from __future__ import annotations

import dataclasses
import unittest

from irene_brain.types import (
    ActionEnvelope,
    GenericControl,
    HidKey,
    MODEL_OBSERVATION_FIELDS,
    ModelObservation,
    Observation,
    RgbFrame,
    StepOutcome,
    TextInput,
    TextProvenance,
)


class GenericControlTests(unittest.TestCase):
    def test_control_state_is_canonical(self) -> None:
        control = GenericControl(
            keys_down=(int(HidKey.W), int(HidKey.A), int(HidKey.W)),
            mouse_buttons=(2, 1, 2),
            gamepad_buttons=(4, 3, 4),
        )

        self.assertEqual(control.keys_down, (int(HidKey.A), int(HidKey.W)))
        self.assertEqual(control.mouse_buttons, (1, 2))
        self.assertEqual(control.gamepad_buttons, (3, 4))
        self.assertEqual(GenericControl.from_dict(control.to_dict()), control)

    def test_non_finite_or_out_of_range_axes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GenericControl(mouse_dx=float("nan"))
        with self.assertRaises(ValueError):
            GenericControl(gamepad_axes=(1.01,) + (0.0,) * 7)

    def test_unknown_or_lossy_fields_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GenericControl.from_dict([])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            GenericControl.from_dict({"keys_down": [True]})
        with self.assertRaises(ValueError):
            GenericControl.from_dict({"keys_down": [1.5]})
        with self.assertRaises(ValueError):
            GenericControl.from_dict({"future_action": "leak"})
        with self.assertRaises(ValueError):
            GenericControl.from_dict({"keys_down": [int(HidKey.W), int(HidKey.W)]})

    def test_negative_zero_has_one_canonical_encoding(self) -> None:
        positive = GenericControl(mouse_dx=0.0, gamepad_axes=(0.0,) * 8)
        negative = GenericControl(mouse_dx=-0.0, gamepad_axes=(-0.0,) * 8)
        self.assertEqual(positive.canonical_bytes(), negative.canonical_bytes())

    def test_control_dimensions_match_the_actuator_contract(self) -> None:
        self.assertEqual(len(GenericControl(gamepad_axes=(0.0,) * 8).gamepad_axes), 8)
        with self.assertRaises(ValueError):
            GenericControl(gamepad_axes=(0.0,) * 7)
        with self.assertRaises(ValueError):
            GenericControl(mouse_buttons=(8,))
        with self.assertRaises(ValueError):
            GenericControl(gamepad_buttons=(32,))

    def test_uint64_wire_fields_are_bounded(self) -> None:
        with self.assertRaises(ValueError):
            GenericControl(impulse_sequence=1 << 64)
        with self.assertRaises(ValueError):
            GenericControl(intended_hold_ns=1 << 64)

    def test_neutral_has_no_active_control(self) -> None:
        self.assertEqual(GenericControl.neutral(), GenericControl())


class ObservationTests(unittest.TestCase):
    def test_rgb_length_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            RgbFrame(width=2, height=2, pixels=b"\x00" * 11)
        with self.assertRaises(ValueError):
            RgbFrame(width=1 << 32, height=1, pixels=b"")

    def test_content_hash_is_stable_and_sensitive(self) -> None:
        frame = RgbFrame(width=2, height=2, pixels=b"\x00" * 12)
        base = Observation(
            frame_id=1,
            capture_tick=3,
            elapsed_ns=16_666_667,
            rgb=frame,
            previous_control=GenericControl.neutral(),
        )
        same = Observation(
            frame_id=1,
            capture_tick=3,
            elapsed_ns=16_666_667,
            rgb=frame,
            previous_control=GenericControl.neutral(),
        )
        changed = Observation(
            frame_id=1,
            capture_tick=3,
            elapsed_ns=16_666_667,
            rgb=frame,
            previous_control=GenericControl(keys_down=(int(HidKey.W),)),
        )

        self.assertEqual(base.content_hash, same.content_hash)
        self.assertNotEqual(base.content_hash, changed.content_hash)

    def test_frame_hash_includes_shape_and_format_contract(self) -> None:
        wide = RgbFrame(width=2, height=1, pixels=b"\x00" * 6)
        tall = RgbFrame(width=1, height=2, pixels=b"\x00" * 6)
        self.assertNotEqual(wide.sha256, tall.sha256)

    def test_absent_and_empty_audio_have_different_hashes(self) -> None:
        frame = RgbFrame(width=1, height=1, pixels=b"\x00" * 3)
        without_audio = Observation(
            frame_id=0,
            capture_tick=0,
            elapsed_ns=0,
            rgb=frame,
            previous_control=GenericControl.neutral(),
        )
        empty_audio = Observation(
            frame_id=0,
            capture_tick=0,
            elapsed_ns=0,
            rgb=frame,
            previous_control=GenericControl.neutral(),
            audio_pcm_s16le=b"",
        )
        self.assertNotEqual(without_audio.content_hash, empty_audio.content_hash)

    def test_model_text_requires_visible_provenance(self) -> None:
        frame = RgbFrame(width=1, height=1, pixels=b"\x00" * 3)
        observation = Observation(
            frame_id=0,
            capture_tick=0,
            elapsed_ns=0,
            rgb=frame,
            previous_control=GenericControl.neutral(),
            text_inputs=(
                TextInput(TextProvenance.USER_INSTRUCTION, "move toward the square"),
            ),
        )
        self.assertIsInstance(observation.text_inputs[0].provenance, TextProvenance)

    def test_live_envelope_materializes_the_canonical_model_boundary(self) -> None:
        frame = RgbFrame(width=1, height=1, pixels=b"\x01\x02\x03")
        raw = Observation(
            frame_id=7,
            capture_tick=999_999,
            elapsed_ns=123,
            rgb=frame,
            previous_control=GenericControl(keys_down=(int(HidKey.W),)),
        )

        model_input = raw.to_model_observation(
            observation_age_ns=4,
            dropped_frames=2,
        )

        self.assertIsInstance(model_input, ModelObservation)
        self.assertEqual(
            {field.name for field in dataclasses.fields(model_input)},
            MODEL_OBSERVATION_FIELDS,
        )
        self.assertNotIn("capture_tick", MODEL_OBSERVATION_FIELDS)
        self.assertIs(model_input.rgb, frame)
        self.assertEqual(model_input.previous_control, raw.previous_control)

    def test_packed_observation_counters_reject_uint64_overflow(self) -> None:
        with self.assertRaises(ValueError):
            Observation(
                frame_id=1 << 64,
                capture_tick=0,
                elapsed_ns=0,
                rgb=RgbFrame(width=1, height=1, pixels=b"\x00" * 3),
                previous_control=GenericControl.neutral(),
            )


class ActionEnvelopeTests(unittest.TestCase):
    def test_timestamp_order_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            ActionEnvelope(
                action_sequence=1,
                source_frame_id=1,
                model_state_version=1,
                created_qpc=10,
                ready_qpc=9,
                submit_deadline_qpc=12,
                expires_qpc=13,
                control=GenericControl.neutral(),
            )

    def test_uint64_wire_bound_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            ActionEnvelope(
                action_sequence=1 << 64,
                source_frame_id=1,
                model_state_version=1,
                created_qpc=10,
                ready_qpc=10,
                submit_deadline_qpc=12,
                expires_qpc=13,
                control=GenericControl.neutral(),
            )


class StepOutcomeTests(unittest.TestCase):
    def test_reward_is_strict_finite_numeric_data(self) -> None:
        observation = Observation(
            frame_id=0,
            capture_tick=0,
            elapsed_ns=0,
            rgb=RgbFrame(width=1, height=1, pixels=b"\x00" * 3),
            previous_control=GenericControl.neutral(),
        )
        with self.assertRaises(ValueError):
            StepOutcome(
                observation=observation,
                requested_control=GenericControl.neutral(),
                applied_control=GenericControl.neutral(),
                reward="1.0",  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            StepOutcome(
                observation=observation,
                requested_control=GenericControl.neutral(),
                applied_control=GenericControl.neutral(),
                reward=True,
            )


if __name__ == "__main__":
    unittest.main()
