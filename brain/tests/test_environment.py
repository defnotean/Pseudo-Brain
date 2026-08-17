from __future__ import annotations

import dataclasses
import itertools
import unittest

from irene_brain.environments import EnvironmentProtocol, MovingShapesEnv
from irene_brain.environments.moving_shapes import _paths_collide_at_same_time
from irene_brain.types import GenericControl, HidKey, Observation


def control_for_step(step: int) -> GenericControl:
    pattern = (
        (int(HidKey.W),),
        (int(HidKey.D),),
        (int(HidKey.S),),
        (int(HidKey.A),),
        (int(HidKey.W), int(HidKey.D)),
        (),
    )
    return GenericControl(keys_down=pattern[step % len(pattern)])


def outcome_identity(env: MovingShapesEnv, outcome: object) -> tuple[object, ...]:
    typed = outcome
    return (
        typed.observation.content_hash,  # type: ignore[attr-defined]
        typed.applied_control.canonical_bytes(),  # type: ignore[attr-defined]
        typed.reward,  # type: ignore[attr-defined]
        typed.events,  # type: ignore[attr-defined]
        typed.terminated,  # type: ignore[attr-defined]
        typed.truncated,  # type: ignore[attr-defined]
        env.state_hash(),
    )


class MovingShapesEnvironmentTests(unittest.TestCase):
    def test_environment_satisfies_branchable_protocol(self) -> None:
        self.assertIsInstance(MovingShapesEnv(), EnvironmentProtocol)

    def test_same_seed_has_identical_initial_state_and_pixels(self) -> None:
        first = MovingShapesEnv()
        second = MovingShapesEnv()
        first_observation = first.reset(123456)
        second_observation = second.reset(123456)

        self.assertEqual(first.state_hash(), second.state_hash())
        self.assertEqual(first_observation.content_hash, second_observation.content_hash)
        self.assertEqual(first_observation.rgb.pixels, second_observation.rgb.pixels)

    def test_collision_paths_use_exact_same_time_intersection(self) -> None:
        # Opposing diagonals meet exactly at the half-cell point. Neither actor
        # occupies the other's endpoint, so endpoint/swap checks miss this.
        self.assertTrue(
            _paths_collide_at_same_time((0, 0), (1, 1), (0, 1), (1, 0))
        )
        self.assertTrue(
            _paths_collide_at_same_time((0, 0), (1, 0), (1, 0), (0, 0))
        )
        # These swept paths cross spatially, but at different times.
        self.assertFalse(
            _paths_collide_at_same_time((0, 0), (1, 0), (1, -1), (0, 0))
        )

    def test_one_thousand_step_replay_is_exact(self) -> None:
        source = MovingShapesEnv(max_ticks=2_000)
        initial = source.reset(83)
        self.assertEqual(
            initial.content_hash,
            "e65ad1e1e1842f38a917aa55235fcf604dc6cf2ce9b1ffdd98e98e48755309f2",
        )
        self.assertEqual(
            source.state_hash(),
            "9e11738ddfa2026484604243cb9c8443e04b1b1170c394cbfbc3593e96bd6ae6",
        )
        root = source.snapshot()
        expected: list[tuple[object, ...]] = []
        for step in range(1_000):
            expected.append(outcome_identity(source, source.step(control_for_step(step))))

        replay = MovingShapesEnv(max_ticks=2_000)
        replay.restore(root)
        actual = [
            outcome_identity(replay, replay.step(control_for_step(step)))
            for step in range(1_000)
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(
            replay.current_observation.content_hash,
            "c5ba0d515ac6f33cc589ce1a5ec8a043c52cd342b739b7af414b19fff45f83c9",
        )
        self.assertEqual(
            replay.state_hash(),
            "ed510ee6f6e659d77f933feb860612fe32addbb0d9782d957035b88f12393fe5",
        )

    def test_snapshot_restore_is_atomic_on_corruption(self) -> None:
        environment = MovingShapesEnv()
        environment.reset(11)
        for step in range(20):
            environment.step(control_for_step(step))
        snapshot = environment.snapshot()
        before = environment.state_hash()
        damaged = snapshot[:-1] + bytes((snapshot[-1] ^ 0xFF,))

        with self.assertRaises(ValueError):
            environment.restore(damaged)
        self.assertEqual(environment.state_hash(), before)

    def test_branch_order_does_not_change_any_future(self) -> None:
        environment = MovingShapesEnv()
        environment.reset(9001)
        for step in range(30):
            environment.step(control_for_step(step))
        root = environment.snapshot()
        root_hash = environment.state_hash()
        branches = {
            "north": [GenericControl(keys_down=(int(HidKey.W),))] * 8,
            "east": [GenericControl(keys_down=(int(HidKey.D),))] * 8,
            "south": [GenericControl(keys_down=(int(HidKey.S),))] * 8,
            "west": [GenericControl(keys_down=(int(HidKey.A),))] * 8,
        }

        reference: dict[str, tuple[tuple[object, ...], ...]] | None = None
        for order in itertools.permutations(branches):
            observed: dict[str, tuple[tuple[object, ...], ...]] = {}
            for name in order:
                environment.restore(root)
                self.assertEqual(environment.state_hash(), root_hash)
                observed[name] = tuple(
                    outcome_identity(environment, environment.step(control))
                    for control in branches[name]
                )
            environment.restore(root)
            self.assertEqual(environment.state_hash(), root_hash)
            if reference is None:
                reference = observed
            else:
                self.assertEqual(observed, reference)

    def test_previous_control_is_applied_not_requested(self) -> None:
        environment = MovingShapesEnv()
        environment.reset(1)
        requested = GenericControl(
            keys_down=(int(HidKey.W), int(HidKey.SPACE)),
            mouse_dx=5.0,
            impulse_sequence=7,
        )
        outcome = environment.step(requested)

        self.assertEqual(outcome.requested_control, requested)
        self.assertEqual(outcome.applied_control.keys_down, (int(HidKey.W),))
        self.assertEqual(outcome.observation.previous_control, outcome.applied_control)
        self.assertNotEqual(outcome.observation.previous_control, requested)

    def test_terminal_boundary_requires_reset_or_restore(self) -> None:
        environment = MovingShapesEnv(max_ticks=2)
        environment.reset(5)
        root = environment.snapshot()
        self.assertFalse(environment.step(GenericControl.neutral()).truncated)
        self.assertTrue(environment.step(GenericControl.neutral()).truncated)
        with self.assertRaises(RuntimeError):
            environment.step(GenericControl.neutral())

        restored = environment.restore(root)
        self.assertEqual(restored.frame_id, 0)
        self.assertFalse(environment.step(GenericControl.neutral()).truncated)

    def test_invalid_control_is_rejected_before_state_changes(self) -> None:
        environment = MovingShapesEnv()
        before = environment.state_hash()

        with self.assertRaises(TypeError):
            environment.step(object())  # type: ignore[arg-type]

        self.assertEqual(environment.state_hash(), before)

    def test_model_observation_has_only_allowlisted_fields(self) -> None:
        allowed = {
            "frame_id",
            "capture_tick",
            "elapsed_ns",
            "rgb",
            "previous_control",
            "audio_pcm_s16le",
            "text_inputs",
        }
        self.assertEqual(
            {field.name for field in dataclasses.fields(Observation)},
            allowed,
        )
        forbidden = {
            "reward",
            "events",
            "seed",
            "game_id",
            "control_mapping",
            "snapshot",
            "privileged_state",
            "future",
        }
        self.assertTrue(forbidden.isdisjoint(allowed))


if __name__ == "__main__":
    unittest.main()
