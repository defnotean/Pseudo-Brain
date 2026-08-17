from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.environments import EnvironmentProtocol, OcclusionEnv
from irene_brain.evaluation.closed_loop_play import (
    ClosedLoopPlayConfig,
    run_policy_closed_loop_episode,
)
from irene_brain.evaluation.diagnostic_policies import RandomMovementPolicy
from irene_brain.types import GenericControl, HidKey


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


def outcome_identity(env: OcclusionEnv, outcome: object) -> tuple[object, ...]:
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


def pixel_color(env: OcclusionEnv, x: int, y: int) -> tuple[int, int, int]:
    pixels = env.current_observation.rgb.pixels
    offset = (y * OcclusionEnv.GRID_SIZE + x) * 3
    return (pixels[offset], pixels[offset + 1], pixels[offset + 2])


class OcclusionEnvironmentTests(unittest.TestCase):
    def test_environment_satisfies_branchable_protocol(self) -> None:
        self.assertIsInstance(OcclusionEnv(), EnvironmentProtocol)

    def test_same_seed_has_identical_initial_state_and_pixels(self) -> None:
        first = OcclusionEnv()
        second = OcclusionEnv()
        first_observation = first.reset(123456)
        second_observation = second.reset(123456)
        self.assertEqual(first.state_hash(), second.state_hash())
        self.assertEqual(
            first_observation.content_hash, second_observation.content_hash
        )
        self.assertEqual(first_observation.rgb.pixels, second_observation.rgb.pixels)

    def test_fog_covers_exactly_the_cells_beyond_the_view_radius(self) -> None:
        env = OcclusionEnv(view_radius=2)
        env.reset(5)
        player = (env._player_x, env._player_y)
        for y in range(OcclusionEnv.GRID_SIZE):
            for x in range(OcclusionEnv.GRID_SIZE):
                beyond = (
                    abs(x - player[0]) > 2 or abs(y - player[1]) > 2
                )
                if (x, y) == player:
                    continue
                color = pixel_color(env, x, y)
                if beyond:
                    self.assertEqual(color, OcclusionEnv.FOG_RGB)
                else:
                    self.assertNotEqual(color, OcclusionEnv.FOG_RGB)

    def test_distant_hazards_are_invisible_but_still_collide(self) -> None:
        # Player at the center with radius 1: hazards spawn anywhere else, so
        # most start hidden; mechanics must not depend on visibility.
        env = OcclusionEnv(hazard_count=3, view_radius=1)
        env.reset(9)
        hidden = [
            hazard
            for hazard in env._hazards
            if abs(hazard.x - env._player_x) > 1
            or abs(hazard.y - env._player_y) > 1
        ]
        self.assertTrue(hidden)
        for hazard in hidden:
            self.assertEqual(
                pixel_color(env, hazard.x, hazard.y), OcclusionEnv.FOG_RGB
            )
        # The hidden hazards still advanced under a no-op step.
        positions = [(hazard.x, hazard.y) for hazard in env._hazards]
        env.step(GenericControl())
        self.assertNotEqual(
            positions, [(hazard.x, hazard.y) for hazard in env._hazards]
        )

    def test_fog_window_follows_the_player(self) -> None:
        env = OcclusionEnv(view_radius=1)
        env.reset(5)
        # Walk right until the player actually moves (it starts mid-grid).
        for _ in range(3):
            env.step(GenericControl(keys_down=(int(HidKey.D),)))
        player = (env._player_x, env._player_y)
        self.assertEqual(
            pixel_color(env, player[0] + 2, player[1]), OcclusionEnv.FOG_RGB
        )
        self.assertNotEqual(
            pixel_color(env, player[0] + 1, player[1]), OcclusionEnv.FOG_RGB
        )

    def test_one_thousand_step_replay_is_exact(self) -> None:
        source = OcclusionEnv(max_ticks=2_000)
        initial = source.reset(83)
        self.assertEqual(
            initial.content_hash,
            "639c56e9adabc33acf884e86b223752907b2f5ffe12b06e6cfca082a93782825",
        )
        root = source.snapshot()
        expected: list[tuple[object, ...]] = []
        for step in range(1_000):
            expected.append(outcome_identity(source, source.step(control_for_step(step))))

        replay = OcclusionEnv(max_ticks=2_000)
        replay.restore(root)
        actual = [
            outcome_identity(replay, replay.step(control_for_step(step)))
            for step in range(1_000)
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(
            replay.current_observation.content_hash,
            "1b6a0f39372a5407f52f755abf8f9094d9c8bdf6b1a5c726a3a41c0e2c8b0703",
        )
        self.assertEqual(
            replay.state_hash(),
            "f4e9944bfa14d928874b6c2130357959696b8fc2d19acfab2be51c62ef61dff6",
        )

    def test_snapshot_restore_is_atomic_on_corruption(self) -> None:
        environment = OcclusionEnv()
        environment.reset(11)
        for step in range(20):
            environment.step(control_for_step(step))
        snapshot = environment.snapshot()
        before = environment.state_hash()
        damaged = snapshot[:-1] + bytes((snapshot[-1] ^ 0xFF,))
        with self.assertRaises(ValueError):
            environment.restore(damaged)
        self.assertEqual(environment.state_hash(), before)

    def test_restore_rejects_configuration_mismatches(self) -> None:
        source = OcclusionEnv(hazard_count=2, view_radius=3)
        source.reset(5)
        snapshot = source.snapshot()
        with self.assertRaises(ValueError):
            OcclusionEnv(hazard_count=3, view_radius=3).restore(snapshot)
        with self.assertRaises(ValueError):
            OcclusionEnv(hazard_count=2, view_radius=4).restore(snapshot)
        with self.assertRaises(ValueError):
            OcclusionEnv(hazard_count=2, view_radius=3, max_ticks=999).restore(snapshot)

    def test_terminal_boundary_requires_reset_or_restore(self) -> None:
        env = OcclusionEnv(max_ticks=4)
        env.reset(1)
        for _ in range(4):
            env.step(GenericControl())
        with self.assertRaises(RuntimeError):
            env.step(GenericControl())
        env.reset(2)
        env.step(GenericControl())

    def test_invalid_control_is_rejected_before_state_changes(self) -> None:
        env = OcclusionEnv()
        env.reset(1)
        before = env.state_hash()
        with self.assertRaises(TypeError):
            env.step(object())  # type: ignore[arg-type]
        self.assertEqual(env.state_hash(), before)

    def test_constructor_validation(self) -> None:
        with self.assertRaises(TypeError):
            OcclusionEnv(view_radius=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            OcclusionEnv(view_radius=0)
        with self.assertRaises(ValueError):
            OcclusionEnv(view_radius=OcclusionEnv.MAX_VIEW_RADIUS + 1)
        with self.assertRaises(ValueError):
            OcclusionEnv(hazard_count=0)
        with self.assertRaises(ValueError):
            OcclusionEnv(max_ticks=0)
        with self.assertRaises(ValueError):
            OcclusionEnv(tick_period_ns=0)
        env = OcclusionEnv()
        with self.assertRaises(TypeError):
            env.reset(True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            env.reset(-1)

    def test_model_observation_has_only_allowlisted_fields(self) -> None:
        env = OcclusionEnv()
        observation = env.reset(9)
        self.assertIsNone(observation.audio_pcm_s16le)
        self.assertEqual(observation.text_inputs, ())
        self.assertEqual(observation.rgb.width, OcclusionEnv.GRID_SIZE)
        self.assertEqual(observation.rgb.height, OcclusionEnv.GRID_SIZE)


class OcclusionClosedLoopTests(unittest.TestCase):
    def test_occlusion_episode_is_deterministic(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(7,), max_ticks=120)
        factory = lambda: OcclusionEnv(  # noqa: E731
            hazard_count=3, view_radius=4, max_ticks=120
        )
        first = run_policy_closed_loop_episode(
            RandomMovementPolicy(), seed=7, config=config, environment_factory=factory
        )
        second = run_policy_closed_loop_episode(
            RandomMovementPolicy(), seed=7, config=config, environment_factory=factory
        )
        self.assertEqual(first, second)
        self.assertEqual(first.ticks_advanced, 120)


if __name__ == "__main__":
    unittest.main()
