from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.environments import EnvironmentProtocol, PursuitEnv
from irene_brain.evaluation.closed_loop_play import (
    ClosedLoopPlayConfig,
    run_policy_closed_loop_episode,
)
from irene_brain.evaluation.diagnostic_policies import (
    NoOpPolicy,
    ScriptedTargetChasePolicy,
)
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


def outcome_identity(env: PursuitEnv, outcome: object) -> tuple[object, ...]:
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


class PursuitEnvironmentTests(unittest.TestCase):
    def test_environment_satisfies_branchable_protocol(self) -> None:
        self.assertIsInstance(PursuitEnv(), EnvironmentProtocol)

    def test_same_seed_has_identical_initial_state_and_pixels(self) -> None:
        first = PursuitEnv()
        second = PursuitEnv()
        first_observation = first.reset(123456)
        second_observation = second.reset(123456)

        self.assertEqual(first.state_hash(), second.state_hash())
        self.assertEqual(
            first_observation.content_hash, second_observation.content_hash
        )
        self.assertEqual(first_observation.rgb.pixels, second_observation.rgb.pixels)

    def test_different_seeds_change_the_initial_state(self) -> None:
        first = PursuitEnv()
        second = PursuitEnv()
        first.reset(1)
        second.reset(2)
        self.assertNotEqual(first.state_hash(), second.state_hash())

    def test_pursuers_spawn_at_least_four_cells_away(self) -> None:
        for seed in range(20):
            env = PursuitEnv(pursuer_count=4)
            env.reset(seed)
            for pursuer in env._pursuers:
                distance = abs(pursuer.x - env._player_x) + abs(
                    pursuer.y - env._player_y
                )
                self.assertGreaterEqual(distance, 4)

    def test_pursuers_close_distance_on_a_stationary_player(self) -> None:
        env = PursuitEnv(pursuer_count=1, pursuer_period=1)
        env.reset(7)
        before = abs(env._pursuers[0].x - env._player_x) + abs(
            env._pursuers[0].y - env._player_y
        )
        caught = False
        for step in range(40):
            outcome = env.step(GenericControl())
            if "caught" in outcome.events:
                caught = True
                break
        self.assertTrue(caught)
        self.assertEqual(outcome.reward, -1.0)
        self.assertGreaterEqual(before, 4)

    def test_pursuer_period_two_moves_slower(self) -> None:
        fast = PursuitEnv(pursuer_count=1, pursuer_period=1)
        slow = PursuitEnv(pursuer_count=1, pursuer_period=2)
        fast.reset(7)
        slow.reset(7)
        self.assertEqual(
            (fast._pursuers[0].x, fast._pursuers[0].y),
            (slow._pursuers[0].x, slow._pursuers[0].y),
        )
        for step in range(4):
            fast.step(GenericControl())
            slow.step(GenericControl())
        fast_distance = abs(fast._pursuers[0].x - fast._player_x) + abs(
            fast._pursuers[0].y - fast._player_y
        )
        slow_distance = abs(slow._pursuers[0].x - slow._player_x) + abs(
            slow._pursuers[0].y - slow._player_y
        )
        self.assertLess(fast_distance, slow_distance)

    def test_catch_respawns_the_player_on_an_empty_cell(self) -> None:
        env = PursuitEnv(pursuer_count=1, pursuer_period=1)
        env.reset(7)
        for _ in range(40):
            outcome = env.step(GenericControl())
            if "caught" in outcome.events:
                break
        else:  # pragma: no cover - the closing-distance test proves the catch
            self.fail("pursuer never caught a stationary player")
        occupied = {(env._target_x, env._target_y)} | {
            (pursuer.x, pursuer.y) for pursuer in env._pursuers
        }
        self.assertNotIn((env._player_x, env._player_y), occupied)

    def test_target_collection_scores_and_relocates(self) -> None:
        env = PursuitEnv(pursuer_count=1, pursuer_period=PursuitEnv.MAX_PURSUER_PERIOD)
        env.reset(3)
        # Walk straight at the target along whichever axis is farther.
        collected = False
        for _ in range(200):
            dx = env._target_x - env._player_x
            dy = env._target_y - env._player_y
            keys = []
            if dy < 0:
                keys.append(int(HidKey.W))
            elif dy > 0:
                keys.append(int(HidKey.S))
            if dx < 0:
                keys.append(int(HidKey.A))
            elif dx > 0:
                keys.append(int(HidKey.D))
            outcome = env.step(GenericControl(keys_down=tuple(keys)))
            if "target_collected" in outcome.events:
                collected = True
                self.assertEqual(outcome.reward, 1.0)
                break
        self.assertTrue(collected)

    def test_one_thousand_step_replay_is_exact(self) -> None:
        source = PursuitEnv(max_ticks=2_000)
        initial = source.reset(83)
        self.assertEqual(
            initial.content_hash,
            "1e34b3e9975cc0c9e5dc29cdc5d9bbedabdc922298d708028ce39fcb4727bf8a",
        )
        root = source.snapshot()
        expected: list[tuple[object, ...]] = []
        for step in range(1_000):
            expected.append(outcome_identity(source, source.step(control_for_step(step))))

        replay = PursuitEnv(max_ticks=2_000)
        replay.restore(root)
        actual = [
            outcome_identity(replay, replay.step(control_for_step(step)))
            for step in range(1_000)
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(
            replay.current_observation.content_hash,
            "247b3faa0a8984e1037c0a0f9edd4d8220eeba8b9869b4e04dcd21d51cec3921",
        )
        self.assertEqual(
            replay.state_hash(),
            "3cf6301d6ad38ee0f8fbc2cfa9fb018a4cd43be38e523927194c37250118b68c",
        )

    def test_snapshot_restore_is_atomic_on_corruption(self) -> None:
        environment = PursuitEnv()
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
        source = PursuitEnv(pursuer_count=2)
        source.reset(5)
        snapshot = source.snapshot()
        with self.assertRaises(ValueError):
            PursuitEnv(pursuer_count=3).restore(snapshot)
        with self.assertRaises(ValueError):
            PursuitEnv(pursuer_count=2, pursuer_period=1).restore(snapshot)
        with self.assertRaises(ValueError):
            PursuitEnv(pursuer_count=2, max_ticks=999).restore(snapshot)

    def test_terminal_boundary_requires_reset_or_restore(self) -> None:
        env = PursuitEnv(max_ticks=4)
        env.reset(1)
        for _ in range(4):
            env.step(GenericControl())
        with self.assertRaises(RuntimeError):
            env.step(GenericControl())
        env.reset(2)
        env.step(GenericControl())

    def test_invalid_control_is_rejected_before_state_changes(self) -> None:
        env = PursuitEnv()
        env.reset(1)
        before = env.state_hash()
        with self.assertRaises(TypeError):
            env.step(object())  # type: ignore[arg-type]
        self.assertEqual(env.state_hash(), before)

    def test_constructor_validation(self) -> None:
        with self.assertRaises(TypeError):
            PursuitEnv(pursuer_count=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            PursuitEnv(pursuer_count=0)
        with self.assertRaises(ValueError):
            PursuitEnv(pursuer_period=0)
        with self.assertRaises(ValueError):
            PursuitEnv(pursuer_period=PursuitEnv.MAX_PURSUER_PERIOD + 1)
        with self.assertRaises(ValueError):
            PursuitEnv(max_ticks=0)
        with self.assertRaises(ValueError):
            PursuitEnv(tick_period_ns=0)
        env = PursuitEnv()
        with self.assertRaises(TypeError):
            env.reset(True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            env.reset(-1)

    def test_model_observation_has_only_allowlisted_fields(self) -> None:
        env = PursuitEnv()
        observation = env.reset(9)
        self.assertIsNone(observation.audio_pcm_s16le)
        self.assertEqual(observation.text_inputs, ())
        self.assertEqual(
            observation.rgb.width, PursuitEnv.GRID_SIZE
        )
        self.assertEqual(
            observation.rgb.height, PursuitEnv.GRID_SIZE
        )


class PursuitClosedLoopTests(unittest.TestCase):
    """The closed-loop evaluator drives any branchable in-repo world."""

    def test_noop_policy_gets_caught_in_pursuit_world(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(7,), max_ticks=120)
        report = run_policy_closed_loop_episode(
            NoOpPolicy(),
            seed=7,
            config=config,
            environment_factory=lambda: PursuitEnv(
                pursuer_count=1, pursuer_period=1, max_ticks=120
            ),
        )
        self.assertEqual(report.ticks_advanced, 120)
        self.assertGreaterEqual(report.collisions, 1)
        self.assertEqual(report.targets_collected, 0)

    def test_scripted_chaser_collects_in_pursuit_world(self) -> None:
        # The pixel-only chaser reads the shared render contract, so it plays
        # the pursuit world unmodified; slow pursuers keep it alive.
        config = ClosedLoopPlayConfig(episode_seeds=(3,), max_ticks=240)
        report = run_policy_closed_loop_episode(
            ScriptedTargetChasePolicy(),
            seed=3,
            config=config,
            environment_factory=lambda: PursuitEnv(
                pursuer_count=1,
                pursuer_period=PursuitEnv.MAX_PURSUER_PERIOD,
                max_ticks=240,
            ),
        )
        self.assertGreaterEqual(report.targets_collected, 1)

    def test_pursuit_episode_is_deterministic(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(7,), max_ticks=120)
        factory = lambda: PursuitEnv(  # noqa: E731
            pursuer_count=2, pursuer_period=2, max_ticks=120
        )
        first = run_policy_closed_loop_episode(
            ScriptedTargetChasePolicy(),
            seed=7,
            config=config,
            environment_factory=factory,
        )
        second = run_policy_closed_loop_episode(
            ScriptedTargetChasePolicy(),
            seed=7,
            config=config,
            environment_factory=factory,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
