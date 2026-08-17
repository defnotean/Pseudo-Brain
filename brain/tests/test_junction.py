from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.environments import EnvironmentProtocol, JunctionEnv
from irene_brain.environments.junction import _bfs_distances, _carve_maze
from irene_brain.evaluation.closed_loop_play import (
    ClosedLoopPlayConfig,
    run_policy_closed_loop_episode,
)
from irene_brain.evaluation.diagnostic_policies import NoOpPolicy
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


def outcome_identity(env: JunctionEnv, outcome: object) -> tuple[object, ...]:
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


class MazeGeneratorTests(unittest.TestCase):
    def test_maze_is_connected_and_deterministic(self) -> None:
        for seed in range(10):
            maze = _carve_maze(seed, JunctionEnv.GRID_SIZE)
            self.assertEqual(maze, _carve_maze(seed, JunctionEnv.GRID_SIZE))
            distances = _bfs_distances((1, 1), maze)
            # A perfect maze: every carved cell reachable, no unreachable rooms.
            self.assertEqual(set(distances), set(maze))

    def test_maze_varies_with_seed(self) -> None:
        self.assertNotEqual(
            _carve_maze(1, JunctionEnv.GRID_SIZE),
            _carve_maze(2, JunctionEnv.GRID_SIZE),
        )


class JunctionEnvironmentTests(unittest.TestCase):
    def test_environment_satisfies_branchable_protocol(self) -> None:
        self.assertIsInstance(JunctionEnv(), EnvironmentProtocol)

    def test_same_seed_has_identical_initial_state_and_pixels(self) -> None:
        first = JunctionEnv()
        second = JunctionEnv()
        first_observation = first.reset(123456)
        second_observation = second.reset(123456)
        self.assertEqual(first.state_hash(), second.state_hash())
        self.assertEqual(
            first_observation.content_hash, second_observation.content_hash
        )
        self.assertEqual(first_observation.rgb.pixels, second_observation.rgb.pixels)

    def test_walls_refuse_movement(self) -> None:
        env = JunctionEnv()
        env.reset(5)
        # Find a walled neighbor of the player and try to walk into it.
        blocked = None
        for key, (dx, dy) in (
            (HidKey.W, (0, -1)),
            (HidKey.A, (-1, 0)),
            (HidKey.S, (0, 1)),
            (HidKey.D, (1, 0)),
        ):
            neighbor = (env._player_x + dx, env._player_y + dy)
            if neighbor not in env._maze:
                blocked = key
                break
        self.assertIsNotNone(blocked)
        before = (env._player_x, env._player_y)
        env.step(GenericControl(keys_down=(int(blocked),)))
        self.assertEqual((env._player_x, env._player_y), before)

    def test_reset_places_everything_on_corridors(self) -> None:
        for seed in range(10):
            env = JunctionEnv(chaser_count=3)
            env.reset(seed)
            self.assertIn((env._player_x, env._player_y), env._maze)
            self.assertIn((env._target_x, env._target_y), env._maze)
            for chaser in env._chasers:
                self.assertIn((chaser.x, chaser.y), env._maze)

    def test_chaser_follows_shortest_path_and_catches_stationary_player(self) -> None:
        env = JunctionEnv(chaser_count=1, chaser_period=1)
        env.reset(7)
        caught = False
        for _ in range(200):
            outcome = env.step(GenericControl())
            if "caught" in outcome.events:
                caught = True
                self.assertEqual(outcome.reward, -1.0)
                break
        self.assertTrue(caught)

    def test_catch_respawns_the_player_on_an_empty_corridor(self) -> None:
        env = JunctionEnv(chaser_count=1, chaser_period=1)
        env.reset(7)
        for _ in range(200):
            outcome = env.step(GenericControl())
            if "caught" in outcome.events:
                break
        else:  # pragma: no cover - the shortest-path test proves the catch
            self.fail("chaser never caught a stationary player")
        occupied = {(env._target_x, env._target_y)} | {
            (chaser.x, chaser.y) for chaser in env._chasers
        }
        self.assertNotIn((env._player_x, env._player_y), occupied)
        self.assertIn((env._player_x, env._player_y), env._maze)

    def test_bfs_teacher_collects_targets(self) -> None:
        # A shortest-path walk to the target must score, proving the maze is
        # traversable end to end under the step mechanics.
        env = JunctionEnv(chaser_count=1, chaser_period=JunctionEnv.MAX_CHASER_PERIOD)
        env.reset(3)
        collected = False
        for _ in range(400):
            distances = _bfs_distances(
                (env._player_x, env._player_y), env._maze
            )
            here = distances[(env._player_x, env._player_y)]
            target_distance = _bfs_distances(
                (env._target_x, env._target_y), env._maze
            )
            keys = []
            for key, (dx, dy) in (
                (HidKey.W, (0, -1)),
                (HidKey.A, (-1, 0)),
                (HidKey.S, (0, 1)),
                (HidKey.D, (1, 0)),
            ):
                neighbor = (env._player_x + dx, env._player_y + dy)
                if target_distance.get(neighbor) == target_distance[
                    (env._player_x, env._player_y)
                ] - 1:
                    keys.append(int(key))
                    break
            outcome = env.step(GenericControl(keys_down=tuple(keys)))
            if "target_collected" in outcome.events:
                collected = True
                self.assertEqual(outcome.reward, 1.0)
                break
        self.assertTrue(collected)

    def test_one_thousand_step_replay_is_exact(self) -> None:
        source = JunctionEnv(max_ticks=2_000)
        initial = source.reset(83)
        self.assertEqual(
            initial.content_hash,
            "6caf2fa5a3e5a7509a7f40888ab3ee9e036cc9989ac11b220a6d0fb5e478bc01",
        )
        root = source.snapshot()
        expected: list[tuple[object, ...]] = []
        for step in range(1_000):
            expected.append(outcome_identity(source, source.step(control_for_step(step))))

        replay = JunctionEnv(max_ticks=2_000)
        replay.restore(root)
        actual = [
            outcome_identity(replay, replay.step(control_for_step(step)))
            for step in range(1_000)
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(
            replay.current_observation.content_hash,
            "dd6a5fbfbb1e6f05039abf3b6833e52e1ec3594ba493fe868b6c20591174cc0a",
        )
        self.assertEqual(
            replay.state_hash(),
            "d3a1bea4eb7eb68a125e430f93250b73d78e59297a392315153bdcd884160a72",
        )

    def test_snapshot_restore_is_atomic_on_corruption(self) -> None:
        environment = JunctionEnv()
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
        source = JunctionEnv(chaser_count=2)
        source.reset(5)
        snapshot = source.snapshot()
        with self.assertRaises(ValueError):
            JunctionEnv(chaser_count=1).restore(snapshot)
        with self.assertRaises(ValueError):
            JunctionEnv(chaser_count=2, chaser_period=1).restore(snapshot)
        with self.assertRaises(ValueError):
            JunctionEnv(chaser_count=2, max_ticks=999).restore(snapshot)

    def test_terminal_boundary_requires_reset_or_restore(self) -> None:
        env = JunctionEnv(max_ticks=4)
        env.reset(1)
        for _ in range(4):
            env.step(GenericControl())
        with self.assertRaises(RuntimeError):
            env.step(GenericControl())
        env.reset(2)
        env.step(GenericControl())

    def test_invalid_control_is_rejected_before_state_changes(self) -> None:
        env = JunctionEnv()
        env.reset(1)
        before = env.state_hash()
        with self.assertRaises(TypeError):
            env.step(object())  # type: ignore[arg-type]
        self.assertEqual(env.state_hash(), before)

    def test_constructor_validation(self) -> None:
        with self.assertRaises(TypeError):
            JunctionEnv(chaser_count=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            JunctionEnv(chaser_count=0)
        with self.assertRaises(ValueError):
            JunctionEnv(chaser_count=9)
        with self.assertRaises(ValueError):
            JunctionEnv(chaser_period=0)
        with self.assertRaises(ValueError):
            JunctionEnv(max_ticks=0)
        with self.assertRaises(ValueError):
            JunctionEnv(tick_period_ns=0)
        env = JunctionEnv()
        with self.assertRaises(TypeError):
            env.reset(True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            env.reset(-1)

    def test_model_observation_has_only_allowlisted_fields(self) -> None:
        env = JunctionEnv()
        observation = env.reset(9)
        self.assertIsNone(observation.audio_pcm_s16le)
        self.assertEqual(observation.text_inputs, ())
        self.assertEqual(observation.rgb.width, JunctionEnv.GRID_SIZE)
        self.assertEqual(observation.rgb.height, JunctionEnv.GRID_SIZE)


class JunctionClosedLoopTests(unittest.TestCase):
    def test_noop_policy_gets_caught_in_junction_world(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(7,), max_ticks=200)
        report = run_policy_closed_loop_episode(
            NoOpPolicy(),
            seed=7,
            config=config,
            environment_factory=lambda: JunctionEnv(
                chaser_count=1, chaser_period=1, max_ticks=200
            ),
        )
        self.assertEqual(report.ticks_advanced, 200)
        self.assertGreaterEqual(report.collisions, 1)
        self.assertEqual(report.targets_collected, 0)

    def test_junction_episode_is_deterministic(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(7,), max_ticks=120)
        factory = lambda: JunctionEnv(  # noqa: E731
            chaser_count=1, chaser_period=2, max_ticks=120
        )
        first = run_policy_closed_loop_episode(
            NoOpPolicy(), seed=7, config=config, environment_factory=factory
        )
        second = run_policy_closed_loop_episode(
            NoOpPolicy(), seed=7, config=config, environment_factory=factory
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
