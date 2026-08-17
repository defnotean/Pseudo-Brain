from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.environments import EnvironmentProtocol, KeysDoorsEnv
from irene_brain.environments.junction import _bfs_distances
from irene_brain.evaluation.closed_loop_play import (
    ClosedLoopPlayConfig,
    run_policy_closed_loop_episode,
)
from irene_brain.evaluation.diagnostic_policies import RandomMovementPolicy
from irene_brain.types import GenericControl, HidKey

_KEYS_FOR_DELTA = {
    (0, -1): int(HidKey.W),
    (-1, 0): int(HidKey.A),
    (0, 1): int(HidKey.S),
    (1, 0): int(HidKey.D),
}


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


def outcome_identity(env: KeysDoorsEnv, outcome: object) -> tuple[object, ...]:
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


def step_toward(env: KeysDoorsEnv, goal: tuple[int, int]) -> object:
    """One greedy shortest-path step toward ``goal`` (walls and closed door
    included in the distance field only when passable)."""
    distances = _bfs_distances(goal, env._maze)
    here = distances[(env._player_x, env._player_y)]
    for (dx, dy), key in _KEY_FOR_DELTA_ORDERED:
        neighbor = (env._player_x + dx, env._player_y + dy)
        if distances.get(neighbor) == here - 1:
            return env.step(GenericControl(keys_down=(key,)))
    raise AssertionError("no descending neighbor")


_KEY_FOR_DELTA_ORDERED = tuple(_KEYS_FOR_DELTA.items())


def walk_to(env: KeysDoorsEnv, goal: tuple[int, int], limit: int = 400) -> list:
    events = []
    for _ in range(limit):
        if (env._player_x, env._player_y) == goal:
            return events
        outcome = step_toward(env, goal)
        events.extend(outcome.events)
    raise AssertionError(f"never reached {goal}")


class KeysDoorsEnvironmentTests(unittest.TestCase):
    def test_environment_satisfies_branchable_protocol(self) -> None:
        self.assertIsInstance(KeysDoorsEnv(), EnvironmentProtocol)

    def test_same_seed_has_identical_initial_state_and_pixels(self) -> None:
        first = KeysDoorsEnv()
        second = KeysDoorsEnv()
        first_observation = first.reset(123456)
        second_observation = second.reset(123456)
        self.assertEqual(first.state_hash(), second.state_hash())
        self.assertEqual(
            first_observation.content_hash, second_observation.content_hash
        )
        self.assertEqual(first_observation.rgb.pixels, second_observation.rgb.pixels)

    def test_layout_invariants_hold_across_seeds(self) -> None:
        for seed in range(10):
            env = KeysDoorsEnv()
            env.reset(seed)
            maze = env._maze
            self.assertIn((env._player_x, env._player_y), maze)
            self.assertIn((env._target_x, env._target_y), maze)
            self.assertIn((env._key_x, env._key_y), maze)
            self.assertIn((env._door_x, env._door_y), maze)
            # The closed door separates the player (and key) from the target.
            blocked = maze - {(env._door_x, env._door_y)}
            reachable = _bfs_distances((env._player_x, env._player_y), blocked)
            self.assertNotIn((env._target_x, env._target_y), reachable)
            self.assertIn((env._key_x, env._key_y), reachable)

    def test_door_refuses_movement_without_the_key(self) -> None:
        # Walking to the door may legitimately cross the key cell on some
        # seeds; find a seed where the player arrives keyless and verify the
        # door then refuses the final step into it.
        for seed in range(20):
            env = KeysDoorsEnv()
            env.reset(seed)
            door = (env._door_x, env._door_y)
            distances = _bfs_distances(door, env._maze)
            collected_key = False
            while distances[(env._player_x, env._player_y)] > 1:
                outcome = step_toward(env, door)
                if "key_collected" in outcome.events:
                    collected_key = True
                    break
            if collected_key:
                continue
            position_before = (env._player_x, env._player_y)
            outcome = step_toward(env, door)  # would step onto the door cell
            self.assertEqual((env._player_x, env._player_y), position_before)
            self.assertNotIn("door_opened", outcome.events)
            return
        self.fail("no seed in range reached the door without the key")

    def test_key_door_target_sequence_scores(self) -> None:
        env = KeysDoorsEnv()
        env.reset(5)
        key = (env._key_x, env._key_y)
        door = (env._door_x, env._door_y)
        target = (env._target_x, env._target_y)

        events = walk_to(env, key)
        self.assertIn("key_collected", events)
        self.assertEqual(env._has_key, 1)

        events = walk_to(env, door)
        self.assertIn("door_opened", events)
        self.assertEqual(env._door_open, 1)

        outcome_events = walk_to(env, target)
        self.assertIn("target_collected", outcome_events)
        self.assertEqual(env._targets_collected, 1)

    def test_key_pixel_disappears_after_pickup(self) -> None:
        env = KeysDoorsEnv()
        env.reset(5)
        key = (env._key_x, env._key_y)
        walk_to(env, key)
        pixels = env.current_observation.rgb.pixels
        offset = (key[1] * KeysDoorsEnv.GRID_SIZE + key[0]) * 3
        color = (pixels[offset], pixels[offset + 1], pixels[offset + 2])
        self.assertNotEqual(color, KeysDoorsEnv.KEY_RGB)

    def test_one_thousand_step_replay_is_exact(self) -> None:
        source = KeysDoorsEnv(max_ticks=2_000)
        initial = source.reset(83)
        self.assertEqual(
            initial.content_hash,
            "e1561ed2e4320a4b52b6e34713e56c91becf0e5314b097fc182764d82a816fc6",
        )
        root = source.snapshot()
        expected: list[tuple[object, ...]] = []
        for step in range(1_000):
            expected.append(outcome_identity(source, source.step(control_for_step(step))))

        replay = KeysDoorsEnv(max_ticks=2_000)
        replay.restore(root)
        actual = [
            outcome_identity(replay, replay.step(control_for_step(step)))
            for step in range(1_000)
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(
            replay.current_observation.content_hash,
            "daed54d34da1301e6e1c3a602334368692eb8f1ba24d0e21a4063a0b2289f85b",
        )
        self.assertEqual(
            replay.state_hash(),
            "cdcf9ec1918615bec1b51cc14f2d4ee6af6a47c08b8b51a01078bc9b41118dc4",
        )

    def test_snapshot_restore_is_atomic_on_corruption(self) -> None:
        environment = KeysDoorsEnv()
        environment.reset(11)
        for step in range(20):
            environment.step(control_for_step(step))
        snapshot = environment.snapshot()
        before = environment.state_hash()
        damaged = snapshot[:-1] + bytes((snapshot[-1] ^ 0xFF,))
        with self.assertRaises(ValueError):
            environment.restore(damaged)
        self.assertEqual(environment.state_hash(), before)

    def test_restore_preserves_key_and_door_state(self) -> None:
        env = KeysDoorsEnv()
        env.reset(5)
        walk_to(env, (env._key_x, env._key_y))
        walk_to(env, (env._door_x, env._door_y))
        restored = KeysDoorsEnv()
        restored.restore(env.snapshot())
        self.assertEqual(restored._has_key, 1)
        self.assertEqual(restored._door_open, 1)
        self.assertEqual(restored.state_hash(), env.state_hash())

    def test_restore_rejects_inconsistent_key_flags(self) -> None:
        env = KeysDoorsEnv()
        env.reset(5)
        snapshot = bytearray(env.snapshot())
        # Flip has_key (field offset: magic4 + H2 + H2 + I4 + Q8*4 + B*5 + I4
        # + key B*2 + door B*2 = 4+2+2+4+32+5+4+2+2 = 57) and re-checksum.
        from hashlib import sha256

        snapshot[57] = 1
        digest = sha256(bytes(snapshot[:-32])).digest()
        snapshot[-32:] = digest
        with self.assertRaises(ValueError):
            KeysDoorsEnv().restore(bytes(snapshot))

    def test_terminal_boundary_requires_reset_or_restore(self) -> None:
        env = KeysDoorsEnv(max_ticks=4)
        env.reset(1)
        for _ in range(4):
            env.step(GenericControl())
        with self.assertRaises(RuntimeError):
            env.step(GenericControl())
        env.reset(2)
        env.step(GenericControl())

    def test_invalid_control_is_rejected_before_state_changes(self) -> None:
        env = KeysDoorsEnv()
        env.reset(1)
        before = env.state_hash()
        with self.assertRaises(TypeError):
            env.step(object())  # type: ignore[arg-type]
        self.assertEqual(env.state_hash(), before)

    def test_constructor_validation(self) -> None:
        with self.assertRaises(ValueError):
            KeysDoorsEnv(max_ticks=0)
        with self.assertRaises(ValueError):
            KeysDoorsEnv(tick_period_ns=0)
        with self.assertRaises(TypeError):
            KeysDoorsEnv(max_ticks=True)  # type: ignore[arg-type]
        env = KeysDoorsEnv()
        with self.assertRaises(TypeError):
            env.reset(True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            env.reset(-1)

    def test_model_observation_has_only_allowlisted_fields(self) -> None:
        env = KeysDoorsEnv()
        observation = env.reset(9)
        self.assertIsNone(observation.audio_pcm_s16le)
        self.assertEqual(observation.text_inputs, ())
        self.assertEqual(observation.rgb.width, KeysDoorsEnv.GRID_SIZE)
        self.assertEqual(observation.rgb.height, KeysDoorsEnv.GRID_SIZE)


class KeysDoorsClosedLoopTests(unittest.TestCase):
    def test_keys_doors_episode_is_deterministic(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(7,), max_ticks=120)
        factory = lambda: KeysDoorsEnv(max_ticks=120)  # noqa: E731
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
