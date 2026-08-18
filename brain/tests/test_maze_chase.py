from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.environments import EnvironmentProtocol, MazeChaseEnv
from irene_brain.environments.junction import _bfs_distances
from irene_brain.evaluation.closed_loop_play import (
    ClosedLoopPlayConfig,
    run_policy_closed_loop_episode,
)
from irene_brain.evaluation.diagnostic_policies import (
    NoOpPolicy,
    ScriptedPelletTeacherPolicy,
)
from irene_brain.types import GenericControl, HidKey

_KEYS_FOR_DELTA = (
    ((0, -1), int(HidKey.W)),
    ((-1, 0), int(HidKey.A)),
    ((0, 1), int(HidKey.S)),
    ((1, 0), int(HidKey.D)),
)


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


def outcome_identity(env: MazeChaseEnv, outcome: object) -> tuple[object, ...]:
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


def greedy_pellet_teacher_step(env: MazeChaseEnv) -> object:
    """Walk the shortest safe path to the nearest remaining pellet.

    Cells within one step of a ghost are treated as blocked for pathing and
    stepping; if that makes every pellet unreachable the teacher falls back
    to the plain shortest path (a catch only costs the test time, not
    correctness).
    """
    player = (env._player_x, env._player_y)
    threatened: set[tuple[int, int]] = set()
    for ghost in env._ghosts:
        for cell, distance in _bfs_distances(ghost, env._maze).items():
            if distance <= 1:
                threatened.add(cell)

    def nearest_goal(blocked: set[tuple[int, int]]) -> tuple[int, int] | None:
        walkable = env._maze - blocked
        if player not in walkable:
            return None
        best = None
        for cell, distance in _bfs_distances(player, walkable).items():
            if distance == 0 or not env._has_pellet(*cell):
                continue
            key = (distance, cell[1], cell[0])
            if best is None or key < best:
                best = key
        return None if best is None else (best[2], best[1])

    goal = nearest_goal(threatened)
    if goal is None:
        goal = nearest_goal(set())
    if goal is None:
        return env.step(GenericControl())

    from_goal = _bfs_distances(goal, env._maze)
    here = from_goal[player]
    fallback = None
    for (dx, dy), key in _KEYS_FOR_DELTA:
        neighbor = (player[0] + dx, player[1] + dy)
        if from_goal.get(neighbor) == here - 1:
            if neighbor in threatened:
                fallback = key
                continue
            return env.step(GenericControl(keys_down=(key,)))
    if fallback is not None:
        return env.step(GenericControl(keys_down=(fallback,)))
    raise AssertionError("no descending neighbor")


class MazeChaseEnvironmentTests(unittest.TestCase):
    def test_environment_satisfies_branchable_protocol(self) -> None:
        self.assertIsInstance(MazeChaseEnv(), EnvironmentProtocol)

    def test_same_seed_has_identical_initial_state_and_pixels(self) -> None:
        first = MazeChaseEnv()
        second = MazeChaseEnv()
        first_observation = first.reset(123456)
        second_observation = second.reset(123456)
        self.assertEqual(first.state_hash(), second.state_hash())
        self.assertEqual(
            first_observation.content_hash, second_observation.content_hash
        )
        self.assertEqual(first_observation.rgb.pixels, second_observation.rgb.pixels)

    def test_pellets_cover_every_corridor_except_the_spawn(self) -> None:
        env = MazeChaseEnv()
        env.reset(5)
        self.assertEqual(env._pellets_remaining, len(env._maze) - 1)
        self.assertFalse(env._has_pellet(1, 1))
        for cell in env._maze:
            if cell != (1, 1):
                self.assertTrue(env._has_pellet(*cell))

    def test_eating_a_pellet_scores_and_clears_the_cell(self) -> None:
        env = MazeChaseEnv(ghost_count=1, ghost_period=MazeChaseEnv.MAX_GHOST_PERIOD)
        env.reset(5)
        before = env._pellets_remaining
        for _ in range(50):
            outcome = greedy_pellet_teacher_step(env)
            if "pellet_eaten" in outcome.events:
                break
        else:
            self.fail("teacher never reached a pellet")
        self.assertEqual(outcome.reward, 1.0)
        self.assertEqual(env._pellets_remaining, before - 1)
        self.assertFalse(env._has_pellet(env._player_x, env._player_y))

    def test_ghost_catches_stationary_player_and_respawns(self) -> None:
        env = MazeChaseEnv(ghost_count=1, ghost_period=1)
        env.reset(7)
        caught = False
        for _ in range(200):
            outcome = env.step(GenericControl())
            if "caught" in outcome.events:
                caught = True
                self.assertEqual(outcome.reward, -10.0)
                break
        self.assertTrue(caught)
        # The ghost ends on the player's old cell, so the respawn lands on a
        # ghost-free corridor cell (the spawn when it is free).
        self.assertIn((env._player_x, env._player_y), env._maze)
        self.assertNotIn((env._player_x, env._player_y), set(env._ghosts))

    def test_greedy_teacher_clears_the_maze_and_terminates(self) -> None:
        env = MazeChaseEnv(
            ghost_count=1,
            ghost_period=MazeChaseEnv.MAX_GHOST_PERIOD,
            max_ticks=5_000,
        )
        env.reset(7)
        cleared = False
        for _ in range(4_000):
            outcome = greedy_pellet_teacher_step(env)
            if outcome.terminated:
                cleared = True
                self.assertIn("cleared", outcome.events)
                break
        self.assertTrue(cleared)
        self.assertEqual(env._pellets_remaining, 0)
        with self.assertRaises(RuntimeError):
            env.step(GenericControl())

    def test_extra_loops_create_cycles(self) -> None:
        env = MazeChaseEnv(extra_loops=16)
        env.reset(5)
        # Count independent cycles: E - V + 1 over the corridor graph.
        cells = set(env._maze)
        edges = 0
        for x, y in cells:
            if (x + 1, y) in cells:
                edges += 1
            if (x, y + 1) in cells:
                edges += 1
        cycles = edges - len(cells) + 1
        self.assertGreaterEqual(cycles, 12)

    def test_walls_refuse_movement(self) -> None:
        env = MazeChaseEnv()
        env.reset(5)
        blocked = None
        for key, (dx, dy) in (
            (HidKey.W, (0, -1)),
            (HidKey.A, (-1, 0)),
            (HidKey.S, (0, 1)),
            (HidKey.D, (1, 0)),
        ):
            if (env._player_x + dx, env._player_y + dy) not in env._maze:
                blocked = key
                break
        self.assertIsNotNone(blocked)
        before = (env._player_x, env._player_y)
        env.step(GenericControl(keys_down=(int(blocked),)))
        self.assertEqual((env._player_x, env._player_y), before)

    def test_shy_ghost_never_catches_a_stationary_player(self) -> None:
        for seed in (7, 5, 9, 13, 42):
            env = MazeChaseEnv(
                ghost_count=1, ghost_period=1, ghost_rule="shy", max_ticks=400
            )
            env.reset(seed)
            for _ in range(300):
                outcome = env.step(GenericControl())
                self.assertNotIn("caught", outcome.events)

    def test_ambush_diverges_from_direct(self) -> None:
        ambush = MazeChaseEnv(
            ghost_count=1, ghost_period=1, ghost_rule="ambush", max_ticks=400
        )
        direct = MazeChaseEnv(
            ghost_count=1, ghost_period=1, ghost_rule="direct", max_ticks=400
        )
        ambush.reset(7)
        direct.reset(7)
        keys = (int(HidKey.W), int(HidKey.D), int(HidKey.S), int(HidKey.A))
        for step in range(120):
            control = GenericControl(keys_down=(keys[step % 4],))
            ambush.step(control)
            direct.step(control)
        self.assertNotEqual(ambush.state_hash(), direct.state_hash())

    def test_mixed_diverges_from_direct(self) -> None:
        mixed = MazeChaseEnv(
            ghost_count=2, ghost_period=1, ghost_rule="mixed", max_ticks=400
        )
        direct = MazeChaseEnv(
            ghost_count=2, ghost_period=1, ghost_rule="direct", max_ticks=400
        )
        mixed.reset(7)
        direct.reset(7)
        keys = (int(HidKey.W), int(HidKey.D), int(HidKey.S), int(HidKey.A))
        for step in range(120):
            control = GenericControl(keys_down=(keys[step % 4],))
            mixed.step(control)
            direct.step(control)
        self.assertNotEqual(mixed.state_hash(), direct.state_hash())

    def test_facing_tracks_control_intent_through_wall_refusals(self) -> None:
        env = MazeChaseEnv()
        env.reset(5)
        blocked = None
        for key, delta in (
            (HidKey.W, (0, -1)),
            (HidKey.A, (-1, 0)),
            (HidKey.S, (0, 1)),
            (HidKey.D, (1, 0)),
        ):
            if (env._player_x + delta[0], env._player_y + delta[1]) not in env._maze:
                blocked = (key, delta)
                break
        self.assertIsNotNone(blocked)
        key, delta = blocked
        before = (env._player_x, env._player_y)
        env.step(GenericControl(keys_down=(int(key),)))
        self.assertEqual((env._player_x, env._player_y), before)
        self.assertEqual((env._player_dx, env._player_dy), delta)

    def test_facing_and_rule_survive_a_snapshot_roundtrip(self) -> None:
        env = MazeChaseEnv(ghost_rule="mixed")
        env.reset(9)
        env.step(GenericControl(keys_down=(int(HidKey.D),)))
        restored = MazeChaseEnv(ghost_rule="mixed")
        restored.restore(env.snapshot())
        self.assertEqual(
            (restored._player_dx, restored._player_dy),
            (env._player_dx, env._player_dy),
        )
        self.assertEqual(restored.state_hash(), env.state_hash())

    def test_restore_rejects_a_ghost_rule_mismatch(self) -> None:
        source = MazeChaseEnv(ghost_rule="direct")
        source.reset(5)
        snapshot = source.snapshot()
        with self.assertRaises(ValueError):
            MazeChaseEnv(ghost_rule="shy").restore(snapshot)

    def test_player_period_gates_movement_and_input_sampling(self) -> None:
        env = MazeChaseEnv(player_period=2, max_ticks=100)
        env.reset(5)
        open_key = None
        for key, delta in (
            (HidKey.D, (1, 0)),
            (HidKey.S, (0, 1)),
            (HidKey.W, (0, -1)),
            (HidKey.A, (-1, 0)),
        ):
            if (env._player_x + delta[0], env._player_y + delta[1]) in env._maze:
                open_key = (key, delta)
                break
        self.assertIsNotNone(open_key)
        key, delta = open_key
        start = (env._player_x, env._player_y)
        env.step(GenericControl(keys_down=(int(key),)))  # tick 0: a move tick
        self.assertEqual(
            (env._player_x, env._player_y),
            (start[0] + delta[0], start[1] + delta[1]),
        )
        after_move = (env._player_x, env._player_y)
        env.step(GenericControl(keys_down=(int(key),)))  # tick 1: gated
        self.assertEqual((env._player_x, env._player_y), after_move)
        # A different direction pressed only on a gated tick changes neither
        # the position nor the facing: input is sampled on move ticks only.
        other = HidKey.W if key != HidKey.W else HidKey.A
        env.step(GenericControl(keys_down=(int(key),)))  # tick 2: a move tick
        env.step(GenericControl(keys_down=(int(other),)))  # tick 3: gated
        self.assertEqual((env._player_dx, env._player_dy), delta)

    def test_elroy_curve_speeds_ghosts_up_after_half_the_pellets(self) -> None:
        env = MazeChaseEnv(ghost_period=2, ghost_elroy=True)
        env.reset(5)
        self.assertEqual(env._effective_ghost_period(), 2)
        total = len(env._maze) - 1
        for y in range(env.GRID_SIZE):
            for x in range(env.GRID_SIZE):
                if env._pellets_remaining * 2 <= total:
                    break
                if env._has_pellet(x, y):
                    env._eat_pellet(x, y)
                    env._pellets_eaten += 1
        self.assertEqual(env._effective_ghost_period(), 1)
        # The curve derives from pellets_remaining, so a restore recomputes it.
        restored = MazeChaseEnv(ghost_period=2, ghost_elroy=True)
        restored.restore(env.snapshot())
        self.assertEqual(restored._effective_ghost_period(), 1)
        # Without the flag the period never changes.
        plain = MazeChaseEnv(ghost_period=2, ghost_elroy=False)
        plain.reset(5)
        self.assertEqual(plain._effective_ghost_period(), 2)

    def test_restore_rejects_speed_curve_mismatches(self) -> None:
        source = MazeChaseEnv(
            player_period=3, ghost_elroy=True, ghost_rule="mixed"
        )
        source.reset(9)
        snapshot = source.snapshot()
        with self.assertRaises(ValueError):
            MazeChaseEnv(
                player_period=2, ghost_elroy=True, ghost_rule="mixed"
            ).restore(snapshot)
        with self.assertRaises(ValueError):
            MazeChaseEnv(
                player_period=3, ghost_elroy=False, ghost_rule="mixed"
            ).restore(snapshot)

    def test_input_delay_postpones_controls(self) -> None:
        env = MazeChaseEnv(
            input_delay_ticks=2,
            ghost_period=MazeChaseEnv.MAX_GHOST_PERIOD,
            max_ticks=100,
        )
        env.reset(5)
        open_key = None
        for key, delta in (
            (HidKey.D, (1, 0)),
            (HidKey.S, (0, 1)),
            (HidKey.W, (0, -1)),
            (HidKey.A, (-1, 0)),
        ):
            if (env._player_x + delta[0], env._player_y + delta[1]) in env._maze:
                open_key = (key, delta)
                break
        self.assertIsNotNone(open_key)
        key, delta = open_key
        start = (env._player_x, env._player_y)
        env.step(GenericControl(keys_down=(int(key),)))  # enters the queue
        self.assertEqual((env._player_x, env._player_y), start)
        env.step(GenericControl())  # still draining zeros
        self.assertEqual((env._player_x, env._player_y), start)
        outcome = env.step(GenericControl())  # the press takes effect
        self.assertEqual(
            (env._player_x, env._player_y),
            (start[0] + delta[0], start[1] + delta[1]),
        )
        self.assertEqual(outcome.applied_control.keys_down, (int(key),))

    def test_sticky_direction_persists_until_replaced(self) -> None:
        kwargs = dict(  # noqa: C408
            ghost_period=MazeChaseEnv.MAX_GHOST_PERIOD,
            max_ticks=100,
        )
        sticky = MazeChaseEnv(sticky_direction=True, **kwargs)
        plain = MazeChaseEnv(sticky_direction=False, **kwargs)
        sticky.reset(5)
        plain.reset(5)
        press = GenericControl(keys_down=(int(HidKey.S),))
        sticky.step(press)
        plain.step(press)
        # Released on both: the sticky mask persists, the plain mask clears.
        sticky_outcome = sticky.step(GenericControl())
        plain_outcome = plain.step(GenericControl())
        self.assertEqual(sticky_outcome.applied_control.keys_down, (int(HidKey.S),))
        self.assertEqual(plain_outcome.applied_control.keys_down, ())
        # A new direction replaces the sticky one.
        outcome = sticky.step(GenericControl(keys_down=(int(HidKey.W),)))
        self.assertEqual(outcome.applied_control.keys_down, (int(HidKey.W),))

    def test_delay_and_sticky_survive_a_snapshot_roundtrip(self) -> None:
        env = MazeChaseEnv(
            input_delay_ticks=3,
            sticky_direction=True,
            ghost_rule="mixed",
            player_period=2,
            ghost_elroy=True,
        )
        env.reset(11)
        keys = (int(HidKey.W), int(HidKey.D), int(HidKey.S), int(HidKey.A))
        for step in range(60):
            env.step(GenericControl(keys_down=(keys[step % 4],)))
        restored = MazeChaseEnv(
            input_delay_ticks=3,
            sticky_direction=True,
            ghost_rule="mixed",
            player_period=2,
            ghost_elroy=True,
        )
        restored.restore(env.snapshot())
        self.assertEqual(restored.state_hash(), env.state_hash())
        for step in range(60, 120):
            control = GenericControl(keys_down=(keys[step % 4],))
            self.assertEqual(restored.step(control), env.step(control))
        self.assertEqual(restored.state_hash(), env.state_hash())

    def test_restore_rejects_input_pipeline_mismatches(self) -> None:
        source = MazeChaseEnv(input_delay_ticks=3, sticky_direction=True)
        source.reset(9)
        snapshot = source.snapshot()
        with self.assertRaises(ValueError):
            MazeChaseEnv(input_delay_ticks=2, sticky_direction=True).restore(snapshot)
        with self.assertRaises(ValueError):
            MazeChaseEnv(input_delay_ticks=3, sticky_direction=False).restore(snapshot)

    def test_restore_rejects_tampered_delay_queue_bits(self) -> None:
        from hashlib import sha256

        env = MazeChaseEnv(input_delay_ticks=2)
        env.reset(5)
        snapshot = bytearray(env.snapshot())
        # Queue bytes sit immediately before the trailing digest.
        snapshot[-33] |= 0xF0
        snapshot[-32:] = sha256(bytes(snapshot[:-32])).digest()
        with self.assertRaises(ValueError):
            MazeChaseEnv(input_delay_ticks=2).restore(bytes(snapshot))

    def test_one_thousand_step_replay_is_exact(self) -> None:
        source = MazeChaseEnv(max_ticks=2_000)
        initial = source.reset(83)
        self.assertEqual(
            initial.content_hash,
            "a4fffc1298d81dc725e3744530045cdd9f611df08573305a4ec9d4fda00083ca",
        )
        root = source.snapshot()
        expected: list[tuple[object, ...]] = []
        for step in range(1_000):
            expected.append(outcome_identity(source, source.step(control_for_step(step))))

        replay = MazeChaseEnv(max_ticks=2_000)
        replay.restore(root)
        actual = [
            outcome_identity(replay, replay.step(control_for_step(step)))
            for step in range(1_000)
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(
            replay.current_observation.content_hash,
            "272cb411297cf84d79758990449449bfb146cb86ce7b5393880f87e931355bcd",
        )
        self.assertEqual(
            replay.state_hash(),
            "3613c576422f0d54c1631dd00760a458b9358f2b433c47a8e6b40ab84f1931ea",
        )

    def test_snapshot_restore_is_atomic_on_corruption(self) -> None:
        environment = MazeChaseEnv()
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
        source = MazeChaseEnv(ghost_count=2)
        source.reset(5)
        snapshot = source.snapshot()
        with self.assertRaises(ValueError):
            MazeChaseEnv(ghost_count=3).restore(snapshot)
        with self.assertRaises(ValueError):
            MazeChaseEnv(ghost_count=2, ghost_period=1).restore(snapshot)
        with self.assertRaises(ValueError):
            MazeChaseEnv(ghost_count=2, max_ticks=999).restore(snapshot)

    def test_restore_validates_pellet_accounting(self) -> None:
        env = MazeChaseEnv()
        env.reset(5)
        snapshot = bytearray(env.snapshot())
        # Eat one pellet for real, then present the pre-eat snapshot with a
        # tampered eaten counter: accounting must not add up.
        env.step(GenericControl(keys_down=(int(HidKey.D),)))
        from hashlib import sha256

        # pellets_eaten offset: 4+2+2+4+32+4 = 48 (after the four B fields).
        eaten = int.from_bytes(snapshot[48:52], "little")
        snapshot[48:52] = (eaten + 1).to_bytes(4, "little")
        snapshot[-32:] = sha256(bytes(snapshot[:-32])).digest()
        with self.assertRaises(ValueError):
            MazeChaseEnv().restore(bytes(snapshot))

    def test_terminal_boundary_requires_reset_or_restore(self) -> None:
        env = MazeChaseEnv(max_ticks=4)
        env.reset(1)
        for _ in range(4):
            env.step(GenericControl())
        with self.assertRaises(RuntimeError):
            env.step(GenericControl())
        env.reset(2)
        env.step(GenericControl())

    def test_invalid_control_is_rejected_before_state_changes(self) -> None:
        env = MazeChaseEnv()
        env.reset(1)
        before = env.state_hash()
        with self.assertRaises(TypeError):
            env.step(object())  # type: ignore[arg-type]
        self.assertEqual(env.state_hash(), before)

    def test_constructor_validation(self) -> None:
        with self.assertRaises(TypeError):
            MazeChaseEnv(ghost_count=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MazeChaseEnv(ghost_count=0)
        with self.assertRaises(ValueError):
            MazeChaseEnv(ghost_count=9)
        with self.assertRaises(ValueError):
            MazeChaseEnv(ghost_period=0)
        with self.assertRaises(ValueError):
            MazeChaseEnv(extra_loops=-1)
        with self.assertRaises(ValueError):
            MazeChaseEnv(extra_loops=MazeChaseEnv.MAX_EXTRA_LOOPS + 1)
        with self.assertRaises(TypeError):
            MazeChaseEnv(extra_loops=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MazeChaseEnv(max_ticks=0)
        with self.assertRaises(ValueError):
            MazeChaseEnv(tick_period_ns=0)
        with self.assertRaises(ValueError):
            MazeChaseEnv(ghost_rule="random")
        with self.assertRaises(ValueError):
            MazeChaseEnv(ghost_rule=3)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MazeChaseEnv(player_period=0)
        with self.assertRaises(ValueError):
            MazeChaseEnv(player_period=MazeChaseEnv.MAX_PLAYER_PERIOD + 1)
        with self.assertRaises(TypeError):
            MazeChaseEnv(player_period=True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            MazeChaseEnv(ghost_elroy=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            MazeChaseEnv(input_delay_ticks=-1)
        with self.assertRaises(ValueError):
            MazeChaseEnv(input_delay_ticks=MazeChaseEnv.MAX_INPUT_DELAY + 1)
        with self.assertRaises(TypeError):
            MazeChaseEnv(input_delay_ticks=True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            MazeChaseEnv(sticky_direction=1)  # type: ignore[arg-type]
        env = MazeChaseEnv()
        with self.assertRaises(TypeError):
            env.reset(True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            env.reset(-1)

    def test_model_observation_has_only_allowlisted_fields(self) -> None:
        env = MazeChaseEnv()
        observation = env.reset(9)
        self.assertIsNone(observation.audio_pcm_s16le)
        self.assertEqual(observation.text_inputs, ())
        self.assertEqual(observation.rgb.width, MazeChaseEnv.GRID_SIZE)
        self.assertEqual(observation.rgb.height, MazeChaseEnv.GRID_SIZE)


class MazeChaseClosedLoopTests(unittest.TestCase):
    def test_noop_policy_gets_caught_and_never_clears(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(7,), max_ticks=200)
        report = run_policy_closed_loop_episode(
            NoOpPolicy(),
            seed=7,
            config=config,
            environment_factory=lambda: MazeChaseEnv(
                ghost_count=1, ghost_period=1, max_ticks=200
            ),
        )
        self.assertGreaterEqual(report.collisions, 1)

    def test_maze_chase_episode_is_deterministic(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(7,), max_ticks=120)
        factory = lambda: MazeChaseEnv(  # noqa: E731
            ghost_count=2, ghost_period=2, max_ticks=120
        )
        first = run_policy_closed_loop_episode(
            NoOpPolicy(), seed=7, config=config, environment_factory=factory
        )
        second = run_policy_closed_loop_episode(
            NoOpPolicy(), seed=7, config=config, environment_factory=factory
        )
        self.assertEqual(first, second)

    def test_scripted_pellet_teacher_eats_from_pixels_only(self) -> None:
        config = ClosedLoopPlayConfig(episode_seeds=(5,), max_ticks=240)
        report = run_policy_closed_loop_episode(
            ScriptedPelletTeacherPolicy(),
            seed=5,
            config=config,
            environment_factory=lambda: MazeChaseEnv(
                ghost_count=1,
                ghost_period=MazeChaseEnv.MAX_GHOST_PERIOD,
                max_ticks=240,
            ),
        )
        self.assertGreater(report.reward_sum, 0.0)
        self.assertEqual(report.opposite_conflicts, 0)


if __name__ == "__main__":
    unittest.main()
