"""Same-state action branch audit for the canonical V2.1 maze world.

This is a bounded, deterministic diagnostic.  It opens only TRAIN and
VALIDATION seed namespaces, never TEST, and performs no model construction or
training.  A pixel-only planner advances each factual trajectory.  Before
every factual step, the environment is snapshotted and each of the five
canonical controls (idle/W/A/S/D) is evaluated from that identical state.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable

# Keep local diagnostics CPU-only and single-threaded even if a caller has a
# broader accelerator environment configured.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from irene_brain.data.moving_shapes_dataset import (  # noqa: E402
    DatasetSplit,
    split_episode_seed,
    split_for_episode_seed,
)
from irene_brain.environments.maze_chase import MazeChaseEnv  # noqa: E402
from irene_brain.evaluation.diagnostic_policies import (  # noqa: E402
    ScriptedMazeChasePlannerPolicy,
)
from irene_brain.types import GenericControl, HidKey  # noqa: E402


SCHEMA_VERSION = 1
SEED_NAMESPACE_SIZE = 1 << 62
CANONICAL_SEED_OFFSET = 8_388_608
CANONICAL_TRAIN_EPISODES = 96
CANONICAL_VALIDATION_EPISODES = 32
CANONICAL_EPISODE_TICKS = 16
CANONICAL_BURN_IN_STEPS = 4

ACTION_CONTROLS: tuple[tuple[str, GenericControl], ...] = (
    ("idle", GenericControl()),
    ("W", GenericControl(keys_down=(int(HidKey.W),))),
    ("A", GenericControl(keys_down=(int(HidKey.A),))),
    ("S", GenericControl(keys_down=(int(HidKey.S),))),
    ("D", GenericControl(keys_down=(int(HidKey.D),))),
)
ACTION_NAMES = tuple(name for name, _ in ACTION_CONTROLS)
CONTROL_NAMES = {control: name for name, control in ACTION_CONTROLS}
AUDITED_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VALIDATION)


@dataclass(slots=True)
class ActionCounts:
    branches: int = 0
    safe: int = 0
    caught: int = 0
    rewards: Counter[float] = field(default_factory=Counter)

    def add(self, *, caught: bool, reward: float) -> None:
        self.branches += 1
        self.safe += int(not caught)
        self.caught += int(caught)
        self.rewards[float(reward)] += 1

    def to_dict(self) -> dict[str, object]:
        return {
            "branches": self.branches,
            "safe": self.safe,
            "caught": self.caught,
            "reward_counts": [
                {"reward": reward, "count": count}
                for reward, count in sorted(self.rewards.items())
            ],
        }


@dataclass(slots=True)
class SameStateCounts:
    states: int = 0
    states_with_caught_difference: int = 0
    states_with_reward_difference: int = 0
    states_with_any_outcome_difference: int = 0
    action_pairs: int = 0
    action_pairs_with_caught_difference: int = 0
    action_pairs_with_reward_difference: int = 0
    action_pairs_with_any_outcome_difference: int = 0

    def add(self, outcomes: tuple[tuple[bool, float], ...]) -> None:
        if len(outcomes) != len(ACTION_CONTROLS):
            raise ValueError("one branch outcome is required for every action")
        caught_values = {caught for caught, _ in outcomes}
        reward_values = {reward for _, reward in outcomes}
        self.states += 1
        self.states_with_caught_difference += int(len(caught_values) > 1)
        self.states_with_reward_difference += int(len(reward_values) > 1)
        self.states_with_any_outcome_difference += int(len(set(outcomes)) > 1)
        for left, right in combinations(outcomes, 2):
            caught_differs = left[0] != right[0]
            reward_differs = left[1] != right[1]
            self.action_pairs += 1
            self.action_pairs_with_caught_difference += int(caught_differs)
            self.action_pairs_with_reward_difference += int(reward_differs)
            self.action_pairs_with_any_outcome_difference += int(
                caught_differs or reward_differs
            )

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "states": self.states,
            "states_with_caught_difference": self.states_with_caught_difference,
            "states_with_caught_difference_rate": self._rate(
                self.states_with_caught_difference, self.states
            ),
            "states_with_reward_difference": self.states_with_reward_difference,
            "states_with_reward_difference_rate": self._rate(
                self.states_with_reward_difference, self.states
            ),
            "states_with_any_outcome_difference": (
                self.states_with_any_outcome_difference
            ),
            "states_with_any_outcome_difference_rate": self._rate(
                self.states_with_any_outcome_difference, self.states
            ),
            "action_pairs": self.action_pairs,
            "action_pairs_with_caught_difference": (
                self.action_pairs_with_caught_difference
            ),
            "action_pairs_with_caught_difference_rate": self._rate(
                self.action_pairs_with_caught_difference, self.action_pairs
            ),
            "action_pairs_with_reward_difference": (
                self.action_pairs_with_reward_difference
            ),
            "action_pairs_with_reward_difference_rate": self._rate(
                self.action_pairs_with_reward_difference, self.action_pairs
            ),
            "action_pairs_with_any_outcome_difference": (
                self.action_pairs_with_any_outcome_difference
            ),
            "action_pairs_with_any_outcome_difference_rate": self._rate(
                self.action_pairs_with_any_outcome_difference, self.action_pairs
            ),
        }


@dataclass(slots=True)
class AuditWindow:
    per_action: dict[str, ActionCounts] = field(
        default_factory=lambda: {name: ActionCounts() for name in ACTION_NAMES}
    )
    same_state: SameStateCounts = field(default_factory=SameStateCounts)

    def add(self, outcomes: tuple[tuple[bool, float], ...]) -> None:
        self.same_state.add(outcomes)
        for name, (caught, reward) in zip(ACTION_NAMES, outcomes, strict=True):
            self.per_action[name].add(caught=caught, reward=reward)

    def to_dict(self) -> dict[str, object]:
        return {
            "per_action": {
                name: self.per_action[name].to_dict() for name in ACTION_NAMES
            },
            "same_state_differences": self.same_state.to_dict(),
        }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def _branch_outcomes(
    environment: MazeChaseEnv,
    snapshot: bytes,
) -> tuple[tuple[bool, float], ...]:
    outcomes: list[tuple[bool, float]] = []
    for _, control in ACTION_CONTROLS:
        environment.restore(snapshot)
        outcome = environment.step(control)
        if outcome.applied_control != control:
            raise RuntimeError(
                "canonical V2.1 branch action was not applied exactly"
            )
        outcomes.append(("caught" in outcome.events, float(outcome.reward)))
    environment.restore(snapshot)
    return tuple(outcomes)


def _factual_action_name(control: GenericControl) -> str:
    try:
        return CONTROL_NAMES[control]
    except KeyError as error:
        raise RuntimeError(
            f"pixel planner emitted a non-canonical control: {control!r}"
        ) from error


def audit_split(
    split: DatasetSplit,
    *,
    episode_count: int,
    episode_ticks: int,
    burn_in_steps: int,
    seed_offset: int,
) -> dict[str, object]:
    if split not in AUDITED_SPLITS:
        raise ValueError("same-state audit permits TRAIN and VALIDATION only")

    all_ticks = AuditWindow()
    post_burn = AuditWindow()
    factual_action_counts: Counter[str] = Counter()
    factual_caught = 0
    factual_reward_counts: Counter[float] = Counter()
    factual_ticks = 0
    terminated_episodes = 0
    first_seed = split_episode_seed(split, seed_offset)
    last_seed = split_episode_seed(split, seed_offset + episode_count - 1)

    for local_episode in range(episode_count):
        episode_seed = split_episode_seed(split, seed_offset + local_episode)
        if split_for_episode_seed(episode_seed) is not split:
            raise RuntimeError("episode seed escaped the requested split namespace")
        environment = MazeChaseEnv(
            ghost_count=5,
            ghost_period=1,
            player_period=1,
            extra_loops=16,
            ghost_rule="direct",
            ghost_elroy=False,
            input_delay_ticks=0,
            sticky_direction=False,
            tick_period_ns=MazeChaseEnv.DEFAULT_TICK_PERIOD_NS,
            max_ticks=episode_ticks,
        )
        planner = ScriptedMazeChasePlannerPolicy(
            ghost_period=1,
            input_delay_ticks=0,
            player_period=1,
            ghost_elroy=False,
        )
        planner.reset(episode_seed)
        observation = environment.reset(episode_seed)

        for tick in range(episode_ticks):
            factual_action = planner.act(observation)
            snapshot = environment.snapshot()
            outcomes = _branch_outcomes(environment, snapshot)
            all_ticks.add(outcomes)
            if tick >= burn_in_steps:
                post_burn.add(outcomes)

            factual_outcome = environment.step(factual_action)
            factual_action_counts[_factual_action_name(factual_action)] += 1
            factual_caught += int("caught" in factual_outcome.events)
            factual_reward_counts[float(factual_outcome.reward)] += 1
            factual_ticks += 1
            observation = factual_outcome.observation
            if factual_outcome.terminated:
                terminated_episodes += 1
                break

    return {
        "episode_count": episode_count,
        "seed_range": {"first": first_seed, "last": last_seed},
        "factual_trajectory": {
            "ticks": factual_ticks,
            "planner_action_counts": {
                name: factual_action_counts[name] for name in ACTION_NAMES
            },
            "caught": factual_caught,
            "reward_counts": [
                {"reward": reward, "count": count}
                for reward, count in sorted(factual_reward_counts.items())
            ],
            "terminated_episodes": terminated_episodes,
        },
        "all_ticks": all_ticks.to_dict(),
        "post_burn": post_burn.to_dict(),
    }


def build_payload(
    *,
    train_episodes: int,
    validation_episodes: int,
    episode_ticks: int,
    burn_in_steps: int,
    seed_offset: int,
) -> dict[str, object]:
    if train_episodes > CANONICAL_TRAIN_EPISODES:
        raise ValueError(
            f"train_episodes cannot exceed {CANONICAL_TRAIN_EPISODES}"
        )
    if validation_episodes > CANONICAL_VALIDATION_EPISODES:
        raise ValueError(
            "validation_episodes cannot exceed "
            f"{CANONICAL_VALIDATION_EPISODES}"
        )
    if episode_ticks > CANONICAL_EPISODE_TICKS:
        raise ValueError(
            f"episode_ticks cannot exceed {CANONICAL_EPISODE_TICKS}"
        )
    if burn_in_steps >= episode_ticks:
        raise ValueError("burn_in_steps must be smaller than episode_ticks")
    if seed_offset != CANONICAL_SEED_OFFSET:
        raise ValueError(
            f"seed_offset must remain the canonical {CANONICAL_SEED_OFFSET}"
        )
    if seed_offset + max(train_episodes, validation_episodes) > SEED_NAMESPACE_SIZE:
        raise ValueError("episode range exceeds its split seed namespace")

    split_counts = {
        DatasetSplit.TRAIN: train_episodes,
        DatasetSplit.VALIDATION: validation_episodes,
    }
    splits = {
        split.value: audit_split(
            split,
            episode_count=split_counts[split],
            episode_ticks=episode_ticks,
            burn_in_steps=burn_in_steps,
            seed_offset=seed_offset,
        )
        for split in AUDITED_SPLITS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "v21_same_state_action_branch_audit",
        "purpose": (
            "measure action-conditioned caught/reward variation from identical "
            "pre-action simulator states"
        ),
        "execution": {
            "cpu_only": True,
            "threads": 1,
            "deterministic": True,
            "training_performed": False,
        },
        "environment": {
            "family": "maze_chase",
            "ghost_count": 5,
            "ghost_period": 1,
            "player_period": 1,
            "extra_loops": 16,
            "ghost_rule": "direct",
            "ghost_elroy": False,
            "input_delay_ticks": 0,
            "sticky_direction": False,
            "tick_period_ns": MazeChaseEnv.DEFAULT_TICK_PERIOD_NS,
        },
        "planner": {
            "identity": ScriptedMazeChasePlannerPolicy.identity,
            "inputs": "visible_rgb_only",
            "uses_privileged_state": False,
            "role": "continue_factual_trajectory_only",
        },
        "bounds": {
            "train_episodes": train_episodes,
            "validation_episodes": validation_episodes,
            "episode_ticks": episode_ticks,
            "burn_in_steps": burn_in_steps,
            "seed_offset": seed_offset,
            "actions_per_state": len(ACTION_CONTROLS),
            "maximum_factual_steps": (
                train_episodes + validation_episodes
            )
            * episode_ticks,
            "maximum_branch_steps": (
                train_episodes + validation_episodes
            )
            * episode_ticks
            * len(ACTION_CONTROLS),
        },
        "split_policy": {
            "opened": [split.value for split in AUDITED_SPLITS],
            "test_split_opened": False,
        },
        "actions": list(ACTION_NAMES),
        "splits": splits,
    }


def _write_atomic_json(path: Path, payload: dict[str, object]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--train-episodes",
        type=_positive_int,
        default=CANONICAL_TRAIN_EPISODES,
    )
    parser.add_argument(
        "--validation-episodes",
        type=_positive_int,
        default=CANONICAL_VALIDATION_EPISODES,
    )
    parser.add_argument(
        "--episode-ticks",
        type=_positive_int,
        default=CANONICAL_EPISODE_TICKS,
    )
    parser.add_argument(
        "--burn-in-steps",
        type=_nonnegative_int,
        default=CANONICAL_BURN_IN_STEPS,
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_payload(
        train_episodes=args.train_episodes,
        validation_episodes=args.validation_episodes,
        episode_ticks=args.episode_ticks,
        burn_in_steps=args.burn_in_steps,
        seed_offset=CANONICAL_SEED_OFFSET,
    )
    _write_atomic_json(args.output, payload)
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
