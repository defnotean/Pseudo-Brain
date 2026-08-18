"""Play-gated maze-chase closed-loop measurement for the distill campaign.

This is the campaign instrument, not a teacher-agreement metric. The frozen
no-op floor comes from the 2026-08-18 transfer-battery constant-LR table
(seeds 5/9, 240 ticks): reward_sum -161, collisions 17, zero pellets.
Play has moved only when reward exceeds that floor. Action-loss drops
alone are a fail for this campaign.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..environments.maze_chase import MazeChaseEnv
from ..evaluation.closed_loop_play import (
    ClosedLoopPlayConfig,
    evaluate_closed_loop_play,
)

CAMPAIGN_ID = "play_gated_maze_chase_distill_v1"
PLAY_SEEDS = (5, 9)
PLAY_TICKS = 240
NOOP_REWARD_FLOOR = -161.0
NOOP_COLLISION_FLOOR = 17


def maze_chase_play_config() -> ClosedLoopPlayConfig:
    return ClosedLoopPlayConfig(episode_seeds=PLAY_SEEDS, max_ticks=PLAY_TICKS)


def maze_chase_environment_factory() -> MazeChaseEnv:
    return MazeChaseEnv(
        ghost_count=3,
        ghost_period=2,
        extra_loops=16,
        max_ticks=PLAY_TICKS,
    )


def evaluate_maze_chase_play(model: object) -> dict[str, object]:
    report = evaluate_closed_loop_play(
        model,
        config=maze_chase_play_config(),
        model_description=CAMPAIGN_ID,
        environment_factory=maze_chase_environment_factory,
    )
    totals = report.to_dict()["totals"]
    reward = float(totals["reward_sum"])
    collisions = int(totals["collisions"])
    pellets = int(totals["targets_collected"])
    play_moved = reward > NOOP_REWARD_FLOOR
    return {
        "campaign_id": CAMPAIGN_ID,
        "play_seeds": list(PLAY_SEEDS),
        "play_ticks": PLAY_TICKS,
        "noop_reward_floor": NOOP_REWARD_FLOOR,
        "noop_collision_floor": NOOP_COLLISION_FLOOR,
        "reward_sum": reward,
        "collisions": collisions,
        "pellets_eaten": pellets,
        "decisions_rejected": int(totals["decisions_rejected"]),
        "play_moved": play_moved,
        "gate": "passed" if play_moved else "failed",
        "report_sha256": report.sha256,
    }


def write_maze_chase_play_gate(model: object, run_dir: Path) -> dict[str, object]:
    payload = evaluate_maze_chase_play(model)
    path = Path(run_dir) / "play-gate.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps({"play_gate": payload}, allow_nan=False, sort_keys=True),
        flush=True,
    )
    return payload


__all__ = [
    "CAMPAIGN_ID",
    "NOOP_COLLISION_FLOOR",
    "NOOP_REWARD_FLOOR",
    "PLAY_SEEDS",
    "PLAY_TICKS",
    "evaluate_maze_chase_play",
    "maze_chase_environment_factory",
    "maze_chase_play_config",
    "write_maze_chase_play_gate",
]
