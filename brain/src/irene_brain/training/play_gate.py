"""Play-gated maze-chase closed-loop measurement for the distill campaign.

This is the campaign instrument, not a teacher-agreement metric. The frozen
no-op floor comes from the 2026-08-18 transfer-battery constant-LR table
(seeds 5/9, 240 ticks): reward_sum -161, collisions 17. Play has moved only
when reward exceeds that floor. Action-loss drops alone are a fail.

``pellets_eaten`` counts maze_chase ``pellet_eaten`` events. Historical
probe JSON files mapped the world-flat ``target_collected`` column and
always printed 0; reward arithmetic on those runs is the pellet record.

Campaign success is eating pellets well above the no-op / sticky-D band
(~9–10 pellets) or clearing a maze. ``play_moved`` remains the thin
reward-floor instrument; sticky D and idle no-op fail the campaign gate
even when reward_sum is one catch better than no-op.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..environments.maze_chase import MazeChaseEnv
from ..evaluation.closed_loop_play import (
    ClosedLoopPlayConfig,
    EXCLUSIVE_ARGMAX_WASD_IDLE_MARGIN,
    EXCLUSIVE_ARGMAX_WASD_V1,
    INDEPENDENT_LOGIT_GT_ZERO_V1,
    evaluate_closed_loop_play,
)

CAMPAIGN_ID = "play_gated_maze_chase_distill_v1"
PLAY_SEEDS = (5, 9)
PLAY_TICKS = 240
NOOP_REWARD_FLOOR = -161.0
NOOP_COLLISION_FLOOR = 17
# No-op eats 9 pellets; sticky D ate ~10. Campaign pass is well above that.
CAMPAIGN_PELLET_FLOOR = 32
IDLE_MOVEMENT_MASK = 0
D_ONLY_MOVEMENT_MASK = 8
PLAY_PEAK_V1 = "play_peak_v1"
# Exclusive-CE / 128-step turn-weighted sticky-S band. A drop into this
# band after a better peak is an early-stop, not a 128-scale.
STICKY_PELLET_BAND = 20
# Stop when pellets fall this many below the kept peak.
PLAY_PEAK_DROP_PELLETS = 8


def maze_chase_play_config(
    decode_kind: str = INDEPENDENT_LOGIT_GT_ZERO_V1,
) -> ClosedLoopPlayConfig:
    return ClosedLoopPlayConfig(
        episode_seeds=PLAY_SEEDS,
        max_ticks=PLAY_TICKS,
        decode_kind=decode_kind,
    )


def maze_chase_environment_factory() -> MazeChaseEnv:
    return MazeChaseEnv(
        ghost_count=3,
        ghost_period=2,
        extra_loops=16,
        max_ticks=PLAY_TICKS,
    )


def _is_sticky_or_idle(histogram: dict[int, int]) -> bool:
    active = {mask for mask, count in histogram.items() if count > 0}
    return not active or active <= {IDLE_MOVEMENT_MASK, D_ONLY_MOVEMENT_MASK}


def movement_histogram_from_payload(payload: dict[str, object]) -> dict[int, int]:
    raw = payload.get("movement_mask_histogram", [])
    if not isinstance(raw, list):
        raise ValueError("movement_mask_histogram must be a list of [mask, count] pairs")
    histogram: dict[int, int] = {}
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("movement_mask_histogram entries must be [mask, count]")
        mask, count = item
        histogram[int(mask)] = int(count)
    return histogram


def is_one_key_wasd(histogram: dict[int, int]) -> bool:
    """True when at most one non-idle WASD mask has any ticks."""

    active = {mask for mask, count in histogram.items() if mask != 0 and count > 0}
    return len(active) <= 1


def play_score(payload: dict[str, object]) -> tuple[int, int]:
    """Best play is more pellets, then fewer collisions."""

    return (int(payload["pellets_eaten"]), -int(payload["collisions"]))


def play_peak_should_stop(
    current: dict[str, object],
    peak: dict[str, object] | None,
) -> bool:
    """Stop when closed-loop play drops from the kept peak.

    Frozen ``play_peak_v1``: pellets fall by ``PLAY_PEAK_DROP_PELLETS`` or
    into the sticky-S band, the histogram collapses to one WASD key, or
    play becomes sticky/idle after a mixed peak. No peak yet means continue.
    """

    if peak is None:
        return False
    current_pellets = int(current["pellets_eaten"])
    peak_pellets = int(peak["pellets_eaten"])
    if current_pellets <= peak_pellets - PLAY_PEAK_DROP_PELLETS:
        return True
    if current_pellets <= STICKY_PELLET_BAND and peak_pellets > STICKY_PELLET_BAND:
        return True
    current_hist = movement_histogram_from_payload(current)
    peak_hist = movement_histogram_from_payload(peak)
    if is_one_key_wasd(current_hist) and not is_one_key_wasd(peak_hist):
        return True
    if bool(current["sticky_or_idle"]) and not bool(peak["sticky_or_idle"]):
        return True
    return False


def evaluate_maze_chase_play(
    model: object,
    *,
    decode_kind: str = INDEPENDENT_LOGIT_GT_ZERO_V1,
) -> dict[str, object]:
    report = evaluate_closed_loop_play(
        model,
        config=maze_chase_play_config(decode_kind),
        model_description=CAMPAIGN_ID,
        environment_factory=maze_chase_environment_factory,
    )
    totals = report.to_dict()["totals"]
    reward = float(totals["reward_sum"])
    collisions = int(totals["collisions"])
    pellets = int(totals["pellets_eaten"])
    histogram: dict[int, int] = {}
    for episode in report.episodes:
        for mask, count in episode.movement_mask_histogram:
            histogram[int(mask)] = histogram.get(int(mask), 0) + int(count)
    play_moved = reward > NOOP_REWARD_FLOOR
    sticky_or_idle = _is_sticky_or_idle(histogram)
    # maze_chase only terminates early on a clear. Early ticks without
    # pellets are a crash/abort, not a win, so clears are reported but
    # campaign success still requires the pellet floor.
    mazes_cleared = sum(
        1 for episode in report.episodes if episode.ticks_advanced < PLAY_TICKS
    )
    campaign_success = pellets >= CAMPAIGN_PELLET_FLOOR and not sticky_or_idle
    payload: dict[str, object] = {
        "campaign_id": CAMPAIGN_ID,
        "play_seeds": list(PLAY_SEEDS),
        "play_ticks": PLAY_TICKS,
        "noop_reward_floor": NOOP_REWARD_FLOOR,
        "noop_collision_floor": NOOP_COLLISION_FLOOR,
        "campaign_pellet_floor": CAMPAIGN_PELLET_FLOOR,
        "reward_sum": reward,
        "collisions": collisions,
        "pellets_eaten": pellets,
        "mazes_cleared": mazes_cleared,
        # W=bit0, A=bit1, S=bit2, D=bit3. Mask 8 is D-only; mask 0 is idle.
        "movement_mask_histogram": [
            [mask, count] for mask, count in sorted(histogram.items())
        ],
        "decisions_rejected": int(totals["decisions_rejected"]),
        "play_moved": play_moved,
        "sticky_or_idle": sticky_or_idle,
        "campaign_success": campaign_success,
        "gate": "passed" if campaign_success else "failed",
        "play_decode_kind": decode_kind,
        "report_sha256": report.sha256,
    }
    if decode_kind == EXCLUSIVE_ARGMAX_WASD_V1:
        payload["play_decode_idle_margin"] = EXCLUSIVE_ARGMAX_WASD_IDLE_MARGIN
    return payload


def write_maze_chase_play_gate(
    model: object,
    run_dir: Path,
    *,
    decode_kind: str = INDEPENDENT_LOGIT_GT_ZERO_V1,
) -> dict[str, object]:
    payload = evaluate_maze_chase_play(model, decode_kind=decode_kind)
    path = Path(run_dir) / "play-gate.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps({"play_gate": payload}, allow_nan=False, sort_keys=True),
        flush=True,
    )
    return payload


__all__ = [
    "CAMPAIGN_ID",
    "CAMPAIGN_PELLET_FLOOR",
    "NOOP_COLLISION_FLOOR",
    "NOOP_REWARD_FLOOR",
    "PLAY_PEAK_DROP_PELLETS",
    "PLAY_PEAK_V1",
    "PLAY_SEEDS",
    "PLAY_TICKS",
    "STICKY_PELLET_BAND",
    "evaluate_maze_chase_play",
    "is_one_key_wasd",
    "maze_chase_environment_factory",
    "maze_chase_play_config",
    "movement_histogram_from_payload",
    "play_peak_should_stop",
    "play_score",
    "write_maze_chase_play_gate",
]
