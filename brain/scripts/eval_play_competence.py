"""Named closed-loop maze-chase play competence vs the scripted planner.

Spark CPU-only. Do not pass --gpus. Measures whether a neural checkpoint
actually plays (pellets / ghost hits / clears), not diagnostic knockouts.

Protocols are labeled and never mixed:

- ``distill-direct-5-9-240``: play-gated distill gate (seeds 5/9, 240 ticks,
  ``exclusive_argmax_wasd_v1``, maze_chase play env).
- ``e-heldout-direct-2001-2020-120``: Variant E seed/tick grid (seeds
  2001-2020, 120 ticks) but **direct** exclusive-argmax play, not the
  original LatentLookaheadPolicy wrap that produced the 5.65-pellet table.

Usage (CUDA hidden):

    python3 -I brain/scripts/eval_play_competence.py \\
      --config brain/configs/training/dgx-play-maze-chase-distill-turn-weighted.toml \\
      --checkpoint /path/to/step-00000032.pt \\
      --out-dir /workspace/run \\
      --agent-id turn-weighted-v1-champion
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

BRAIN_ROOT = Path(__file__).resolve().parents[1]
_ENV_SRC = os.environ.get("IRENE_BRAIN_SRC")
if _ENV_SRC:
    SRC = Path(_ENV_SRC)
elif Path("/workspace/repo/brain/src").is_dir():
    SRC = Path("/workspace/repo/brain/src")
else:
    SRC = BRAIN_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TURN_WEIGHTED_V1_CKPT_SHA256 = (
    "e58f323fb90893c4953b04211002753dd3162f398d1b3b4152390203ba8131cf"
)
DISTILL_SEEDS = (5, 9)
DISTILL_TICKS = 240
E_HELDOUT_SEEDS = tuple(range(2001, 2021))
E_HELDOUT_TICKS = 120
EXCLUSIVE_ARGMAX = "exclusive_argmax_wasd_v1"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8")
    print(json.dumps({"wrote": str(path), "bytes": len(encoded)}, sort_keys=True), flush=True)


def _load_trained_model(*, config_path: Path, checkpoint_path: Path):
    import torch
    from irene_brain.training.config import load_training_config
    from irene_brain.training.factory import build_thesis_model
    from irene_brain.training.objective import ThoughtFieldObjective

    config = load_training_config(config_path)
    model = build_thesis_model(config)
    objective = ThoughtFieldObjective(
        model,
        action_loss_kind=config.objective.action_loss_kind,
        button_support_control_indices=tuple(
            config.objective.button_support_control_indices
        ),
        button_support_weight=config.objective.button_support_weight,
        button_background_weight=config.objective.button_background_weight,
        button_background_tail_mix=config.objective.button_background_tail_mix,
        button_background_tail_temperature=(
            config.objective.button_background_tail_temperature
        ),
        continuous_action_weight=config.objective.continuous_action_weight,
        action_weight=config.objective.action_weight,
        value_weight=config.objective.value_weight,
        world_weight=config.objective.world_weight,
        diversity_weight=config.objective.diversity_weight,
        deadzone_hinge_weight=config.objective.continuous_deadzone_hinge_weight,
        deadzone_hinge_margin=config.objective.continuous_deadzone_hinge_margin,
        opposite_pair_weight=config.objective.opposite_key_pair_weight,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "system_state" not in payload:
        raise ValueError("checkpoint envelope is missing system_state")
    objective.load_state_dict(payload["system_state"]["objective"], strict=True)
    objective.eval()
    return objective.model, config


def _maze_env_factory(max_ticks: int):
    from irene_brain.environments.maze_chase import MazeChaseEnv

    def factory():
        return MazeChaseEnv(
            ghost_count=3,
            ghost_period=2,
            extra_loops=16,
            max_ticks=max_ticks,
        )

    return factory


def _episode_row(episode: object, *, max_ticks: int) -> dict[str, object]:
    ticks = int(episode.ticks_advanced)
    return {
        "seed": int(episode.episode_seed),
        "ticks_advanced": ticks,
        "pellets_eaten": int(episode.pellets_eaten),
        "collisions": int(episode.collisions),
        "reward_sum": float(episode.reward_sum),
        "cleared": ticks < max_ticks,
        "movement_mask_histogram": [
            [int(mask), int(count)] for mask, count in episode.movement_mask_histogram
        ],
    }


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    pellets = [int(row["pellets_eaten"]) for row in rows]
    collisions = [int(row["collisions"]) for row in rows]
    return {
        "episodes": n,
        "pellets_total": sum(pellets),
        "pellets_mean": (sum(pellets) / n) if n else 0.0,
        "collisions_total": sum(collisions),
        "collisions_mean": (sum(collisions) / n) if n else 0.0,
        "clears": sum(1 for row in rows if row["cleared"]),
        "reward_sum": sum(float(row["reward_sum"]) for row in rows),
        "ticks_advanced": sum(int(row["ticks_advanced"]) for row in rows),
    }


def _run_neural(
    model: object,
    *,
    seeds: tuple[int, ...],
    max_ticks: int,
    decode_kind: str,
    protocol: str,
) -> dict[str, object]:
    from irene_brain.evaluation.closed_loop_play import (
        ClosedLoopPlayConfig,
        run_closed_loop_episode,
    )

    device = next(model.parameters()).device
    config = ClosedLoopPlayConfig(
        episode_seeds=seeds,
        max_ticks=max_ticks,
        decode_kind=decode_kind,
    )
    factory = _maze_env_factory(max_ticks)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for seed in seeds:
        episode = run_closed_loop_episode(
            model,
            seed=seed,
            config=config,
            device=device,
            environment_factory=factory,
        )
        row = _episode_row(episode, max_ticks=max_ticks)
        rows.append(row)
        print(
            json.dumps(
                {"protocol": protocol, "agent": "neural", **row},
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "protocol": protocol,
        "agent": "neural",
        "decode_kind": decode_kind,
        "seeds": list(seeds),
        "max_ticks": max_ticks,
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "episodes": rows,
        "summary": _summarize(rows),
    }


def _run_planner(
    *,
    seeds: tuple[int, ...],
    max_ticks: int,
    protocol: str,
) -> dict[str, object]:
    from irene_brain.evaluation.closed_loop_play import (
        ClosedLoopPlayConfig,
        run_policy_closed_loop_episode,
    )
    from irene_brain.evaluation.diagnostic_policies import ScriptedMazeChasePlannerPolicy

    config = ClosedLoopPlayConfig(
        episode_seeds=seeds,
        max_ticks=max_ticks,
        decode_kind=EXCLUSIVE_ARGMAX,
    )
    factory = _maze_env_factory(max_ticks)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for seed in seeds:
        episode = run_policy_closed_loop_episode(
            ScriptedMazeChasePlannerPolicy(),
            seed=seed,
            config=config,
            environment_factory=factory,
        )
        row = _episode_row(episode, max_ticks=max_ticks)
        rows.append(row)
        print(
            json.dumps(
                {"protocol": protocol, "agent": "planner", **row},
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "protocol": protocol,
        "agent": "planner",
        "policy": "diagnostic.scripted_maze_chase_planner.v1",
        "seeds": list(seeds),
        "max_ticks": max_ticks,
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "episodes": rows,
        "summary": _summarize(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--agent-id", default="turn-weighted-v1-champion")
    parser.add_argument(
        "--expect-sha256",
        default=TURN_WEIGHTED_V1_CKPT_SHA256,
        help="Refuse to eval if the checkpoint digest does not match.",
    )
    parser.add_argument(
        "--protocols",
        default="distill,e-heldout",
        help="Comma list: distill, e-heldout",
    )
    arguments = parser.parse_args()

    checkpoint_path = Path(arguments.checkpoint)
    digest = _file_sha256(checkpoint_path)
    if arguments.expect_sha256 and digest != arguments.expect_sha256:
        raise SystemExit(
            f"checkpoint sha256 {digest} != expected {arguments.expect_sha256}"
        )

    import torch

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    model, config = _load_trained_model(
        config_path=Path(arguments.config),
        checkpoint_path=checkpoint_path,
    )
    decode_kind = config.objective.play_decode_kind
    if decode_kind != EXCLUSIVE_ARGMAX:
        raise SystemExit(
            f"play decode is {decode_kind}, expected {EXCLUSIVE_ARGMAX}"
        )

    wanted = {
        part.strip() for part in arguments.protocols.split(",") if part.strip()
    }
    tables: list[dict[str, object]] = []
    if "distill" in wanted:
        protocol = "distill-direct-5-9-240"
        tables.append(
            _run_neural(
                model,
                seeds=DISTILL_SEEDS,
                max_ticks=DISTILL_TICKS,
                decode_kind=decode_kind,
                protocol=protocol,
            )
        )
        tables.append(
            _run_planner(
                seeds=DISTILL_SEEDS,
                max_ticks=DISTILL_TICKS,
                protocol=protocol,
            )
        )
    if "e-heldout" in wanted:
        protocol = "e-heldout-direct-2001-2020-120"
        tables.append(
            _run_neural(
                model,
                seeds=E_HELDOUT_SEEDS,
                max_ticks=E_HELDOUT_TICKS,
                decode_kind=decode_kind,
                protocol=protocol,
            )
        )
        tables.append(
            _run_planner(
                seeds=E_HELDOUT_SEEDS,
                max_ticks=E_HELDOUT_TICKS,
                protocol=protocol,
            )
        )

    payload = {
        "job": "play-competence",
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "agent_id": arguments.agent_id,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": digest,
        "config_path": str(arguments.config),
        "play_decode_kind": decode_kind,
        "notes": {
            "thought_mediated_campaign_weights": "none persisted on Spark",
            "variant_e_checkpoint": "none persisted on Spark",
            "evaluated": "play-gated distill 32-step turn-weighted champion",
            "e_protocol_is_direct_not_lookahead": True,
        },
        "tables": tables,
    }
    out_dir = Path(arguments.out_dir)
    _write(out_dir / "play-competence.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
