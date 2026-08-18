"""Named CPU-only campaign jobs for play-gated maze-chase distill.

Spark host/CPU work: planner rollouts, teacher-window statistics,
exclusive-direction audits, checkpoint play-gate, and cheap thoughtlet
dumps. These jobs must not request the GB10. Do not import this module
from the Phase-0 package surface.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

JOBS = ("planner", "teacher-hist", "teacher-exclusive", "play-gate", "thoughtlets")
WASD = (
    ("w", 26),
    ("a", 4),
    ("s", 22),
    ("d", 7),
)


def _src_root() -> Path:
    env = os.environ.get("IRENE_BRAIN_SRC")
    if env:
        return Path(env)
    container = Path("/workspace/repo/brain/src")
    if container.is_dir():
        return container
    return Path(__file__).resolve().parents[1] / "src"


def _ensure_src() -> None:
    src = _src_root()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8")
    print(json.dumps({"wrote": str(path), "bytes": len(encoded)}, sort_keys=True), flush=True)


def _wasd_counts(control: object) -> tuple[str, int]:
    keys = set(getattr(control, "keys_down", ()))
    active = [name for name, hid in WASD if hid in keys]
    return (",".join(active) if active else "idle", len(active))


def _maze_batch_config(*, episode_horizon: int, sequences: int):
    from irene_brain.training.batches import MazeChaseBatchConfig

    return MazeChaseBatchConfig(
        train_sequences=sequences,
        validation_sequences=max(2, sequences // 4),
        test_sequences=2,
        sequence_length=8,
        burn_in_steps=2,
        seed_offset=0,
        ghost_count=3,
        ghost_period=2,
        extra_loops=16,
        tick_period_ns=16_666_667,
        discount=0.99,
        episode_horizon=episode_horizon,
    )


def _histogram_from_source(source, *, split: str) -> dict[str, object]:
    counts: Counter[str] = Counter()
    exclusive = 0
    idle = 0
    multi = 0
    ticks = 0
    for batch in source.iter_batches(
        split=split,
        epoch=0,
        start_batch=0,
        batch_size=1,
        max_batches=source.batches_per_epoch(split=split, batch_size=1),
    ):
        for sequence in batch.sequences:
            for transition in sequence.transitions:
                label, n_active = _wasd_counts(transition.action_target)
                counts[label] += 1
                ticks += 1
                if n_active == 0:
                    idle += 1
                elif n_active == 1:
                    exclusive += 1
                else:
                    multi += 1
    return {
        "ticks": ticks,
        "exclusive_wasd": exclusive,
        "idle": idle,
        "multi_wasd": multi,
        "labels": dict(sorted(counts.items())),
        "rates": {
            name: (counts.get(name, 0) / ticks if ticks else 0.0)
            for name, _hid in WASD
        }
        | {"idle": idle / ticks if ticks else 0.0},
        "source_manifest_sha256": source.manifest_sha256,
    }


def job_planner(out_dir: Path) -> None:
    from irene_brain.evaluation.closed_loop_play import run_policy_closed_loop_episode
    from irene_brain.evaluation.diagnostic_policies import ScriptedMazeChasePlannerPolicy
    from irene_brain.training.play_gate import (
        PLAY_SEEDS,
        PLAY_TICKS,
        maze_chase_environment_factory,
        maze_chase_play_config,
    )

    config = maze_chase_play_config()
    episodes = []
    for seed in PLAY_SEEDS:
        episode = run_policy_closed_loop_episode(
            ScriptedMazeChasePlannerPolicy(),
            seed=seed,
            config=config,
            environment_factory=maze_chase_environment_factory,
        )
        episodes.append(
            {
                "seed": seed,
                "ticks_advanced": episode.ticks_advanced,
                "reward_sum": episode.reward_sum,
                "pellets_eaten": episode.pellets_eaten,
                "collisions": episode.collisions,
                "cleared": episode.ticks_advanced < PLAY_TICKS,
                "movement_mask_histogram": [
                    [mask, count] for mask, count in episode.movement_mask_histogram
                ],
            }
        )
    _write(
        out_dir / "planner-closed-loop.json",
        {
            "job": "planner",
            "campaign_id": "play_gated_maze_chase_distill_v1",
            "policy": "diagnostic.scripted_maze_chase_planner.v1",
            "play_seeds": list(PLAY_SEEDS),
            "play_ticks": PLAY_TICKS,
            "episodes": episodes,
        },
    )


def job_teacher_hist(out_dir: Path) -> None:
    from irene_brain.training.batches import MazeChaseBatchSource

    spawn = MazeChaseBatchSource(_maze_batch_config(episode_horizon=0, sequences=16))
    windows = MazeChaseBatchSource(_maze_batch_config(episode_horizon=240, sequences=16))
    _write(
        out_dir / "teacher-wasd-histogram.json",
        {
            "job": "teacher-hist",
            "campaign_id": "play_gated_maze_chase_distill_v1",
            "spawn": {
                split: _histogram_from_source(spawn, split=split)
                for split in ("train", "validation")
            },
            "episode_windows": {
                split: _histogram_from_source(windows, split=split)
                for split in ("train", "validation")
            },
        },
    )


def job_teacher_exclusive(out_dir: Path) -> None:
    from irene_brain.training.batches import MazeChaseBatchSource

    source = MazeChaseBatchSource(_maze_batch_config(episode_horizon=240, sequences=32))
    hist = _histogram_from_source(source, split="train")
    _write(
        out_dir / "teacher-exclusive-audit.json",
        {
            "job": "teacher-exclusive",
            "campaign_id": "play_gated_maze_chase_distill_v1",
            "hypothesis": "planner labels are one of WASD or idle, never two keys",
            "train": hist,
            "exclusive_or_idle": hist["multi_wasd"] == 0,
        },
    )


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


def job_play_gate(out_dir: Path, *, config_path: Path, checkpoint_path: Path) -> None:
    from irene_brain.training.play_gate import write_maze_chase_play_gate

    model, config = _load_trained_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    payload = write_maze_chase_play_gate(
        model,
        out_dir,
        decode_kind=config.objective.play_decode_kind,
    )
    _write(
        out_dir / "cpu-play-gate-meta.json",
        {
            "job": "play-gate",
            "campaign_id": "play_gated_maze_chase_distill_v1",
            "device": "cpu",
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "play_decode_kind": config.objective.play_decode_kind,
            "gate": payload.get("gate"),
            "reward_sum": payload.get("reward_sum"),
            "pellets_eaten": payload.get("pellets_eaten"),
            "collisions": payload.get("collisions"),
            "movement_mask_histogram": payload.get("movement_mask_histogram"),
        },
    )


def job_thoughtlets(
    out_dir: Path,
    *,
    config_path: Path,
    checkpoint_path: Path,
    ticks: int,
) -> None:
    import torch
    from irene_brain.training.batches import control_to_vector
    from irene_brain.training.objective import (
        _rgb_tensor,
        deterministic_eval_thought_noise,
    )
    from irene_brain.training.play_gate import maze_chase_environment_factory

    model, _config = _load_trained_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )
    env = maze_chase_environment_factory()
    observation = env.reset(5)
    device = next(model.parameters()).device
    thought_noise = deterministic_eval_thought_noise(
        thoughtlets=model.config.thoughtlets,
        width=model.config.core_width,
        batch_size=1,
        device=device,
    )
    state = None
    rows = []
    with torch.no_grad():
        for tick in range(ticks):
            pixels = _rgb_tensor(
                (observation.rgb,),
                device=device,
                resolution=model.input_resolution,
            )
            previous = torch.tensor(
                [control_to_vector(observation.previous_control)],
                dtype=torch.float32,
                device=device,
            )
            elapsed = torch.zeros(1, dtype=torch.float32, device=device)
            output = model(
                pixels,
                previous,
                elapsed,
                state,
                thought_noise=thought_noise,
            )
            state = output.next_state.detach()
            attention = output.diagnostics.actuator_thought_attention.float()[0]
            mass = attention.clamp_min(1e-12)
            normalized = mass / mass.sum(dim=-1, keepdim=True)
            entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(dim=-1)
            mean_entropy = float(entropy.mean())
            logits = output.action.button_logits[0].float()
            wasd = {
                name: float(logits[hid])
                for name, hid in WASD
            }
            rows.append(
                {
                    "tick": tick,
                    "frame_id": observation.frame_id,
                    "mean_thought_attention_entropy": mean_entropy,
                    "wasd_logits": wasd,
                    "value": float(output.value[0].float()),
                }
            )
            observation = env.step(observation.previous_control).observation
    _write(
        out_dir / "thoughtlet-routing-dump.json",
        {
            "job": "thoughtlets",
            "campaign_id": "play_gated_maze_chase_distill_v1",
            "device": "cpu",
            "seed": 5,
            "ticks": ticks,
            "rows": rows,
            "mean_thought_attention_entropy": (
                sum(row["mean_thought_attention_entropy"] for row in rows) / len(rows)
                if rows
                else 0.0
            ),
        },
    )


def main() -> int:
    _ensure_src()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, choices=JOBS)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--ticks", type=int, default=8)
    arguments = parser.parse_args()
    out_dir = Path(arguments.out_dir)
    if arguments.job == "planner":
        job_planner(out_dir)
    elif arguments.job == "teacher-hist":
        job_teacher_hist(out_dir)
    elif arguments.job == "teacher-exclusive":
        job_teacher_exclusive(out_dir)
    elif arguments.job == "play-gate":
        if not arguments.config or not arguments.checkpoint:
            raise SystemExit("play-gate requires --config and --checkpoint")
        job_play_gate(
            out_dir,
            config_path=Path(arguments.config),
            checkpoint_path=Path(arguments.checkpoint),
        )
    else:
        if not arguments.config or not arguments.checkpoint:
            raise SystemExit("thoughtlets requires --config and --checkpoint")
        if arguments.ticks < 1 or arguments.ticks > 32:
            raise SystemExit("--ticks must be in [1, 32]")
        job_thoughtlets(
            out_dir,
            config_path=Path(arguments.config),
            checkpoint_path=Path(arguments.checkpoint),
            ticks=arguments.ticks,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
