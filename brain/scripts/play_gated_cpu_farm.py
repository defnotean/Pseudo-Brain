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

JOBS = (
    "planner",
    "planner-seeds",
    "teacher-hist",
    "teacher-exclusive",
    "coverage",
    "tiled-coverage",
    "tiled-hist",
    "multi-episode-coverage",
    "offpolicy-teacher",
    "window-majority",
    "play-gate",
    "thoughtlets",
)
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


def _maze_batch_config(
    *,
    episode_horizon: int,
    sequences: int,
    window_sampling: str = "uniform",
):
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
        window_sampling=window_sampling,
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


def _parse_seeds(raw: str, *, default: tuple[int, ...]) -> tuple[int, ...]:
    text = raw.strip()
    if not text:
        return default
    if "-" in text and "," not in text:
        low_text, high_text = text.split("-", 1)
        low = int(low_text)
        high = int(high_text)
        if high < low or high - low + 1 > 16:
            raise SystemExit("seed range must cover 1-16 seeds")
        return tuple(range(low, high + 1))
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not seeds or len(seeds) > 16:
        raise SystemExit("need 1-16 seeds")
    return seeds


def _write_planner_episodes(out_dir: Path, *, seeds: tuple[int, ...], job: str) -> None:
    from irene_brain.evaluation.closed_loop_play import run_policy_closed_loop_episode
    from irene_brain.evaluation.diagnostic_policies import ScriptedMazeChasePlannerPolicy
    from irene_brain.training.play_gate import (
        PLAY_TICKS,
        maze_chase_environment_factory,
        maze_chase_play_config,
    )

    config = maze_chase_play_config()
    episodes = []
    for seed in seeds:
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
    filename = (
        "planner-closed-loop.json" if job == "planner" else "planner-seeds.json"
    )
    _write(
        out_dir / filename,
        {
            "job": job,
            "campaign_id": "play_gated_maze_chase_distill_v1",
            "policy": "diagnostic.scripted_maze_chase_planner.v1",
            "play_seeds": list(seeds),
            "play_ticks": PLAY_TICKS,
            "episodes": episodes,
            "clears": sum(1 for episode in episodes if episode["cleared"]),
        },
    )


def job_planner(out_dir: Path) -> None:
    from irene_brain.training.play_gate import PLAY_SEEDS

    _write_planner_episodes(out_dir, seeds=PLAY_SEEDS, job="planner")


def job_planner_seeds(out_dir: Path, *, seeds: tuple[int, ...]) -> None:
    _write_planner_episodes(out_dir, seeds=seeds, job="planner-seeds")


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


def _coverage_payload(
    dataset,
    *,
    job: str,
    hypothesis: str,
    sequence_length: int,
    episode_horizon: int,
) -> dict[str, object]:
    rows = []
    by_episode: dict[int, list[int]] = {}
    for index, sequence in enumerate(dataset):
        start = int(sequence.transitions[0].observation.frame_id)
        rows.append(
            {
                "sequence_index": index,
                "episode_seed": sequence.episode_seed,
                "window_start": start,
            }
        )
        by_episode.setdefault(sequence.episode_seed, []).append(start)
    per_episode = []
    for seed, starts in sorted(by_episode.items()):
        covered: set[int] = set()
        for start in starts:
            covered.update(range(start, start + sequence_length))
        per_episode.append(
            {
                "episode_seed": seed,
                "windows": len(starts),
                "window_starts": starts,
                "ticks_covered": len(covered),
                "coverage_fraction": len(covered) / float(episode_horizon),
            }
        )
    return {
        "job": job,
        "campaign_id": "play_gated_maze_chase_distill_v1",
        "hypothesis": hypothesis,
        "generator_id": dataset.config.generator_id,
        "window_sampling": dataset.config.manifest_dict().get("window_sampling"),
        "source_manifest_sha256": dataset.manifest_sha256,
        "sequence_length": sequence_length,
        "episode_horizon": episode_horizon,
        "sequences": rows,
        "unique_episodes": len(by_episode),
        "per_episode": per_episode,
        "mean_coverage_fraction": (
            sum(row["coverage_fraction"] for row in per_episode) / len(per_episode)
            if per_episode
            else 0.0
        ),
    }


def job_coverage(out_dir: Path) -> None:
    from irene_brain.data.maze_chase_dataset import (
        DatasetSplit,
        MazeChaseDatasetConfig,
        MazeChaseSequenceDataset,
    )

    uniform = MazeChaseSequenceDataset(
        MazeChaseDatasetConfig(
            split=DatasetSplit.TRAIN,
            sequence_count=16,
            sequence_length=8,
            episode_horizon=240,
        )
    )
    _write(
        out_dir / "dataset-coverage.json",
        _coverage_payload(
            uniform,
            job="coverage",
            hypothesis=(
                "uniform 8-tick windows on 16 train sequences never cover a "
                "full 240-tick planner episode 1:1"
            ),
            sequence_length=8,
            episode_horizon=240,
        ),
    )


def job_tiled_hist(out_dir: Path) -> None:
    from irene_brain.training.batches import MazeChaseBatchSource

    tiled = MazeChaseBatchSource(
        _maze_batch_config(
            episode_horizon=240,
            sequences=30,
            window_sampling="tiled",
        )
    )
    _write(
        out_dir / "tiled-teacher-wasd-histogram.json",
        {
            "job": "tiled-hist",
            "campaign_id": "play_gated_maze_chase_distill_v1",
            "hypothesis": (
                "one 240-tick tiled planner episode is mixed WASD, not one-key"
            ),
            "tiled_windows": {
                split: _histogram_from_source(tiled, split=split)
                for split in ("train", "validation")
            },
        },
    )


def job_multi_episode_coverage(out_dir: Path) -> None:
    from irene_brain.data.maze_chase_dataset import (
        DatasetSplit,
        MazeChaseDatasetConfig,
        MazeChaseSequenceDataset,
    )

    tiled = MazeChaseSequenceDataset(
        MazeChaseDatasetConfig(
            split=DatasetSplit.TRAIN,
            sequence_count=90,
            sequence_length=8,
            episode_horizon=240,
            window_sampling="tiled",
        )
    )
    _write(
        out_dir / "multi-episode-tiled-coverage.json",
        _coverage_payload(
            tiled,
            job="multi-episode-coverage",
            hypothesis=(
                "90 tiled 8-tick windows cover three 240-tick planner "
                "episodes 1:1"
            ),
            sequence_length=8,
            episode_horizon=240,
        ),
    )


def job_window_majority(out_dir: Path) -> None:
    from irene_brain.data.maze_chase_dataset import (
        DatasetSplit,
        MazeChaseDatasetConfig,
        MazeChaseSequenceDataset,
    )

    tiled = MazeChaseSequenceDataset(
        MazeChaseDatasetConfig(
            split=DatasetSplit.TRAIN,
            sequence_count=90,
            sequence_length=8,
            episode_horizon=240,
            window_sampling="tiled",
        )
    )
    windows = []
    pure_one_key = 0
    mixed = 0
    majority_keys: Counter[str] = Counter()
    majority_fractions: list[float] = []
    for index, sequence in enumerate(tiled):
        counts: Counter[str] = Counter()
        for transition in sequence.transitions:
            label, n_active = _wasd_counts(transition.action_target)
            if n_active > 1:
                raise RuntimeError("planner issued a multi-key teacher label")
            counts[label] += 1
        ticks = sum(counts.values())
        unique_wasd = [name for name, _hid in WASD if counts.get(name, 0)]
        majority_label, majority_count = counts.most_common(1)[0]
        fraction = majority_count / ticks if ticks else 0.0
        is_pure = len(unique_wasd) <= 1 and counts.get("idle", 0) == 0
        if is_pure:
            pure_one_key += 1
        if len(unique_wasd) >= 2:
            mixed += 1
        majority_keys[majority_label] += 1
        majority_fractions.append(fraction)
        windows.append(
            {
                "sequence_index": index,
                "episode_seed": sequence.episode_seed,
                "labels": dict(sorted(counts.items())),
                "unique_wasd": unique_wasd,
                "majority_label": majority_label,
                "majority_fraction": fraction,
                "pure_one_key": is_pure,
            }
        )
    _write(
        out_dir / "window-majority.json",
        {
            "job": "window-majority",
            "campaign_id": "play_gated_maze_chase_distill_v1",
            "hypothesis": (
                "most 8-tick tiled windows are one-key corridors, so exclusive "
                "CE can copy a window majority even when the episode is mixed"
            ),
            "generator_id": tiled.config.generator_id,
            "source_manifest_sha256": tiled.manifest_sha256,
            "windows": len(windows),
            "pure_one_key_windows": pure_one_key,
            "mixed_wasd_windows": mixed,
            "mean_majority_fraction": (
                sum(majority_fractions) / len(majority_fractions)
                if majority_fractions
                else 0.0
            ),
            "majority_key_histogram": dict(sorted(majority_keys.items())),
            "rows": windows,
        },
    )


def job_offpolicy_teacher(out_dir: Path) -> None:
    from irene_brain.evaluation.diagnostic_policies import (
        ScriptedMazeChasePlannerPolicy,
    )
    from irene_brain.training.play_gate import maze_chase_environment_factory
    from irene_brain.types import GenericControl

    ticks = 32
    seeds = (5, 9)
    behaviors = {
        "idle": GenericControl(),
        "w": GenericControl(keys_down=(26,)),
        "a": GenericControl(keys_down=(4,)),
        "s": GenericControl(keys_down=(22,)),
        "d": GenericControl(keys_down=(7,)),
    }
    rows = []
    for behavior, control in behaviors.items():
        for seed in seeds:
            environment = maze_chase_environment_factory()
            observation = environment.reset(seed)
            planner = ScriptedMazeChasePlannerPolicy()
            planner.reset(seed)
            counts: Counter[str] = Counter()
            agree = 0
            for _tick in range(ticks):
                teacher = planner.act(observation)
                label, n_active = _wasd_counts(teacher)
                counts[label] += 1
                rolled, _rolled_n = _wasd_counts(control)
                if label == rolled:
                    agree += 1
                if n_active > 1:
                    raise RuntimeError("planner issued a multi-key teacher label")
                observation = environment.step(control).observation
            total = sum(counts.values())
            rows.append(
                {
                    "behavior": behavior,
                    "seed": seed,
                    "ticks": ticks,
                    "teacher_labels": dict(sorted(counts.items())),
                    "teacher_unique_wasd": len(
                        {name for name, _hid in WASD if counts.get(name, 0)}
                    ),
                    "teacher_matches_behavior": agree,
                    "mixed_wasd": (
                        len({name for name, _hid in WASD if counts.get(name, 0)})
                        >= 3
                    ),
                    "rates": {
                        name: (counts.get(name, 0) / total if total else 0.0)
                        for name, _hid in WASD
                    }
                    | {"idle": counts.get("idle", 0) / total if total else 0.0},
                }
            )
    _write(
        out_dir / "offpolicy-teacher.json",
        {
            "job": "offpolicy-teacher",
            "campaign_id": "play_gated_maze_chase_distill_v1",
            "hypothesis": (
                "planner labels on sticky-WASD and idle rollouts stay mixed, "
                "so closed-loop BC would not copy the sticky key"
            ),
            "ticks": ticks,
            "seeds": list(seeds),
            "rows": rows,
            "any_mixed": any(row["mixed_wasd"] for row in rows),
            "all_mixed": all(row["mixed_wasd"] for row in rows),
        },
    )


def job_tiled_coverage(out_dir: Path) -> None:
    from irene_brain.data.maze_chase_dataset import (
        DatasetSplit,
        MazeChaseDatasetConfig,
        MazeChaseSequenceDataset,
    )

    tiled = MazeChaseSequenceDataset(
        MazeChaseDatasetConfig(
            split=DatasetSplit.TRAIN,
            sequence_count=30,
            sequence_length=8,
            episode_horizon=240,
            window_sampling="tiled",
        )
    )
    _write(
        out_dir / "tiled-dataset-coverage.json",
        _coverage_payload(
            tiled,
            job="tiled-coverage",
            hypothesis=(
                "tiled 8-tick windows on 30 train sequences cover one "
                "240-tick planner episode 1:1"
            ),
            sequence_length=8,
            episode_horizon=240,
        ),
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
    seeds: tuple[int, ...],
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
    device = next(model.parameters()).device
    thought_noise = deterministic_eval_thought_noise(
        thoughtlets=model.config.thoughtlets,
        width=model.config.core_width,
        batch_size=1,
        device=device,
    )
    episodes = []
    with torch.no_grad():
        for seed in seeds:
            env = maze_chase_environment_factory()
            observation = env.reset(seed)
            state = None
            rows = []
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
                entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(
                    dim=-1
                )
                mean_entropy = float(entropy.mean())
                logits = output.action.button_logits[0].float()
                wasd = {name: float(logits[hid]) for name, hid in WASD}
                ranked = sorted(wasd, key=lambda name: wasd[name], reverse=True)
                rows.append(
                    {
                        "tick": tick,
                        "frame_id": observation.frame_id,
                        "mean_thought_attention_entropy": mean_entropy,
                        "wasd_logits": wasd,
                        "argmax": ranked[0],
                        "value": float(output.value[0].float()),
                    }
                )
                observation = env.step(observation.previous_control).observation
            episodes.append(
                {
                    "seed": seed,
                    "ticks": ticks,
                    "rows": rows,
                    "mean_thought_attention_entropy": (
                        sum(row["mean_thought_attention_entropy"] for row in rows)
                        / len(rows)
                        if rows
                        else 0.0
                    ),
                    "argmax_counts": {
                        name: sum(1 for row in rows if row["argmax"] == name)
                        for name, _hid in WASD
                    },
                }
            )
    _write(
        out_dir / "thoughtlet-routing-dump.json",
        {
            "job": "thoughtlets",
            "campaign_id": "play_gated_maze_chase_distill_v1",
            "device": "cpu",
            "seeds": list(seeds),
            "ticks": ticks,
            "episodes": episodes,
            "mean_thought_attention_entropy": (
                sum(episode["mean_thought_attention_entropy"] for episode in episodes)
                / len(episodes)
                if episodes
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
    parser.add_argument("--seeds", default="")
    arguments = parser.parse_args()
    out_dir = Path(arguments.out_dir)
    if arguments.job == "planner":
        job_planner(out_dir)
    elif arguments.job == "planner-seeds":
        job_planner_seeds(
            out_dir,
            seeds=_parse_seeds(arguments.seeds, default=tuple(range(100, 116))),
        )
    elif arguments.job == "teacher-hist":
        job_teacher_hist(out_dir)
    elif arguments.job == "teacher-exclusive":
        job_teacher_exclusive(out_dir)
    elif arguments.job == "coverage":
        job_coverage(out_dir)
    elif arguments.job == "tiled-coverage":
        job_tiled_coverage(out_dir)
    elif arguments.job == "tiled-hist":
        job_tiled_hist(out_dir)
    elif arguments.job == "multi-episode-coverage":
        job_multi_episode_coverage(out_dir)
    elif arguments.job == "offpolicy-teacher":
        job_offpolicy_teacher(out_dir)
    elif arguments.job == "window-majority":
        job_window_majority(out_dir)
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
            seeds=_parse_seeds(arguments.seeds, default=(5, 9)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
