"""Render maze_chase closed-loop GIFs from a play-gated checkpoint.

Reuses the planner GIF capture loop in ``render_maze_chase_replay.py`` and
the campaign play-gate decode (``exclusive_argmax_wasd_v1``) via
``run_closed_loop_episode``. CPU-only: do not launch this on the GB10.

Usage (Spark CPU container, CUDA hidden):

    python3 -I brain/scripts/render_maze_chase_neural_replay.py \\
      --config brain/configs/training/dgx-play-maze-chase-distill-turn-weighted.toml \\
      --checkpoint /path/to/step-00000032.pt \\
      --out-dir brain/docs/runs/artifacts/play-gated-maze-chase-distill \\
      --seeds 5,9
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

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

# Campaign gate checkpoint for turn-weighted exclusive CE.
TURN_WEIGHTED_V1_CKPT_SHA256 = (
    "e58f323fb90893c4953b04211002753dd3162f398d1b3b4152390203ba8131cf"
)
SCALE = 16
CAPTION_HEIGHT = 20
FRAME_DURATION_MS = 60
WASD_NAMES = {26: "W", 4: "A", 22: "S", 7: "D"}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def _wasd_label(control: object) -> str:
    keys = set(getattr(control, "keys_down", ()))
    active = [WASD_NAMES[key] for key in (26, 4, 22, 7) if key in keys]
    return "".join(active) if active else "-"


def _capture_frame(observation: object, note: str):
    from PIL import Image, ImageDraw
    from irene_brain.environments.maze_chase import MazeChaseEnv

    frame = observation.rgb
    image = Image.frombytes("RGB", (frame.width, frame.height), frame.pixels)
    size = MazeChaseEnv.GRID_SIZE * SCALE
    image = image.resize((size, size), Image.NEAREST)
    canvas = Image.new("RGB", (size, size + CAPTION_HEIGHT), (16, 18, 26))
    canvas.paste(image, (0, CAPTION_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    tick = int(getattr(observation, "frame_id", 0))
    draw.text((4, 4), f"tick {tick:3d}  {note}", fill=(220, 224, 235))
    return canvas


class _CapturingMazeChase:
    """World wrapper that records the same public RGB the planner GIF uses."""

    def __init__(self, environment: object, frames: list) -> None:
        self._environment = environment
        self._frames = frames

    @property
    def tick_period_ns(self) -> int:
        return self._environment.tick_period_ns

    @property
    def current_observation(self):
        return self._environment.current_observation

    def reset(self, seed: int):
        observation = self._environment.reset(seed)
        self._frames.append(_capture_frame(observation, "reset"))
        return observation

    def step(self, control):
        outcome = self._environment.step(control)
        events = ",".join(outcome.events) if outcome.events else ""
        note = _wasd_label(control) if not events else f"{_wasd_label(control)} {events}"
        self._frames.append(_capture_frame(outcome.observation, note))
        return outcome


def _write_gif(path: Path, frames: list) -> None:
    if not frames:
        raise ValueError("no frames to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
    )


def render_seed(
    *,
    model: object,
    seed: int,
    decode_kind: str,
    out_path: Path,
) -> dict[str, object]:
    from irene_brain.evaluation.closed_loop_play import (
        ClosedLoopPlayConfig,
        run_closed_loop_episode,
    )
    from irene_brain.training.play_gate import (
        PLAY_TICKS,
        maze_chase_environment_factory,
    )

    frames: list = []

    def environment_factory() -> object:
        return _CapturingMazeChase(maze_chase_environment_factory(), frames)

    device = next(model.parameters()).device
    report = run_closed_loop_episode(
        model,
        seed=seed,
        config=ClosedLoopPlayConfig(
            episode_seeds=(seed,),
            max_ticks=PLAY_TICKS,
            decode_kind=decode_kind,
        ),
        device=device,
        environment_factory=environment_factory,
    )
    _write_gif(out_path, frames)
    payload = {
        "seed": seed,
        "ticks": report.ticks_advanced,
        "frames": len(frames),
        "pellets_eaten": report.pellets_eaten,
        "collisions": report.collisions,
        "reward_sum": report.reward_sum,
        "movement_mask_histogram": [
            [mask, count] for mask, count in report.movement_mask_histogram
        ],
        "play_decode_kind": decode_kind,
        "gif": str(out_path),
        "bytes": out_path.stat().st_size,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="5,9")
    parser.add_argument("--stem", default="turn-weighted-v1")
    parser.add_argument(
        "--expect-sha256",
        default=TURN_WEIGHTED_V1_CKPT_SHA256,
        help="Refuse to render if the checkpoint digest does not match.",
    )
    arguments = parser.parse_args()
    checkpoint_path = Path(arguments.checkpoint)
    digest = _file_sha256(checkpoint_path)
    if digest != arguments.expect_sha256:
        raise SystemExit(
            f"checkpoint sha256 {digest} != expected {arguments.expect_sha256}"
        )
    model, config = _load_trained_model(
        config_path=Path(arguments.config),
        checkpoint_path=checkpoint_path,
    )
    decode_kind = config.objective.play_decode_kind
    if decode_kind != "exclusive_argmax_wasd_v1":
        raise SystemExit(
            f"play decode is {decode_kind}, expected exclusive_argmax_wasd_v1"
        )
    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(part) for part in arguments.seeds.split(",") if part.strip())
    summaries = []
    for seed in seeds:
        summaries.append(
            render_seed(
                model=model,
                seed=seed,
                decode_kind=decode_kind,
                out_path=out_dir / f"{arguments.stem}-seed{seed}.gif",
            )
        )
    summary_path = out_dir / f"{arguments.stem}-gif-meta.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": digest,
                "play_decode_kind": decode_kind,
                "episodes": summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": str(summary_path)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
