"""Render a deterministic maze_chase replay GIF from the scripted planner.

This is an offline, CPU-only evidence artifact: it replays one canonical
matrix-slot episode (seed 5) with the pixel-only lookahead planner
(``diagnostic.scripted_maze_chase_planner.v1``) and writes an animated GIF
under ``brain/docs/runs/artifacts/``. It reads no screen, injects no input,
and touches no network. Every frame comes from the environment's own
public render contract, upscaled 16x with a tick/event caption.

Usage (from the repository root, play-safe Python):

    python brain/scripts/render_maze_chase_replay.py
"""

from __future__ import annotations

from pathlib import Path
import sys

BRAIN_ROOT = Path(__file__).resolve().parents[1]
SRC = BRAIN_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.diagnostic_policies import ScriptedMazeChasePlannerPolicy

EPISODE_SEED = 5
MAX_TICKS = 600
SCALE = 16
CAPTION_HEIGHT = 20
FRAME_DURATION_MS = 60
OUTPUT = (
    BRAIN_ROOT
    / "docs"
    / "runs"
    / "artifacts"
    / "2026-08-17-maze-chase-planner-clear-seed5.gif"
)


def main() -> None:
    from PIL import Image, ImageDraw

    environment = MazeChaseEnv(
        ghost_count=3,
        ghost_period=2,
        extra_loops=16,
        max_ticks=MAX_TICKS,
    )
    planner = ScriptedMazeChasePlannerPolicy()
    planner.reset(EPISODE_SEED)
    observation = environment.reset(EPISODE_SEED)

    grid = MazeChaseEnv.GRID_SIZE
    size = grid * SCALE
    frames: list[Image.Image] = []
    events: list[str] = []

    def capture(tick: int, note: str) -> None:
        frame = observation.rgb
        image = Image.frombytes("RGB", (frame.width, frame.height), frame.pixels)
        image = image.resize((size, size), Image.NEAREST)
        canvas = Image.new("RGB", (size, size + CAPTION_HEIGHT), (16, 18, 26))
        canvas.paste(image, (0, CAPTION_HEIGHT))
        draw = ImageDraw.Draw(canvas)
        draw.text((4, 4), f"tick {tick:3d}  {note}", fill=(220, 224, 235))
        frames.append(canvas)

    capture(0, "reset")
    outcome = None
    for tick in range(MAX_TICKS):
        outcome = environment.step(planner.act(observation))
        observation = outcome.observation
        note = ",".join(outcome.events) if outcome.events else ""
        capture(tick + 1, note)
        if outcome.terminated:
            break

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
    )
    print(f"wrote {OUTPUT} ({len(frames)} frames)")
    print(f"pellets={environment._pellets_eaten} caught={environment._times_caught} cleared={environment._cleared}")


if __name__ == "__main__":
    main()
