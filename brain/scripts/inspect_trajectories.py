"""Targeted inspection of Seed 43 vs Seed 46 actual trajectories, ghost vectors, and wall collisions."""

import math
import torch
from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.types import HidKey
from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from diagnose_seed_divergence import train_model_on_seed


def inspect_trajectories(seed: int, num_episodes: int = 2, max_ticks: int = 60) -> None:
    print(f"\n========================================================")
    print(f"  DETAILED TRAJECTORY TRACE FOR SEED {seed}")
    print(f"========================================================")
    model = train_model_on_seed(seed)
    policy = LatentLookaheadPolicy(model)

    for ep in range(num_episodes):
        world_seed = 2001 + ep
        env = MazeChaseEnv(
            ghost_count=2,
            player_period=1,
            ghost_period=2,
        )
        obs = env.reset(world_seed)
        policy.reset(world_seed)

        print(f"\n--- Episode {ep+1} (World Seed {world_seed}) ---")
        prev_pos = (env._player_x, env._player_y)

        for tick in range(max_ticks):
            px, py = env._player_x, env._player_y
            ghost_dists = [math.hypot(px - gx, py - gy) for (gx, gy) in env._ghosts]
            min_dist = min(ghost_dists)
            nearest_idx = ghost_dists.index(min_dist)
            gx, gy = env._ghosts[nearest_idx]

            control = policy.act(obs)
            keys = [k for k in [HidKey.W, HidKey.A, HidKey.S, HidKey.D] if int(k) in control.keys_down]
            key_names = "".join([k.name for k in keys]) or "NONE"

            is_w = int(HidKey.W) in control.keys_down
            is_a = int(HidKey.A) in control.keys_down
            is_s = int(HidKey.S) in control.keys_down
            is_d = int(HidKey.D) in control.keys_down
            chosen_dx = (1 if is_d else 0) - (1 if is_a else 0)
            chosen_dy = (1 if is_s else 0) - (1 if is_w else 0)

            v_gx, v_gy = gx - px, gy - py
            dot = chosen_dx * v_gx + chosen_dy * v_gy
            direction_desc = "NEUTRAL"
            if dot > 0:
                direction_desc = "TOWARD_GHOST (SUICIDE)"
            elif dot < 0:
                direction_desc = "AWAY_FROM_GHOST (EVADE)"

            outcome = env.step(control)
            obs = outcome.observation
            new_pos = (env._player_x, env._player_y)
            moved = (new_pos != prev_pos)
            is_wall_hit = (not moved) and (chosen_dx != 0 or chosen_dy != 0)

            is_caught = "caught" in outcome.events

            if min_dist <= 3.0 or is_caught:
                print(
                    f"Tick {tick:02d} | Pos: ({px:2d},{py:2d}) | Ghost: ({gx:2d},{gy:2d}) dist={min_dist:4.2f} | "
                    f"Action: {key_names:4s} ({chosen_dx:+d},{chosen_dy:+d}) | VectorToGhost: ({v_gx:+d},{v_gy:+d}) | "
                    f"Dot: {dot:+d} [{direction_desc:22s}] | Moved: {str(moved):5s} WallHit: {str(is_wall_hit):5s} Caught: {is_caught}"
                )

            prev_pos = new_pos
            if is_caught:
                print(f"  --> CAUGHT AT TICK {tick}! Respawned at ({env._player_x}, {env._player_y})")


if __name__ == "__main__":
    inspect_trajectories(43, num_episodes=1, max_ticks=40)
    inspect_trajectories(46, num_episodes=1, max_ticks=40)
