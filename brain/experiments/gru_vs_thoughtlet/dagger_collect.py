"""DAgger data collection: student rollouts labeled by family-matched planner expert.

Student: argmax + anti-stuck (eval distribution). Expert: ScriptedMazeChasePlannerPolicy
with corpus-frozen knobs (candidate_pellets=6, horizon=24, matched ghost_period /
player_period / ghost_elroy). Seeds: TRAIN partition beyond corpus first-50.
Saves corpus-compatible frames/actions + manifest for mixed training.
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import torch

BRAIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRAIN_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import make_model, N_FRAMES
from evaluate import _rgb_to_np
from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.environments.pacman_harness import (
    FAMILIES, partition_seeds, control_for_class,
)
from irene_brain.evaluation.diagnostic_policies import ScriptedMazeChasePlannerPolicy

KEY_TO_CLASS = {0x1A: 1, 0x04: 2, 0x16: 3, 0x07: 4}  # W,A,S,D; empty -> 0 (noop)

def control_to_class(ctrl) -> int:
    keys = tuple(ctrl.keys_down)
    if len(keys) == 0:
        return 0
    return KEY_TO_CLASS.get(int(keys[0]), 0)


@torch.no_grad()
def collect_episode(student, student_type, expert, env, seed, device, max_collect_ticks=2000,
                    temperature=0.0):
    """Roll out student (argmax + anti-stuck), label every step with expert."""
    obs = env.reset(seed=seed)
    try:
        expert.reset(seed)
    except Exception:
        pass
    first = _rgb_to_np(obs.rgb)
    frames_buf = [first] * N_FRAMES
    prev_action = 0
    h = None
    stuck_count = 0
    last_frame = first
    banned_until = {}
    raw_frames = []   # uint8 (16,16,3)
    prev_actions = [] # student action taken previously
    expert_labels = []
    pellets = 0

    for tick in range(max_collect_ticks):
        frames_np = np.stack(frames_buf)
        frames_t = torch.from_numpy(frames_np.transpose(0, 3, 1, 2)).unsqueeze(0).to(device)
        prev_t = torch.tensor([prev_action], dtype=torch.long, device=device)
        if student_type == "reactive":
            logits, _ = student(frames_t, prev_t, None)
        elif student_type == "gru":
            if h is None and hasattr(student, "init_hidden"):
                h = student.init_hidden(1, device)
            logits, h = student(frames_t, prev_t, h)
        else:
            if h is None and hasattr(student, "init_thoughts"):
                h = student.init_thoughts(1, device)
            logits, h = student(frames_t, prev_t, h)
        order = torch.argsort(logits, dim=-1, descending=True).squeeze().tolist()
        if isinstance(order, int):
            order = [order]
        for a in list(banned_until.keys()):
            if banned_until[a] <= tick:
                del banned_until[a]
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1).squeeze()
            action = torch.multinomial(probs, 1).item()
        else:
            action = next((a for a in order if a not in banned_until), order[0])
        if stuck_count >= 6:
            if action not in banned_until:
                banned_until[action] = tick + 12
            stuck_count = 0
            action = next((a for a in order if a not in banned_until), order[0])

        # Expert label on the CURRENT observation (before stepping)
        expert_ctrl = expert.act(obs)
        expert_action = control_to_class(expert_ctrl)

        step_out = env.step(control_for_class(action))
        if "pellet_eaten" in step_out.events:
            pellets += 1

        new_frame = _rgb_to_np(step_out.observation.rgb)
        raw_frames.append((new_frame * 255).astype(np.uint8))
        prev_actions.append(prev_action)
        expert_labels.append(expert_action)

        frame_diff = float(np.abs(new_frame - last_frame).sum())
        repeated = (action == prev_action)
        if repeated and frame_diff < 5.0:
            stuck_count += 1
        else:
            stuck_count = 0
        last_frame = new_frame
        prev_action = action
        obs = step_out.observation
        frames_buf.append(new_frame)
        frames_buf = frames_buf[-N_FRAMES:]
        if "cleared" in step_out.events:
            break
    return raw_frames, prev_actions, expert_labels, pellets, tick + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-type", default="thoughtlet")
    parser.add_argument("--student-ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--eps-per-family", type=int, default=8)
    parser.add_argument("--seed-offset", type=int, default=550, help="TRAIN offset start (corpus used 500-549)")
    parser.add_argument("--max-ticks", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    student = make_model(args.student_type).to(device)
    ckpt = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    student.load_state_dict(ckpt["state_dict"])
    student.eval()

    out = Path(args.out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "actions").mkdir(parents=True, exist_ok=True)
    episodes = []
    total_trans = 0
    for fi, fam in enumerate(FAMILIES):
        expert = ScriptedMazeChasePlannerPolicy(
            ghost_period=fam.ghost_period, candidate_pellets=6, horizon=24,
            input_delay_ticks=0, player_period=fam.player_period,
            ghost_elroy=fam.ghost_elroy,
        )
        seeds = partition_seeds(fi, "TRAIN")[args.seed_offset - 500: args.seed_offset - 500 + args.eps_per_family]
        for si, seed in enumerate(seeds):
            env = MazeChaseEnv(**fam.env_kwargs())
            frames, prevs, labels, pellets, ticks = collect_episode(
                student, args.student_type, expert, env, seed, device, args.max_ticks,
                temperature=args.temperature)
            name = f"dagger_f{fi}_{si}"
            np.save(out / "frames" / f"{name}.npy", np.stack(frames))
            acts = np.stack([np.array(prevs, dtype=np.int8), np.array(labels, dtype=np.int8)], axis=1)
            np.save(out / "actions" / f"{name}.npy", acts)
            episodes.append({"name": name, "family": fi, "seed": seed, "pellets": pellets, "ticks": ticks})
            total_trans += len(frames)
            print(f"  {name}: ticks={ticks} student_pellets={pellets}", flush=True)
    (out / "manifest.json").write_text(json.dumps(
        {"episodes": episodes, "total_episodes": len(episodes), "total_transitions": total_trans}, indent=2))
    print(f"DONE {len(episodes)} eps {total_trans} transitions -> {out}")


if __name__ == "__main__":
    main()
