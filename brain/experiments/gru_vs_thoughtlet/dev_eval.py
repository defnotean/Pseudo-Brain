"""DEV-partition closed-loop eval: held-out layouts, aligned protocol.

Compares checkpoints (BC init vs PPO) on the SAME partition_seeds DEV
layouts — the honest generalization read. Reuses run_episode metrics
(pel/1k, cat/1k, recovery, latency, nan_ticks).
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

BRAIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRAIN_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import make_model
from train import get_device
from evaluate import run_episode
from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.environments.pacman_harness import FAMILIES, partition_seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--model-type", default="thoughtlet")
    parser.add_argument("--partition", default="DEV")
    parser.add_argument("--eps-per-family", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--anti-stuck", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="dev_eval.json")
    args = parser.parse_args()

    dev = get_device(args.device)
    all_results = {}
    for ckpt_path in args.checkpoints:
        model = make_model(args.model_type).to(dev)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        name = Path(ckpt_path).stem
        all_results[name] = {}
        for fam_idx, fam in enumerate(FAMILIES):
            seeds = partition_seeds(fam_idx, args.partition)[:args.eps_per_family]
            fam_res = []
            for s in seeds:
                env = MazeChaseEnv(**fam.env_kwargs())
                fam_res.append(run_episode(
                    model=model, env=env, seed=s, device=dev,
                    model_type=args.model_type, max_ticks=fam.max_ticks,
                    temperature=args.temperature, anti_stuck=args.anti_stuck))
            tp = sum(r["pellets_eaten"] for r in fam_res)
            tt = sum(r["total_steps"] for r in fam_res)
            tc = sum(r["catches"] for r in fam_res)
            all_results[name][fam.name] = {
                "pellets_per_1k": tp / max(tt, 1) * 1000,
                "catches_per_1k": tc / max(tt, 1) * 1000,
                "mean_recovery": float(np.mean([r["post_catch_pellets_200"] for r in fam_res])),
                "mean_nan_ticks": float(np.mean([r["nan_ticks"] for r in fam_res])),
                "mean_latency_ms": float(np.mean([r["mean_latency_ms"] for r in fam_res])),
            }
        ov = all_results[name]
        fams = [fam.name for fam in FAMILIES]
        all_results[name]["overall"] = {
            "mean_pellets_per_1k": float(np.mean([ov[f]["pellets_per_1k"] for f in fams])),
            "mean_catches_per_1k": float(np.mean([ov[f]["catches_per_1k"] for f in fams])),
            "mean_recovery": float(np.mean([ov[f]["mean_recovery"] for f in fams])),
        }
        o = all_results[name]["overall"]
        print(f"{name}: pel/1k={o['mean_pellets_per_1k']:.1f} "
              f"cat/1k={o['mean_catches_per_1k']:.1f} rec={o['mean_recovery']:.2f}", flush=True)

    out = BRAIN_ROOT / "runs" / "gru_vs_thoughtlet" / args.out
    json.dump(all_results, open(out, "w"), indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
