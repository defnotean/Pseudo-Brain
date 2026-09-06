"""Same-protocol closed-loop eval: BC init vs PPO+KL vs PPO-noKL."""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

BRAIN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRAIN_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import evaluate_model

RUN = BRAIN_ROOT / "runs" / "gru_vs_thoughtlet"
MODELS = {
    "bc_init": RUN / "long15k-thoughtlet-142.pt",
    "ppo_kl": RUN / "ppo-kl-142.pt",
    "ppo_nokl": RUN / "ppo-nokl-142.pt",
}

out = {}
for name, path in MODELS.items():
    r = evaluate_model(str(path), "thoughtlet", 142, n_seeds_per_family=4,
                       device="cpu", temperature=1.0, anti_stuck=6)
    out[name] = r
    o = r["overall"]
    print(f"{name}: pel/1k={o['mean_pellets_per_1k']:.1f} "
          f"cat/1k={o['mean_catches_per_1k']:.1f} rec={o['mean_recovery']:.2f} "
          f"lat={o['mean_latency_ms']:.2f}ms", flush=True)

json.dump(out, open(RUN / "ppo_vs_bc_eval.json", "w"), indent=1)
print("saved ppo_vs_bc_eval.json")
