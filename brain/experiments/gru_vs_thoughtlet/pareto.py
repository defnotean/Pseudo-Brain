"""Aggregate ablation eval JSONs into Pareto table."""
import json
from pathlib import Path
import numpy as np

ad = Path("brain/runs/gru_vs_thoughtlet/ablations")
bd = Path("brain/runs/gru_vs_thoughtlet")

res = {}
for f in ["eval_A", "eval_B", "eval_C", "eval_D"]:
    res.update(json.load(open(ad / f"{f}.json")))
res.update(json.load(open(bd / "eval_k32.json")))
import torch

def ckpt_params(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return sum(p.numel() for p in ck["state_dict"].values())

params = {}
for v in ["k1", "k2", "k4", "k8", "k16",
          "no_persistence", "no_attention", "independent_slots"]:
    for s in (42, 142, 242):
        p = ad / f"seed_{s}_{v}.pt"
        if p.exists():
            params[(v, s)] = ckpt_params(p)
for s in (42, 142, 242):
    params[("thoughtlet", s)] = ckpt_params(bd / f"seed_{s}_thoughtlet.pt")

fams = ["F0_calm", "F1_standard", "F2_pressured", "F3_fast", "F4_open_slow"]
variants = ["k1", "k2", "k4", "k8", "k16", "thoughtlet",
            "no_persistence", "no_attention", "independent_slots"]

table = []
for v in variants:
    pel, cat, rec, lat, nan = [], [], [], [], []
    pf = []
    pcount = None
    for s in (42, 142, 242):
        key = f"{v}_seed{s}"
        if key not in res:
            continue
        r = res[key]
        o = r["overall"]
        pel.append(o["mean_pellets_per_1k"])
        cat.append(o["mean_catches_per_1k"])
        rec.append(o["mean_recovery"])
        lat.append(o["mean_latency_ms"])
        pf.append(o["mean_pellet_fraction"])
        fams_nan = np.mean([r[f]["mean_nan_ticks"] for f in fams])
        nan.append(fams_nan)
        pcount = params.get((v, s), pcount)
    table.append(dict(variant=v, n=len(pel),
                      pel_mean=np.mean(pel), pel_std=np.std(pel),
                      cat_mean=np.mean(cat),
                      rec_mean=np.mean(rec),
                      lat_mean=np.mean(lat),
                      nan_mean=np.mean(nan),
                      pf_mean=np.mean(pf),
                      params=pcount))

print(f"{'variant':18s} {'n':>2s} {'pel/1k':>12s} {'cat/1k':>8s} {'recov':>7s} {'latms':>7s} {'nantk':>7s} {'params':>9s}")
for t in sorted(table, key=lambda t: -t["pel_mean"]):
    print(f"{t['variant']:18s} {t['n']:2d} "
          f"{t['pel_mean']:6.1f}+-{t['pel_std']:4.1f} {t['cat_mean']:8.1f} "
          f"{t['rec_mean']:7.2f} {t['lat_mean']:7.2f} {t['nan_mean']:7.1f} "
          f"{t['params'] or -1:>9d}")

json.dump(table, open(ad / "ablation_pareto.json", "w"), indent=1)
print("\nsaved ablation_pareto.json")

# per-family pel/1k for the winner-relevant variants
print("\nper-family pel/1k (mean over 3 seeds):")
print(f"{'variant':18s} " + " ".join(f"{f:>12s}" for f in fams))
for v in variants:
    row = []
    for f in fams:
        vals = [res[f"{v}_seed{s}"][f]["pellets_per_1k"] for s in (42, 142, 242) if f"{v}_seed{s}" in res]
        row.append(np.mean(vals) if vals else float("nan"))
    print(f"{v:18s} " + " ".join(f"{x:12.1f}" for x in row))
