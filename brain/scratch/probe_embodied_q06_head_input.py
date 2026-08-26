"""READ-ONLY experiment: pick the EmbodiedInterfaceV1 head input that lets the
Q0.6 gate (>=0.90 composite held-out accuracy on the synthetic pixel->record
task, core frozen) pass with margin.

Candidates (all derived only from the frame + frozen core, no privileged info):
  L  = latent only                     (core.encoder(frame))
  LB = cat([latent, belief])           (current obs + recurrent world belief)
  B  = belief only                     (current interface; expected to fail)

Reproduces the gate's exact training procedure (40 epochs, per-epoch
randperm, single-sample Adam lr=3e-3, loss on locomotion+yaw+buttons) and
reports composite held-out accuracy.  Deterministic: fixed global seed +
fixed perm seed, CUDA hidden, single thread.
"""
from __future__ import annotations

import os
import sys
from hashlib import sha256
from pathlib import Path

BRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN_ROOT / "scripts"), str(BRAIN_ROOT / "src")]

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import run_provenance  # noqa: E402
run_provenance.apply_deterministic_mode()

from irene_brain.training.objective import _rgb_tensor  # noqa: E402
from irene_brain.types import RgbFrame  # noqa: E402
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Config  # noqa: E402
from irene_brain.v2.core import CoreV2Model  # noqa: E402
from irene_brain.v2.embodied_interface import (  # noqa: E402
    BUTTON_NAMES, LOCOMOTION, LOOK_YAW_DEG,
)

PARENT_CHECKPOINT = (BRAIN_ROOT / "runs" / "v21i-development"
                     / "2026-08-24-seed42-v3.json.uncalibrated.pt")
PARENT_SHA256 = "e5a4c1e92cdf5649679f344ce9a032397e69a0a5c93bb9d3a7bebc4bc1352081"
_FRAME = 32
_RED_BLOCK = (200, 0, 0)
_NEUTRAL_BG = (24, 24, 24)
W = 120


def _load_core() -> CoreV2Model:
    assert sha256(PARENT_CHECKPOINT.read_bytes()).hexdigest() == PARENT_SHA256
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    config = CoreV2Config(**dict(payload["config"]))
    model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE)
    model.load_state_dict(dict(payload["model_state_dict"]), strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def _make_frame(*, red_x: int, red_y: int, bright_corner: bool = False) -> RgbFrame:
    arr = np.full((_FRAME, _FRAME, 3), _NEUTRAL_BG, dtype=np.uint8)
    arr[red_y: red_y + 4, red_x: red_x + 4] = _RED_BLOCK
    if bright_corner:
        arr[0:8, 24:32] = (255, 255, 0)
    return RgbFrame(width=_FRAME, height=_FRAME, pixels=arr.tobytes())


def target_for(rx, ry, bright):
    third = rx // (_FRAME // 3 + 1)
    loc = "LEFT" if third == 0 else ("RIGHT" if third == 2 else "NOOP")
    vt = ry // (_FRAME // 3 + 1)
    yaw = -30 if vt == 0 else (30 if vt == 2 else 0)
    buttons = [False] * len(BUTTON_NAMES)
    if bright:
        buttons[BUTTON_NAMES.index("attack")] = True
    return loc, yaw, tuple(buttons)


def build_pairs(seed, n, hold_every=5):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for i in range(n):
        rx, ry = int(rng.integers(0, 28)), int(rng.integers(0, 28))
        bright = bool(rng.integers(0, 2))
        (te if i % hold_every == 0 else tr).append(
            (_make_frame(red_x=rx, red_y=ry, bright_corner=bright),
             target_for(rx, ry, bright)))
    return tr, te


def encode_all(core, pairs, device):
    """Run the frozen recurrent core; return per-tick latent + belief stacks."""
    lat, bel = [], []
    state = core.init_state(1, device)
    for frame, _ in pairs:
        pixels = _rgb_tensor((frame,), device=device, resolution=(32, 32))
        prev = torch.tensor([0], dtype=torch.long, device=device)
        out, state = core(pixels, state, prev_action=prev)
        lat.append(out.latent.squeeze(0).detach())
        b = out.belief
        if b.dim() == 2:
            b = b.squeeze(0)
        bel.append(b.detach())
    return torch.stack(lat), torch.stack(bel)


def gate_train_eval(feat_train, labels_train, feat_test, labels_test,
                    in_dim, out_dims, perm_seed=7):
    """Exact gate procedure on a single feature vector per tick.
    out_dims = {field: n_classes}.  Supervised loss on locomotion, yaw,
    attack-button (the gate's 3 checked fields)."""
    heads = {f: torch.nn.Parameter(torch.randn(in_dim, d) * 0.01)
             for f, d in out_dims.items()}
    opt = torch.optim.Adam(heads.values(), lr=3e-3, betas=(0.9, 0.999))
    gen = torch.Generator().manual_seed(perm_seed)
    for _epoch in range(40):
        for idx in torch.randperm(feat_train.shape[0], generator=gen):
            i = int(idx)
            x = feat_train[i:i + 1]
            loc, yaw, buttons = labels_train[i]
            loc_lp = F.log_softmax(x @ heads["locomotion"], dim=1)
            yaw_lp = F.log_softmax(x @ heads["yaw"], dim=1)
            btn = x @ heads["buttons"]
            loss = (-loc_lp[0, LOCOMOTION.index(loc)]
                    -yaw_lp[0, LOOK_YAW_DEG.index(yaw)]
                    + F.binary_cross_entropy_with_logits(
                        btn[0], torch.tensor([float(v) for v in buttons])))
            opt.zero_grad(); loss.backward(); opt.step()
    # evaluate the 3 supervised fields
    hits = {"locomotion": 0, "yaw": 0, "buttons": 0}
    with torch.no_grad():
        for i in range(feat_test.shape[0]):
            x = feat_test[i:i + 1]
            loc, yaw, buttons = labels_test[i]
            if int((x @ heads["locomotion"]).argmax()) == LOCOMOTION.index(loc):
                hits["locomotion"] += 1
            if int((x @ heads["yaw"]).argmax()) == LOOK_YAW_DEG.index(yaw):
                hits["yaw"] += 1
            pred = tuple(v > 0.0 for v in (x @ heads["buttons"]).squeeze(0).tolist())
            if pred == tuple(buttons):
                hits["buttons"] += 1
    n = feat_test.shape[0]
    comp = (hits["locomotion"] + hits["yaw"] + hits["buttons"]) / (3 * n)
    return comp, {k: round(v / n, 4) for k, v in hits.items()}


def main():
    torch.manual_seed(999)
    np.random.seed(999)
    device = torch.device("cpu")
    core = _load_core()
    tr, te = build_pairs(0, 280, hold_every=5)
    print(f"train={len(tr)} test={len(te)}", flush=True)
    lat_tr, bel_tr = encode_all(core, tr, device)
    lat_te, bel_te = encode_all(core, te, device)
    labels_tr = [p[1] for p in tr]
    labels_te = [p[1] for p in te]
    out_dims = {"locomotion": len(LOCOMOTION), "yaw": len(LOOK_YAW_DEG),
                "buttons": len(BUTTON_NAMES)}

    cands = {
        "B_belief_only": (bel_tr, bel_te, W),
        "L_latent_only": (lat_tr, lat_te, W),
        "LB_latent+belief": (torch.cat([lat_tr, bel_tr], 1),
                             torch.cat([lat_te, bel_te], 1), 2 * W),
    }
    print("\nQ0.6 gate procedure, per candidate head input:", flush=True)
    for name, (ft, x, in_dim) in cands.items():
        ft, fe = ft, x
        comp, per = gate_train_eval(ft, labels_tr, fe, labels_te, in_dim, out_dims)
        verdict = "PASS" if comp >= 0.90 else "fail"
        print(f"  {name:18s} in_dim={in_dim:3d}  composite={comp:.4f} "
              f"({verdict})  per-field={per}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
