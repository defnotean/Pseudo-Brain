"""READ-ONLY: margin check for the Q0.6 head-feasibility gate under the
latent+belief head input.  Vary the training procedure (epochs, batch, lr)
and confirm the composite held-out accuracy clears 0.90 with margin.

Also reports the information ceiling (400-epoch probe) on the same feature
so we know how far the gate is from the best achievable.
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


def _load_core() -> CoreV2Model:
    assert sha256(PARENT_CHECKPOINT.read_bytes()).hexdigest() == PARENT_SHA256
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    config = CoreV2Config(**dict(payload["config"]))
    model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE)
    model.load_state_dict(dict(payload["model_state_dict"]), strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def _make_frame(*, red_x, red_y, bright_corner=False) -> RgbFrame:
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
    lat, bel = [], []
    state = core.init_state(1, device)
    for frame, _ in pairs:
        pixels = _rgb_tensor((frame,), device=device, resolution=(32, 32))
        prev = torch.tensor([0], dtype=torch.long, device=device)
        out, state = core(pixels, state, prev_action=prev)
        lat.append(out.latent.squeeze(0).detach())
        b = out.belief
        b = b.squeeze(0) if b.dim() == 2 else b
        bel.append(b.detach())
    return torch.stack(lat), torch.stack(bel)


def evaluate(feat, labels, heads):
    hits = {"locomotion": 0, "yaw": 0, "buttons": 0}
    with torch.no_grad():
        for i in range(feat.shape[0]):
            x = feat[i:i + 1]
            loc, yaw, buttons = labels[i]
            if int((x @ heads["locomotion"]).argmax()) == LOCOMOTION.index(loc):
                hits["locomotion"] += 1
            if int((x @ heads["yaw"]).argmax()) == LOOK_YAW_DEG.index(yaw):
                hits["yaw"] += 1
            pred = tuple(v > 0.0 for v in (x @ heads["buttons"]).squeeze(0).tolist())
            if pred == tuple(buttons):
                hits["buttons"] += 1
    n = feat.shape[0]
    return ((hits["locomotion"] + hits["yaw"] + hits["buttons"]) / (3 * n),
            {k: round(v / n, 4) for k, v in hits.items()})


def train_heads(feat, labels, in_dim, out_dims, epochs, batch, lr, perm_seed):
    heads = {f: torch.nn.Parameter(torch.randn(in_dim, d) * 0.01)
             for f, d in out_dims.items()}
    opt = torch.optim.Adam(heads.values(), lr=lr, betas=(0.9, 0.999))
    gen = torch.Generator().manual_seed(perm_seed)
    n = feat.shape[0]
    for _epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            x = feat[idx]
            loc_t = torch.tensor([LOCOMOTION.index(l) for l, _, _ in
                                  [labels[i] for i in idx]], device=feat.device)
            yaw_t = torch.tensor([LOOK_YAW_DEG.index(y) for _, y, _ in
                                  [labels[i] for i in idx]], device=feat.device)
            btn_t = torch.tensor([[float(v) for v in b] for _, _, b in
                                  [labels[i] for i in idx]], device=feat.device)
            loss = (F.cross_entropy(x @ heads["locomotion"], loc_t)
                    + F.cross_entropy(x @ heads["yaw"], yaw_t)
                    + F.binary_cross_entropy_with_logits(x @ heads["buttons"], btn_t))
            opt.zero_grad(); loss.backward(); opt.step()
    return heads


def main():
    torch.manual_seed(999)
    np.random.seed(999)
    device = torch.device("cpu")
    core = _load_core()
    tr, te = build_pairs(0, 280, hold_every=5)
    lat_tr, bel_tr = encode_all(core, tr, device)
    lat_te, bel_te = encode_all(core, te, device)
    labels_tr = [p[1] for p in tr]
    labels_te = [p[1] for p in te]
    out_dims = {"locomotion": len(LOCOMOTION), "yaw": len(LOOK_YAW_DEG),
                "buttons": len(BUTTON_NAMES)}

    feat_tr = torch.cat([lat_tr, bel_tr], 1)
    feat_te = torch.cat([lat_te, bel_te], 1)
    in_dim = feat_tr.shape[1]

    # information ceiling
    heads = train_heads(feat_tr, labels_tr, in_dim, out_dims,
                        epochs=400, batch=64, lr=1e-2, perm_seed=7)
    comp, per = evaluate(feat_te, labels_te, heads)
    print(f"ceiling (400ep b64 lr1e-2): composite={comp:.4f} per-field={per}", flush=True)

    print("\nprocedure sweep (epochs x batch x lr):", flush=True)
    for epochs, batch, lr in [
        (40, 1, 3e-3),      # original gate procedure
        (40, 16, 3e-3),
        (80, 16, 3e-3),
        (120, 32, 3e-3),
        (120, 32, 1e-3),
        (200, 64, 1e-3),
        (300, 64, 3e-4),
    ]:
        heads = train_heads(feat_tr, labels_tr, in_dim, out_dims,
                            epochs=epochs, batch=batch, lr=lr, perm_seed=7)
        comp, per = evaluate(feat_te, labels_te, heads)
        verdict = "PASS" if comp >= 0.90 else "fail"
        print(f"  ep={epochs:3d} batch={batch:2d} lr={lr:.0e}  "
              f"composite={comp:.4f} ({verdict})  {per}", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
