"""READ-ONLY diagnostic: why does the EmbodiedInterfaceV1 Q0.6 head-feasibility
gate fail (0.6429 vs 0.90)?

No new partition, no publish, no frozen-state change. Mirrors the frozen
runner's synthetic task exactly (seed 0, 280 pairs, 1-in-5 held out) and
measures, for the frozen W120/K32/C3 core + seed-43 heads:

  A. belief dynamics: does the frozen belief actually change across frames
     (variance of belief across the training sequence, per-dim and total)?
     If the belief is ~constant, no head can read pixel information from it.
  B. per-field gate accuracy: decompose the gate's 0.6429 into locomotion /
     yaw / buttons component accuracies under the gate's own training.
  C. probe ceilings: independent ridge/logistic probes (much more data,
     more epochs, batched) fit on (i) belief, (ii) encoder latent, against
     the three supervised fields.  If latent probes are near-perfect but
     belief probes are not, the belief update is discarding position
     information (representation bottleneck, not a head/optimizer bug).
     If both are low, the frozen encoder itself lacks the readout.
  D. gate procedure bug check: re-run the gate's exact training procedure
     with per-field logging; also test whether button BCE + argmax/0.0
     decoding is the weak field.

Deterministic: CUDA hidden, single-thread, global torch seed fixed here
(diagnostic only; the gate's own non-determinism is reported separately).
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
from irene_brain.v2 import CONFIG_B_PREDICTIVE  # noqa: E402
from irene_brain.v2.core import CoreV2Model  # noqa: E402
from irene_brain.v2.embodied_interface import (  # noqa: E402
    BUTTON_NAMES,
    LOCOMOTION,
    LOOK_YAW_DEG,
    EmbodiedInterfaceRecord,
    EmbodiedInterfaceV1,
)

PARENT_CHECKPOINT = (
    BRAIN_ROOT / "runs" / "v21i-development"
    / "2026-08-24-seed42-v3.json.uncalibrated.pt"
)
PARENT_SHA256 = "e5a4c1e92cdf5649679f344ce9a032397e69a0a5c93bb9d3a7bebc4bc1352081"
HEAD_INIT_SEED = 43
_FRAME = 32
_RED_BLOCK = (200, 0, 0)
_NEUTRAL_BG = (24, 24, 24)


def _load_frozen_parent() -> CoreV2Model:
    actual = sha256(PARENT_CHECKPOINT.read_bytes()).hexdigest()
    assert actual == PARENT_SHA256, actual
    payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    from irene_brain.v2 import CoreV2Config
    from dataclasses import asdict
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


def target_for(red_x: int, red_y: int, bright_corner: bool) -> EmbodiedInterfaceRecord:
    third = red_x // (_FRAME // 3 + 1)
    loc = "LEFT" if third == 0 else ("RIGHT" if third == 2 else "NOOP")
    vthird = red_y // (_FRAME // 3 + 1)
    yaw = -30 if vthird == 0 else (30 if vthird == 2 else 0)
    buttons = [False] * len(BUTTON_NAMES)
    if bright_corner:
        buttons[BUTTON_NAMES.index("attack")] = True
    return EmbodiedInterfaceRecord(
        locomotion=loc,
        look_yaw_deg=yaw,
        look_pitch_deg=0,
        cursor_dx_px=0,
        cursor_dy_px=0,
        buttons=tuple(buttons),
        hotbar="HOLD",
    )


def build_pairs(seed: int, n: int, hold_every: int = 5):
    rng = np.random.default_rng(seed)
    train_pairs, test_pairs = [], []
    for i in range(n):
        rx, ry = int(rng.integers(0, 28)), int(rng.integers(0, 28))
        bright = bool(rng.integers(0, 2))
        pair = (_make_frame(red_x=rx, red_y=ry, bright_corner=bright),
                target_for(rx, ry, bright))
        (test_pairs if i % hold_every == 0 else train_pairs).append(pair)
    return train_pairs, test_pairs


def run_sequence(core, frames, device, prev_action=0):
    """Recurrent pass; returns per-tick (belief, latent, mean_thoughts)."""
    state = core.init_state(1, device)
    beliefs, latents, thoughts = [], [], []
    for frame in frames:
        pixels = _rgb_tensor((frame,), device=device, resolution=(32, 32))
        prev = torch.tensor([prev_action], dtype=torch.long, device=device)
        out, state = core(pixels, state, prev_action=prev)
        b = out.belief
        if b.dim() == 2:
            b = b.squeeze(0)
        beliefs.append(b.detach())
        latents.append(out.latent.squeeze(0).detach())
        thoughts.append(out.thoughts.squeeze(0).mean(dim=0).detach())
    return (torch.stack(beliefs), torch.stack(latents),
            torch.stack(thoughts))


def probe_ceiling(feats: torch.Tensor, labels: np.ndarray, num_classes: int,
                  epochs: int = 400, lr: float = 1e-2, tag: str = "", name: str = ""):
    """Batched logistic probe on a frozen feature vector for ONE field;
    returns held-out accuracy."""
    Wd = feats.shape[1]
    w = torch.nn.Parameter(torch.randn(Wd, num_classes) * 0.01)
    b = torch.nn.Parameter(torch.zeros(num_classes))
    opt = torch.optim.Adam([w, b], lr=lr)
    log_softmax = torch.nn.LogSoftmax(dim=1)
    tgt = torch.from_numpy(labels)  # [N]
    for _ in range(epochs):
        logp = log_softmax(feats.float() @ w + b)
        loss = -logp.gather(1, tgt.long().unsqueeze(1)).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        pred = (feats.float() @ w + b).argmax(1).numpy()
    acc = float((pred == labels).mean())
    print(f"  [probe{tag}] {name}={acc:.4f}", flush=True)
    return round(acc, 4)


def main() -> None:
    torch.manual_seed(12345)  # diagnostic determinism only
    np.random.seed(12345)
    device = torch.device("cpu")
    core = _load_frozen_parent()
    interface = EmbodiedInterfaceV1(core, init_seed=HEAD_INIT_SEED)
    print(f"heads_sha256={interface.heads_sha256()[:16]}...", flush=True)

    train_pairs, test_pairs = build_pairs(0, 280, hold_every=5)
    print(f"train={len(train_pairs)} test={len(test_pairs)}", flush=True)

    # ---- A. belief dynamics on the training sequence -------------------
    t_bel, t_lat, t_tho = run_sequence(core, [p[0] for p in train_pairs], device)
    b_var = float(t_bel.var(dim=0).sum()) / t_bel.shape[1]
    l_var = float(t_lat.var(dim=0).sum()) / t_lat.shape[1]
    o_var = float(t_tho.var(dim=0).sum()) / t_tho.shape[1]
    print(f"\nA. frozen-recurrent variance (train seq): belief={b_var:.4f} latent={l_var:.4f} mean_thought={o_var:.4f}",
          flush=True)
    # pairwise: does belief separate distinct frames?
    i, j = 0, 100
    d_same_seq = torch.cdist(t_bel[None, i][None], t_bel[None, j][None]).item()
    d_lat = torch.cdist(t_lat[None, i][None], t_lat[None, j][None]).item()
    print(f"   cdist(frame {i}, frame {j}): belief={d_same_seq:.4f} latent={d_lat:.4f}",
          flush=True)

    # ---- C. probe ceilings: belief vs latent, all 4 fields -------------
    print("\nC. probe ceilings (400 epochs, batched, on same data as gate):",
          flush=True)
    loc_idx = {n: i for i, n in enumerate(LOCOMOTION)}
    yaw_idx = {d: i for i, d in enumerate(LOOK_YAW_DEG)}
    btn_idx = {n: i for i, n in enumerate(BUTTON_NAMES)}
    loc = np.array([loc_idx[p[1].locomotion] for p in train_pairs])
    yaw = np.array([yaw_idx[p[1].look_yaw_deg] for p in train_pairs])
    btn = np.array([[float(p[1].buttons[btn_idx[n]]) for n in BUTTON_NAMES]
                    for p in train_pairs])
    print("  (per-field readout probes, belief / latent / mean-thought)", flush=True)
    probe_ceiling(t_bel, loc, 5, tag=" belief", name="locomotion")
    probe_ceiling(t_lat, loc, 5, tag=" latent", name="locomotion")
    probe_ceiling(t_tho, loc, 5, tag=" thought", name="locomotion")
    probe_ceiling(t_bel, yaw, len(LOOK_YAW_DEG), tag=" belief", name="yaw")
    probe_ceiling(t_lat, yaw, len(LOOK_YAW_DEG), tag=" latent", name="yaw")
    probe_ceiling(t_tho, yaw, len(LOOK_YAW_DEG), tag=" thought", name="yaw")
    attack = np.asarray(btn[:, btn_idx['attack']]).astype(int)
    probe_ceiling(t_bel, attack, 2, tag=" belief", name="attack")
    probe_ceiling(t_lat, attack, 2, tag=" latent", name="attack")
    probe_ceiling(t_tho, attack, 2, tag=" thought", name="attack")

    # one-hot full button vector probe (6-dim multilabel)
    print("  (all-6 button multilabel, per-bit accuracy)", flush=True)
    for tag_, feats in (("belief", t_bel), ("latent", t_lat)):
        w = torch.nn.Parameter(torch.randn(feats.shape[1], 6) * 0.01)
        opt = torch.optim.Adam([w], lr=1e-2)
        tgt = torch.from_numpy(btn.astype(np.float32))
        for _ in range(400):
            F.binary_cross_entropy_with_logits(feats.float() @ w, tgt).backward()
            opt.step(); opt.zero_grad()
        with torch.no_grad():
            pred = ((feats.float() @ w) > 0).float().numpy()
        accs = [round(float((pred[:, k] == btn[:, k]).mean()), 4) for k in range(6)]
        print(f"   {tag_}: " + " ".join(f"{BUTTON_NAMES[k]}={a}" for k, a in enumerate(accs)),
              flush=True)

    # ---- B/D. gate procedure re-run with per-field logging -------------
    print("\nB/D. exact gate procedure re-run (seed as in gate: unseeded global "
          "RNG; here we FIX the perm for determinism and report both)", flush=True)
    device = torch.device("cpu")
    state = interface.core.init_state(1, device)
    beliefs, labels = [], []
    for frame, target in train_pairs:
        pixels = _rgb_tensor((frame,), device=device, resolution=(32, 32))
        prev = torch.tensor([0], dtype=torch.long, device=device)
        _, state = interface.core(pixels, state, prev_action=prev)
        b = state.fast.belief
        if b.dim() == 2:
            b = b.squeeze(0)
        beliefs.append(b.detach())
        labels.append(target)
    beliefs = torch.stack(beliefs)
    opt = torch.optim.Adam(list(interface.heads.values()), lr=3e-3, betas=(0.9, 0.999))
    perm_fixed = torch.randperm(beliefs.shape[0], generator=torch.Generator().manual_seed(7))
    for _epoch in range(40):
        total = 0.0
        for idx in perm_fixed:
            i = int(idx)
            belief = beliefs[i : i + 1]
            target = labels[i]
            logits = {f: belief @ interface.heads[f] for f in ("locomotion", "look_yaw_deg", "buttons")}
            loc_logprobs = torch.log_softmax(logits["locomotion"], dim=1)
            yaw_logprobs = torch.log_softmax(logits["look_yaw_deg"], dim=1)
            loss = (
                -loc_logprobs[0, loc_idx[target.locomotion]]
                -yaw_logprobs[0, yaw_idx[target.look_yaw_deg]]
                + F.binary_cross_entropy_with_logits(
                    logits["buttons"][0],
                    torch.tensor([float(target.buttons[btn_idx[btn]]) for btn in BUTTON_NAMES]),
                )
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
    # per-field held-out accuracy
    state = interface.core.init_state(1, device)
    n = len(test_pairs)
    hit = {f: 0 for f in ("locomotion", "yaw", "buttons")}
    for frame, target in test_pairs:
        pixels = _rgb_tensor((frame,), device=device, resolution=(32, 32))
        prev = torch.tensor([0], dtype=torch.long, device=device)
        record, state = interface.emit(frame, state, prev_action=0)
        if record.locomotion == target.locomotion:
            hit["locomotion"] += 1
        if record.look_yaw_deg == target.look_yaw_deg:
            hit["yaw"] += 1
        if record.buttons == target.buttons:
            hit["buttons"] += 1
    print(f"   gate-style 40-epoch per-field held-out (n={n}): "
          + "  ".join(f"{k}={hit[k]}/{n}" for k in hit), flush=True)
    print(f"   composite = {(sum(hit.values()))/(3*n):.4f} vs threshold 0.90", flush=True)

    # what do the heads predict on each test field — confusion summary
    conf = {}
    state = interface.core.init_state(1, device)
    for frame, target in test_pairs:
        pixels = _rgb_tensor((frame,), device=device, resolution=(32, 32))
        prev = torch.tensor([0], dtype=torch.long, device=device)
        record, state = interface.emit(frame, state, prev_action=0)
        key = (target.locomotion, record.locomotion)
        conf[key] = conf.get(key, 0) + 1
    print("   locomotion confusion (target -> pred): "
          + "  ".join(f"{k[0]}->{k[1]}:{v}" for k, v in sorted(conf.items(), key=lambda kv: -kv[1])[:8]),
          flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
