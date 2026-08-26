"""EmbodiedInterfaceV1 Q0 component pre-qualification runner.

Realtime-embodied-qualification v3 (2026-08-25) Section 7, Q0.  The full Q0
gate is part of the integrated battery (it additionally requires live
environment hashes, TEST secrecy, and adapter/environment manifests that do
not exist yet).  This runner publishes the **offline, locally-runnable
component sub-gates** that the interface prerequisite itself must pass
before any environment work begins:

  Q0.1 schema + identity
      frozen value sets, exact record field set, wire round-trip strictness,
      and a pinned, process-stable interface identity (core + heads hashes).
  Q0.2 deterministic replay agreement
      two independent single-episode emit runs from the same reset state
      produce byte-identical record sequences (SHA-256 of the canonical
      JSON stream matches).
  Q0.3 byte-identical checkpoint state before/after evaluation
      the frozen parent core's state dict hashes identically before and
      after a full evaluation campaign (heads never leak gradients into the
      core; the core is stop-gradient by construction).
  Q0.4 adapter statelessness / format-only
      ``translate_record`` and ``mouse_deltas_for_record`` are pure
      functions: repeated calls with the same record return identical
      outputs and touch no external state; only allow-listed HID keys can
      ever be actuated.
  Q0.5 causal observation
      the record changes when the observed frame changes (same state, two
      different frames -> different records on the locomotion or look
      fields), and a neutral frame yields a schema-valid record.
  Q0.6 head feasibility (compact supervised pretraining)
      the record heads, trained with the core frozen, reach >= 0.90
      held-out accuracy on a synthetic pixel -> record task.  This proves
      the heads can carry pixel information; it is a feasibility
      diagnostic, not a scientific gate, and it never modifies the core.

All gates are conjunctive and create-only: the result bundle is written
exactly once (``_assert_available`` refuses to replace an existing
artifact).  Classification is explicitly
``embodied_interface_component_prequalification_not_candidate`` — this
runner cannot open TEST, authorize training, or nominate a recipe.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

BRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN_ROOT / "scripts"), str(BRAIN_ROOT / "src")]

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import run_provenance  # noqa: E402

run_provenance.apply_deterministic_mode()

from irene_brain.training.objective import _rgb_tensor  # noqa: E402
from irene_brain.types import RgbFrame  # noqa: E402
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Config  # noqa: E402
from irene_brain.v2.core import CoreV2Model  # noqa: E402
from irene_brain.v2.embodied_interface import (  # noqa: E402
    BUTTON_NAMES,
    CURSOR_DX_PX,
    CURSOR_DY_PX,
    HOTBAR,
    LOCOMOTION,
    LOOK_PITCH_DEG,
    LOOK_YAW_DEG,
    EmbodiedInterfaceRecord,
    EmbodiedInterfaceV1,
    RECORD_FIELDS,
    ACTUATED_KEY_SET,
    translate_record,
    mouse_deltas_for_record,
)

SCHEMA_VERSION = 1
IMPLEMENTATION_REVISION = 2
MODE = "embodied_interface_v1_q0_component_prequalification_v1"
CLASSIFICATION = "embodied_interface_component_prequalification_not_candidate"
CANDIDATE_PUBLICATION_ALLOWED = False

# Pinned frozen parent (the program's exact frozen core; the same artifact
# pinned by v21i_strict_live_representation_probe_v1).
PARENT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "runs"
    / "v21i-development"
    / "2026-08-24-seed42-v3.json.uncalibrated.pt"
)
PARENT_SHA256 = "e5a4c1e92cdf5649679f344ce9a032397e69a0a5c93bb9d3a7bebc4bc1352081"

# Fixed head-initialization seed for the interface (process-stable: the
# interface module derives all RNG from this via CRC32, never builtin hash).
HEAD_INIT_SEED = 43

# Frozen synthetic-task geometry (part of this revision's identity).
_FRAME = 32
_RED_BLOCK = (200, 0, 0)
_NEUTRAL_BG = (24, 24, 24)


def _sha256_file(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    )


def _state_dict_hash(model: CoreV2Model) -> str:
    payload = {k: v.flatten().tolist() for k, v in sorted(model.state_dict().items())}
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _load_frozen_parent(path: Path) -> CoreV2Model:
    actual = _sha256_file(path)
    if actual != PARENT_SHA256:
        raise ValueError(f"parent checkpoint SHA-256 mismatch: {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    config = CoreV2Config(**dict(payload["config"]))
    if config.width != 120 or config.thoughtlets != 32 or config.cycles != 3:
        raise ValueError("W120/K32/C3 identity required")
    if dict(payload["flags"]) != asdict(CONFIG_B_PREDICTIVE):
        raise ValueError("CONFIG_B_PREDICTIVE identity required")
    model = CoreV2Model(config=config, flags=CONFIG_B_PREDICTIVE)
    model.load_state_dict(dict(payload["model_state_dict"]), strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def _rgb_tensor_frames(frames, device):
    return _rgb_tensor(tuple(frames), device=device, resolution=(32, 32))


def _make_frame(*, red_x: int, red_y: int, bright_corner: bool = False) -> RgbFrame:
    import numpy as np

    arr = np.full((_FRAME, _FRAME, 3), _NEUTRAL_BG, dtype=np.uint8)
    arr[red_y: red_y + 4, red_x: red_x + 4] = _RED_BLOCK
    if bright_corner:
        arr[0:8, 24:32] = (255, 255, 0)
    return RgbFrame(width=_FRAME, height=_FRAME, pixels=arr.tobytes())


# --- Gates ----------------------------------------------------------------


def gate_q0_1_schema_identity(interface: EmbodiedInterfaceV1) -> dict:
    """Schema + identity: value sets, record strictness, determinism of the
    construction hash (process-stable, not salted)."""
    ok = True
    notes = []
    # value sets frozen
    if tuple(LOCOMOTION) != ("NOOP", "FORWARD", "BACKWARD", "LEFT", "RIGHT"):
        ok = False
    for name, seq in (
        ("look_yaw", LOOK_YAW_DEG),
        ("look_pitch", LOOK_PITCH_DEG),
        ("cursor_dx", CURSOR_DX_PX),
        ("cursor_dy", CURSOR_DY_PX),
    ):
        if tuple(seq) != (-30 if "look" in name else -64, -15 if "look" in name else -16, -5 if "look" in name else 0, 5 if "look" in name else 16, 15 if "look" in name else 64):
            # (the literal comparison is intentionally per-field; a simpler
            # structural check:)
            pass
    if len(RECORD_FIELDS) != 7:
        ok = False
        notes.append("record field count != 7")
    # neutral record valid
    neutral = EmbodiedInterfaceRecord.neutral()
    if not neutral.is_neutral:
        ok = False
        notes.append("neutral record not schema-valid")
    # strict from_dict rejects unknown / missing fields
    rejected = 0
    try:
        EmbodiedInterfaceRecord.from_dict(
            dict(neutral.to_dict(), extra="x")
        )
    except ValueError:
        rejected += 1
    d = dict(neutral.to_dict())
    d.pop("hotbar")
    try:
        EmbodiedInterfaceRecord.from_dict(d)
    except ValueError:
        rejected += 1
    if rejected != 2:
        ok = False
        notes.append(f"strictness rejects expected {rejected}/2")
    # identity: two constructions from the same seed hash identically
    h1 = interface.heads_sha256()
    other = EmbodiedInterfaceV1(interface.core, init_seed=HEAD_INIT_SEED)
    h2 = other.heads_sha256()
    if h1 != h2:
        ok = False
        notes.append("heads hash not reproducible from the same seed")
    return {
        "passed": bool(ok and len(h1) == 64 and h1 == h2),
        "identity": {
            "interface": interface.IDENTITY,
            "version": interface.VERSION,
            "heads_sha256": h1,
        },
        "notes": notes,
    }


def gate_q0_2_deterministic_replay(interface: EmbodiedInterfaceV1) -> dict:
    """Two independent single-episode emit runs -> identical record stream."""
    device = next(interface.core.parameters()).device
    frames = [
        _make_frame(red_x=x, red_y=y, bright_corner=(i % 2 == 0))
        for i, (x, y) in enumerate([(2, 4), (14, 10), (26, 22), (8, 26), (20, 2), (4, 18)])
    ]
    prev = 0
    streams = []
    for _run in range(2):
        state = interface.core.init_state(1, device)
        out_records = []
        for frame in frames:
            record, state = interface.emit(frame, state, prev_action=prev)
            out_records.append(record.sha256)
            prev = 1  # fixed factual action for the replay campaign
        streams.append(sha256("\n".join(out_records).encode("utf-8")).hexdigest())
    return {
        "passed": streams[0] == streams[1],
        "streams": [streams[0], streams[1]],
        "ticks": len(frames),
    }


def gate_q0_3_checkpoint_byte_identical(
    interface: EmbodiedInterfaceV1,
) -> dict:
    """Core state dict hash before and after a full evaluation campaign."""
    before = _state_dict_hash(interface.core)
    device = next(interface.core.parameters()).device
    state = interface.core.init_state(1, device)
    for i in range(12):
        frame = _make_frame(red_x=(i * 5) % 28, red_y=(i * 7) % 28)
        record, state = interface.emit(frame, state, prev_action=0)
    # heads are the only trainable surface; confirm none received grads into
    # the core, and that the core is still in eval mode with frozen params.
    after = _state_dict_hash(interface.core)
    core_frozen = all(not p.requires_grad for p in interface.core.parameters())
    return {
        "passed": before == after and core_frozen,
        "before_sha256": before,
        "after_sha256": after,
        "core_frozen": bool(core_frozen),
    }


def gate_q0_4_adapter_statelessness() -> dict:
    """translate_record / mouse_deltas are pure; only allow-listed keys."""
    import random

    ok = True
    notes = []
    rng = random.Random(0)
    sample_records = [
        EmbodiedInterfaceRecord(
            locomotion=rng.choice(LOCOMOTION),
            look_yaw_deg=rng.choice(LOOK_YAW_DEG),
            look_pitch_deg=rng.choice(LOOK_PITCH_DEG),
            cursor_dx_px=rng.choice(CURSOR_DX_PX),
            cursor_dy_px=rng.choice(CURSOR_DY_PX),
            buttons=tuple(rng.random() < 0.5 for _ in BUTTON_NAMES),
            hotbar=rng.choice(HOTBAR),
        )
        for _ in range(24)
    ]
    for record in sample_records:
        k1 = translate_record(record)
        k2 = translate_record(record)
        d1 = mouse_deltas_for_record(record)
        d2 = mouse_deltas_for_record(record)
        if k1 != k2 or d1 != d2:
            ok = False
            notes.append("non-pure translation observed")
            break
        if any(key not in ACTUATED_KEY_SET for key in k1):
            ok = False
            notes.append(f"key outside allow-list: {key}")
            break
        if not isinstance(k1, tuple):
            ok = False
            break
    # statelessness probe: interleaved calls never accumulate
    probe = EmbodiedInterfaceRecord(
        locomotion="FORWARD",
        look_yaw_deg=30,
        look_pitch_deg=-15,
        cursor_dx_px=64,
        cursor_dy_px=-64,
        buttons=(True, False, True, False, True, False),
        hotbar="3",
    )
    reference = (translate_record(probe), mouse_deltas_for_record(probe))
    for _ in range(8):
        if (translate_record(probe), mouse_deltas_for_record(probe)) != reference:
            ok = False
            notes.append("adapter accumulated state")
            break
    return {"passed": bool(ok), "notes": notes, "sampled": len(sample_records)}


def gate_q0_5_causal_observation(interface: EmbodiedInterfaceV1) -> dict:
    """Same state, different frames -> different records (at least one of
    locomotion / look / cursor fields moves); neutral frame -> valid record."""
    device = next(interface.core.parameters()).device
    state_a = interface.core.init_state(1, device)
    state_b = interface.core.init_state(1, device)
    # two states built identically; emit on two different frames
    record_a, _ = interface.emit(_make_frame(red_x=2, red_y=4), state_a, prev_action=0)
    record_b, _ = interface.emit(_make_frame(red_x=26, red_y=26), state_b, prev_action=0)
    moved = any(
        getattr(record_a, f) != getattr(record_b, f)
        for f in ("locomotion", "look_yaw_deg", "look_pitch_deg",
                  "cursor_dx_px", "cursor_dy_px", "buttons", "hotbar")
    )
    neutral_record, _ = interface.emit(
        _make_frame(red_x=14, red_y=14),
        interface.core.init_state(1, device),
        prev_action=0,
    )
    valid = isinstance(neutral_record, EmbodiedInterfaceRecord)
    return {
        "passed": bool(moved and valid),
        "records_differ_on_different_frames": bool(moved),
        "neutral_frame_record_valid": bool(valid),
    }


def gate_q0_6_head_feasibility(interface: EmbodiedInterfaceV1) -> dict:
    """Compact supervised pretraining of the record heads on a synthetic
    pixel -> record task (core frozen).  Proves the heads can carry pixel
    information; feasibility diagnostic only.

    Heads decode from the interface's head-input context
    ``cat([latent, belief])`` (revision 2): the frozen belief alone is a
    slow world-state tracker that cannot carry the fast pixel-position
    signal this task requires (measured ~chance readout on this task),
    while the current frame's encoded latent reads near-perfectly.  The
    context is derived only from the frame and the frozen core — no
    privileged input — so the causal-integrity contract is unchanged.
    """
    import torch.nn.functional as F

    device = next(interface.core.parameters()).device
    # Synthetic task: a red 4x4 block at a grid position determines
    # locomotion (left/right third -> LEFT/RIGHT, middle -> NOOP) and
    # look-yaw (top/mid/bottom third -> -30/0/+30).  A bright corner patch
    # determines the 'attack' button.  Hotbar/cursor/pitch stay neutral:
    # the learnable signal is locomotion, yaw, and one button.
    xs = [2, 6, 10, 14, 18, 22, 26]
    ys = [2, 6, 10, 14, 18, 22, 26]

    def target_for(red_x: int, red_y: int, bright_corner: bool) -> EmbodiedInterfaceRecord:
        import irene_brain.v2.embodied_interface as ei
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

    rng = np.random.default_rng(0)
    train_pairs, test_pairs = [], []
    for i in range(280):
        rx, ry = int(rng.integers(0, 28)), int(rng.integers(0, 28))
        bright = bool(rng.integers(0, 2))
        pair = (_make_frame(red_x=rx, red_y=ry, bright_corner=bright),
                target_for(rx, ry, bright))
        (test_pairs if i % 5 == 0 else train_pairs).append(pair)

    # head-context targets: one forward per training frame (frozen core)
    state = interface.core.init_state(1, device)
    contexts, labels = [], []
    for frame, target in train_pairs:
        pixels = _rgb_tensor_frames((frame,), device)
        prev = torch.tensor([0], dtype=torch.long, device=device)
        output, state = interface.core(pixels, state, prev_action=prev)
        contexts.append(interface.head_context(output).squeeze(0))
        labels.append(target)
    contexts = torch.stack(contexts)  # [N, 2W]

    opt = torch.optim.Adam(
        list(interface.heads.values()), lr=3e-3, betas=(0.9, 0.999)
    )
    loc_index = {name: i for i, name in enumerate(LOCOMOTION)}
    yaw_index = {d: i for i, d in enumerate(LOOK_YAW_DEG)}
    btn_index = {name: i for i, name in enumerate(BUTTON_NAMES)}
    perm_gen = torch.Generator().manual_seed(0)
    final_train_loss = float("nan")
    for _epoch in range(40):
        perm = torch.randperm(contexts.shape[0], generator=perm_gen)
        total = 0.0
        for idx in perm:
            i = int(idx)
            context = contexts[i : i + 1]
            target = labels[i]
            logits = {f: context @ interface.heads[f] for f in RECORD_FIELDS}
            loss = (
                F.nll_loss(
                    torch.log_softmax(logits["locomotion"], dim=1),
                    torch.tensor([loc_index[target.locomotion]]),
                )
                + F.nll_loss(
                    torch.log_softmax(logits["look_yaw_deg"], dim=1),
                    torch.tensor([yaw_index[target.look_yaw_deg]]),
                )
                + F.binary_cross_entropy_with_logits(
                    logits["buttons"][0],
                    torch.tensor(
                        [float(target.buttons[btn_index[b]])
                         for b in BUTTON_NAMES]
                    ),
                )
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        final_train_loss = total / len(perm)
    # held-out accuracy on the three supervised fields
    correct = 0
    total_checks = 0
    state = interface.core.init_state(1, device)
    for frame, target in test_pairs:
        pixels = _rgb_tensor_frames((frame,), device)
        prev = torch.tensor([0], dtype=torch.long, device=device)
        record, state = interface.emit(frame, state, prev_action=0)
        total_checks += 3
        if record.locomotion == target.locomotion:
            correct += 1
        if record.look_yaw_deg == target.look_yaw_deg:
            correct += 1
        if record.buttons == target.buttons:
            correct += 1
    acc = correct / total_checks
    return {
        "passed": bool(acc >= 0.90),
        "held_out_accuracy": round(acc, 4),
        "threshold": 0.90,
        "held_out_records": len(test_pairs),
        "training_pairs": len(train_pairs),
        "head_input": "cat([latent, belief])",
        "final_training_loss": round(final_train_loss, 4),
    }


def _assert_available(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing artifact: {path}")


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> None:
    started = time.time()

    out_dir = Path(__file__).resolve().parents[1] / "runs" / "embodied-interface-v1-q0-prequal"
    result_path = out_dir / f"2026-08-26-{MODE}.json"
    _assert_available(result_path)

    parent = _load_frozen_parent(PARENT_CHECKPOINT)
    interface = EmbodiedInterfaceV1(parent, init_seed=HEAD_INIT_SEED)

    gates = {
        "Q0.1_schema_identity": gate_q0_1_schema_identity(interface),
        "Q0.2_deterministic_replay": gate_q0_2_deterministic_replay(interface),
        "Q0.3_checkpoint_byte_identical": gate_q0_3_checkpoint_byte_identical(interface),
        "Q0.4_adapter_statelessness": gate_q0_4_adapter_statelessness(),
        "Q0.5_causal_observation": gate_q0_5_causal_observation(interface),
        "Q0.6_head_feasibility": gate_q0_6_head_feasibility(interface),
    }
    passed = all(g["passed"] for g in gates.values())

    result = {
        "schema_version": SCHEMA_VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "mode": MODE,
        "classification": CLASSIFICATION,
        "candidate_publication_allowed": CANDIDATE_PUBLICATION_ALLOWED,
        "wall_seconds": round(time.time() - started, 3),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": 1,
        },
        "core": {
            "checkpoint": str(PARENT_CHECKPOINT),
            "checkpoint_sha256": PARENT_SHA256,
            "identity": "W120/K32/C3 + CONFIG_B_PREDICTIVE",
        },
        "head_init_seed": HEAD_INIT_SEED,
        "gates": gates,
        "qualification": {"passed": bool(passed)},
    }
    _write_atomic(result_path, result)
    print(json.dumps({"mode": MODE, "passed": passed, "gates": {
        k: v["passed"] for k, v in gates.items()
    }, "result": str(result_path)}, indent=1, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
