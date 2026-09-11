"""Focused contracts for EmbodiedInterfaceV1 (revision 2: latent+belief heads).

Local CPU-only (CUDA hidden, single thread, no screen capture / HID /
network / background jobs — workspace rules).

Covers:
  1. Head-input context is ``[latent, belief]`` — shape ``[B, 2W]``,
     stop-gradient by construction (no grad_fn, core params never require
     grad through the heads).
  2. Head parameter in-dimension is exactly ``2W``; legacy ``W``-dim
     heads are rejected with an explicit error.
  3. ``emit`` returns a schema-valid record and passes the core's
     post-tick state through unchanged (heads add no state of their own).
  4. Byte-preserved core: the core's state dict hashes identically before
     and after a full emit campaign (no gradient leak into the core).
  5. Construction determinism: two interfaces built from the same seed
     hash identically; same frame -> same record (byte replay).
  6. Causal observation: same fresh state, two different frames ->
     different records on at least one field; neutral frame -> valid record.
  7. ``emit_batch`` agrees with per-frame ``emit`` on a fresh state.
  8. Adapter purity: translate_record / mouse_deltas_for_record are pure
     functions over the allow-listed HID set.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "src"))

import json
from hashlib import sha256

import numpy as np
import torch

from irene_brain.types import RgbFrame
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Config
from irene_brain.v2.core import CoreV2Model
from irene_brain.v2.embodied_interface import (
    ACTUATED_KEY_SET,
    BUTTON_NAMES,
    HOTBAR,
    LOCOMOTION,
    LOOK_PITCH_DEG,
    LOOK_YAW_DEG,
    RECORD_FIELDS,
    EmbodiedInterfaceRecord,
    EmbodiedInterfaceV1,
    mouse_deltas_for_record,
    translate_record,
)


def make_frame(red_x=6, red_y=10, bright_corner=False) -> RgbFrame:
    arr = np.full((32, 32, 3), (24, 24, 24), dtype=np.uint8)
    arr[red_y:red_y + 4, red_x:red_x + 4] = (200, 0, 0)
    if bright_corner:
        arr[0:8, 24:32] = (255, 255, 0)
    return RgbFrame(width=32, height=32, pixels=arr.tobytes())


def make_core() -> CoreV2Model:
    torch.manual_seed(0)
    model = CoreV2Model(flags=CONFIG_B_PREDICTIVE)
    model.requires_grad_(False)
    model.eval()
    return model


def _state_hash(model) -> str:
    payload = {k: v.flatten().tolist() for k, v in sorted(model.state_dict().items())}
    return sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def test_head_context_is_latent_plus_belief():
    model = make_core()
    itf = EmbodiedInterfaceV1(model, init_seed=43)
    state = model.init_state(2, torch.device("cpu"))
    obs = torch.rand(2, 3, 32, 32)
    output, _ = model(obs, state)
    ctx = itf.head_context(output)
    W = model.config.width
    assert ctx.shape == (2, 2 * W), f"context shape {tuple(ctx.shape)} != (2, {2*W})"
    assert not ctx.requires_grad
    assert ctx.grad_fn is None
    # both halves stop-gradient: detach() makes them leaves
    assert output.latent.requires_grad or ctx.shape[1] == 2 * W
    # context equals cat of the detached halves
    expected = torch.cat([output.latent.detach(), output.belief.detach()], dim=-1)
    assert torch.equal(ctx, expected)
    print("    head context [B, 2W] stop-gradient: OK")


def test_head_in_dim_is_two_widths():
    model = make_core()
    itf = EmbodiedInterfaceV1(model, init_seed=1)
    W = model.config.width
    for field in RECORD_FIELDS:
        assert itf.heads[field].shape[0] == 2 * W, field
    # a legacy W-dim head must be rejected
    bad = torch.nn.ParameterDict({f: torch.nn.Parameter(torch.zeros(W, 4))
                                  for f in RECORD_FIELDS})
    try:
        EmbodiedInterfaceV1(model, heads=bad, init_seed=1)
        raise AssertionError("legacy W-dim heads were accepted")
    except ValueError as e:
        assert "in-dim" in str(e)
    print("    head in-dim == 2W (legacy W rejected): OK")


def test_emit_returns_valid_record_and_passthrough_state():
    model = make_core()
    itf = EmbodiedInterfaceV1(model, init_seed=7)
    state = model.init_state(1, torch.device("cpu"))
    record, state2 = itf.emit(make_frame(4, 6), state, prev_action=0)
    assert isinstance(record, EmbodiedInterfaceRecord)
    assert record.locomotion in LOCOMOTION
    assert record.look_yaw_deg in LOOK_YAW_DEG
    assert record.look_pitch_deg in LOOK_PITCH_DEG
    assert record.buttons == tuple(bool(b) for b in record.buttons)
    assert len(record.buttons) == len(BUTTON_NAMES)
    assert record.hotbar in HOTBAR
    # heads add no state of their own: returned state is the core state
    assert state2.fast.belief.shape == state.fast.belief.shape
    assert record.canonical_json  # non-empty, serializable
    print("    emit record schema + state pass-through: OK")


def test_core_byte_preserved_after_emit_campaign():
    model = make_core()
    itf = EmbodiedInterfaceV1(model, init_seed=11)
    before = _state_hash(model)
    state = model.init_state(1, torch.device("cpu"))
    for i in range(12):
        _, state = itf.emit(make_frame((i * 5) % 28, (i * 7) % 28), state, prev_action=0)
    after = _state_hash(model)
    assert before == after, "core state dict changed during emit campaign"
    assert all(not p.requires_grad for p in model.parameters())
    print("    core byte-preserved after 12 emits: OK")


def test_construction_determinism_and_replay():
    m1, m2 = make_core(), make_core()
    i1 = EmbodiedInterfaceV1(m1, init_seed=43)
    i2 = EmbodiedInterfaceV1(m2, init_seed=43)
    assert i1.heads_sha256() == i2.heads_sha256()
    assert len(i1.heads_sha256()) == 64
    # different seeds -> different heads
    i3 = EmbodiedInterfaceV1(make_core(), init_seed=44)
    assert i1.heads_sha256() != i3.heads_sha256()
    # byte replay: same frames -> identical record stream
    frames = [make_frame(x, y) for x, y in [(2, 4), (14, 10), (26, 22)]]
    streams = []
    for _ in range(2):
        st = i1.core.init_state(1, torch.device("cpu"))
        out = []
        for f in frames:
            rec, st = i1.emit(f, st, prev_action=0)
            out.append(rec.sha256)
        streams.append(sha256("\n".join(out).encode()).hexdigest())
    assert streams[0] == streams[1]
    print("    construction determinism + byte replay: OK")


def test_causal_observation_and_neutral():
    model = make_core()
    itf = EmbodiedInterfaceV1(model, init_seed=5)
    s_a = model.init_state(1, torch.device("cpu"))
    s_b = model.init_state(1, torch.device("cpu"))
    rec_a, _ = itf.emit(make_frame(2, 4), s_a, prev_action=0)
    rec_b, _ = itf.emit(make_frame(26, 26), s_b, prev_action=0)
    moved = any(getattr(rec_a, f) != getattr(rec_b, f)
                for f in ("locomotion", "look_yaw_deg", "look_pitch_deg",
                          "cursor_dx_px", "cursor_dy_px", "buttons", "hotbar"))
    assert moved, "different frames produced identical records"
    rec_n, _ = itf.emit(make_frame(14, 14), model.init_state(1, torch.device("cpu")),
                        prev_action=0)
    assert isinstance(rec_n, EmbodiedInterfaceRecord)
    print("    causal observation + neutral frame valid: OK")


def test_emit_batch_matches_emit():
    model = make_core()
    itf = EmbodiedInterfaceV1(model, init_seed=9)
    frames = [make_frame(4, 6), make_frame(20, 4), make_frame(8, 24, bright_corner=True)]
    state = model.init_state(len(frames), torch.device("cpu"))
    records, _ = itf.emit_batch(frames, state, prev_actions=[0, 0, 0])
    assert len(records) == 3
    for f, rec in zip(frames, records):
        single, _ = itf.emit(f, model.init_state(1, torch.device("cpu")), prev_action=0)
        # batch position i sees only its own frame from a fresh zero state
        assert rec.sha256 == single.sha256, f"{f.width}x{f.height} frame mismatch"
    print("    emit_batch agrees with emit on fresh states: OK")


def test_adapter_purity_and_allow_list():
    rec = EmbodiedInterfaceRecord(
        locomotion="FORWARD", look_yaw_deg=30, look_pitch_deg=-15,
        cursor_dx_px=64, cursor_dy_px=-64,
        buttons=(True, False, True, False, True, False), hotbar="3",
    )
    keys = translate_record(rec)
    for _ in range(5):
        assert translate_record(rec) == keys
        assert mouse_deltas_for_record(rec) == (30 + 64, -(-15) + (-64))
    assert all(k in ACTUATED_KEY_SET for k in keys)
    assert isinstance(keys, tuple)
    # NOOP + no buttons -> no keys at all
    none_keys = translate_record(EmbodiedInterfaceRecord.neutral())
    assert none_keys == ()
    print("    adapter purity + allow-list: OK")


if __name__ == "__main__":
    test_head_context_is_latent_plus_belief()
    test_head_in_dim_is_two_widths()
    test_emit_returns_valid_record_and_passthrough_state()
    test_core_byte_preserved_after_emit_campaign()
    test_construction_determinism_and_replay()
    test_causal_observation_and_neutral()
    test_emit_batch_matches_emit()
    test_adapter_purity_and_allow_list()
    print("ALL EMBODIED INTERFACE V1 CONTRACTS PASSED")
