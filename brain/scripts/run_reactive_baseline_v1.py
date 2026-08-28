"""T1-4: train ReactiveBaselineV1 on the shared TRAIN corpus (preregistration).

Preregistration: ``brain/docs/preregistrations/2026-08-28-pacman-reactive-\
baseline-v1.md`` (frozen 2026-08-28).

Trains the frozen ``ReactiveBaselineV1`` (``brain/src/irene_brain/v2/\
reactive_baseline.py``, 133,773 params, 4-frame + prev-action context) on the
T1-3 corpus (``brain/datasets/embodied-corpus-v1/``,
``corpus_sha256=59c41b6dc71f…``) under the candidate's budget class:

- AdamW(lr=5e-4, weight_decay=1e-4), grad-clip max_norm=1.0, batch 16;
- exactly 6000 optimizer steps per seed;
- confirmatory seed cohort [42, 142, 242, 342];
- 5-class cross-entropy on the teacher's applied action class (same target
  distribution as the candidate's BC).

Supervision (prereg §2, frozen): sample at tick t (t >= 1) is
  - frames: last 4 rendered frames up to and including tick t (right-aligned;
    left-padded with the episode's first frame when history < 4);
  - prev_action: the applied action at tick t-1 (corpus ``prev_action[t]``);
  - label: the teacher's applied action at tick t (corpus ``target_action[t]``).

Determinism (prereg §4): strict deterministic mode; single-thread
(``torch.set_num_threads(1)`` + OMP/MKL pinned to 1); model init from
``torch.Generator`` seeded by ``zlib.crc32(f"reactive.<seed>")`` (NEVER
builtin hash()); corpus sample order shuffled by a ``torch.Generator`` seeded
``zlib.crc32(f"reactive.sample_order.<seed>")``. Gate R verifies two
in-process runs at seed 42 byte-match (final parameter tensors + per-step loss
curve).

Compute policy: local CPU ONLY (2026-08-26 decision record); CUDA hidden;
runs at below-normal OS priority so interactive desktop work (gaming) is
unaffected. No DEV/CAL/TEST, no candidate training, no DGX dispatch.

Outputs:
  brain/runs/embodied-reactive-baseline-v1/seed_<s>.pt   (create-only, x4)
  brain/runs/embodied-reactive-baseline-v1/
      2026-08-28-reactive-baseline-v1.json               (create-only report)

Usage (from brain/):
  py -3.11 scripts/run_reactive_baseline_v1.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import zlib
from hashlib import sha256
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

_HERE = Path(__file__).resolve()
BRAIN_ROOT = _HERE.parent.parent
sys.path.insert(0, str(BRAIN_ROOT / "src"))
sys.path.insert(0, str(BRAIN_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from run_provenance import apply_deterministic_mode, provenance  # noqa: E402

apply_deterministic_mode()

# Single-thread, pinned: determinism AND gaming-polite (one core, below-normal
# priority). The corpus generator (T1-3) ran under the same thread pinning.
# Order matters: interop before compute threads. Guarded: a caller may have
# already pinned threads (harness/re-run context).
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
try:
    torch.set_num_threads(1)
except RuntimeError:
    pass

# Below-normal OS priority so interactive desktop work (gaming) is unaffected.
# Best-effort: ignored if the platform has no API; never fatal.
try:  # Windows
    from ctypes import windll  # type: ignore

    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    windll.kernel32.SetPriorityClass(
        windll.kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS
    )
except Exception:  # non-Windows or unsupported
    pass

from irene_brain.v2.reactive_baseline import (  # noqa: E402
    ACTION_CLASSES,
    N_FRAMES,
    ReactiveBaselineV1,
)

# --- Frozen constants (preregistration 2026-08-28-pacman-reactive-baseline-\
# --- v1.md §1–§3; change requires a NEW registration, never an edit) -------

TOTAL_STEPS = 6000          # §3: the V2.0j TOTAL_STEPS budget
BATCH_SIZE = 16             # §3
LR = 5e-4                   # §3
WEIGHT_DECAY = 1e-4         # §3
GRAD_CLIP_MAX_NORM = 1.0    # §3
SEEDS = (42, 142, 242, 342)  # §3 confirmatory seed cohort
CORPUS_RELPATH = BRAIN_ROOT / "datasets" / "embodied-corpus-v1"
CORPUS_SHA256_PREFIX = "59c41b6dc71f"  # T1-3 manifest digest prefix
RESULT_RELPATH = (
    BRAIN_ROOT / "runs" / "embodied-reactive-baseline-v1"
    / "2026-08-28-reactive-baseline-v1.json"
)


def stable_seed(name: str) -> int:
    """Process-stable seed (zlib.crc32; NEVER builtin hash()). Prereg §4."""
    return int(zlib.crc32(name.encode()) & 0xFFFFFFFF)


def load_corpus() -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load the T1-3 corpus: per-episode frames [T,16,16,3] uint8 and
    (prev_action, target_action) int8 [T,2]. Only these two field families
    feed the model (C8); meta/ (reward/event/terminated/truncated) is never
    read by this script."""
    manifest = json.loads((CORPUS_RELPATH / "manifest.json").read_text())
    frames, actions = [], []
    for rec in manifest["episodes"]:
        frames.append(np.load(CORPUS_RELPATH / "frames" / f"{rec['name']}.npy"))
        actions.append(np.load(CORPUS_RELPATH / "actions" / f"{rec['name']}.npy"))
    return frames, actions


def build_sample_indices(frames: list[np.ndarray]) -> list[tuple[int, int]]:
    """Global (episode, tick) sample list, prereg §2: tick t runs 1..T-1."""
    idx: list[tuple[int, int]] = []
    for e, fr in enumerate(frames):
        for t in range(1, fr.shape[0]):
            idx.append((e, t))
    return idx


def fetch_batch(
    frames: list[np.ndarray],
    actions: list[np.ndarray],
    batch: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One frozen sample batch: (frames [B,4,3,16,16] float32 in [0,1],
    prev_action [B] long, labels [B] long). Input tensors are built ONLY from
    the corpus frame tensors and the previous-applied-action column (C8)."""
    frs, prevs, labs = [], [], []
    for e, t in batch:
        fr = frames[e]
        start = t - (N_FRAMES - 1)  # oldest tick in the 4-frame window ending at t
        # Right-aligned last-4 window ending at tick t; any tick < 0 left-pads
        # with the episode's reset frame (prereg §2). fr[max(start+i, 0)] is
        # fr[0] when start+i < 0, else fr[start+i].
        win = [fr[max(start + i, 0)] for i in range(N_FRAMES)]
        # corpus frames are [16,16,3] RGB; the model expects [n,3,16,16]
        # channels-first (ReactiveBaselineV1.forward contract).
        frs.append(np.stack(win).transpose(0, 3, 1, 2).astype(np.float32) / 255.0)
        prevs.append(int(actions[e][t, 0]))  # applied action at tick t-1
        labs.append(int(actions[e][t, 1]))  # teacher's applied action at tick t
    # C8 audit tag: the ONLY model-input construction lines in this script
    # (tagged below; the gate-S audit greps this tag and expects exactly
    # three matches, so nothing else in the file may contain the literal).
    frames_t = torch.from_numpy(np.stack(frs))  # c8-input: frames
    prevs_t = torch.tensor(prevs, dtype=torch.long)  # c8-input: prev_action
    return frames_t, prevs_t, torch.tensor(labs, dtype=torch.long)  # c8-input: labels


def param_fingerprint(model: torch.nn.Module) -> str:
    """Byte fingerprint of every parameter tensor (gate R byte-match)."""
    h = sha256()
    for p in model.parameters():
        h.update(p.detach().to("cpu").numpy().tobytes())
    return h.hexdigest()


def train_seed(
    seed: int,
    sample_indices: list[tuple[int, int]],
    frames: list[np.ndarray],
    actions: list[np.ndarray],
) -> dict:
    """One frozen training run: 6000 steps of batch-16 cross-entropy on the
    seed-shuffled corpus sample order."""
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    order_rng = torch.Generator()
    order_rng.manual_seed(stable_seed(f"reactive.sample_order.{seed}"))
    perm = torch.randperm(len(sample_indices), generator=order_rng).tolist()

    model = ReactiveBaselineV1(seed=stable_seed(f"reactive.{seed}"))
    assert model.num_parameters == 133773, (
        f"frozen param count violated: {model.num_parameters}"
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    losses = []
    step = 0
    t0 = time.perf_counter()
    while step < TOTAL_STEPS:
        chunk = [sample_indices[j] for j in perm[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]]
        frames_t, prevs_t, labels_t = fetch_batch(frames, actions, chunk)
        out = model(frames_t, prevs_t)
        loss = loss_fn(out, labels_t)
        loss_value = float(loss.item())
        if not np.isfinite(loss_value):
            raise FloatingPointError(f"non-finite loss at step {step} seed {seed}")
        opt.zero_grad()
        loss.backward()
        gnorm = float(
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM).item()
        )
        if not np.isfinite(gnorm):
            raise FloatingPointError(f"non-finite grad norm at step {step} seed {seed}")
        opt.step()
        losses.append(loss_value)
        step += 1
        if step % 500 == 0:
            print(
                f"  seed {seed} step {step}/{TOTAL_STEPS} "
                f"loss={loss_value:.5f} ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )

    ckpt_path = RESULT_RELPATH.parent / f"seed_{seed}.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "seed": seed,
         "param_sha256": param_fingerprint(model)},
        ckpt_path,
    )
    return {
        "seed": seed,
        "param_sha256": param_fingerprint(model),
        "checkpoint": str(ckpt_path.relative_to(BRAIN_ROOT.parent)),
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "initial_loss": round(losses[0], 5),
        "final_loss": round(losses[-1], 5),
        "mean_first100": round(float(np.mean(losses[:100])), 5),
        "mean_last100": round(float(np.mean(losses[-100:])), 5),
        "loss_curve": [round(x, 6) for x in losses],
    }


def corpus_recall(
    seed: int,
    sample_indices: list[tuple[int, int]],
    frames: list[np.ndarray],
    actions: list[np.ndarray],
) -> dict:
    """Self-evaluation on its own TRAIN corpus (gate U; NOT a qualification
    claim — prereg §6.4): per-class recall over all t>=1 samples."""
    model = ReactiveBaselineV1(seed=stable_seed(f"reactive.{seed}"))
    sd = torch.load(RESULT_RELPATH.parent / f"seed_{seed}.pt")["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    correct = np.zeros(ACTION_CLASSES, dtype=np.int64)
    total = np.zeros(ACTION_CLASSES, dtype=np.int64)
    B = 256
    with torch.no_grad():
        for i in range(0, len(sample_indices), B):
            batch = sample_indices[i:i + B]
            frames_t, prevs_t, labels_t = fetch_batch(frames, actions, batch)
            pred = model(frames_t, prevs_t).argmax(dim=1).numpy()
            labs = labels_t.numpy()
            for c in range(ACTION_CLASSES):
                m = labs == c
                total[c] += int(m.sum())
                correct[c] += int((pred[m] == c).sum())
    per_class = {
        str(c): {"n": int(total[c]), "recall": round(float(correct[c] / total[c]), 6)}
        for c in range(ACTION_CLASSES)
    }
    return {
        "seed": seed,
        "per_class": per_class,
        "overall": round(float(sum(correct) / sum(total)), 6),
    }


def main() -> int:
    started = time.time()
    if RESULT_RELPATH.exists():
        print(f"REFUSED: {RESULT_RELPATH} already exists (create-only).",
              file=sys.stderr)
        return 1
    RESULT_RELPATH.parent.mkdir(parents=True, exist_ok=True)

    frames, actions = load_corpus()
    n_trans = sum(fr.shape[0] - 1 for fr in frames)
    print(f"corpus: {len(frames)} episodes, {n_trans} samples (t>=1)", flush=True)

    sample_indices = build_sample_indices(frames)
    assert len(sample_indices) == n_trans

    seeds_records: list[dict] = []
    for seed in SEEDS:
        print(f"[train] seed {seed} ...", flush=True)
        seeds_records.append(train_seed(seed, sample_indices, frames, actions))

    # --- Gate R: determinism — two in-process runs at seed 42 ----------------
    print("[gate R] second seed-42 run (determinism byte-match) ...", flush=True)
    run2 = train_seed(42, sample_indices, frames, actions)
    run1 = seeds_records[0]
    gate_r = {
        "passed": (run1["param_sha256"] == run2["param_sha256"]
                   and run1["loss_curve"] == run2["loss_curve"]),
        "seed42_run1_param_sha256": run1["param_sha256"],
        "seed42_run2_param_sha256": run2["param_sha256"],
        "loss_curves_identical": run1["loss_curve"] == run2["loss_curve"],
        "note": "in-process repeat at seed 42; byte-match of final parameter "
                "tensors (sha256) and full per-step loss curve",
    }

    # --- Gate S: causal boundary (C8) static audit ----------------------------
    # The audit greps the marker tag that labels ONLY the three model-input
    # construction lines above. The filter statement and these comments are
    # deliberately unmarked, so the audit matches exactly three lines.
    import inspect
    src = inspect.getsource(sys.modules[__name__])
    # The tag literal is split across the concatenation so this filter line's
    # own source text does NOT contain the contiguous marker (which would
    # make the audit match itself).
    _tag = "c8-" + "input"
    input_lines = [ln for ln in src.splitlines() if _tag in ln]
    privileged = [
        ln for ln in input_lines
        if "meta" in ln or "reward" in ln or "event" in ln
        or "terminated" in ln or "truncated" in ln or "m_p" in ln
    ]
    gate_s = {
        "passed": not privileged and len(input_lines) == 3,
        "model_input_construction_lines": input_lines,
        "privileged_reads_in_model_input_lines": privileged,
        "note": "input tensors built only from corpus frame tensors + "
                "previous-applied-action column; meta/ (reward/event/"
                "terminated/truncated) is never loaded into the model path",
    }

    # --- Gate T: budget fidelity ---------------------------------------------
    gate_t = {
        "passed": (
            all(r["seed"] == s for r, s in zip(seeds_records, SEEDS))
            and all(len(r["loss_curve"]) == TOTAL_STEPS for r in seeds_records)
            and len(SEEDS) == 4
        ),
        "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM, "batch_size": BATCH_SIZE,
        "total_steps": TOTAL_STEPS, "seeds": list(SEEDS),
        "steps_per_seed": [len(r["loss_curve"]) for r in seeds_records],
    }

    # --- Gate U: non-degenerate (all 5 action classes emitted, loss down) ----
    print("[gate U] per-seed corpus self-evaluation ...", flush=True)
    recalls = [corpus_recall(r["seed"], sample_indices, frames, actions)
               for r in seeds_records]
    all_classes = all(
        rc["per_class"][str(c)]["recall"] > 0 for rc in recalls for c in range(5)
    )
    loss_down = all(
        r["mean_first100"] > r["mean_last100"] for r in seeds_records
    )
    gate_u = {
        "passed": all_classes and loss_down,
        "all_5_classes_emitted_every_seed": all_classes,
        "training_loss_decreased_every_seed": loss_down,
        "per_seed": recalls,
        "note": "self-evaluation on its own TRAIN corpus; not a qualification "
                "claim (candidate comparison is the Tier-3 Q2 arm)",
    }

    gates = {
        "R_determinism": gate_r,
        "S_causal_boundary": gate_s,
        "T_budget_fidelity": gate_t,
        "U_non_degenerate": gate_u,
    }
    passed_all = all(g["passed"] for g in gates.values())

    manifest = json.loads((CORPUS_RELPATH / "manifest.json").read_text())
    result = {
        "mode": "reactive_baseline_v1_construction_acceptance",
        "preregistration": "brain/docs/preregistrations/"
                            "2026-08-28-pacman-reactive-baseline-v1.md",
        "started_unix": int(started),
        "wall_seconds": round(time.time() - started, 3),
        "corpus_sha256": manifest["corpus_sha256"],
        "corpus_matches_t1_3": manifest["corpus_sha256"].startswith(
            CORPUS_SHA256_PREFIX),
        "param_count_frozen": 133773,
        "seeds": seeds_records,
        "gates": gates,
        "all_passed": passed_all,
        "compute_policy": {
            "platform": "local CPU only (2026-08-26 decision record)",
            "cuda_hidden": True,
            "torch_threads": 1,
            "os_priority": "below-normal (gaming-polite)",
            "dgx_dispatch": False,
        },
        "provenance": {
            **provenance(deterministic=True),
            "note": "T1-4 reactive baseline; no candidate, no DEV/CAL/TEST",
        },
    }
    RESULT_RELPATH.write_text(json.dumps(result, indent=2, sort_keys=True),
                              encoding="utf-8")
    for name, gate in gates.items():
        print(f"[{'PASS' if gate['passed'] else 'FAIL'}] {name}")
    print(f"all_passed={passed_all} wall={result['wall_seconds']}s")
    return 0 if passed_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
