"""Run provenance: mandatory metadata for every experiment (owner directive).

Every future run saves:
  - training bank digest (pinned crc32-offset banks)
  - initial parameter digest
  - training seed / evaluation seed
  - deterministic-mode flag (+ which flags were set)
  - torch / CUDA versions

Usage:
    from run_provenance import provenance, PROVENANCE_VERSION
    rec = provenance(model=model, train_seed=42, eval_seed=20260822,
                     bank_digest="b3bb5fc33fd5f605", deterministic=True)
    payload["provenance"] = rec
"""
from __future__ import annotations

import hashlib
import platform
import zlib

import numpy as np


def stable_task_offset(task_name: str) -> int:
    """Process-stable task offset. NEVER use hash() for this."""
    return zlib.crc32(task_name.encode()) % 99991


def build_bank_pinned(seed, tasks, eps_per_task=24, episode_factory=None):
    """Episode bank with process-stable seeding.

    episode_factory(rng, index) -> episode; defaults to fn(rng) where tasks
    is a list of callables.
    """
    bank = []
    for fn in tasks:
        for i in range(eps_per_task):
            r = np.random.default_rng(seed + stable_task_offset(fn.__name__) + i)
            bank.append(fn(r))
    return bank


def bank_digest(bank) -> str:
    h = hashlib.sha256()
    for ep in bank:
        h.update(str(ep.label).encode())
        h.update(str(getattr(ep, "decision_frame_index", "")).encode())
        fr = np.stack(ep.frames)
        h.update(np.asarray(fr, dtype=np.uint8).tobytes())
    return h.hexdigest()[:16]


def param_digest(model) -> dict:
    """Order-stable float64 stats over all parameters."""
    s = 0.0
    sq = 0.0
    n = 0
    for p in model.parameters():
        pd = p.detach().to("cpu").to(torch.float64) if hasattr(p, "detach") else None
        if pd is None:
            continue
        s += float(pd.sum())
        sq += float((pd ** 2).sum())
        n += pd.numel()
    return {"sum": round(s, 6), "l2": round(sq ** 0.5, 6), "n": n}


def apply_deterministic_mode():
    """Strict deterministic execution (research/comparison mode)."""
    import os
    import torch
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def provenance(*, model=None, train_seed=None, eval_seed=None,
               bank_digest=None, deterministic=False) -> dict:
    import torch
    rec = {
        "PROVENANCE_VERSION": 1,
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device": str(next(model.parameters()).device) if model is not None else None,
        "python": platform.python_version(),
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "bank_digest": bank_digest,
        "deterministic_mode": bool(deterministic),
        "deterministic_flags": {
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        },
    }
    if model is not None:
        rec["init_param_digest"] = param_digest(model)
    return rec
