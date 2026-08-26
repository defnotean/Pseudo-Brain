"""PB21M-POSTHOC-MECHANISM-AUDIT-v1: two-arm post-hoc consumed-data audit.

Classification: post_hoc_consumed_data_nonqualifying, permanently.

Arm A: fixed MIX35 cross-fit calibrator (focal factual weight w=0.35) on the
sealed PB21M raw CAL logits. No sweep, no scorer training.
Arm B: prior-action residual strata on deterministically reconstructed
consumed CAL tapes. No training, no re-inference beyond the exact parent
model (loaded read-only, verified byte-identical before and after).

Neither arm may open DEV, CPU-QUAL, PLAY-QUAL, or TEST. At most one single
change may be nominated for a later fresh PB21N preregistration.

Preregistration: brain/docs/preregistrations/
2026-08-25-pb21m-posthoc-mechanism-audit-v1.md
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import struct
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

BRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN_ROOT / "scripts"), str(BRAIN_ROOT / "src")]

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

import run_provenance  # noqa: E402
import v21i_development_runner as development  # noqa: E402
import v21i_strict_live_representation_probe_v1 as strict  # noqa: E402
import v21k_nz_dualcal_diagnostic_v1 as pb21k  # noqa: E402
import v21l_nz_upmix_consumed_feasibility_v1 as pb21l  # noqa: E402
import v21m_fresh_bal_cal_first_qualification_v1 as pb21m  # noqa: E402
from irene_brain.evaluation.v21_qualification_metrics import (  # noqa: E402
    binary_probability_metrics,
)
from irene_brain.v2 import hazard_calibration as hc  # noqa: E402
from irene_brain.v2.trajectory_objective import control_action_class  # noqa: E402

run_provenance.apply_deterministic_mode()
torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)

# ---------------------------------------------------------------------------
# Frozen audit constants (prereg 2026-08-25-pb21m-posthoc-mechanism-audit-v1)
# ---------------------------------------------------------------------------
MODE = "pb21m_posthoc_mechanism_audit_v1"
RUN_ID = "2026-08-25-pb21m-posthoc-mechanism-audit-v1"
CLASSIFICATION = "post_hoc_consumed_data_nonqualifying"
EVIDENCE_DIR = BRAIN_ROOT / "runs" / "pb21m-posthoc-audit"
PARENT_EVIDENCE = (
    BRAIN_ROOT / "runs/v21m-qualification/"
    "2026-08-25-pb21m-fresh-bal-cal-first-v1.cal-evidence.npz"
)
PB21K_EVIDENCE = (
    BRAIN_ROOT / "runs/v21k-diagnostics/"
    "2026-08-25-pb21k-nz-dualcal-v1.evidence.npz"
)

# Sealed PB21M parent file digests (bound by its registration).
PARENT_SHA = {
    "registration": "f2ca97558697e4597d57578261a82f13621808758cfd2ec5e1b8542ba516dee6",
    "attempt": "e3322c79cd979d6b3b486680360c1eeafb961ad528a194d4706e68d2434d8758",
    "cal_evidence": "5bdf26f0988a1bd79faa07b650eede2fadd08c37616a97b5b21ee739069ffe02",
    "cal_decision": "20bc265e17c02e25a78cb76d31ceb281e253b44cfb1478db88dd64bc912706df",
    "result": "0833b6c848003f9639c7e69250724639948b6cda334f992b0ff32fb0abd4d18f",
}
PARENT_PATHS = {
    "registration": BRAIN_ROOT / "runs/v21m-qualification/"
    "2026-08-25-pb21m-fresh-bal-cal-first-v1.registration.json",
    "attempt": BRAIN_ROOT / "runs/v21m-qualification/"
    "2026-08-25-pb21m-fresh-bal-cal-first-v1.attempt.json",
    "cal_evidence": PARENT_EVIDENCE,
    "cal_decision": BRAIN_ROOT / "runs/v21m-qualification/"
    "2026-08-25-pb21m-fresh-bal-cal-first-v1.cal-decision.json",
    "result": BRAIN_ROOT / "runs/v21m-qualification/"
    "2026-08-25-pb21m-fresh-bal-cal-first-v1.json",
}

# Calibration solver constants (identical to the sealed PB21J/PB21K/PB21M runs:
# pb21j.CALIBRATION_* = L2 1e-6, min scale 1e-4, >=20 examples/action,
# >=2 per class, max 100 iterations, tolerance 1e-10).
L2 = 1.0e-6
MIN_SCALE = 1.0e-4
MIN_EXAMPLES = 20
MIN_CLASS = 2
MAX_ITER = 100
TOL = 1.0e-10

# Arm A focal weight and context variants (context may never drive the gate).
MIX35_FOCAL_WEIGHT = 0.35
MIX35_CONTEXT_WEIGHTS = (0.5, 1.0)

# Frozen gates (from prereg).
FACTUAL_AGGREGATE_BIAS_LIMIT = 0.05
FACTUAL_AGGREGATE_ECE_LIMIT = 0.05
ALL_ACTION_CONTROL_WORSE_MARGIN = 0.01
FACTORIAL_AUC_DROP_LIMIT = 0.05
B2_CELLS_REQUIRED = 3
B2_STRATUM_BIAS = 0.03
B2_STRATUM_ECE = 0.05
B3_PERMUTATIONS = 30
B3_SHUFFLE_SEED = 81_042
MIN_STRATUM_ROWS = 20
ONE_SIDED_ALPHA = 0.025
QUANTILE_METHOD = "linear"
BOOTSTRAP_BLOCK = 1024

CAL_FOLDS = ("A", "B")
SCORER_BASE = "BASE"
SCORER_BAL = "BAL-UPMIX"
CAL_COHORT_PAIRS = (
    ("C0A", "C0B"), ("C1A", "C1B"), ("C2A", "C2B"),
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, np.integer)):
        return value.item()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (dict, Mapping)):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"not JSON-serializable: {type(value)!r}")


def resource_guard() -> dict[str, object]:
    """Mirror of the PB21M resource guard (Windows), informational only."""
    record: dict[str, object] = {"platform": platform.platform()}
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        record["memory_load_percent"] = float(status.dwMemoryLoad)
        record["available_physical_ram_bytes"] = int(status.ullAvailPhys)
        record["process_working_set_bytes"] = int(_process_working_set())
    record["torch_threads"] = torch.get_num_threads()
    record["torch_interop_threads"] = torch.get_num_interop_threads()
    return _jsonable(record)


def _process_working_set() -> int:
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaults", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("QuotaPagefileUsage", ctypes.c_size_t),
        ]
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return int(counters.WorkingSetSize)


# ---------------------------------------------------------------------------
# Parent integrity
# ---------------------------------------------------------------------------

def verify_parent_integrity() -> dict[str, object]:
    checks: dict[str, object] = {}
    for role in ("registration", "attempt", "cal_decision", "result"):
        digest = _sha256_file(PARENT_PATHS[role])
        checks[role] = {"sha256": digest, "matches_sealed": digest == PARENT_SHA[role]}
    evidence_digest = _sha256_file(PARENT_EVIDENCE)
    checks["cal_evidence"] = {"sha256": evidence_digest,
                              "matches_sealed": evidence_digest == PARENT_SHA["cal_evidence"]}
    passed = all(v["matches_sealed"] for v in checks.values())
    if not passed:
        raise RuntimeError("PB21M parent artifacts drifted; audit aborts before any scoring")
    return {"checks": checks, "passed": True}


def load_parent_evidence() -> dict[str, np.ndarray]:
    arrays = {k: np.asarray(v) for k, v in np.load(PARENT_EVIDENCE, allow_pickle=False).items()}
    # Load-time OOF reconstruction self-check (seal integrity).
    raw = arrays["cal_raw_logits"]
    oof = arrays["oof_calibrated_logits"]
    xs, xb = arrays["crossfit_scales"], arrays["crossfit_biases"]
    evfold = arrays["crossfit_eval_fold_index"]
    for sc in range(2):
        for cell in range(9):
            for f in range(2):
                j = int(np.where(evfold == f)[0][0])
                rec = raw[sc, cell, f] * xs[sc, cell, j] + xb[sc, cell, j]
                lo, hi = f * 3072, (f + 1) * 3072
                if not np.array_equal(rec, oof[sc, cell, lo:hi]):
                    raise RuntimeError(f"sealed OOF reconstruction drifted at sc{sc}/cell{cell}/fold{f}")
    return arrays


# ---------------------------------------------------------------------------
# Weighted per-action affine Newton solver (generalization of v21k MIX solver)
# ---------------------------------------------------------------------------

def _mixed_weights(all_count: int, factual_mask: np.ndarray, weight: float) -> np.ndarray:
    """(1-w)/n + w * I[i in F_a]/n_a, summed over i exactly one."""
    n = int(all_count)
    fa = int(factual_mask.sum())
    if fa < 1:
        raise ValueError("no factual examples in fitting fold")
    weights = np.full(n, (1.0 - weight) / n, dtype=np.float64)
    weights[factual_mask] += weight / fa
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1.0e-14):
        raise RuntimeError("weighted calibration weights do not sum to one")
    return weights


def _weighted_objective(
    logits: Tensor, targets: Tensor, weights: Tensor,
    scale: Tensor, bias: Tensor,
) -> Tensor:
    calibrated = scale * logits + bias
    data = torch.sum(weights * (torch.nn.functional.softplus(calibrated) - targets * calibrated))
    penalty = 0.5 * L2 * ((scale - 1.0).square() + bias.square())
    return data + penalty


def fit_weighted_per_action(
    logits: np.ndarray, targets: np.ndarray, weights: np.ndarray,
) -> tuple[float, float, str, int, float]:
    """Deterministic constrained Newton fit with the scale-bound KKT rule.

    Mirrors v21k._fit_one_mixed_action exactly (same objective, Hessian,
    KKT projection, bias-only direction at the bound, halving line search).
    Returns (scale, bias, termination, iterations, final_objective).
    """
    x = torch.as_tensor(logits, dtype=torch.float64, device="cpu")
    y = torch.as_tensor(targets, dtype=torch.float64, device="cpu")
    w = torch.as_tensor(weights, dtype=torch.float64, device="cpu")
    if x.ndim != 1 or x.shape != y.shape or x.shape != w.shape or len(x) < 1:
        raise ValueError("weighted calibration arrays must be non-empty aligned vectors")
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(w).all()):
        raise ValueError("weighted calibration arrays must be finite")
    if not bool(((y == 0.0) | (y == 1.0)).all()) or bool((w < 0.0).any()):
        raise ValueError("targets must be binary and weights non-negative")
    if not np.isclose(float(w.sum()), 1.0, rtol=0.0, atol=1.0e-14):
        raise ValueError("weights must sum to one")
    scale = x.new_tensor(1.0)
    bias = x.new_tensor(0.0)
    current = _weighted_objective(x, y, w, scale, bias)
    if not bool(torch.isfinite(current)):
        raise FloatingPointError("non-finite initial objective")
    iterations = 0
    terminated = "maximum_iterations"
    for iteration in range(MAX_ITER):
        iterations = iteration + 1
        calibrated = scale * x + bias
        probability = torch.sigmoid(calibrated)
        residual = probability - y
        curvature = probability * (1.0 - probability)
        grad_scale = torch.sum(w * residual * x) + L2 * (scale - 1.0)
        grad_bias = torch.sum(w * residual) + L2 * bias
        if not bool(torch.isfinite(grad_scale) and torch.isfinite(grad_bias)):
            raise FloatingPointError("non-finite projected gradient")
        at_bound = float(scale) <= MIN_SCALE + TOL
        scale_kkt = at_bound and float(grad_scale) >= 0.0
        projected_scale_gradient = 0.0 if scale_kkt else float(grad_scale)
        if max(abs(projected_scale_gradient), abs(float(grad_bias))) <= TOL:
            terminated = "projected_gradient_KKT"
            break
        h_ss = torch.sum(w * curvature * x.square()) + L2
        h_sb = torch.sum(w * curvature * x)
        h_bb = torch.sum(w * curvature) + L2
        if scale_kkt:
            delta_scale = x.new_tensor(0.0)
            delta_bias = grad_bias / h_bb
        else:
            determinant = h_ss * h_bb - h_sb.square()
            if float(determinant) <= 0.0:
                raise RuntimeError("calibration Hessian is not positive definite")
            delta_scale = (h_bb * grad_scale - h_sb * grad_bias) / determinant
            delta_bias = (h_ss * grad_bias - h_sb * grad_scale) / determinant
        accepted_step = False
        step = 1.0
        for _ in range(50):
            candidate_scale = torch.clamp(scale - step * delta_scale, min=MIN_SCALE)
            candidate_bias = bias - step * delta_bias
            candidate = _weighted_objective(x, y, w, candidate_scale, candidate_bias)
            if not bool(torch.isfinite(candidate)):
                step *= 0.5
                continue
            if float(candidate) <= float(current):
                improvement = float(current - candidate)
                parameter_move = max(
                    abs(float(candidate_scale - scale)), abs(float(candidate_bias - bias)),
                )
                scale = candidate_scale
                bias = candidate_bias
                current = candidate
                accepted_step = True
                if improvement <= TOL and parameter_move <= TOL:
                    terminated = "objective_tolerance"
                break
            step *= 0.5
        if not accepted_step:
            raise RuntimeError("calibration line search exhausted")
        if terminated == "objective_tolerance":
            break
    if terminated == "maximum_iterations":
        raise RuntimeError("calibration exhausted maximum iterations")
    return float(scale), float(bias), terminated, iterations, float(current)


# ---------------------------------------------------------------------------
# Solver controls
# ---------------------------------------------------------------------------

def control_aa_refit(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Re-derive the 36 sealed cross-fit AA calibrators with the canonical
    repo solver (hc._fit_one_action) and confirm byte-exact match."""
    worst = 0.0
    count = 0
    evfold = arrays["crossfit_eval_fold_index"]
    raw = arrays["cal_raw_logits"]
    targets = arrays["cal_targets"]
    cell_fit = arrays["cell_fit_index"]
    for sc in range(2):
        for cell in range(9):
            cohort = int(cell_fit[cell])
            for entry in range(2):
                fit_fold = int(arrays["crossfit_fit_fold_index"][entry])
                fit_rows = targets[cohort][0:3072] if fit_fold == 0 else targets[cohort][3072:6144]
                for act in range(5):
                    s, b = hc._fit_one_action(
                        torch.from_numpy(raw[sc, cell, fit_fold][:, act]),
                        torch.from_numpy(fit_rows[:, act]),
                        l2_regularization=L2, minimum_scale=MIN_SCALE,
                        max_iterations=MAX_ITER, tolerance=TOL,
                        fit_mode="per_action_affine",
                    )
                    worst = max(worst, abs(s - float(arrays["crossfit_scales"][sc, cell, entry, act])),
                                abs(b - float(arrays["crossfit_biases"][sc, cell, entry, act])))
                    count += 1
    passed = worst < 1e-9
    return {
        "refits": count,
        "worst_parameter_difference": float(worst),
        "passed": bool(passed),
    }


def control_pb21k_mix() -> dict[str, object]:
    """Re-derive the 45 sealed PB21K MIX action fits with this runner's
    weighted solver at weight 0.5 and confirm match."""
    arrays = np.load(PB21K_EVIDENCE, allow_pickle=False)
    worst = 0.0
    acceptance_mismatches = 0
    for cell in range(9):
        logits = np.asarray(arrays["cal_nz_raw_logits"][cell], dtype=np.float64)
        cohort = int(arrays["cell_cal_index"][cell])
        targets = np.asarray(arrays["cal_targets"][cohort], dtype=np.float64)
        actions = np.asarray(arrays["cal_actions"][cohort], dtype=np.int64)
        for act in range(5):
            weights = _mix_weights_v21k(actions, act)
            s, b, *_ = fit_weighted_per_action(logits[:, act], targets[:, act], weights)
            worst = max(worst,
                        abs(s - float(arrays["mix_proposed_scales"][cell, act])),
                        abs(b - float(arrays["mix_proposed_biases"][cell, act])))
            if bool(_mix_acceptance(logits, targets, actions, act, s, b)) != \
                    bool(arrays["mix_per_action_accepted"][cell, act]):
                acceptance_mismatches += 1
    passed = worst < 1e-9 and acceptance_mismatches == 0
    return {
        "refits": 45,
        "weight": 0.5,
        "worst_parameter_difference": float(worst),
        "acceptance_mismatches": int(acceptance_mismatches),
        "passed": bool(passed),
    }


def _mix_weights_v21k(factual_actions: np.ndarray, action: int) -> np.ndarray:
    """Exactly the v21k MIX weights at weight 0.5."""
    actions = np.asarray(factual_actions, dtype=np.int64)
    mask = actions == action
    factual_count = int(mask.sum())
    if factual_count < MIN_EXAMPLES:
        raise ValueError(f"action {action} has insufficient factual CAL support")
    return _mix_weights_v21k_weights(len(actions), mask, 0.5)


def _mix_weights_v21k_weights(n: int, mask: np.ndarray, weight: float) -> np.ndarray:
    return _mixed_weights(n, mask, weight)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def _mix_acceptance(
    logits: np.ndarray, targets: np.ndarray, actions: np.ndarray,
    action: int, scale: float, bias: float,
) -> bool:
    """Replicate the sealed PB21K MIX per-action acceptance rule."""
    weights = _mix_weights_v21k(actions, action)
    mask = actions == action
    x = torch.from_numpy(logits[:, action])
    y = torch.from_numpy(targets[:, action])
    w = torch.from_numpy(weights)
    identity_mixed = float(_weighted_objective(
        x, y, w, torch.tensor(1.0, dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64)))
    proposed_mixed = float(_weighted_objective(
        x, y, w, torch.tensor(scale, dtype=torch.float64), torch.tensor(bias, dtype=torch.float64)))
    raw_p = _sigmoid(logits[:, action])
    prop_p = _sigmoid(np.asarray(scale * logits[:, action] + bias))
    def bce(t, p):
        return float(np.mean(
            -(t * np.log(np.clip(p, 1e-12, None)) + (1 - t) * np.log1p(-np.clip(p, None, 1 - 1e-12)))
        ))
    def brier(t, p):
        return float(np.mean((p - t) ** 2))
    return bool(
        identity_mixed - proposed_mixed > TOL
        and bce(targets[:, action], prop_p) < bce(targets[:, action], raw_p)
        and brier(targets[:, action], prop_p) < brier(targets[:, action], raw_p)
        and bce(targets[mask, action], prop_p[mask]) < bce(targets[mask, action], raw_p[mask])
        and brier(targets[mask, action], prop_p[mask]) < brier(targets[mask, action], raw_p[mask])
    )


# ---------------------------------------------------------------------------
# Arm A — MIX35 cross-fit calibrator
# ---------------------------------------------------------------------------

def _fit_mixed_action_on_fold(
    raw_fold: np.ndarray, targets_fold: np.ndarray, actions_fold: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Fit per-action weighted affine calibrator on one fold.

    Returns (scales[5], biases[5], per-action fit reports).
    """
    scales = np.empty(5, dtype=np.float64)
    biases = np.empty(5, dtype=np.float64)
    reports: list[dict[str, object]] = []
    for act in range(5):
        mask = actions_fold == act
        if int(mask.sum()) < MIN_EXAMPLES:
            raise RuntimeError(f"insufficient factual support for action {act}")
        n_pos = int(targets_fold[mask, act].sum())
        if n_pos < MIN_CLASS or (int(mask.sum()) - n_pos) < MIN_CLASS:
            raise RuntimeError(f"insufficient class support for action {act}")
        n_all_pos = int(targets_fold[:, act].sum())
        if n_all_pos < MIN_CLASS or (int(len(targets_fold)) - n_all_pos) < MIN_CLASS:
            raise RuntimeError(f"insufficient all-action support for action {act}")
        weights = _mixed_weights(len(actions_fold), mask, weight)
        s, b, term, iters, obj = fit_weighted_per_action(
            raw_fold[:, act], targets_fold[:, act], weights,
        )
        scales[act] = s
        biases[act] = b
        reports.append({
            "action": act, "factual_rows": int(mask.sum()),
            "factual_positives": n_pos, "termination": term,
            "iterations": int(iters), "final_objective": obj,
        })
    return scales, biases, reports


def arm_a(arrays: dict[str, np.ndarray], weight: float) -> dict[str, object]:
    """Cross-fit MIX(caliber weight) on sealed raw CAL logits.

    Returns OOF logits/probabilities [sc, cell, 6144, 5] and fit provenance.
    """
    oof_logits = np.empty((2, 9, 6144, 5), dtype=np.float64)
    provenance: list[dict[str, object]] = []
    evfold = arrays["crossfit_eval_fold_index"]
    raw = arrays["cal_raw_logits"]
    targets = arrays["cal_targets"]
    actions = arrays["cal_actions"]
    cell_fit = arrays["cell_fit_index"]
    for sc in range(2):
        for cell in range(9):
            cohort = int(cell_fit[cell])
            for entry in range(2):
                eval_fold = int(evfold[entry])
                fit_fold = int(arrays["crossfit_fit_fold_index"][entry])
                fit_rows = 0 if fit_fold == 0 else 1
                scales, biases, reports = _fit_mixed_action_on_fold(
                    raw[sc, cell, fit_fold],
                    targets[cohort][0:3072] if fit_rows == 0 else targets[cohort][3072:6144],
                    actions[cohort][0:3072] if fit_rows == 0 else actions[cohort][3072:6144],
                    weight,
                )
                lo, hi = eval_fold * 3072, (eval_fold + 1) * 3072
                oof_logits[sc, cell, lo:hi] = (
                    raw[sc, cell, eval_fold] * scales[None, :] + biases[None, :]
                )
                provenance.append({
                    "scorer_index": sc, "cell_index": cell, "weight": weight,
                    "fit_fold": CAL_FOLDS[fit_fold], "eval_fold": CAL_FOLDS[eval_fold],
                    "scales": scales.tolist(), "biases": biases.tolist(),
                    "actions": reports,
                })
    oof_probabilities = _sigmoid(oof_logits)
    return {"oof_logits": oof_logits, "oof_probabilities": oof_probabilities,
            "provenance": provenance}


def _factual_metrics(probabilities_fold: np.ndarray, targets_fold: np.ndarray,
                     actions_fold: np.ndarray) -> dict[str, object]:
    """Factual-domain aggregate + per-action metrics (repo metric fn)."""
    mask = pb21l._domain_mask(actions_fold, "factual")
    agg = binary_probability_metrics(targets_fold[mask], probabilities_fold[mask],
                                      ece_bins=10).as_dict()
    per_action = []
    for act in range(5):
        rows = mask[:, act]
        m = binary_probability_metrics(targets_fold[rows, act],
                                       probabilities_fold[rows, act], ece_bins=10).as_dict()
        per_action.append(m)
    return {"aggregate": agg, "per_action": per_action}


def _all_action_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, object]:
    flat_p = probabilities.reshape(-1)
    flat_t = targets.reshape(-1)
    return binary_probability_metrics(flat_t, flat_p, ece_bins=10).as_dict()


def _per_root_factual_losses(targets: np.ndarray, probabilities: np.ndarray,
                             actions: np.ndarray) -> np.ndarray:
    """[n, 2] = (factual BCE, factual Brier) per root (factual entry only)."""
    bce, brier = pb21l._loss_elements(targets, probabilities)
    mask = pb21l._domain_mask(actions, "factual")
    count = mask.sum(axis=1)
    return np.stack(((bce * mask).sum(axis=1) / count,
                     (brier * mask).sum(axis=1) / count), axis=1)


def _episode_means(values: np.ndarray, ordinal: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    groups = int(ordinal.max()) + 1
    result = np.asarray([data[ordinal == g].mean(axis=0) for g in range(groups)])
    if any(int((ordinal == g).sum()) != pb21m.ROOTS_PER_EPISODE for g in range(groups)):
        raise ValueError("episode mean geometry drifted")
    return result


def _bootstrap_draws(episode_values: np.ndarray, sampled: np.ndarray) -> np.ndarray:
    episodes = np.asarray(episode_values, dtype=np.float64)
    indices = np.asarray(sampled)
    if indices.ndim == 2:
        indices = indices[:, None, :]
        episodes = episodes.reshape(1, indices.shape[2], *episodes.shape[1:])
    elif indices.ndim == 3:
        episodes = episodes.reshape(indices.shape[1], indices.shape[2], *episodes.shape[1:])
    else:
        raise ValueError("bootstrap indices must be resample x [fold x] episode")
    result = np.empty((len(indices), *episodes.shape[2:]), dtype=np.float64)
    for start in range(0, len(indices), BOOTSTRAP_BLOCK):
        stop = min(start + BOOTSTRAP_BLOCK, len(indices))
        block = indices[start:stop].astype(np.int64, copy=False)
        gathered = np.stack(
            [episodes[fold][block[:, fold]] for fold in range(indices.shape[1])], axis=1,
        )
        result[start:stop] = gathered.mean(axis=(1, 2))
    return result


def arm_a_gates(
    mix: dict[str, object], aa_probs: np.ndarray, arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    """Frozen gates A1/A2/A3 for the focal MIX35 OOF table vs sealed AA OOF."""
    mix_probs = mix["oof_probabilities"]
    targets = arrays["cal_targets"]
    actions = arrays["cal_actions"]
    ordinal = arrays["cal_cluster_ordinal"]
    sampled = arrays["bootstrap_indices"]
    cell_fit = arrays["cell_fit_index"]

    # A1: factual aggregate |bias| <= 0.05 and ECE <= 0.05 in ALL 18 tables.
    a1_tables: list[dict[str, object]] = []
    a1_passed = True
    for sc in range(2):
        for cell in range(9):
            cohort = int(cell_fit[cell])
            fm = _factual_metrics(mix_probs[sc, cell], targets[cohort], actions[cohort])
            agg = fm["aggregate"]
            ok = (abs(float(agg["calibration_bias"])) <= FACTUAL_AGGREGATE_BIAS_LIMIT
                  and float(agg["ece_equal_mass"]) <= FACTUAL_AGGREGATE_ECE_LIMIT)
            a1_passed = a1_passed and ok
            a1_tables.append({
                "scorer": int(sc), "cell": int(cell),
                "factual_bias": float(agg["calibration_bias"]),
                "factual_ece": float(agg["ece_equal_mass"]),
                "factual_bce": float(agg["bce"]), "factual_brier": float(agg["brier"]),
                "passed": bool(ok),
            })

    # A2: paired grand factual improvement, per-cell fold points, cluster LCBs.
    a2_cells: dict[str, object] = {}
    a2_passed = True
    cohort_episodes = []
    for sc in range(2):
        if sc != 1:
            continue  # candidate arm is BAL-UPMIX = index 1; baseline = index 0
        for cell in range(9):
            cohort = int(cell_fit[cell])
            base = _per_root_factual_losses(targets[cohort], aa_probs[sc, cell], actions[cohort])
            cand = _per_root_factual_losses(targets[cohort], mix_probs[sc, cell], actions[cohort])
            root_delta = base - cand  # positive = candidate better
            episode = _episode_means(root_delta, ordinal[cohort])
            draws = _bootstrap_draws(episode, sampled[:, cohort])
            point = episode.mean(axis=0)
            lower = np.quantile(draws, ONE_SIDED_ALPHA, axis=0, method=QUANTILE_METHOD)
            fold_points = {}
            for f, f_name in enumerate(CAL_FOLDS):
                lo, hi = f * 3072, (f + 1) * 3072
                fp = root_delta[lo:hi].mean(axis=0)
                fold_points[f_name] = {
                    "BCE": float(fp[0]), "Brier": float(fp[1]),
                    "passed": bool(fp[0] > 0.0 and fp[1] > 0.0),
                }
            ok = (bool(point[0] > 0.0 and lower[0] > 0.0
                       and point[1] > 0.0 and lower[1] > 0.0)
                  and all(v["passed"] for v in fold_points.values()))
            a2_passed = a2_passed and ok
            a2_cells[f"{SCORER_BASE}_to_{SCORER_BAL}/cell{cell}"] = {
                "point": {"BCE": float(point[0]), "Brier": float(point[1])},
                "LCB": {"BCE": float(lower[0]), "Brier": float(lower[1])},
                "heldout_fold_points": fold_points,
                "passed": bool(ok),
            }

    # A3: all-action controls not worse than sealed AA by > 0.01; no factual
    # per-action AUC drop > 0.05.
    a3_cells: list[dict[str, object]] = []
    a3_passed = True
    for sc in range(2):
        for cell in range(9):
            cohort = int(cell_fit[cell])
            aa_agg = _all_action_metrics(aa_probs[sc, cell], targets[cohort])
            mx_agg = _all_action_metrics(mix_probs[sc, cell], targets[cohort])
            bias_delta = abs(float(mx_agg["calibration_bias"])) - abs(float(aa_agg["calibration_bias"]))
            ece_delta = float(mx_agg["ece_equal_mass"]) - float(aa_agg["ece_equal_mass"])
            dom = pb21l._domain_mask(actions[cohort], "factual")
            auc_ok = True
            auc_worst_drop = 0.0
            for act in range(5):
                rows = dom[:, act]
                if int(rows.sum()) < 20:
                    continue
                aa_auc = float(binary_probability_metrics(
                    targets[cohort][rows, act], aa_probs[sc, cell][rows, act],
                    ece_bins=10).as_dict()["roc_auc"])
                mx_auc = float(binary_probability_metrics(
                    targets[cohort][rows, act], mix_probs[sc, cell][rows, act],
                    ece_bins=10).as_dict()["roc_auc"])
                drop = aa_auc - mx_auc
                auc_worst_drop = max(auc_worst_drop, drop)
                auc_ok = auc_ok and drop <= FACTORIAL_AUC_DROP_LIMIT
            ok = (bias_delta <= ALL_ACTION_CONTROL_WORSE_MARGIN
                  and ece_delta <= ALL_ACTION_CONTROL_WORSE_MARGIN and auc_ok)
            a3_passed = a3_passed and ok
            a3_cells.append({
                "scorer": int(sc), "cell": int(cell),
                "all_action_bias_abs_delta": float(bias_delta),
                "all_action_ece_delta": float(ece_delta),
                "worst_factual_auc_drop": float(auc_worst_drop),
                "passed": bool(ok),
            })

    passed = bool(a1_passed and a2_passed and a3_passed)
    return {
        "A1_factual_absolute": {"tables": a1_tables, "passed": bool(a1_passed)},
        "A2_factual_paired_vs_AA": {"cells": a2_cells, "passed": bool(a2_passed)},
        "A3_all_action_controls": {"cells": a3_cells, "passed": bool(a3_passed)},
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Arm B — prior-action residual strata
# ---------------------------------------------------------------------------

def replay_partition_applied(
    partition_label: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Deterministic data-only replay of one sealed PB21M CAL partition.

    Returns (root_state_ids, applied_actions_per_recorded_root,
    prior_actions_per_recorded_root). The prior of the first recorded root of
    each episode is the burn-in tick-3 applied action (collector semantics).
    """
    contracts = pb21m.partition_contracts()
    manifests = pb21m.partition_manifests()
    contract = contracts[partition_label]
    source = development.PartitionSource(contract)
    if source.manifest_sha256 != manifests[partition_label]:
        raise RuntimeError(f"manifest drift for {partition_label}")
    burn = contract.burn_in_steps
    seq_len = contract.dataset_config.sequence_length
    roots: list[str] = []
    episodes_applied: list[list[int]] = []
    for batch in source.iter_all_action_batches(epoch=0, batch_size=1):
        ep_applied: list[int] = []
        for tick in range(seq_len):
            transitions = tuple(seq.transitions[tick] for seq in batch.sequences)
            applied = [control_action_class(t.applied_control) for t in transitions]
            for i, t in enumerate(transitions):
                if tick >= burn:
                    roots.append(t.root_state_sha256)
                ep_applied.append(applied[i])
        episodes_applied.append(ep_applied)
    applied_arr = np.asarray(episodes_applied, dtype=np.int64)
    if applied_arr.shape != (256, seq_len):
        raise RuntimeError(f"applied geometry drifted for {partition_label}: {applied_arr.shape}")
    expected = 256 * (seq_len - burn)
    if len(roots) != expected:
        raise RuntimeError(f"replay geometry drifted for {partition_label}")
    recorded = applied_arr[:, burn:]
    prior = np.empty_like(recorded)
    prior[:, 0] = applied_arr[:, burn - 1]  # burn-in tick-3 applied action
    prior[:, 1:] = recorded[:, :-1]
    return roots, recorded.reshape(-1), prior.reshape(-1)


def arm_b(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Reconstruct consumed CAL tapes (data-only) and score prior-action
    residual strata on the sealed AA OOF probabilities."""
    cell_fit = arrays["cell_fit_index"]
    replay_cache: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}

    def replay(label: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        if label not in replay_cache:
            replay_cache[label] = replay_partition_applied(label)
        return replay_cache[label]

    replay_report: dict[str, object] = {}
    b1_passed = True
    for ci, (label_a, label_b) in enumerate(CAL_COHORT_PAIRS):
        for fold_idx, label in enumerate((label_a, label_b)):
            roots, applied, prior = replay(label)
            roots_np = np.asarray([r.encode("ascii") for r in roots], dtype="S64")
            sealed_roots = arrays["cal_root_ids"][ci][fold_idx * 3072:(fold_idx + 1) * 3072]
            sealed_actions = arrays["cal_actions"][ci][fold_idx * 3072:(fold_idx + 1) * 3072]
            roots_ok = bool(np.array_equal(roots_np, sealed_roots))
            actions_ok = bool(np.array_equal(applied, sealed_actions))
            b1_passed = b1_passed and roots_ok and actions_ok
            replay_report[label] = {
                "rows": int(len(roots)),
                "roots_byte_identical": roots_ok,
                "actions_byte_identical": actions_ok,
            }
    if not b1_passed:
        return {"B1_reconstruction": {"report": replay_report, "passed": False},
                "passed": False, "failure": "replay_drift"}

    cohort_prior = [
        np.concatenate([replay(label_a)[2], replay(label_b)[2]])
        for label_a, label_b in CAL_COHORT_PAIRS
    ]

    oof = arrays["oof_calibrated_probabilities"]
    targets = arrays["cal_targets"]
    actions = arrays["cal_actions"]

    strata_report: dict[str, object] = {"cells": []}
    b2_passed = True
    for sc in range(2):
        cells_passing = 0
        for cell in range(9):
            cohort = int(cell_fit[cell])
            p = oof[sc, cell]
            t = targets[cohort]
            a = actions[cohort]
            pr = cohort_prior[cohort]
            stratum_rows: list[dict[str, object]] = []
            worst_bias = 0.0
            worst_ece = 0.0
            structured_cells = 0
            for act in range(5):
                fm = a == act
                for pv in range(5):
                    m = fm & (pr == pv)
                    n = int(m.sum())
                    if n < MIN_STRATUM_ROWS:
                        stratum_rows.append({"action": act, "prior": pv, "n": n,
                                             "status": "insufficient_support"})
                        continue
                    probs = p[m, act]
                    targs = t[m, act]
                    agg = binary_probability_metrics(targs, probs, ece_bins=10).as_dict()
                    bias = float(agg["calibration_bias"])
                    ece = float(agg["ece_equal_mass"])
                    stratum_rows.append({
                        "action": act, "prior": pv, "n": n,
                        "mean_probability": float(probs.mean()),
                        "factual_prevalence": float(targs.mean()),
                        "calibration_bias": bias,
                        "ece_equal_mass": ece,
                        "status": "ok",
                    })
                    worst_bias = max(worst_bias, abs(bias))
                    worst_ece = max(worst_ece, ece)
            structured_cells = sum(
                1 for r in stratum_rows if r["status"] == "ok"
                and abs(r["calibration_bias"]) >= B2_STRATUM_BIAS
                and r["ece_equal_mass"] >= B2_STRATUM_ECE
            )
            cells_passing = cells_passing + (1 if structured_cells >= 1 else 0)
            strata_report["cells"].append({
                "scorer": int(sc), "cell": int(cell), "cohort": cohort,
                "worst_stratum_abs_bias": float(worst_bias),
                "worst_stratum_ece": float(worst_ece),
                "structured_stratum_count": int(structured_cells),
                "strata": stratum_rows,
            })
        b2_passed = b2_passed and cells_passing >= B2_CELLS_REQUIRED
        strata_report[f"cells_passing_b2_sc{sc}"] = int(cells_passing)

    # B3: prior-label shuffle specificity control (candidate arm only).
    sc = 1
    observed_range = []
    permuted_ranges = []
    for cell in range(9):
        cohort = int(cell_fit[cell])
        p = oof[sc, cell]
        t = targets[cohort]
        a = actions[cohort]
        pr = cohort_prior[cohort]
        ordinal = arrays["cal_cluster_ordinal"][cohort]
        # observed per-stratum factual bias (factual rows of each action/prior)
        biases: list[float] = []
        for act in range(5):
            fm = a == act
            for pv in range(5):
                m = fm & (pr == pv)
                if int(m.sum()) >= MIN_STRATUM_ROWS:
                    agg = binary_probability_metrics(t[m, act], p[m, act],
                                                     ece_bins=10).as_dict()
                    biases.append(float(agg["calibration_bias"]))
        observed_range.append(float(max(biases) - min(biases)))
        # shuffled ranges: permute prior labels within each episode cluster
        rng = np.random.default_rng(B3_SHUFFLE_SEED + cell)
        for rep in range(B3_PERMUTATIONS):
            permuted = np.empty_like(pr)
            for g in range(int(ordinal.max()) + 1):
                gm = ordinal == g
                permuted[gm] = rng.permutation(pr[gm])
            sb: list[float] = []
            for act in range(5):
                fm = a == act
                for pv in range(5):
                    m = fm & (permuted == pv)
                    if int(m.sum()) >= MIN_STRATUM_ROWS:
                        agg = binary_probability_metrics(t[m, act], p[m, act],
                                                         ece_bins=10).as_dict()
                        sb.append(float(agg["calibration_bias"]))
            permuted_ranges.append(float(max(sb) - min(sb)))
    observed_max_range = float(max(observed_range))
    permuted_max_range = float(max(permuted_ranges))
    permuted_p95 = float(np.quantile(permuted_ranges, 0.95))
    b3_passed = bool(observed_max_range > permuted_p95)

    return {
        "B1_reconstruction": {"report": replay_report, "passed": True},
        "B2_structure": {
            "cells_passing_per_scorer": strata_report.get("cells_passing_b2_sc0"),
            "cells": strata_report["cells"],
            "passed": bool(b2_passed),
        },
        "B3_shuffle_specificity": {
            "observed_range_by_cell": observed_range,
            "permuted_range_p95": permuted_p95,
            "permuted_range_max": permuted_max_range,
            "passed": b3_passed,
        },
        "passed": bool(b2_passed and b3_passed),
    }


# ---------------------------------------------------------------------------
# Nomination
# ---------------------------------------------------------------------------

def nominate(arm_a_result: dict[str, object], arm_b_result: dict[str, object]) -> dict[str, object]:
    a = bool(arm_a_result.get("passed", False))
    b = bool(arm_b_result.get("passed", False))
    if a and not b:
        change = ("replace the standard AA CAL calibrator with a fixed MIX35 "
                  "weighted per-action affine calibrator (focal factual weight 0.35)")
        return {"nominated_arm": "A", "nominated_change": change,
                "ledger": {"B_prior_action_strata": "see run report"}}
    if b and not a:
        change = ("condition the hazard representation on the prior applied "
                  "action (one-step lag, burn-in-aware)")
        return {"nominated_arm": "B", "nominated_change": change,
                "ledger": {"A_mix35_calibration": "see run report"}}
    if a and b:
        change = ("replace the standard AA CAL calibrator with a fixed MIX35 "
                  "weighted per-action affine calibrator (focal factual weight 0.35)")
        return {"nominated_arm": "A", "nominated_change": change,
                "ledger": {"B_prior_action_strata": "logged for later preregistration; "
                                                    "never combined with Arm A change"}}
    return {"nominated_arm": None, "nominated_change": None,
            "ledger": {"A_mix35_calibration": "measured negative; see run report",
                        "B_prior_action_strata": "measured; see run report"}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_audit() -> dict[str, object]:
    started = time.time()
    provenance = run_provenance.provenance(
        deterministic=True, train_seed=None, eval_seed=None,
        bank_digest=None,
    )
    provenance["mode"] = MODE
    provenance["run_id"] = RUN_ID
    provenance["classification"] = CLASSIFICATION

    parent = verify_parent_integrity()
    arrays = load_parent_evidence()
    resource_before = resource_guard()

    # Solver controls (fail-closed).
    aa_control = control_aa_refit(arrays)
    pb21k_control = control_pb21k_mix()
    controls = {"aa_refit": aa_control, "pb21k_mix_weight_05": pb21k_control,
                "passed": bool(aa_control["passed"] and pb21k_control["passed"])}
    if not controls["passed"]:
        result = {
            "mode": MODE, "run_id": RUN_ID, "classification": CLASSIFICATION,
            "status": "solver_control_failed", "passed": False,
            "controls": controls, "arm_a": None, "arm_b": None,
            "nomination": {"nominated_arm": None, "nominated_change": None},
            "retry_allowed": False, "provenance": provenance,
            "resource_before": resource_before,
            "wall_seconds": time.time() - started,
        }
        _publish(result, controls_only=True)
        return result

    # Arm A: focal MIX35 + context variants (context not gated).
    focal = arm_a(arrays, MIX35_FOCAL_WEIGHT)
    aa_probs = arrays["oof_calibrated_probabilities"]
    gates = arm_a_gates(focal, aa_probs, arrays)
    context = {}
    for w in MIX35_CONTEXT_WEIGHTS:
        variant = arm_a(arrays, w)
        cm = _factual_metrics(
            variant["oof_probabilities"][1, 0],
            arrays["cal_targets"][0], arrays["cal_actions"][0],
        )
        context[str(w)] = {
            "cell0_factual_bias": float(cm["aggregate"]["calibration_bias"]),
            "cell0_factual_ece": float(cm["aggregate"]["ece_equal_mass"]),
        }
    arm_a_result = {
        "focal_weight": MIX35_FOCAL_WEIGHT,
        "oof_logits_sha256": _sha256_bytes(np.ascontiguousarray(
            focal["oof_logits"]).tobytes()),
        "focal_params": focal["provenance"],
        "context_variants": context,
        "gates": gates,
        "passed": bool(gates["passed"]),
    }

    # Arm B.
    arm_b_result = arm_b(arrays)

    nom = nominate(arm_a_result, arm_b_result)
    resource_after = resource_guard()
    result = {
        "mode": MODE, "run_id": RUN_ID, "classification": CLASSIFICATION,
        "status": "completed",
        "passed": bool(arm_a_result["passed"] or arm_b_result["passed"]),
        "parent_integrity": parent,
        "controls": controls,
        "arm_a": arm_a_result,
        "arm_b": arm_b_result,
        "nomination": nom,
        "retry_allowed": False,
        "provenance": provenance,
        "resource_before": resource_before,
        "resource_after": resource_after,
        "wall_seconds": float(time.time() - started),
    }
    _publish(result, focal_oof=focal["oof_logits"])
    return result


def _publish(result: dict[str, object], *, focal_oof: np.ndarray | None = None,
             controls_only: bool = False) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    result_path = EVIDENCE_DIR / f"{RUN_ID}.json"
    if result_path.exists():
        raise RuntimeError("create-only: result already published")
    payload = json.dumps(_jsonable(result), sort_keys=True, allow_nan=False,
                         separators=(",", ":"))
    result_path.write_text(payload + "\n", encoding="utf-8")
    record = {
        "run_id": RUN_ID,
        "mode": MODE,
        "published_before_arm_scoring": not controls_only,
        "result_sha256": _sha256_bytes(payload.encode("utf-8")),
        "focal_oof_logits_sha256": (
            _sha256_bytes(np.ascontiguousarray(focal_oof).tobytes())
            if focal_oof is not None else None
        ),
        "retry_allowed": False,
        "wall_seconds": result.get("wall_seconds"),
    }
    attempt_path = EVIDENCE_DIR / f"{RUN_ID}.attempt.json"
    if attempt_path.exists():
        raise RuntimeError("create-only: attempt already published")
    attempt_payload = json.dumps(_jsonable(record), sort_keys=True, allow_nan=False,
                                 separators=(",", ":"))
    attempt_path.write_text(attempt_payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    audit = run_audit()
    print(json.dumps(_jsonable({
        "status": audit["status"],
        "passed": audit["passed"],
        "controls": audit.get("controls"),
        "arm_a_passed": bool(audit.get("arm_a", {}).get("passed")) if audit.get("arm_a") else None,
        "arm_b_passed": bool(audit.get("arm_b", {}).get("passed")) if audit.get("arm_b") else None,
        "nomination": audit.get("nomination"),
        "wall_seconds": audit.get("wall_seconds"),
    }, ), indent=1))
