"""PB21O — isotonic (PAV) per-action hazard calibrator on the prior-action
conditioned path (single mechanism: the calibrator function class).

Classification: fresh_isotonic_hazard_calibration_single_mechanism
(planned), non-qualifying until a separate scaling/qualification prereg
passes.

Source of the mechanism: architecture ideas ledger L4, licensed by the
PB21N trial #1 frozen negative (the affine combination on the conditioned
path failed G1/G3). Prereg:
brain/docs/preregistrations/2026-08-25-pb21o-isotonic-hazard-calibration-v1.md

Arms (all on the fresh PB21O-CAL partition; upstream belief frozen
byte-exact):
  H_base      — frozen parent hazard path, raw logits, AA cross-fit cal.
  H_prior_aa  — prior-action conditioned path, AFFINE (AA) cross-fit cal
                (control arm; identical conditioning to the candidate).
  H_prior_iso — prior-action conditioned path, ISOTONIC (PAV) cross-fit
                cal (candidate).
H_prior_aa and H_prior_iso share the SAME conditioned OOF raw logit
tables, so G5 isolates the calibrator function class.

PASS = G1 AND G2 AND G3 AND G4 AND G5 AND G6 (frozen in the prereg).
"""
from __future__ import annotations

import json
import math
import os
import time
from hashlib import sha256
from pathlib import Path
from typing import Mapping

BRAIN_ROOT = Path(__file__).resolve().parents[1]
sys_path_fix = [str(BRAIN_ROOT / "scripts"), str(BRAIN_ROOT / "src")]
import sys  # noqa: E402
sys.path[:0] = sys_path_fix

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import importlib  # noqa: E402

import run_provenance  # noqa: E402

from irene_brain.evaluation.v21_qualification_metrics import (  # noqa: E402
    binary_probability_metrics as bpm,
)

# Reuse the PB21N runner as a library (its __main__ guard prevents the
# trial from executing on import). It supplies the conditioned-path
# mechanism, the AA calibrator, the gate helpers, and the house digest.
pb21n = importlib.import_module("v21n_prior_action_hazard_conditioning_v1")

run_provenance.apply_deterministic_mode()
torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)

# ---------------------------------------------------------------------------
# Frozen trial constants (prereg 2026-08-25-pb21o-isotonic-hazard-
# calibration-v1)
# ---------------------------------------------------------------------------
MODE = "pb21o_isotonic_hazard_calibration_v1"
RUN_ID = "2026-08-25-pb21o-isotonic-hazard-calibration-v1"
CLASSIFICATION = "fresh_isotonic_hazard_calibration_single_mechanism"
EVIDENCE_DIR = BRAIN_ROOT / "runs" / "pb21o-isotonic-hazard-cal"

# Parent seals — identical to PB21N trial #1 (the same frozen parent).
V3_RESULT_SHA256 = pb21n.V3_RESULT_SHA256
PARENT_SHA256 = pb21n.PARENT_SHA256
PARENT_STATE_SHA256 = pb21n.PARENT_STATE_SHA256

# Fresh partition (PB21O-CAL), contiguous after PB21N-CAL.
CAL_SPLIT = pb21n.CAL_SPLIT  # DatasetSplit.TRAIN
CAL_SEED_OFFSET = 167_774_720
CAL_EPISODES = 256
CAL_BURN_IN = 4
CAL_SEQ_LEN = 16
ROOTS_PER_EPISODE = 12
TOTAL_ROOTS = CAL_EPISODES * ROOTS_PER_EPISODE  # 3072
FOLD_ROWS = 1536
WIDTH = pb21n.WIDTH
HIDDEN = pb21n.HIDDEN
ACTION_COUNT = pb21n.ACTION_COUNT

# Gates (frozen, prereg §5).
FACTUAL_AGGREGATE_BIAS_LIMIT = 0.05
FACTUAL_AGGREGATE_ECE_LIMIT = 0.05
ALL_ACTION_CONTROL_WORSE_MARGIN = 0.01
FACTORIAL_AUC_DROP_LIMIT = 0.05
ONE_SIDED_ALPHA = 0.025
QUANTILE_METHOD = "linear"
BOOTSTRAP_RESAMPLES = 10_000
MIN_STRATUM_ROWS = 20
TRIAL_SEED = 43  # identical conditioning seed to PB21N trial #1

# Training schedule: bound to the same V2.1i refinement contract as PB21N.
REFINEMENT_PASSES = pb21n.REFINEMENT_PASSES
REFINEMENT_ROOT_BATCH_SIZE = pb21n.REFINEMENT_ROOT_BATCH_SIZE
REFINEMENT_LEARNING_RATE = pb21n.REFINEMENT_LEARNING_RATE
REFINEMENT_WEIGHT_DECAY = pb21n.REFINEMENT_WEIGHT_DECAY
REFINEMENT_CLIP_NORM = pb21n.REFINEMENT_CLIP_NORM


def _pb21o_cal_contract():
    """PB21O-CAL contract: identical to PB21N-CAL but seed_offset
    167,774,720 (the contract dataclass carries the seed only inside
    dataset_config)."""
    from dataclasses import replace
    base = _ORIG_PBN_CAL_CONTRACT()
    return pb21n.PB21NContract(
        "PB21O-CAL",
        replace(base.dataset_config, seed_offset=CAL_SEED_OFFSET),
    )


_ORIG_PBN_CAL_CONTRACT = pb21n.cal_contract


# ---------------------------------------------------------------------------
# Isotonic regression (PAV), deterministic, hand-rolled.
# ---------------------------------------------------------------------------

def pav_fit(values: np.ndarray, targets: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, int]:
    """Deterministic pool-adjacent-violators isotonic fit.

    Returns (node_values, node_probs, step_count):
      node_values[i] = the logit boundary (i-th block's value range start)
      node_probs[i]  = the block's mean target (piecewise-constant prob)
    The mapping is monotone non-decreasing in the input value. Ties are
    broken by stable original order (sort by (value, index)).
    """
    v = np.ascontiguousarray(values, dtype=np.float64)
    t = np.ascontiguousarray(targets, dtype=np.float64)
    n = v.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), 0
    order = np.lexsort((np.arange(n), v))  # stable by index on ties
    sv = v[order]
    st = t[order]
    cnt = np.ones(n, dtype=np.float64)
    mean = st.copy()
    # forward pass with a stack of (start_index, count, weighted_mean)
    stack: list[tuple[int, float, float]] = []
    for i in range(n):
        stack.append((i, 1.0, st[i]))
        while len(stack) >= 2:
            a = stack[-2]
            b = stack[-1]
            merged_mean = (a[2] * a[1] + b[2] * b[1]) / (a[1] + b[1])
            if a[2] <= b[2] + 0.0:  # non-decreasing -> keep
                break
            # merge b into a
            new_start = a[0]
            new_count = a[1] + b[1]
            new_mean = merged_mean
            stack.pop()
            stack.pop()
            stack.append((new_start, new_count, new_mean))
    # materialize blocks
    node_values = np.array([sv[b[0]] for b in stack], dtype=np.float64)
    node_probs = np.array([b[2] for b in stack], dtype=np.float64)
    return node_values, node_probs, len(stack)


def pav_evaluate(x: np.ndarray, node_values: np.ndarray,
                 node_probs: np.ndarray) -> np.ndarray:
    """Piecewise-constant isotonic evaluation (right-continuous)."""
    if node_values.shape[0] == 0:
        raise ValueError("empty PAV map")
    x = np.ascontiguousarray(x, dtype=np.float64)
    # idx = number of node_values strictly <= x, minus 1; clamp to [0, K-1]
    idx = np.searchsorted(node_values, x, side="right") - 1
    idx = np.clip(idx, 0, node_values.shape[0] - 1)
    return node_probs[idx]


def pav_crossfit_calibration(
    raw_oof: np.ndarray, ev,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Fit per-action PAV on the complementary fold's OOF (logit, target),
    apply to the target fold. Same fold convention as the AA calibrator.
    Returns OOF *probabilities* [2, 3072, 5] (PAV maps into probability
    space directly; complement rows stay the defined 0.0).
    """
    calibrated = np.zeros_like(raw_oof)
    provenance: list[dict[str, object]] = []
    for fold in range(2):
        fit_fold = 1 - fold
        fit_mask = ev.fold_index == fit_fold
        eval_rows = ev.fold_index == fold
        step_counts: list[int] = []
        node_values_all: list[list[float]] = []
        node_probs_all: list[list[float]] = []
        for act in range(ACTION_COUNT):
            x_fit = np.ascontiguousarray(raw_oof[fit_fold][fit_mask, act])
            t_fit = np.ascontiguousarray(ev.targets[fit_mask, act])
            node_values, node_probs, steps = pav_fit(x_fit, t_fit)
            step_counts.append(int(steps))
            node_values_all.append(node_values.tolist())
            node_probs_all.append(node_probs.tolist())
            x_eval = np.ascontiguousarray(raw_oof[fold][eval_rows, act])
            calibrated[fold][eval_rows, act] = pav_evaluate(
                x_eval, node_values, node_probs)
        provenance.append({
            "cal_fold": fold,
            "fit_fold": fit_fold,
            "pav_step_counts": step_counts,
            "pav_node_values": node_values_all,
            "pav_node_probs": node_probs_all,
        })
    return calibrated, provenance


# ---------------------------------------------------------------------------
# Conditioned-path conditioning (one pass; reused for determinism G6).
# Returns the prior-arm OOF raw logit table [2, 3072, 5].
# ---------------------------------------------------------------------------

def condition_prior_path(parent_outcome, ev: "pb21n.CALEvidence") -> np.ndarray:
    """Retrain the prior-action path per OOF fold; return OOF raw logits
    [fold, 3072, 5] (complement half zero-defined)."""
    oof = np.zeros((2, TOTAL_ROOTS, ACTION_COUNT), dtype=np.float64)
    for fold in range(2):
        module = pb21n.graft_prior(parent_outcome)
        pb21n.train_hazard_fold(
            module, ev.beliefs, ev.targets, ev.prior_applied,
            fold_index=ev.fold_index, fold=fold,
        )
        eval_rows = ev.fold_index == fold
        ids = torch.arange(ACTION_COUNT).unsqueeze(0).expand(
            int(eval_rows.sum()), -1)
        logits = module(
            torch.from_numpy(ev.beliefs[eval_rows]).to(torch.float32).detach(),
            ids,
            torch.from_numpy(ev.prior_applied[eval_rows]).to(torch.long),
        )
        oof[fold][eval_rows] = logits.detach().to(
            "cpu", dtype=torch.float64).numpy()
    return oof


# ---------------------------------------------------------------------------
# Gates (frozen, prereg §5). Reuses PB21N's metric + bootstrap helpers.
# ---------------------------------------------------------------------------

def evaluate_gates(
    p_iso: np.ndarray, p_aa: np.ndarray, p_base: np.ndarray,
    ev,
) -> dict[str, object]:
    # p_iso / p_aa / p_base are OOF *probability* tables [2, 3072, 5].
    g1_tables = []
    g1_passed = True
    for fold in range(2):
        rows = ev.fold_index == fold
        m = pb21n._factual_metrics(
            p_iso[fold][rows], ev.targets[rows], ev.applied[rows])
        bias = float(m["aggregate"]["calibration_bias"])
        ece = float(m["aggregate"]["ece_equal_mass"])
        ok = abs(bias) <= FACTUAL_AGGREGATE_BIAS_LIMIT and \
            ece <= FACTUAL_AGGREGATE_ECE_LIMIT
        g1_passed = g1_passed and ok
        g1_tables.append({
            "fold": fold, "factual_bias": bias, "factual_ece": ece,
            "factual_bce": float(m["aggregate"]["bce"]),
            "factual_brier": float(m["aggregate"]["brier"]),
            "per_action": m["per_action"], "passed": bool(ok),
        })

    # G2: paired P_iso vs P_base, factual BCE+Brier, episode-clustered.
    delta_2 = np.stack([
        pb21n._factual_losses(
            ev.targets[ev.fold_index == f],
            p_base[f][ev.fold_index == f], ev.applied[ev.fold_index == f])
        - pb21n._factual_losses(
            ev.targets[ev.fold_index == f],
            p_iso[f][ev.fold_index == f], ev.applied[ev.fold_index == f])
        for f in range(2)
    ], axis=0)
    g2_cells = {}
    g2_passed = True
    for fold in range(2):
        point = delta_2[fold].mean(axis=0)
        draws = pb21n._episode_cluster_bootstrap(
            delta_2[fold], ev.episode_ordinal[ev.fold_index == fold],
            BOOTSTRAP_RESAMPLES, seed=TRIAL_SEED + 500 * fold + 1)
        lower = np.quantile(draws, ONE_SIDED_ALPHA, axis=0,
                            method=QUANTILE_METHOD)
        ok = bool(point[0] > 0.0 and lower[0] > 0.0 and point[1] > 0.0
                  and lower[1] > 0.0)
        g2_passed = g2_passed and ok
        g2_cells[f"fold{fold}"] = {
            "point": {"BCE": float(point[0]), "Brier": float(point[1])},
            "LCB_97.5": {"BCE": float(lower[0]), "Brier": float(lower[1])},
            "passed": bool(ok),
        }

    # G3: all-action control not degraded vs P_base.
    g3_cells = []
    g3_passed = True
    for fold in range(2):
        rows = ev.fold_index == fold
        aa_iso = pb21n._all_action_metrics(p_iso[fold][rows],
                                           ev.targets[rows])
        aa_base = pb21n._all_action_metrics(p_base[fold][rows],
                                            ev.targets[rows])
        bias_delta = abs(float(aa_iso["calibration_bias"])) - abs(
            float(aa_base["calibration_bias"]))
        ece_delta = float(aa_iso["ece_equal_mass"]) - float(
            aa_base["ece_equal_mass"])
        dom = pb21n._factual_mask(ev.applied[rows])
        auc_ok = True
        auc_worst_drop = 0.0
        for act in range(ACTION_COUNT):
            r = dom[:, act]
            if int(r.sum()) < 20:
                continue
            t_aa = ev.targets[rows][r, act]
            if not (0.0 < float(np.mean(t_aa)) < 1.0):
                continue
            aa_auc = float(bpm(t_aa, p_base[fold][rows][r, act],
                               ece_bins=10).as_dict()["roc_auc"])
            p_auc = float(bpm(t_aa, p_iso[fold][rows][r, act],
                              ece_bins=10).as_dict()["roc_auc"])
            drop = aa_auc - p_auc
            auc_worst_drop = max(auc_worst_drop, drop)
            auc_ok = auc_ok and drop <= FACTORIAL_AUC_DROP_LIMIT
        ok = bool(bias_delta <= ALL_ACTION_CONTROL_WORSE_MARGIN
                  and ece_delta <= ALL_ACTION_CONTROL_WORSE_MARGIN and auc_ok)
        g3_passed = g3_passed and ok
        g3_cells.append({
            "fold": fold,
            "all_action_bias_abs_delta": float(bias_delta),
            "all_action_ece_delta": float(ece_delta),
            "worst_factual_auc_drop": float(auc_worst_drop),
            "passed": bool(ok),
        })

    # G5: function-class isolation P_iso vs P_aa (same conditioned path).
    delta_5 = np.stack([
        pb21n._factual_losses(
            ev.targets[ev.fold_index == f],
            p_aa[f][ev.fold_index == f], ev.applied[ev.fold_index == f])
        - pb21n._factual_losses(
            ev.targets[ev.fold_index == f],
            p_iso[f][ev.fold_index == f], ev.applied[ev.fold_index == f])
        for f in range(2)
    ], axis=0)
    g5_cells = {}
    g5_passed = True
    for fold in range(2):
        point = delta_5[fold].mean(axis=0)
        draws = pb21n._episode_cluster_bootstrap(
            delta_5[fold], ev.episode_ordinal[ev.fold_index == fold],
            BOOTSTRAP_RESAMPLES, seed=TRIAL_SEED + 700 * fold + 3)
        lower = np.quantile(draws, ONE_SIDED_ALPHA, axis=0,
                            method=QUANTILE_METHOD)
        ok = bool(point[0] > 0.0 and lower[0] > 0.0)
        g5_passed = g5_passed and ok
        g5_cells[f"fold{fold}"] = {
            "point": {"BCE": float(point[0]), "Brier": float(point[1])},
            "LCB_97.5": {"BCE": float(lower[0]), "Brier": float(lower[1])},
            "passed": bool(ok),
        }

    gates = {
        "G1_factual_absolute": {"tables": g1_tables, "passed": bool(g1_passed)},
        "G2_paired_vs_base": {"cells": g2_cells, "passed": bool(g2_passed)},
        "G3_all_action_controls": {"cells": g3_cells, "passed": bool(g3_passed)},
        "G5_function_class_isolation": {"cells": g5_cells, "passed": bool(g5_passed)},
    }
    gates["passed"] = bool(g1_passed and g2_passed and g3_passed and g5_passed)
    return gates


# ---------------------------------------------------------------------------
# Trial orchestration + create-only publish.
# ---------------------------------------------------------------------------

def _jsonable(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    raise TypeError(f"unserializable: {type(value)}")


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def run_trial() -> dict[str, object]:
    started = time.time()
    provenance = pb21n._provenance()
    provenance["mode"] = MODE
    provenance["run_id"] = RUN_ID
    provenance["classification"] = CLASSIFICATION

    model, result, parent_provenance = pb21n.load_parent()
    outcome = model.world_model.outcome_model
    parent_state_sha_before = pb21n._state_dict_sha256(model.state_dict())
    if parent_state_sha_before != PARENT_STATE_SHA256:
        raise RuntimeError("parent state digest drifted at trial start")

    # PB21O-CAL: collect_cal_evidence builds its contract from the
    # module-level cal_contract(); patch it to the PB21O-CAL contract for
    # this call only and restore afterwards.
    orig_contract = pb21n.cal_contract
    try:
        pb21n.cal_contract = _pb21o_cal_contract
        ev = pb21n.collect_cal_evidence(model)
    finally:
        pb21n.cal_contract = orig_contract

    # One conditioning pass (shared by both calibrator arms).
    oof_prior_raw = condition_prior_path(outcome, ev)

    # H_base: frozen parent raw logits, AA cross-fit cal -> probabilities.
    base_raw_oof = np.stack([ev.base_raw_logits, ev.base_raw_logits])
    h_base_aa_logits, base_cal = pb21n.aa_crossfit_calibration(base_raw_oof, ev)
    p_base = 1.0 / (1.0 + np.exp(-h_base_aa_logits))

    # H_prior_aa: affine cal on the conditioned OOF logits -> probs.
    h_prior_aa_logits, prior_aa_cal = pb21n.aa_crossfit_calibration(
        oof_prior_raw, ev)
    p_aa = 1.0 / (1.0 + np.exp(-h_prior_aa_logits))

    # H_prior_iso: PAV cal on the same conditioned OOF logits -> probs.
    p_iso, iso_cal = pav_crossfit_calibration(oof_prior_raw, ev)

    # G4: parent preservation.
    parent_state_sha_after = pb21n._state_dict_sha256(model.state_dict())
    g4_passed = bool(parent_state_sha_after == PARENT_STATE_SHA256)
    g4 = {
        "parent_state_sha256_before": parent_state_sha_before,
        "parent_state_sha256_after": parent_state_sha_after,
        "parent_state_sha256_sealed": PARENT_STATE_SHA256,
        "passed": g4_passed,
    }

    gates = evaluate_gates(p_iso, p_aa, p_base, ev)
    gates["G4_parent_preservation"] = g4
    gates["passed"] = bool(gates["passed"] and g4_passed)

    # G6: determinism — re-condition and re-fit PAV from saved arrays.
    oof_prior_raw_2 = condition_prior_path(outcome, ev)
    conditioning_rederived = bool(np.array_equal(oof_prior_raw, oof_prior_raw_2))
    p_iso_2, _ = pav_crossfit_calibration(oof_prior_raw, ev)
    iso_rederived = bool(np.array_equal(p_iso, p_iso_2))
    g6 = bool(conditioning_rederived and iso_rederived)
    del oof_prior_raw_2, p_iso_2

    result_doc = {
        "mode": MODE,
        "run_id": RUN_ID,
        "classification": CLASSIFICATION,
        "status": "completed",
        "passed": bool(gates["passed"] and g6),
        "determinism": {
            "prior_conditioning_rederived_byte_equal": conditioning_rederived,
            "iso_oof_prob_rederived_byte_equal": iso_rederived,
            "g6_passed": g6,
        },
        "parent_provenance": {
            "v3_result_sha256": V3_RESULT_SHA256,
            "parent_sha256": PARENT_SHA256,
            "parent_state_sha256": PARENT_STATE_SHA256,
            "provenance": _jsonable(parent_provenance),
        },
        "partition": {
            "label": "PB21O-CAL",
            "split": CAL_SPLIT.value,
            "seed_offset": CAL_SEED_OFFSET,
            "episodes": CAL_EPISODES,
            "burn_in_steps": CAL_BURN_IN,
            "sequence_length": CAL_SEQ_LEN,
            "roots_per_episode": ROOTS_PER_EPISODE,
            "total_roots": TOTAL_ROOTS,
            "fold_rows": FOLD_ROWS,
            "partition_manifest_sha256": ev.partition_manifest_sha256,
            "episode_permutation_sha256": ev.episode_permutation_sha256,
        },
        "arms": {
            "H_base": "frozen parent raw hazard logits, AA cross-fit calibrated",
            "H_prior_aa": "prior-action conditioned path, affine (AA) cross-fit calibrated (control)",
            "H_prior_iso": "prior-action conditioned path, isotonic (PAV) cross-fit calibrated (candidate)",
        },
        "gates": gates,
        "calibration_provenance": {
            "base": base_cal, "prior_aa": prior_aa_cal, "iso": iso_cal,
        },
        "oof_probability_sha256": {
            "H_base": _sha256_bytes(np.ascontiguousarray(p_base).tobytes()),
            "H_prior_aa": _sha256_bytes(np.ascontiguousarray(p_aa).tobytes()),
            "H_prior_iso": _sha256_bytes(np.ascontiguousarray(p_iso).tobytes()),
        },
        "conditioning_oof_logit_sha256": _sha256_bytes(
            np.ascontiguousarray(oof_prior_raw).tobytes()),
        "retry_allowed": False,
        "provenance": provenance,
        "wall_seconds": time.time() - started,
    }
    _publish(result_doc, ev, oof_prior_raw, p_iso, p_aa, p_base)
    return result_doc


def _publish(result_doc, ev, oof_prior_raw, p_iso, p_aa, p_base) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(result_doc), sort_keys=True,
                         allow_nan=False, separators=(",", ":"))
    result_path = EVIDENCE_DIR / f"{RUN_ID}.json"
    if result_path.exists():
        raise RuntimeError("create-only: result already published")
    result_path.write_text(payload + "\n", encoding="utf-8")
    record = {
        "run_id": RUN_ID,
        "mode": MODE,
        "result_sha256": _sha256_bytes(payload.encode("utf-8")),
        "retry_allowed": False,
        "wall_seconds": result_doc.get("wall_seconds"),
    }
    attempt_path = EVIDENCE_DIR / f"{RUN_ID}.attempt.json"
    if attempt_path.exists():
        raise RuntimeError("create-only: attempt already published")
    attempt_path.write_text(
        json.dumps(_jsonable(record), sort_keys=True,
                   allow_nan=False, separators=(",", ":")) + "\n",
        encoding="utf-8")
    np.savez(
        EVIDENCE_DIR / f"{RUN_ID}.evidence.npz",
        beliefs=ev.beliefs,
        targets=ev.targets,
        applied=ev.applied,
        prior_applied=ev.prior_applied,
        episode_ordinal=ev.episode_ordinal,
        root_ids=ev.root_ids,
        base_raw_logits=ev.base_raw_logits,
        fold_index=ev.fold_index,
        oof_prior_raw=oof_prior_raw,
        oof_H_prior_iso_prob=p_iso,
        oof_H_prior_aa_prob=p_aa,
        oof_H_base_prob=p_base,
        partition_manifest_sha256=ev.partition_manifest_sha256.encode(),
        episode_permutation_sha256=ev.episode_permutation_sha256.encode(),
    )
    reg = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "mode": MODE,
        "classification": CLASSIFICATION,
        "prereg": "brain/docs/preregistrations/"
                  "2026-08-25-pb21o-isotonic-hazard-calibration-v1.md",
        "parent": {
            "v3_result_sha256": V3_RESULT_SHA256,
            "parent_sha256": PARENT_SHA256,
            "parent_state_sha256": PARENT_STATE_SHA256,
        },
        "partition": {
            "split": CAL_SPLIT.value,
            "seed_offset": CAL_SEED_OFFSET,
            "episodes": CAL_EPISODES,
            "burn_in_steps": CAL_BURN_IN,
            "sequence_length": CAL_SEQ_LEN,
        },
        "gates": {
            "factual_bias_limit": FACTUAL_AGGREGATE_BIAS_LIMIT,
            "factual_ece_limit": FACTUAL_AGGREGATE_ECE_LIMIT,
            "all_action_worse_margin": ALL_ACTION_CONTROL_WORSE_MARGIN,
            "factual_auc_drop_limit": FACTORIAL_AUC_DROP_LIMIT,
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "quantile_method": QUANTILE_METHOD,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "min_stratum_rows": MIN_STRATUM_ROWS,
        },
        "trial_seed": TRIAL_SEED,
        "training_schedule": {
            "passes": REFINEMENT_PASSES,
            "root_batch_size": REFINEMENT_ROOT_BATCH_SIZE,
            "learning_rate": REFINEMENT_LEARNING_RATE,
            "weight_decay": REFINEMENT_WEIGHT_DECAY,
            "clip_norm": REFINEMENT_CLIP_NORM,
        },
    }
    reg_path = EVIDENCE_DIR / f"{RUN_ID}.registration.json"
    if reg_path.exists():
        raise RuntimeError("create-only: registration already published")
    reg_path.write_text(
        json.dumps(_jsonable(reg), sort_keys=True, allow_nan=False,
                   separators=(",", ":")) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    trial = run_trial()
    print(json.dumps(_jsonable({
        "status": trial["status"],
        "passed": trial["passed"],
        "determinism": trial["determinism"],
        "gates": {k: v.get("passed") for k, v in trial["gates"].items()
                  if isinstance(v, dict)},
        "wall_seconds": trial["wall_seconds"],
    }), indent=1))
