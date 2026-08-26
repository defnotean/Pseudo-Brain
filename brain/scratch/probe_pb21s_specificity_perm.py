"""PB21S-SIG specificity check: permutation test of the rec-vs-base gain.

H0 (no mechanism): the applied-trajectory (action, hazard) sequence carries
no hazard-ranking information; the rec arm's factual AUC gain over base is
just extra trunk capacity / fit variance.

Test: using the SAME OOF-trained rec modules that produced the reference
rec AUC (a freshly grafted module per fold, exactly as crossfit_arm), take
the rec arm's per-root GRU history states and PERMUTE the applied-trajectory
(action, hazard) SEQUENCE across episodes, breaking the hazard-persistence
structure the rec arm is supposed to exploit, while keeping the belief
context, the OOF OOF partition, and the trained GRU parameters intact.
Recompute the rec arm's factual OOF AUC.

Reference: base OOF factual AUC (no history channel); rec OOF factual AUC
(unshuffled history, same modules).

A nonparametric p-value = fraction of random episode-permutations at which
the shuffled-rec factual AUC >= the observed unshuffled rec AUC.

Read-only; reuses the PB21S-SIG probe module.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

BRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN / "scripts"), str(BRAIN / "src")]

spec = importlib.util.spec_from_file_location(
    "pb21s", BRAIN / "scratch" / "probe_pb21s_recurrent_history_signal.py")
pb21s = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pb21s
spec.loader.exec_module(pb21s)

N_PERMS = 1000
PERM_SEED = 77_341
A = pb21s.A
ROOTS_PER_EPISODE = pb21s.ROOTS_PER_EPISODE
BURN_IN = pb21s.BURN_IN
CAL_EPISODES = pb21s.CAL_EPISODES
TOTAL_ROOTS = pb21s.TOTAL_ROOTS


def factual_auc(tab, ev):
    s, y = pb21s.factual_select(tab, ev)
    return pb21s.rank_auc(y, s)


def oof_rec_modules(parent_outcome, ev, seed_offset):
    """OOF cross-fit for the rec arm: a freshly grafted module per fold,
    each trained on its complement fold (mirrors crossfit_arm exactly).
    Returns [module_fold0, module_fold1] (kept for reuse)."""
    modules = []
    for fold in range(2):
        m = pb21s.graft(parent_outcome, True, seed_offset)
        pb21s.train_fold(m, ev, fold, seed_offset)
        modules.append(m)
    return modules


def rec_oof(modules, ev, perm):
    """Full 2-fold rec OOF logit table, with the applied-trajectory sequence
    permuted across episodes by ``perm`` (a length-256 permutation;
    perm[e_now] = the episode whose (action,hazard) sequence is placed at
    episode e_now). Uses the OOF-trained modules (no retraining)."""
    oof = np.zeros((2, TOTAL_ROOTS, A), dtype=np.float64)
    all_eps = np.arange(CAL_EPISODES, dtype=np.int64)
    for fold in range(2):
        m = modules[fold]
        rows = np.where(ev["fold_index"] == fold)[0]
        rows = np.asarray(rows, dtype=np.int64)
        ep_of_row = rows // ROOTS_PER_EPISODE
        tick = BURN_IN + rows % ROOTS_PER_EPISODE
        donor = perm[ep_of_row]                      # [n_rows], any episode
        hs_all = pb21s.gru_states(m, ev, fold, all_eps)  # [256,16,128]
        hs = hs_all[donor, tick - 1]                # causal state through t-1
        hs_t = torch.from_numpy(hs).to(torch.float32)
        bel = torch.from_numpy(ev["beliefs"][rows]).to(torch.float32).detach()
        ids = torch.arange(A).unsqueeze(0).expand(len(rows), -1)
        logits = m(bel, ids, hs_t)
        oof[fold][rows] = logits.detach().to(
            dtype=torch.float64).cpu().numpy()
    return oof


def main() -> None:
    t0 = time.time()
    model, _r, _p = pb21s.pb21n.load_parent()
    parent_outcome = model.world_model.outcome_model
    ev = pb21s.collect_ep_evidence(model)
    seed_offset = 0
    base = pb21s.crossfit_arm(parent_outcome, ev, False, seed_offset)
    modules = oof_rec_modules(parent_outcome, ev, seed_offset)
    a_base = factual_auc(base, ev)
    # Reference unshuffled rec AUC with the same OOF modules:
    ident = np.arange(CAL_EPISODES)
    a_rec = factual_auc(rec_oof(modules, ev, ident), ev)
    print(f"[i] reference (seed_offset={seed_offset}): base={a_base:.4f}  "
          f"rec={a_rec:.4f}  gain={a_rec - a_base:+.4f}")

    rng = np.random.default_rng(PERM_SEED)
    aces = []
    for it in range(N_PERMS):
        perm = rng.permutation(CAL_EPISODES)
        aces.append(factual_auc(rec_oof(modules, ev, perm), ev))
    aces = np.array(aces)
    p = float((aces >= a_rec).mean())
    print(f"[i] {N_PERMS} perms; shuffled-rec mean={aces.mean():.4f} "
          f"(base={a_base:.4f}, rec={a_rec:.4f})")
    print(f"[i] empirical p (shuffled-rec >= observed rec) = {p:.4f}")
    print(f"[i] wall {time.time()-t0:.1f}s (READ-ONLY)")
    print("INTERPRETATION:")
    print("  shuffled-rec ~ base, p small  -> gain is SPECIFICALLY from the")
    print("    applied-trajectory history (a real mechanism; preregister a")
    print("    trial with a ranking gate).")
    print("  shuffled-rec ~ rec, p ~ 1     -> gain is from trunk capacity,")
    print("    not the history; refuted at probe level; no trial.")


if __name__ == "__main__":
    main()
