"""Probe: data-only replay of one sealed PB21M CAL partition.

Verifies that roots/actions/targets (data-derived, model-independent)
reconstruct byte-identically from the deterministic dataset, and derives
the prior-action sequence needed by the Arm B strata.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN / "scripts"), str(BRAIN / "src")]

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import run_provenance  # noqa: E402

run_provenance.apply_deterministic_mode()

import v21i_development_runner as development  # noqa: E402
import v21i_strict_live_representation_probe_v1 as strict  # noqa: E402
import v21m_fresh_bal_cal_first_qualification_v1 as pb21m  # noqa: E402
from irene_brain.v2.trajectory_objective import control_action_class  # noqa: E402

EVIDENCE = BRAIN / "runs/v21m-qualification/2026-08-25-pb21m-fresh-bal-cal-first-v1.cal-evidence.npz"
TARGET = "C0A"


def main() -> None:
    spec = {s.label: s for s in pb21m.PARTITION_SPECS}[TARGET]
    contracts = pb21m.partition_contracts()
    manifests = pb21m.partition_manifests()
    source = development.PartitionSource(contracts[TARGET])
    if source.manifest_sha256 != manifests[TARGET]:
        raise SystemExit(f"manifest drift: {source.manifest_sha256} != {manifests[TARGET]}")
    burn = contracts[TARGET].burn_in_steps
    seq_len = contracts[TARGET].dataset_config.sequence_length

    roots: list[str] = []
    actions: list[int] = []
    for batch in source.iter_all_action_batches(epoch=0, batch_size=1):
        for tick in range(batch.sequence_length):
            transitions = tuple(s.transitions[tick] for s in batch.sequences)
            applied = [
                control_action_class(t.applied_control) for t in transitions
            ]
            if tick >= burn:
                for i, t in enumerate(transitions):
                    roots.append(t.root_state_sha256)
                    actions.append(applied[i])
    roots_np = np.asarray([r.encode("ascii") for r in roots], dtype="S64")
    actions_np = np.asarray(actions, dtype=np.int64)

    z = np.load(EVIDENCE, allow_pickle=False)
    # find cohort index for TARGET
    cohort = None
    for c in range(3):
        ids = z["cal_cluster_ids"][c]
        lab = bytes(ids[0]).rstrip(b"\x00").decode()
        if lab.startswith(TARGET):
            cohort = c
            break
    assert cohort is not None, "cohort not found"
    sealed_roots = z["cal_root_ids"][cohort][:3072]
    sealed_actions = z["cal_actions"][cohort][:3072]
    sealed_targets = z["cal_targets"][cohort][:3072]
    print(f"cohort {cohort} ({TARGET}): replayed rows={len(roots)} sealed(first fold)={sealed_roots.shape[0]}")
    print("roots byte-identical:", np.array_equal(roots_np, sealed_roots))
    print("actions byte-identical:", np.array_equal(actions_np, sealed_actions))

    # derive prior action: first recorded root per episode has prior=0 (collector initial)
    roots_per_ep = seq_len - burn
    ep = len(roots) // roots_per_ep
    prior = np.zeros(len(roots), dtype=np.int64)
    for e in range(ep):
        base = e * roots_per_ep
        for t in range(1, roots_per_ep):
            prior[base + t] = actions_np[base + t - 1]
    print("episodes:", ep, "roots/ep:", roots_per_ep)
    # per-episode applied sequence check
    from collections import Counter
    pa = Counter()
    for p, a in zip(prior, actions_np):
        pa[(int(p), int(a))] += 1
    print("prior==factual fraction:", pa[(0, 0)] + pa[(1, 1)] + pa[(2, 2)] + pa[(3, 3)] + pa[(4, 4)],
          "of", len(roots))
    # prior action distribution
    print("prior action dist:", np.bincount(prior, minlength=5).tolist())
    print("factual action dist:", np.bincount(actions_np, minlength=5).tolist())

    # quick ECE probe: do sealed AA OOF factual probabilities, stratum by prior action,
    # show structured miscalibration? (informal preview, not the audit)
    oof = z["oof_calibrated_probabilities"]  # [sc, cell, 6144, 5]; rows 0:3072 = C0A fold
    cell = 0  # F0/init-91042
    for sc in range(2):
        p = oof[sc, cell, :3072]
        t = sealed_targets
        fact = actions_np
        for a in range(5):
            m = fact == a
            pm = p[m, a]
            tm = t[m, a]
            sm = prior[m]
            for pv in range(5):
                mm = sm == pv
                if mm.sum() >= 20:
                    bias = float(pm[mm].mean() - tm[mm].mean())
                    print(f"  sc{sc} cell{cell} factual a={a} prior={pv}: n={int(mm.sum())} "
                          f"prob_mean={pm[mm].mean():.4f} prev={tm[mm].mean():.4f} bias={bias:+.4f}")


if __name__ == "__main__":
    main()
