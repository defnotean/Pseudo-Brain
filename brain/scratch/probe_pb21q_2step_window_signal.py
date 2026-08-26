"""PB21Q-SIG — 2-step prior-action window SIGNAL probe (READ-ONLY, no publish).

Purpose
-------
L8 (see ARCHITECTURE_IDEAS_LEDGER) established that the hazard factual
failure is *ranking/signal-density-limited*: the prior-conditioned hazard
logit tops out at factual aggregate AUC ~0.708, and every monotone
calibrator (affine / PAV / block-capped PAV) saturates below the 0.05
factual-ECE gate. The only unexplored mechanism in the hazard branch is
representation-level: change what the hazard path *sees*.

The single-change extension of the already-measured 1-step prior-action
mechanism (PB21N, confirmed specific G5 but not calibrating) is a **2-step
prior-action window**: condition the hazard representation on the action
applied one tick earlier *and* the action applied two ticks earlier.

This probe is a DESIGN decision aid, not a trial. It is READ-ONLY on a
fresh deterministic replay of the frozen parent over the PB21O-CAL
partition (seed_offset 167,774,720; 256 episodes x 12 roots = 3072;
2 folds of 1536). It:
  * retrains three arms per OOF fold (deterministic, CPU, single-thread,
    CUDA hidden) — base (2H, no prior), 1-step (3H, one prior channel),
    2-step (4H, two prior channels) — all zero-extended so each arm's new
    channels contribute zero at init;
  * reports factual aggregate + per-action AUC (the binding constraint),
    factual ECE, and all-action calibration bias for each arm;
  * reports the in-sample PAV ECE ceiling for each arm (what any monotone
    calibrator could ever reach on that logit), reusing PB21O's hand-rolled
    PAV (sklearn cross-checked at 1e-16).

The 1-step arm is a harness self-validation: it must reproduce PB21O's
prior-conditioned factual aggregate AUC ~0.708 on this partition. If the
2-step arm cannot clear the 1-step ranking ceiling by a margin beyond
cross-fit noise, the representation-level single change is refuted at
probe level and NO PB21Q trial is warranted (same discipline that saved
the PB21P block-cap trial).

NO publish. NO frozen-state change. NO new partition. Deterministic.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

os_env_setup = True
import os  # noqa: E402
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

BRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN / "scripts"), str(BRAIN / "src")]

import run_provenance  # noqa: E402
import v21n_prior_action_hazard_conditioning_v1 as pb21n  # noqa: E402
import v21o_isotonic_hazard_calibration_v1 as pb21o  # noqa: E402
import v21i_development_runner as development  # noqa: E402
from irene_brain.data import DatasetSplit  # noqa: E402
from irene_brain.v2.trajectory_objective import (  # noqa: E402
    control_action_class,
)

run_provenance.apply_deterministic_mode()
torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)

WIDTH = pb21n.WIDTH            # 120
HIDDEN = pb21n.HIDDEN          # 120
A = pb21n.ACTION_COUNT         # 5
TRIAL_SEED = pb21n.TRIAL_SEED  # 43
TOTAL_ROOTS = pb21n.TOTAL_ROOTS
FOLD_ROWS = pb21n.FOLD_ROWS
CAL_EPISODES = pb21n.CAL_EPISODES
ROOTS_PER_EPISODE = pb21n.ROOTS_PER_EPISODE


# ---------------------------------------------------------------------------
# 2-step evidence collection (replay the frozen parent over the PB21O-CAL
# partition; add the 2-step prior channel).
# ---------------------------------------------------------------------------
def collect_2step_evidence(model) -> dict[str, np.ndarray]:
    """Deterministic replay of the PB21O-CAL partition through the frozen
    parent. Returns full-partition arrays (3072 rows, epoch-0 order,
    batch_size=1). Records applied, prior(1-step), prior2(2-step) applied
    actions and the frozen parent raw hazard logits.

    Burn-in = 4, so the first recorded tick is 4; every recorded root has a
    valid tick-2 predecessor (transitions[t-2] exists for t>=2).
    """
    contract = pb21o._pb21o_cal_contract()
    dev_contract = pb21n._to_dev_contract(contract)
    source = development.PartitionSource(dev_contract)
    device = torch.device("cpu")
    was_training = model.training
    model.eval()

    beliefs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    applied: list[int] = []
    prior1: list[int] = []
    prior2: list[int] = []
    base_raw: list[np.ndarray] = []
    episode_index = 0

    for batch in source.iter_all_action_batches(epoch=0, batch_size=1):
        episode_index += 1
        sequence = batch.sequences[0]
        state = model.init_state(batch.batch_size, device)
        for tick in range(batch.sequence_length):
            transition = sequence.transitions[tick]
            prev = None if tick == 0 else sequence.transitions[tick - 1]
            pixels, applied_t, previous, prior_reward, prior_hazard = (
                development._model_inputs(
                    (transition,),
                    prior_transitions=(prev,) if prev is not None else None,
                    device=device,
                )
            )
            output, state = model(
                pixels,
                state,
                prev_action=previous,
                actual_reward=prior_reward,
                actual_hazard=prior_hazard,
                world_model_action=applied_t,
                intervention_pe="normal",
            )
            if tick < batch.burn_in_steps:
                continue
            table = output.outcome_table
            if table is None or table.raw_hazard_logits is None:
                raise RuntimeError("V2.1i requires raw all-action hazard logits")
            beliefs.append(
                output.belief[0].detach().to("cpu", dtype=torch.float64).numpy()
            )
            targets.append(
                np.asarray(
                    [
                        pb21n.transition_hazard(t.event_targets)
                        for t in transition.counterfactual_targets
                    ],
                    dtype=np.float64,
                )
            )
            applied.append(int(applied_t.to("cpu").item()))
            # 1-step prior: action applied at tick-1 (prev.applied_control).
            p1 = control_action_class(prev.applied_control)
            prior1.append(p1)
            # 2-step prior: action applied at tick-2.
            prev2 = None if tick < 2 else sequence.transitions[tick - 2]
            p2 = control_action_class(prev2.applied_control)
            prior2.append(p2)
            base_raw.append(
                development._canonicalize_table(
                    table.raw_hazard_logits, table.action_ids
                )
            )
    if was_training:
        model.train()
    if episode_index != CAL_EPISODES:
        raise RuntimeError(f"episode count drifted: {episode_index}")

    b = np.asarray(beliefs, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    a = np.asarray(applied, dtype=np.int64)
    p1 = np.asarray(prior1, dtype=np.int64)
    p2 = np.asarray(prior2, dtype=np.int64)
    base = np.concatenate(base_raw, axis=0).astype(np.float64)
    fold = np.repeat(np.arange(2, dtype=np.int64), FOLD_ROWS)
    if b.shape != (TOTAL_ROOTS, WIDTH) or base.shape != (TOTAL_ROOTS, A):
        raise RuntimeError(f"evidence geometry drifted: {b.shape} {base.shape}")
    return {
        "beliefs": b, "targets": t, "applied": a, "prior1": p1,
        "prior2": p2, "base_raw": base, "fold_index": fold,
        "rows": TOTAL_ROOTS,
    }


# ---------------------------------------------------------------------------
# 2-step hazard path (4-channel): cat([state, action, prior1, prior2]).
# ---------------------------------------------------------------------------
class WindowHazardPath(nn.Module):
    """Standalone stop-gradient hazard path with n_prior lagged-action
    channels (0, 1, or 2). Zero-extended outcome trunk so every new channel
    contributes exactly zero at init (the arm computes the 2H base function
    until trained)."""

    def __init__(self, *, n_prior: int) -> None:
        super().__init__()
        self.n_prior = int(n_prior)
        self.state_trunk = nn.Sequential(nn.Linear(WIDTH, HIDDEN), nn.ReLU())
        self.action_embedding = nn.Embedding(A, HIDDEN)
        in_dim = HIDDEN * (2 + n_prior)
        self.outcome_trunk = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU()
        )
        self.head = nn.Linear(HIDDEN, 1)
        if n_prior >= 1:
            self.prior1_embedding = nn.Embedding(A, HIDDEN)
        else:
            self.prior1_embedding = None
        if n_prior >= 2:
            self.prior2_embedding = nn.Embedding(A, HIDDEN)
        else:
            self.prior2_embedding = None

    def forward(
        self, belief: Tensor, ids: Tensor,
        prior1_ids: Tensor | None, prior2_ids: Tensor | None,
    ) -> Tensor:
        state = F.layer_norm(
            self.state_trunk(belief), (self.state_trunk[0].out_features,)
        )
        state = state.unsqueeze(1).expand(-1, A, -1)
        action = F.layer_norm(
            self.action_embedding(ids), (self.action_embedding.embedding_dim,)
        )
        parts = [state, action]
        if self.n_prior >= 1:
            p1 = F.layer_norm(
                self.prior1_embedding(prior1_ids),
                (self.prior1_embedding.embedding_dim,),
            )
            parts.append(p1.unsqueeze(1).expand(-1, A, -1))
        if self.n_prior >= 2:
            p2 = F.layer_norm(
                self.prior2_embedding(prior2_ids),
                (self.prior2_embedding.embedding_dim,),
            )
            parts.append(p2.unsqueeze(1).expand(-1, A, -1))
        x = torch.cat(parts, dim=-1)
        return self.head(self.outcome_trunk(x)).squeeze(-1)


_PARENT_MAP = (
    ("state_trunk.0.weight", "hazard_state_trunk.0.weight"),
    ("state_trunk.0.bias", "hazard_state_trunk.0.bias"),
    ("action_embedding.weight", "hazard_action_embedding.weight"),
    ("head.weight", "hazard_head.weight"),
    ("head.bias", "hazard_head.bias"),
)


def _param(module, dotted):
    parts = dotted.split(".")
    target = module
    for part in parts[:-1]:
        target = getattr(target, part)
    return getattr(target, parts[-1])


def graft_window(parent_outcome, n_prior, embed_seed) -> WindowHazardPath:
    """Zero-extended graft: first 2H trunk rows = parent 2H trunk, remaining
    n_prior*H rows = 0. Prior embeddings N(0,1) with a fixed generator."""
    m = WindowHazardPath(n_prior=n_prior)
    with torch.no_grad():
        for dst, src in _PARENT_MAP:
            _param(m, dst).copy_(_param(parent_outcome, src))
        parent_trunk = parent_outcome.hazard_outcome_trunk[0]
        extended = torch.zeros_like(m.outcome_trunk[0].weight)
        extended[:, : parent_trunk.weight.shape[1]] = parent_trunk.weight
        m.outcome_trunk[0].weight.copy_(extended)
        m.outcome_trunk[0].bias.copy_(parent_trunk.bias)
        if n_prior >= 1:
            g = torch.Generator(device="cpu")
            g.manual_seed(embed_seed)
            m.prior1_embedding.weight.copy_(
                torch.empty_like(m.prior1_embedding.weight)
                .normal_(mean=0.0, std=1.0, generator=g)
            )
        if n_prior >= 2:
            g = torch.Generator(device="cpu")
            g.manual_seed(embed_seed + 7)
            m.prior2_embedding.weight.copy_(
                torch.empty_like(m.prior2_embedding.weight)
                .normal_(mean=0.0, std=1.0, generator=g)
            )
    return m


# ---------------------------------------------------------------------------
# Deterministic OOF fold retrain (mirror of pb21n.train_hazard_fold, but for
# WindowHazardPath with up to two prior channels and a per-arm seed offset).
# ---------------------------------------------------------------------------
def train_fold(
    module: WindowHazardPath,
    ev: dict[str, np.ndarray],
    fold: int,
    arm_seed_offset: int,
) -> None:
    rows = ev["fold_index"] != fold
    n = int(rows.sum())
    belief = torch.from_numpy(ev["beliefs"][rows]).to(dtype=torch.float32)
    target = torch.from_numpy(ev["targets"][rows]).to(dtype=torch.float32)
    p1 = None
    p2 = None
    if module.n_prior >= 1:
        p1 = torch.from_numpy(ev["prior1"][rows]).to(torch.long)
    if module.n_prior >= 2:
        p2 = torch.from_numpy(ev["prior2"][rows]).to(torch.long)
    ids = torch.arange(A).unsqueeze(0).expand(n, -1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(TRIAL_SEED + 100 * fold + arm_seed_offset)
    module.train()
    optimizer = torch.optim.AdamW(
        [p for p in module.parameters()],
        lr=pb21n.REFINEMENT_LEARNING_RATE,
        weight_decay=pb21n.REFINEMENT_WEIGHT_DECAY,
    )
    expected_steps_per_pass = int(np.ceil(n / pb21n.REFINEMENT_ROOT_BATCH_SIZE))
    total = 0
    for _ in range(pb21n.REFINEMENT_PASSES):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, pb21n.REFINEMENT_ROOT_BATCH_SIZE):
            idx = perm[start:start + pb21n.REFINEMENT_ROOT_BATCH_SIZE]
            logits = module(
                belief[idx].detach(), ids[idx],
                None if p1 is None else p1[idx],
                None if p2 is None else p2[idx],
            )
            loss = F.binary_cross_entropy_with_logits(logits, target[idx])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite hazard loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(nn.utils.clip_grad_norm_(
                [p for p in module.parameters()], pb21n.REFINEMENT_CLIP_NORM
            ))
            if not np.isfinite(gn):
                raise FloatingPointError("non-finite gradient")
            optimizer.step()
            total += 1
    module.eval()
    if total != expected_steps_per_pass * pb21n.REFINEMENT_PASSES:
        raise RuntimeError("optimizer-step count drifted")


def crossfit_arm(parent_outcome, ev, n_prior, arm_seed_offset):
    """Return OOF raw logit table [2, TOTAL_ROOTS, A] for one arm."""
    oof = np.zeros((2, TOTAL_ROOTS, A), dtype=np.float64)
    for fold in range(2):
        module = graft_window(parent_outcome, n_prior,
                              embed_seed=TRIAL_SEED + 500 + 17 * n_prior)
        train_fold(module, ev, fold, arm_seed_offset)
        eval_rows = ev["fold_index"] == fold
        n_ev = int(eval_rows.sum())
        ids = torch.arange(A).unsqueeze(0).expand(n_ev, -1)
        p1 = None if n_prior < 1 else torch.from_numpy(
            ev["prior1"][eval_rows]).to(torch.long)
        p2 = None if n_prior < 2 else torch.from_numpy(
            ev["prior2"][eval_rows]).to(torch.long)
        logits = module(
            torch.from_numpy(ev["beliefs"][eval_rows]).to(torch.float32)
            .detach(), ids, p1, p2,
        )
        oof[fold][eval_rows] = logits.detach().to(
            "cpu", dtype=torch.float64).numpy()
    return oof


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def factual_select(tab, ev):
    """Stack the two OOF slots' eval rows; per-row applied-action column."""
    scores = []
    ys = []
    for f in range(2):
        rows = ev["fold_index"] == f
        s = tab[f][rows]
        t = ev["targets"][rows]
        a = ev["applied"][rows]
        idx = a
        scores.append(s[np.arange(rows.sum()), idx])
        ys.append(t[np.arange(rows.sum()), idx])
    return np.concatenate(scores), np.concatenate(ys)


def rank_auc(y, s):
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1.0]
    neg = s[y == 0.0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="stable")
    vals = combined[order]
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    i = 0
    while i < vals.size:
        j = i
        while j + 1 < vals.size and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rs = ranks[:pos.size].sum()
    return float((rs - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def in_sample_pav_ece(tab, ev):
    """Per-fold in-sample per-action PAV factual ECE (optimistic ceiling)."""
    from irene_brain.evaluation.v21_qualification_metrics import (
        binary_probability_metrics as bpm,
    )
    out = []
    for f in range(2):
        rows = ev["fold_index"] == f
        per = []
        for a in range(A):
            fa = ev["applied"][rows] == a
            x = np.ascontiguousarray(tab[f][rows][fa, a])
            t = np.ascontiguousarray(ev["targets"][rows][fa, a])
            nv, np_, _ = pb21o.pav_fit(x, t)
            p_in = pb21o.pav_evaluate(x, nv, np_)
            per.append(bpm(t, p_in, ece_bins=10).as_dict()["ece_equal_mass"])
        out.append(per)
    return out


def main() -> None:
    started = time.time()
    print("=" * 74)
    print("PB21Q-SIG — 2-step prior-action window SIGNAL probe (READ-ONLY)")
    print("partition: PB21O-CAL (seed_offset 167,774,720; 256 ep x 12 roots)")
    print("=" * 74)

    model, _result, _prov = pb21n.load_parent()
    parent_outcome = model.world_model.outcome_model
    print(f"[i] parent loaded; state digest = "
          f"{pb21n._state_dict_sha256(model.state_dict())[:16]}...")

    ev = collect_2step_evidence(model)
    print(f"[i] evidence collected: {ev['rows']} roots, "
          f"burn-in {pb21n.CAL_BURN_IN}, fold rows {FOLD_ROWS}")

    arms = {
        "base(2H)": (0, 0),
        "1-step(3H)": (1, 1),
        "2-step(4H)": (2, 2),
    }
    oof = {}
    for name, (n_prior, seed_off) in arms.items():
        t0 = time.time()
        oof[name] = crossfit_arm(parent_outcome, ev, n_prior, seed_off)
        print(f"[i] {name} OOF cross-fit done in {time.time()-t0:.1f}s")

    print("\n--- Factual aggregate AUC (rank ceiling; binding constraint) ---")
    for name in arms:
        s_all, y_all = factual_select(oof[name], ev)
        agg = rank_auc(y_all, s_all)
        per = []
        for a in range(A):
            m = ev["applied"] == a
            per.append(rank_auc(y_all[m], s_all[m]))
        print(f"  {name:12s} agg AUC={agg:.4f}  "
              + "  ".join(f"a{i}={v:.3f}" for i, v in enumerate(per)))

    print("\n--- In-sample PAV factual ECE ceiling (per fold, per action) ---")
    for name in arms:
        f0 = in_sample_pav_ece(oof[name], ev)[0]
        f1 = in_sample_pav_ece(oof[name], ev)[1]
        print(f"  {name:12s} fold0 " +
              "  ".join(f"{v:.4f}" for v in f0) +
              "   fold1 " + "  ".join(f"{v:.4f}" for v in f1))

    print(f"\n[wall] {time.time()-started:.1f}s total (READ-ONLY, no publish)")
    print("GUIDE: 2-step warrants a PB21Q trial only if its factual aggregate")
    print("AUC clears the 1-step ceiling (~0.708) by > cross-fit noise AND")
    print("in-sample PAV ceiling improves; otherwise the window is refuted")
    print("at probe level and no trial is spent.")


if __name__ == "__main__":
    main()
