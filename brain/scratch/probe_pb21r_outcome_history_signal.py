"""PB21R-SIG — outcome-history hazard input SIGNAL probe (READ-ONLY, no
publish).

Motivation (L9b diagnostic, probe_outcome_history_signal.py): the factual
hazard target is NOT exchangeable over ticks — a hazard at t-1..t-4 lifts
P(hazard at t) by ~1.4-1.5x (0.19-0.21 vs 0.1315 marginal), consistently
across lags 1-4. This motivates a representation-level single change
ORTHOGONAL to the L9-refuted prior-action window-length axis: condition the
hazard path on recent applied-trajectory hazard events (outcome-history
features). Past information only (ticks t-4..t-1), consistent with the PB21N
prior-action precedent; no current-tick leakage.

Probe (read-only, no publish, no frozen-state change): on the PB21O-CAL
partition (seed_offset 167,774,720; 256 ep x 12 roots = 3072; 2 folds of
1536; burn-in 4), retrain two zero-extended hazard arms per OOF fold
(deterministic, CPU, single-thread, CUDA hidden):
  base(2H)   : cat([state, action])                       -> trunk(2H)
  hist(2H+1) : cat([state, action, hist_frac])            -> trunk(2H+1)
where hist_frac = (# applied-trajectory hazards in the last K=4 ticks,
excluding the current tick, within the episode) / 4, in [0, 1].

Report factual aggregate + per-action AUC (the L8 binding constraint) and
the in-sample PAV ECE ceiling per arm.

Single-change discipline: the candidate (outcome history) is tested
ALONE against the base control — NOT stacked on the 1-step action channel
(that combination is a "combined change" forbidden by the L1/L7 rules).
The base arm reproduces L9's 0.6706 exactly (same parent, partition,
schedule) as a harness self-validation.

The outcome-history arm warrants a PB21R trial only if its factual
aggregate AUC clears the base (0.6706) toward/above the 1-step ceiling
(0.7060) by more than cross-fit noise; otherwise that direction is refuted
at probe level and no trial is spent.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

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

run_provenance.apply_deterministic_mode()
torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)

WIDTH = pb21n.WIDTH
HIDDEN = pb21n.HIDDEN
A = pb21n.ACTION_COUNT
TRIAL_SEED = pb21n.TRIAL_SEED
TOTAL_ROOTS = pb21n.TOTAL_ROOTS
FOLD_ROWS = pb21n.FOLD_ROWS
CAL_EPISODES = pb21n.CAL_EPISODES
ROOTS_PER_EPISODE = pb21n.ROOTS_PER_EPISODE
HIST_K = 4  # recent-hazard window (lags 1-4 were the consistent-signal band)


# ---------------------------------------------------------------------------
# Evidence collection: add applied-trajectory hazard series + hist_frac.
# ---------------------------------------------------------------------------
def collect_hist_evidence(model) -> dict[str, np.ndarray]:
    contract = pb21o._pb21o_cal_contract()
    dev_contract = pb21n._to_dev_contract(contract)
    source = development.PartitionSource(dev_contract)
    device = torch.device("cpu")
    was_training = model.training
    model.eval()

    beliefs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    applied: list[int] = []
    hist: list[float] = []
    base_raw: list[np.ndarray] = []
    episode_index = 0

    for batch in source.iter_all_action_batches(epoch=0, batch_size=1):
        episode_index += 1
        sequence = batch.sequences[0]
        state = model.init_state(batch.batch_size, device)
        # Applied-trajectory hazard series for ALL ticks (0..seq_len-1),
        # needed to build the recent-hazard history for post-burn-in rows.
        tick_hazard: list[float] = []
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
            # Applied-trajectory hazard at this tick (all ticks, not just
            # post-burn-in): the factual column of the all-action table.
            ah = float(pb21n.transition_hazard(
                transition.counterfactual_targets[
                    int(applied_t.to("cpu").item())
                ].event_targets
            ))
            tick_hazard.append(ah)
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
            # hist_frac: hazards in ticks tick-HIST_K..tick-1 (excluding
            # current), within the episode, normalized by HIST_K.
            lo = tick - HIST_K
            window = tick_hazard[max(0, lo):tick]
            hist.append(float(sum(window[-HIST_K:])) / float(HIST_K))
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
    hv = np.asarray(hist, dtype=np.float64)
    base = np.concatenate(base_raw, axis=0).astype(np.float64)
    fold = np.repeat(np.arange(2, dtype=np.int64), FOLD_ROWS)
    if b.shape != (TOTAL_ROOTS, WIDTH) or base.shape != (TOTAL_ROOTS, A):
        raise RuntimeError(f"evidence geometry drifted: {b.shape} {base.shape}")
    return {"beliefs": b, "targets": t, "applied": a, "hist": hv,
            "base_raw": base, "fold_index": fold, "rows": TOTAL_ROOTS}


# ---------------------------------------------------------------------------
# History hazard path (2H + 1-dim scalar channel).
# ---------------------------------------------------------------------------
class HistHazardPath(nn.Module):
    """cat([state(H), action(H), hist_frac(1)]) -> trunk(2H+1) -> head.
    Zero-extended: the hist weight column is 0 at init, so the arm computes
    the 2H base function until trained."""

    def __init__(self, *, with_hist: bool) -> None:
        super().__init__()
        self.with_hist = bool(with_hist)
        self.state_trunk = nn.Sequential(nn.Linear(WIDTH, HIDDEN), nn.ReLU())
        self.action_embedding = nn.Embedding(A, HIDDEN)
        in_dim = HIDDEN * 2 + (1 if with_hist else 0)
        self.outcome_trunk = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU()
        )
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, belief: Tensor, ids: Tensor,
                hist: Tensor | None) -> Tensor:
        state = F.layer_norm(
            self.state_trunk(belief), (self.state_trunk[0].out_features,)
        )
        state = state.unsqueeze(1).expand(-1, A, -1)
        action = F.layer_norm(
            self.action_embedding(ids), (self.action_embedding.embedding_dim,)
        )
        if self.with_hist:
            if hist is None:
                raise ValueError("hist arm requires hist_frac")
            h = hist.view(-1, 1, 1).expand(-1, A, -1)  # [B, A, 1], raw (no LN)
            x = torch.cat((state, action, h), dim=-1)
        else:
            x = torch.cat((state, action), dim=-1)
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


def graft_hist(parent_outcome, with_hist, embed_seed) -> HistHazardPath:
    m = HistHazardPath(with_hist=with_hist)
    with torch.no_grad():
        for dst, src in _PARENT_MAP:
            _param(m, dst).copy_(_param(parent_outcome, src))
        parent_trunk = parent_outcome.hazard_outcome_trunk[0]
        extended = torch.zeros_like(m.outcome_trunk[0].weight)
        extended[:, : parent_trunk.weight.shape[1]] = parent_trunk.weight
        # The hist column (last 1) stays 0 -> init-identical to base.
        m.outcome_trunk[0].weight.copy_(extended)
        m.outcome_trunk[0].bias.copy_(parent_trunk.bias)
    return m


def assert_init_identity(base, histm, ev):
    rows = np.where(ev["fold_index"] == 0)[0][:32]
    ids = torch.arange(A).unsqueeze(0).expand(len(rows), -1)
    bel = torch.from_numpy(ev["beliefs"][rows]).to(torch.float32).detach()
    h = torch.from_numpy(ev["hist"][rows]).to(torch.float32)
    with torch.no_grad():
        a = base(bel, ids, None)
        b = histm(bel, ids, h)
    delta = float((a - b).abs().max())
    print(f"[i] init-identity base vs hist arm: max|delta|={delta:.2e}")
    if not bool(torch.allclose(a, b, atol=1.0e-6, rtol=1.0e-5)):
        raise RuntimeError(f"hist graft not init-identical (max|d|={delta})")


def train_fold(module, ev, fold, arm_seed_offset):
    rows = ev["fold_index"] != fold
    n = int(rows.sum())
    belief = torch.from_numpy(ev["beliefs"][rows]).to(dtype=torch.float32)
    target = torch.from_numpy(ev["targets"][rows]).to(dtype=torch.float32)
    hv = (torch.from_numpy(ev["hist"][rows]).to(torch.float32)
          if module.with_hist else None)
    ids = torch.arange(A).unsqueeze(0).expand(n, -1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(TRIAL_SEED + 100 * fold + arm_seed_offset)
    module.train()
    optimizer = torch.optim.AdamW(
        [p for p in module.parameters()],
        lr=pb21n.REFINEMENT_LEARNING_RATE,
        weight_decay=pb21n.REFINEMENT_WEIGHT_DECAY,
    )
    expected = int(np.ceil(n / pb21n.REFINEMENT_ROOT_BATCH_SIZE))
    total = 0
    for _ in range(pb21n.REFINEMENT_PASSES):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, pb21n.REFINEMENT_ROOT_BATCH_SIZE):
            idx = perm[start:start + pb21n.REFINEMENT_ROOT_BATCH_SIZE]
            logits = module(belief[idx].detach(), ids[idx],
                            None if hv is None else hv[idx])
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
    if total != expected * pb21n.REFINEMENT_PASSES:
        raise RuntimeError("optimizer-step count drifted")


def crossfit_arm(parent_outcome, ev, with_hist, arm_seed_offset):
    oof = np.zeros((2, TOTAL_ROOTS, A), dtype=np.float64)
    for fold in range(2):
        module = graft_hist(parent_outcome, with_hist, arm_seed_offset)
        train_fold(module, ev, fold, arm_seed_offset)
        eval_rows = ev["fold_index"] == fold
        n_ev = int(eval_rows.sum())
        ids = torch.arange(A).unsqueeze(0).expand(n_ev, -1)
        hv = None
        if with_hist:
            hv = torch.from_numpy(ev["hist"][eval_rows]).to(torch.float32)
        logits = module(
            torch.from_numpy(ev["beliefs"][eval_rows]).to(torch.float32)
            .detach(), ids, hv,
        )
        oof[fold][eval_rows] = logits.detach().to(
            "cpu", dtype=torch.float64).numpy()
    return oof


def factual_select(tab, ev):
    scores, ys = [], []
    for f in range(2):
        rows = ev["fold_index"] == f
        s = tab[f][rows]
        a = ev["applied"][rows]
        scores.append(s[np.arange(rows.sum()), a])
        ys.append(ev["targets"][rows][np.arange(rows.sum()), a])
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
    print("PB21R-SIG — outcome-history hazard input SIGNAL probe (READ-ONLY)")
    print("partition: PB21O-CAL (seed_offset 167,774,720; 256 ep x 12 roots)")
    print(f"history window: K={HIST_K} applied-trajectory ticks (excl. current)")
    print("=" * 74)

    model, _result, _prov = pb21n.load_parent()
    parent_outcome = model.world_model.outcome_model
    print(f"[i] parent loaded; state digest = "
          f"{pb21n._state_dict_sha256(model.state_dict())[:16]}...")

    ev = collect_hist_evidence(model)
    print(f"[i] evidence collected: {ev['rows']} roots; "
          f"hist_frac mean={ev['hist'].mean():.4f}, "
          f"frac(rows hist>0)={(ev['hist'] > 0).mean():.4f}")

    base_m = graft_hist(parent_outcome, False, 0)
    hist_m = graft_hist(parent_outcome, True, 0)
    assert_init_identity(base_m, hist_m, ev)

    arms = {"base(2H)": (False, 0), "hist(2H+1)": (True, 1)}
    oof = {}
    for name, (with_hist, seed_off) in arms.items():
        t0 = time.time()
        oof[name] = crossfit_arm(parent_outcome, ev, with_hist, seed_off)
        print(f"[i] {name} OOF cross-fit done in {time.time()-t0:.1f}s")

    print("\n--- Factual aggregate AUC (rank ceiling; binding constraint) ---")
    for name in arms:
        s_all, y_all = factual_select(oof[name], ev)
        agg = rank_auc(y_all, s_all)
        per = [rank_auc(y_all[ev["applied"] == a], s_all[ev["applied"] == a])
               for a in range(A)]
        print(f"  {name:10s} agg AUC={agg:.4f}  "
              + "  ".join(f"a{i}={v:.3f}" for i, v in enumerate(per)))

    print("\n--- In-sample PAV factual ECE ceiling (per fold, per action) ---")
    for name in arms:
        f0 = in_sample_pav_ece(oof[name], ev)[0]
        f1 = in_sample_pav_ece(oof[name], ev)[1]
        print(f"  {name:10s} fold0 " +
              "  ".join(f"{v:.4f}" for v in f0) +
              "   fold1 " + "  ".join(f"{v:.4f}" for v in f1))

    print(f"\n[wall] {time.time()-started:.1f}s total (READ-ONLY, no publish)")
    print("GUIDE: hist arm warrants a PB21R trial only if its factual aggregate")
    print("AUC clears base (0.6706) toward/above the 1-step ceiling (0.7060)")
    print("by > cross-fit noise; else the outcome-history direction is")
    print("refuted at probe level and no trial is spent.")


if __name__ == "__main__":
    main()
