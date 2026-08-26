"""PB21S-SIG / L11 — recurrent applied-trajectory hazard-history state
SIGNAL probe (READ-ONLY, no publish).

Motivation (L10 measured fact, probe_outcome_history_signal.py): the factual
hazard target has genuine MULTI-TICK autocorrelation — P(h_t=1 | h_{t-k}=1)
lift 1.46x (k=1) up to 1.76x (k=4); any-hazard-in-last-K lift 1.36-1.53x
(K=1-4). A hazard on the applied trajectory tends to PERSIST for a few
ticks. The L9 (2-step window) and L10 (last-4 scalar count) probes are
feedforward: each can only read a single snapshot of that history. They
cannot exploit the SEQUENCE/persistence structure.

Mechanism under test (a NEW mechanism class, orthogonal to L9/L10): a small
RECURRENT hazard-history state — a 1-layer GRU, hidden 128, reset to zero at
every episode start — that accumulates the applied trajectory's (action,
hazard-event) sequence over the episode. Its final hidden state is a new
128-dim channel on the hazard path (cat([state(120), action(120),
hist_state(128)])), zero-extended so at init the arm computes exactly the
base function. The GRU sees ONLY past info: at recorded tick t its input is
(applied_control[t], hazard_event[t]), both available before the hazard of
tick t+1 is predicted; the GRU state at tick t is built from ticks <= t.
Causal, no current-tick leakage (the hazard logit for tick t conditions on
the history state built through tick t's own applied trajectory, which is
information the agent already committed to — the same precedent as the
PB21N prior-action channel).

Single-change discipline: the recurrent history channel is tested ALONE
against the base control. NOT stacked on the 1-step action channel (that
combination is a "combined change", forbidden by the L1/L7 rules).

Probe (read-only, no publish, no frozen-state change): PB21O-CAL partition
(seed_offset 167,774,720; 256 ep x 12 roots = 3072; 2 folds of 1536; burn-in
4). Two arms retrained per OOF fold (deterministic, CPU, single-thread,
CUDA hidden):
  base(2H)  : cat([state, action]) -> trunk(2H) -> head
  rec(2H+128): cat([state, action, hist_state]) -> trunk(2H+128) -> head
Both share the parent init (zero-extended); the GRU + applied-action
embedding are fresh N(0,1) from fixed generators.

Decision guide: the rec arm warrants a fresh PB21S trial ONLY if its
factual aggregate AUC clears the base (0.6706) toward/above the 1-step
action ceiling (0.7060) by more than cross-fit noise. Otherwise the
recurrent-history direction is refuted at probe level and no trial is
spent.
"""
from __future__ import annotations

import os
import sys
import time
import zlib
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
SEQ_LEN = 16
BURN_IN = 4
HIST_DIM = 128          # GRU hidden
GRU_INPUT_EMB = 32      # applied-action embedding dim into the GRU
EPS = 24                # episodes per training batch (root batch 24 = 2 eps)
FOLD_EPS = CAL_EPISODES // 2   # 128 episodes per fold

# Fresh disjoint partition for the TRIAL (contiguous after PB21O-CAL,
# which ends at 167,774,720 + 256 = 167,774,976). The discovery probes
# ran on PB21O-CAL (167,774,720); a publish-once trial must not be tuned
# against its own test partition, so the trial partition is fresh.
PB21S_CAL_SEED_OFFSET = 167_774_976


def _pb21s_cal_contract():
    """PB21S-CAL contract: identical to PB21N-CAL but seed_offset
    PB21S_CAL_SEED_OFFSET (fresh, disjoint from PB21O-CAL)."""
    from dataclasses import replace
    base = pb21n.cal_contract()
    return pb21n.PB21NContract(
        "PB21S-CAL",
        replace(base.dataset_config, seed_offset=PB21S_CAL_SEED_OFFSET),
    )


# ---------------------------------------------------------------------------
# Evidence: per-episode arrays (beliefs, targets, applied, hazard series,
# applied-action series, base raw logits) for the PB21O-CAL partition.
# ---------------------------------------------------------------------------
def collect_ep_evidence(model) -> dict:
    contract = _pb21s_cal_contract()
    dev_contract = pb21n._to_dev_contract(contract)
    source = development.PartitionSource(dev_contract)
    device = torch.device("cpu")
    was_training = model.training
    model.eval()

    beliefs = []      # list of [T, W] float64 per episode
    targets = []      # [T, 5]
    applied = []      # [T] int64
    applied_ctrl = [] # [SEQ_LEN] int64 (full tick series, for the GRU input)
    hazard_series = []# [SEQ_LEN] float64 (applied-trajectory hazard, all ticks)
    base_raw = []     # [T, 5]

    for batch in source.iter_all_action_batches(epoch=0, batch_size=1):
        sequence = batch.sequences[0]
        state = model.init_state(batch.batch_size, device)
        ep_applied_ctrl = []
        ep_hazard = []
        ep_beliefs = []
        ep_targets = []
        ep_applied = []
        ep_base = []
        for tick in range(SEQ_LEN):
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
                pixels, state,
                prev_action=previous,
                actual_reward=prior_reward,
                actual_hazard=prior_hazard,
                world_model_action=applied_t,
                intervention_pe="normal",
            )
            ah = int(applied_t.to("cpu").item())
            ep_applied_ctrl.append(ah)
            ep_hazard.append(float(pb21n.transition_hazard(
                transition.counterfactual_targets[ah].event_targets)))
            if tick < BURN_IN:
                continue
            table = output.outcome_table
            if table is None or table.raw_hazard_logits is None:
                raise RuntimeError("V2.1i requires raw all-action hazard logits")
            ep_beliefs.append(output.belief[0].detach().to(
                "cpu", dtype=torch.float64).numpy())
            ep_targets.append(np.asarray(
                [pb21n.transition_hazard(t.event_targets)
                 for t in transition.counterfactual_targets],
                dtype=np.float64))
            ep_applied.append(ah)
            ep_base.append(development._canonicalize_table(
                table.raw_hazard_logits, table.action_ids))
        beliefs.append(np.asarray(ep_beliefs, dtype=np.float64))
        targets.append(np.asarray(ep_targets, dtype=np.float64))
        applied.append(np.asarray(ep_applied, dtype=np.int64))
        applied_ctrl.append(np.asarray(ep_applied_ctrl, dtype=np.int64))
        hazard_series.append(np.asarray(ep_hazard, dtype=np.float64))
        base_raw.append(np.concatenate(ep_base, axis=0).astype(np.float64))
    if was_training:
        model.train()
    if len(beliefs) != CAL_EPISODES:
        raise RuntimeError(f"episode count drifted: {len(beliefs)}")
    b = np.concatenate([x for x in beliefs], axis=0)
    t = np.concatenate([x for x in targets], axis=0)
    a = np.concatenate([x for x in applied], axis=0)
    base = np.concatenate(base_raw, axis=0)
    fold = np.repeat(np.arange(2, dtype=np.int64), FOLD_ROWS)
    if b.shape != (TOTAL_ROOTS, WIDTH) or base.shape != (TOTAL_ROOTS, A):
        raise RuntimeError(f"geometry drifted: {b.shape} {base.shape}")
    return {"beliefs": b, "targets": t, "applied": a, "base_raw": base,
            "fold_index": fold,
            "applied_ctrl": np.stack(applied_ctrl),
            "hazard_series": np.stack(hazard_series),
            "rows": TOTAL_ROOTS}


# ---------------------------------------------------------------------------
# Hazard path with an optional recurrent-history channel.
# ---------------------------------------------------------------------------
class RecHazardPath(nn.Module):
    """cat([state(120), action(120), hist(128-or-none)]) -> trunk -> head."""

    def __init__(self, *, with_hist: bool) -> None:
        super().__init__()
        self.with_hist = bool(with_hist)
        self.state_trunk = nn.Sequential(nn.Linear(WIDTH, HIDDEN), nn.ReLU())
        self.action_embedding = nn.Embedding(A, HIDDEN)
        in_dim = HIDDEN * 2 + (HIST_DIM if with_hist else 0)
        self.outcome_trunk = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU()
        )
        self.head = nn.Linear(HIDDEN, 1)
        if with_hist:
            self.applied_emb = nn.Embedding(A, GRU_INPUT_EMB)
            self.gru = nn.GRUCell(GRU_INPUT_EMB + 1, HIST_DIM)

    def forward(self, belief, ids, hist_state=None):
        state = F.layer_norm(
            self.state_trunk(belief), (self.state_trunk[0].out_features,))
        state = state.unsqueeze(1).expand(-1, A, -1)
        action = F.layer_norm(
            self.action_embedding(ids),
            (self.action_embedding.embedding_dim,))
        if self.with_hist:
            if hist_state is None:
                raise ValueError("rec arm requires hist_state")
            h = hist_state.unsqueeze(1).expand(-1, A, -1)  # [B, A, 128]
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


def graft(parent_outcome, with_hist, seed_offset) -> RecHazardPath:
    m = RecHazardPath(with_hist=with_hist)
    with torch.no_grad():
        for dst, src in _PARENT_MAP:
            _param(m, dst).copy_(_param(parent_outcome, src))
        parent_trunk = parent_outcome.hazard_outcome_trunk[0]
        extended = torch.zeros_like(m.outcome_trunk[0].weight)
        extended[:, : parent_trunk.weight.shape[1]] = parent_trunk.weight
        m.outcome_trunk[0].weight.copy_(extended)
        m.outcome_trunk[0].bias.copy_(parent_trunk.bias)
        if with_hist:
            g = torch.Generator(device="cpu")
            g.manual_seed(TRIAL_SEED + 777 + seed_offset)
            m.applied_emb.weight.copy_(torch.empty_like(
                m.applied_emb.weight).normal_(0.0, 1.0, generator=g))
            for name, p in m.gru.named_parameters():
                # STABLE per-process hash (zlib.crc32): the builtin
                # hash(str) is salted per process (PYTHONHASHSEED) and
                # would make the GRU init non-deterministic.
                stable = zlib.crc32(name.encode("utf-8")) % 997
                pg = torch.Generator(device="cpu")
                pg.manual_seed(TRIAL_SEED + 888 + seed_offset + stable)
                p.normal_(0.0, 1.0, generator=pg)
    return m


def init_identity_delta(parent_outcome, ev) -> float:
    """At episode start the GRU state is 0 and the hist trunk columns are 0,
    so the rec arm must equal the base arm EXACTLY on tick-0 beliefs."""
    base = graft(parent_outcome, False, 0)
    rec = graft(parent_outcome, True, 0)
    rows = np.where(ev["fold_index"] == 0)[0][:16]
    ids = torch.arange(A).unsqueeze(0).expand(len(rows), -1)
    bel = torch.from_numpy(ev["beliefs"][rows]).to(torch.float32).detach()
    with torch.no_grad():
        a = base(bel, ids, None)
        b = rec(bel, ids, torch.zeros(len(rows), HIST_DIM,
                                      dtype=torch.float32))
    return float((a - b).abs().max())


def gru_states(module: RecHazardPath, ev: dict, fold: int,
               episodes: np.ndarray) -> np.ndarray:
    """Deterministic per-episode GRU roll. Returns [E, SEQ_LEN, HIST_DIM]
    float64: hs_all[:, tau] = the state AFTER processing tick tau (tau =
    0..SEQ_LEN-1), input at tau = (action-emb[applied_ctrl[tau]],
    hazard_series[tau]). Pre-burn-in ticks are included (real history)."""
    n_ep = len(episodes)
    ac = torch.from_numpy(ev["applied_ctrl"][episodes]).to(torch.long)
    hz = torch.from_numpy(ev["hazard_series"][episodes]).to(torch.float32)
    emb = module.applied_emb(ac)                       # [E, 16, EMB]
    x = torch.cat([emb, hz.unsqueeze(-1)], dim=-1)     # [E, 16, EMB+1]
    h = torch.zeros(n_ep, HIST_DIM, dtype=torch.float32)
    out = torch.empty(n_ep, SEQ_LEN, HIST_DIM, dtype=torch.float32)
    for t in range(SEQ_LEN):
        h = module.gru(x[:, t], h)
        out[:, t] = h
    return out.detach().to("cpu", dtype=torch.float64).numpy()


def logits_for_rows(module, ev, fold, rows, hist_per_root):
    """Logits for the given ROOT indices. ``hist_per_root`` is
    [len(rows), HIST_DIM] (float32) for the rec arm, else None."""
    bel = torch.from_numpy(ev["beliefs"][rows]).to(torch.float32).detach()
    ids = torch.arange(A).unsqueeze(0).expand(len(rows), -1)
    hs = None
    if module.with_hist:
        if hist_per_root is None:
            raise ValueError("rec arm requires per-root hist state")
        hs = torch.from_numpy(hist_per_root).to(torch.float32)
    return module(bel, ids, hs)


def _hist_per_root(module, ev, fold, rows):
    """Causal per-root GRU history state: for a root at tick t the state is
    the one built through tick t-1 (the agent's committed past), NEVER
    including tick t's own hazard event (that is the label being
    predicted). No current-tick leakage."""
    ep_of_row = rows // ROOTS_PER_EPISODE
    tick = BURN_IN + rows % ROOTS_PER_EPISODE          # actual tick t
    assert int(tick.min()) >= 1                        # t-1 >= 0 always
    uniq = np.unique(ep_of_row)
    hs_all = gru_states(module, ev, fold, uniq)        # [E, 16, HIST_DIM]
    idx = np.searchsorted(uniq, ep_of_row)
    return np.stack([hs_all[idx[i], tick[i] - 1] for i in range(len(rows))])


def train_fold(module, ev, fold, seed_offset):
    rows = np.where(ev["fold_index"] != fold)[0]     # 1536 root indices
    n = int(rows.sum() if rows.ndim == 0 else len(rows))
    rows = np.asarray(rows, dtype=np.int64)
    ep_of_row = rows // ROOTS_PER_EPISODE
    assert int(np.unique(ep_of_row).shape[0]) == FOLD_EPS
    generator = torch.Generator(device="cpu")
    generator.manual_seed(TRIAL_SEED + 100 * fold + 3000 + seed_offset)
    module.train()
    optimizer = torch.optim.AdamW(
        [p for p in module.parameters()],
        lr=pb21n.REFINEMENT_LEARNING_RATE,
        weight_decay=pb21n.REFINEMENT_WEIGHT_DECAY,
    )
    target = torch.from_numpy(ev["targets"][rows]).to(torch.float32)
    root_batch = pb21n.REFINEMENT_ROOT_BATCH_SIZE   # 24 roots (= 2 episodes)
    steps = 0
    expected = int(np.ceil(n / root_batch))
    for _ in range(pb21n.REFINEMENT_PASSES):
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, root_batch):
            idx = perm[start:start + root_batch].numpy()
            rows_now = rows[idx]
            if module.with_hist:
                hs = _hist_per_root(module, ev, fold, rows_now)
            else:
                hs = None
            logits = logits_for_rows(module, ev, fold, rows_now, hs)
            loss = F.binary_cross_entropy_with_logits(
                logits, target[idx])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(nn.utils.clip_grad_norm_(
                [p for p in module.parameters()],
                pb21n.REFINEMENT_CLIP_NORM))
            if not np.isfinite(gn):
                raise FloatingPointError("non-finite gradient")
            optimizer.step()
            steps += 1
    module.eval()
    if steps != expected * pb21n.REFINEMENT_PASSES:
        raise RuntimeError(f"step count drifted: {steps}")


def crossfit_arm(parent_outcome, ev, with_hist, seed_offset):
    oof = np.zeros((2, TOTAL_ROOTS, A), dtype=np.float64)
    for fold in range(2):
        module = graft(parent_outcome, with_hist, seed_offset)
        train_fold(module, ev, fold, seed_offset)
        rows = np.where(ev["fold_index"] == fold)[0]
        rows = np.asarray(rows, dtype=np.int64)
        if with_hist:
            hs = _hist_per_root(module, ev, fold, rows)
        else:
            hs = None
        logits = logits_for_rows(module, ev, fold, rows, hs)
        oof[fold][rows] = logits.detach().to(
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
    print("PB21S-SIG / L11 — recurrent hazard-history state SIGNAL probe")
    print("partition: PB21O-CAL (seed_offset 167,774,720; 256 ep x 12 roots)")
    print(f"history: GRU(hidden={HIST_DIM}) over applied "
          f"(action-emb {GRU_INPUT_EMB} + hazard-event 1) sequence")
    print("=" * 74)

    model, _result, _prov = pb21n.load_parent()
    parent_outcome = model.world_model.outcome_model
    print(f"[i] parent loaded; state digest = "
          f"{pb21n._state_dict_sha256(model.state_dict())[:16]}...")

    ev = collect_ep_evidence(model)
    print(f"[i] evidence: {ev['rows']} roots; "
          f"hazard series mean={ev['hazard_series'].mean():.4f}")

    delta = init_identity_delta(parent_outcome, ev)
    print(f"[i] init-identity base vs rec (tick-0, zero GRU state): "
          f"max|delta|={delta:.2e}")
    if not delta <= 1.0e-6:
        raise RuntimeError(f"rec graft not init-identical: {delta}")

    arms = {"base(2H)": (False, 0), "rec(2H+128)": (True, 0)}
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
    print("GUIDE: rec arm warrants a fresh PB21S trial only if its factual")
    print("aggregate AUC clears base (0.6706) toward/above the 1-step ceiling")
    print("(0.7060) by > cross-fit noise; else the recurrent-history")
    print("direction is refuted at probe level and no trial is spent.")


if __name__ == "__main__":
    main()
