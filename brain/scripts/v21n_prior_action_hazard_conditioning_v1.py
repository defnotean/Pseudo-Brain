"""PB21N — prior-action-conditioned hazard representation (single change).

Classification: fresh_hazard_prior_action_single_change (planned),
non-qualifying until a separate scaling/qualification prereg passes.

Source of the single change: the frozen Arm B nomination from
2026-08-25-pb21m-posthoc-mechanism-audit-v1.

Arms (all on the fresh PB21N-CAL partition; upstream belief frozen byte-exact):
  H_base     — frozen parent hazard path, raw logits, no retrain.
  H_retrain  — parent 2H hazard path, retrained on the OOF folds (control arm).
  H_prior    — 3H hazard path + prior-action embedding, retrained (candidate).

Gates G1..G6 are frozen in
brain/docs/preregistrations/2026-08-25-pb21n-prior-action-hazard-conditioning-v1.md.
PASS = G1 AND G2 AND G3 AND G4 AND G5 AND G6.
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

BRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN_ROOT / "scripts"), str(BRAIN_ROOT / "src")]

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import run_provenance  # noqa: E402
import v21i_development_runner as development  # noqa: E402
import v21i_strict_live_representation_probe_v1 as strict  # noqa: E402
import v21m_posthoc_mechanism_audit_v1 as audit  # noqa: E402
from irene_brain.data import DatasetSplit, MazeChaseDatasetConfig  # noqa: E402
from irene_brain.evaluation.v21_qualification_metrics import (  # noqa: E402
    binary_probability_metrics,
)
from irene_brain.v2 import CONFIG_B_PREDICTIVE, CoreV2Config  # noqa: E402
from irene_brain.v2.trajectory_objective import (  # noqa: E402
    control_action_class,
    transition_hazard,
)

run_provenance.apply_deterministic_mode()
torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)

# ---------------------------------------------------------------------------
# Frozen trial constants (prereg 2026-08-25-pb21n-prior-action-hazard-
# conditioning-v1)
# ---------------------------------------------------------------------------
MODE = "pb21n_prior_action_hazard_conditioning_v1"
RUN_ID = "2026-08-25-pb21n-prior-action-hazard-conditioning-v1"
CLASSIFICATION = "fresh_hazard_prior_action_single_change"
EVIDENCE_DIR = BRAIN_ROOT / "runs" / "pb21n-hazard-prior-action"

V3_RESULT_PATH = (
    BRAIN_ROOT / "runs" / "v21i-development" / "2026-08-24-seed42-v3.json"
)
V3_RESULT_SHA256 = (
    "3f91921b7ca63148addd3204cdb1e96b130f26240f49277a269ac8f079b13936"
)
PARENT_SHA256 = strict.EXACT_V3_PARENT_SHA256
PARENT_STATE_SHA256 = strict.EXACT_V3_PARENT_STATE_SHA256

# Fresh disjoint CAL partition (prereg §3): immediately after PB21M C2B.
CAL_SPLIT = DatasetSplit.TRAIN
CAL_SEED_OFFSET = 167_774_464
CAL_EPISODES = 256
CAL_BURN_IN = development.BURN_IN_STEPS  # 4
CAL_SEQ_LEN = 16
ROOTS_PER_EPISODE = CAL_SEQ_LEN - CAL_BURN_IN  # 12
TOTAL_ROOTS = CAL_EPISODES * ROOTS_PER_EPISODE  # 3072
FOLD_ROWS = TOTAL_ROOTS // 2  # 1536 = 128 episodes * 12 roots

# Gates (prereg §5).
FACTUAL_AGGREGATE_BIAS_LIMIT = 0.05
FACTUAL_AGGREGATE_ECE_LIMIT = 0.05
ALL_ACTION_CONTROL_WORSE_MARGIN = 0.01
FACTORIAL_AUC_DROP_LIMIT = 0.05
ONE_SIDED_ALPHA = 0.025
QUANTILE_METHOD = "linear"
BOOTSTRAP_RESAMPLES = 10_000
MIN_STRATUM_ROWS = 20
G5_SHUFFLE_PERMUTATIONS = 30
G5_SHUFFLE_SEED = 81_043  # distinct from the audit's 81042

TRIAL_SEED = 43
EVIDENCE_BATCH_SIZE = development.BATCH_SIZE

# Hazard-path training schedule: bound to the V2.1i refinement contract
# (prereg §4 — no new hyperparameters).
REFINEMENT_PASSES = development.HAZARD_REFINEMENT_PASSES
REFINEMENT_ROOT_BATCH_SIZE = development.HAZARD_REFINEMENT_ROOT_BATCH_SIZE
REFINEMENT_LEARNING_RATE = development.HAZARD_REFINEMENT_LEARNING_RATE
REFINEMENT_WEIGHT_DECAY = development.HAZARD_REFINEMENT_WEIGHT_DECAY
REFINEMENT_CLIP_NORM = development.HAZARD_REFINEMENT_CLIP_NORM

ACTION_COUNT = development.ACTION_COUNT
WIDTH = 120
HIDDEN = 120


def _common_dataset_config() -> dict[str, object]:
    return {
        "sequence_length": CAL_SEQ_LEN,
        "ghost_count": 5,
        "ghost_period": 1,
        "ghost_rule": "direct",
        "input_delay_ticks": 0,
        "sticky_direction": False,
        "episode_horizon": 0,
        "behavior_policy": "balanced_intervention_v1",
        "behavior_intervention_rate": 0.5,
        "counterfactual_targets": "all_actions_v1",
    }


@dataclass(frozen=True)
class PB21NContract:
    name: str
    dataset_config: MazeChaseDatasetConfig
    burn_in_steps: int = CAL_BURN_IN


def cal_contract() -> PB21NContract:
    return PB21NContract(
        "PB21N-CAL",
        MazeChaseDatasetConfig(
            split=CAL_SPLIT,
            seed_offset=CAL_SEED_OFFSET,
            sequence_count=CAL_EPISODES,
            **_common_dataset_config(),
        ),
    )


def _sha256_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    """House canonical state-digest (identical to the sealed parent's)."""
    return development._state_dict_sha256(state)


# ---------------------------------------------------------------------------
# Grafted hazard path (standalone; parent stays read-only)
# ---------------------------------------------------------------------------

class HazardPath(nn.Module):
    """Standalone stop-gradient hazard path mirroring the parent's
    dedicated_stopgrad_v1 sub-network.

    base  : cat([state, action]) -> trunk(2H) -> head
    prior : cat([state, action, prior_action]) -> trunk(3H) -> head

    The prior variant's trunk is zero-extended: its first 2H weight rows are
    the parent's exact rows and the last H rows are zero, so at
    initialization the prior variant computes EXACTLY the base function
    (the new channel contributes zero).
    """

    def __init__(self, *, prior: bool) -> None:
        super().__init__()
        self.prior_enabled = bool(prior)
        self.state_trunk = nn.Sequential(
            nn.Linear(WIDTH, HIDDEN), nn.ReLU(),
        )
        self.action_embedding = nn.Embedding(ACTION_COUNT, HIDDEN)
        in_dim = HIDDEN * (3 if prior else 2)
        self.outcome_trunk = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU(),
        )
        self.head = nn.Linear(HIDDEN, 1)
        if prior:
            self.prior_embedding = nn.Embedding(ACTION_COUNT, HIDDEN)
        else:
            self.prior_embedding = None

    def forward(
        self,
        belief: Tensor,
        ids: Tensor,
        prior_ids: Tensor | None,
    ) -> Tensor:
        """belief [B, W] (detached externally), ids [B, A], returns [B, A]."""
        state = F.layer_norm(
            self.state_trunk(belief), (self.state_trunk[0].out_features,)
        )
        state = state.unsqueeze(1).expand(-1, ACTION_COUNT, -1)
        action = F.layer_norm(
            self.action_embedding(ids), (self.action_embedding.embedding_dim,)
        )
        if self.prior_enabled:
            if prior_ids is None:
                raise ValueError("prior variant requires prior_ids")
            prior = F.layer_norm(
                self.prior_embedding(prior_ids),
                (self.prior_embedding.embedding_dim,),
            )
            prior = prior.unsqueeze(1).expand(-1, ACTION_COUNT, -1)
            x = torch.cat((state, action, prior), dim=-1)
        else:
            x = torch.cat((state, action), dim=-1)
        return self.head(self.outcome_trunk(x)).squeeze(-1)


_PARENT_HAZARD_TENSOR_MAP = (
    ("state_trunk.0.weight", "hazard_state_trunk.0.weight"),
    ("state_trunk.0.bias", "hazard_state_trunk.0.bias"),
    ("action_embedding.weight", "hazard_action_embedding.weight"),
    ("head.weight", "hazard_head.weight"),
    ("head.bias", "hazard_head.bias"),
)


def _param(module: nn.Module, dotted: str) -> Tensor:
    """Fetch a parameter/buffer by dotted submodule path."""
    parts = dotted.split(".")
    target = module
    for part in parts[:-1]:
        target = getattr(target, part)
    return getattr(target, parts[-1])


def graft_base(parent_outcome: nn.Module) -> HazardPath:
    base = HazardPath(prior=False)
    with torch.no_grad():
        for dst, src in _PARENT_HAZARD_TENSOR_MAP:
            _param(base, dst).copy_(_param(parent_outcome, src))
        base.outcome_trunk[0].weight.copy_(
            parent_outcome.hazard_outcome_trunk[0].weight
        )
        base.outcome_trunk[0].bias.copy_(
            parent_outcome.hazard_outcome_trunk[0].bias
        )
    return base


def graft_prior(parent_outcome: nn.Module) -> HazardPath:
    prior = HazardPath(prior=True)
    with torch.no_grad():
        for dst, src in _PARENT_HAZARD_TENSOR_MAP:
            _param(prior, dst).copy_(_param(parent_outcome, src))
        parent_trunk = parent_outcome.hazard_outcome_trunk[0]
        extended = torch.zeros_like(prior.outcome_trunk[0].weight)
        extended[:, : parent_trunk.weight.shape[1]] = parent_trunk.weight
        extended[:, parent_trunk.weight.shape[1]:] = 0.0
        prior.outcome_trunk[0].weight.copy_(extended)
        prior.outcome_trunk[0].bias.copy_(parent_trunk.bias)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(TRIAL_SEED + 1_001)
        prior.prior_embedding.weight.copy_(
            torch.empty_like(prior.prior_embedding.weight)
            .normal_(mean=0.0, std=1.0, generator=generator)
        )
    return prior


def assert_prior_init_identity(base: HazardPath, prior: HazardPath) -> dict[str, object]:
    """At init the prior variant must compute the base function up to float32
    GEMM blocking over zero-padded columns (the new channel contributes ~0)."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(TRIAL_SEED + 7)
    belief = torch.randn(64, WIDTH, generator=generator)
    ids = torch.arange(ACTION_COUNT).unsqueeze(0).expand(64, -1)
    prior_ids = torch.randint(0, ACTION_COUNT, (64,), generator=generator)
    with torch.no_grad():
        a = base(belief.detach(), ids, None)
        b = prior(belief.detach(), ids, prior_ids)
    max_abs = float((a - b).abs().max())
    if not bool(torch.allclose(a, b, atol=1.0e-6, rtol=1.0e-5)):
        raise RuntimeError(
            f"prior-action graft not init-identical to base (max|Δ|={max_abs})"
        )
    return {"max_abs_delta_at_init": max_abs,
            "atol": 1.0e-6, "rtol": 1.0e-5, "passed": True}


# ---------------------------------------------------------------------------
# Parent integrity
# ---------------------------------------------------------------------------

def load_parent() -> tuple[object, dict[str, object], dict[str, object]]:
    if _sha256_bytes(V3_RESULT_PATH.read_bytes()) != V3_RESULT_SHA256:
        raise RuntimeError("sealed v3 result drifted; abort before scoring")
    model, result, provenance = strict.load_exact_parent(V3_RESULT_PATH)
    state = model.state_dict()
    if _state_dict_sha256(state) != PARENT_STATE_SHA256:
        raise RuntimeError("parent state digest drifted after load")
    return model, result, provenance


# ---------------------------------------------------------------------------
# Evidence collection (frozen upstream belief + parent raw hazard table)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CALEvidence:
    beliefs: np.ndarray            # [3072, W] float64
    targets: np.ndarray            # [3072, 5] float64
    applied: np.ndarray            # [3072] int64
    prior_applied: np.ndarray      # [3072] int64
    episode_ordinal: np.ndarray    # [3072] int64 (0..255 per row)
    root_ids: np.ndarray           # [3072] S64
    base_raw_logits: np.ndarray    # [3072, 5] float64
    fold_index: np.ndarray         # [3072] int64 (0/1 by row block)
    episode_permutation_sha256: str
    partition_manifest_sha256: str
    rows: int


def _episode_permutation_sha(indices: np.ndarray) -> str:
    return _sha256_bytes(
        indices.astype("<i8", copy=False).tobytes(order="C")
    )


def collect_cal_evidence(model) -> CALEvidence:
    """Replay the fresh PB21N-CAL partition through the frozen parent.

    Row order is the dataset epoch-0 order (batch_size=1, matching the
    audit's byte-verified replay convention). Fold 0 = rows [0:1536]
    (episodes 0..127 in that order), fold 1 = rows [1536:3072].
    """
    source = development.PartitionSource(
        _to_dev_contract(cal_contract()),
    )
    device = torch.device("cpu")
    was_training = model.training
    model.eval()
    beliefs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    applied: list[int] = []
    prior_applied: list[int] = []
    episode_ordinals: list[int] = []
    root_ids: list[str] = []
    base_raw: list[np.ndarray] = []
    episode_index = 0
    perm_seen: list[int] = []

    for batch in source.iter_all_action_batches(
        epoch=0, batch_size=1,
    ):
        perm_seen.append(episode_index)
        episode_index += 1
        sequence = batch.sequences[0]
        state = model.init_state(batch.batch_size, device)
        for tick in range(batch.sequence_length):
            transition = sequence.transitions[tick]
            prev = (
                None
                if tick == 0
                else sequence.transitions[tick - 1]
            )
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
                output.belief[0].detach().to(
                    "cpu", dtype=torch.float64
                ).numpy()
            )
            targets.append(
                np.asarray(
                    [
                        transition_hazard(t.event_targets)
                        for t in transition.counterfactual_targets
                    ],
                    dtype=np.float64,
                )
            )
            applied.append(int(applied_t.to("cpu").item()))
            prior_applied.append(
                control_action_class(prev.applied_control)
            )
            episode_ordinals.append(episode_index - 1)
            root_ids.append(transition.root_state_sha256)
            base_raw.append(
                development._canonicalize_table(
                    table.raw_hazard_logits, table.action_ids
                )
            )
    if was_training:
        model.train()
    if episode_index != CAL_EPISODES:
        raise RuntimeError(f"episode count drifted: {episode_index}")
    return _finish_evidence(
        beliefs, targets, applied, prior_applied, episode_ordinals, root_ids,
        base_raw, perm_seen, source,
    )


def _to_dev_contract(contract: PB21NContract) -> object:
    """PB21N-CAL is not in the V2.1i contract enum; wrap it with the
    dataset/burn-in fields PartitionSource consumes, naming it TRAIN-CAL."""
    wrapper = development.PartitionContract(
        "TRAIN-CAL", contract.dataset_config, burn_in_steps=contract.burn_in_steps
    )
    return wrapper


def _finish_evidence(
    beliefs, targets, applied, prior_applied, episode_ordinals, root_ids,
    base_raw, perm_seen, source,
) -> CALEvidence:
    b = np.asarray(beliefs, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    a = np.asarray(applied, dtype=np.int64)
    p = np.asarray(prior_applied, dtype=np.int64)
    eo = np.asarray(episode_ordinals, dtype=np.int64)
    roots = np.asarray([r.encode("ascii") for r in root_ids], dtype="S64")
    base = np.concatenate(base_raw, axis=0).astype(np.float64)
    if b.shape != (TOTAL_ROOTS, WIDTH):
        raise RuntimeError(f"evidence geometry drifted: {b.shape}")
    fold = np.repeat(np.arange(2, dtype=np.int64), FOLD_ROWS)
    if not bool(np.array_equal(base.shape, (TOTAL_ROOTS, ACTION_COUNT))):
        raise RuntimeError("base table geometry drifted")
    if bool((np.unique(eo, return_counts=True)[1] != ROOTS_PER_EPISODE).any()):
        raise RuntimeError("episode row count drifted")
    # Fold must be episode-disjoint (contiguous 128-episode blocks).
    for f in range(2):
        ep_in_fold = set(eo[fold == f].tolist())
        if len(ep_in_fold) != CAL_EPISODES // 2:
            raise RuntimeError(f"fold {f} is not episode-disjoint")
        if ep_in_fold & set(eo[fold == 1 - f].tolist()):
            raise RuntimeError("folds share an episode")
    return CALEvidence(
        beliefs=b, targets=t, applied=a, prior_applied=p,
        episode_ordinal=eo, root_ids=roots, base_raw_logits=base,
        fold_index=fold,
        episode_permutation_sha256=_episode_permutation_sha(
            np.asarray(perm_seen, dtype=np.int64)
        ),
        partition_manifest_sha256=source.manifest_sha256,
        rows=TOTAL_ROOTS,
    )


# ---------------------------------------------------------------------------
# Hazard-path retraining (mirrors refine_hazard_path; standalone modules)
# ---------------------------------------------------------------------------

def train_hazard_fold(
    module: HazardPath,
    beliefs: np.ndarray,
    targets: np.ndarray,
    prior_applied: np.ndarray | None,
    *,
    fold_index: np.ndarray,
    fold: int,
) -> dict[str, object]:
    """Retrain the hazard path on one OOF fold, deterministic.

    Inputs are the FULL partition arrays (TOTAL_ROOTS rows); the fit set is
    the complement fold, ``rows = fold_index != fold`` (OOF cross-fit: train
    on the other fold, predict on ``fold``). ``fold`` is also the RNG seed
    selector.

    Schedule bound to the V2.1i HAZARD_REFINEMENT_* constants (prereg §4).
    Loss: unweighted BCE over all five actions, batch 24 roots, complete
    deterministic root permutation without replacement per pass.
    """
    if beliefs.shape[0] != fold_index.shape[0]:
        raise ValueError("beliefs/fold_index length mismatch")
    rows = fold_index != fold  # fit fold = complement of the eval fold
    n = int(rows.sum())
    if n == 0:
        raise ValueError("empty fit fold")
    belief = torch.from_numpy(beliefs[rows]).to(dtype=torch.float32)
    target = torch.from_numpy(targets[rows]).to(dtype=torch.float32)
    prior = (
        None
        if prior_applied is None
        else torch.from_numpy(prior_applied[rows]).to(dtype=torch.long)
    )
    ids = torch.arange(ACTION_COUNT).unsqueeze(0).expand(n, -1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(TRIAL_SEED + 100 * fold + (1 if module.prior_enabled else 0))
    module.train()
    optimizer = torch.optim.AdamW(
        [p for p in module.parameters()],
        lr=REFINEMENT_LEARNING_RATE,
        weight_decay=REFINEMENT_WEIGHT_DECAY,
    )
    pass_records: list[dict[str, object]] = []
    optimizer_steps = 0
    expected_steps_per_pass = int(np.ceil(n / REFINEMENT_ROOT_BATCH_SIZE))
    for pass_index in range(REFINEMENT_PASSES):
        permutation = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        sample_count = 0
        max_grad = 0.0
        for start in range(0, n, REFINEMENT_ROOT_BATCH_SIZE):
            idx = permutation[start : start + REFINEMENT_ROOT_BATCH_SIZE]
            logits = module(
                belief[idx].detach(), ids[idx], None if prior is None else prior[idx]
            )
            loss = F.binary_cross_entropy_with_logits(logits, target[idx])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite hazard-refinement loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(
                nn.utils.clip_grad_norm_(
                    [p for p in module.parameters()], REFINEMENT_CLIP_NORM
                )
            )
            if not math.isfinite(grad_norm):
                raise FloatingPointError("non-finite hazard-refinement gradient")
            optimizer.step()
            optimizer_steps += 1
            roots = len(idx)
            loss_sum += float(loss.detach()) * roots
            sample_count += roots
            max_grad = max(max_grad, grad_norm)
        pass_records.append({
            "pass": pass_index + 1,
            "roots": sample_count,
            "optimizer_steps": expected_steps_per_pass,
            "mean_unweighted_all_action_bce": loss_sum / sample_count,
            "maximum_preclip_gradient_norm": max_grad,
            "root_permutation_sha256": _sha256_bytes(
                permutation.to(dtype=torch.int64).numpy()
                .astype("<i8", copy=False).tobytes(order="C")
            ),
        })
    module.eval()
    expected_total = expected_steps_per_pass * REFINEMENT_PASSES
    if optimizer_steps != expected_total:
        raise RuntimeError("hazard refinement optimizer-step count drifted")
    return {
        "fold": int(fold),
        "prior_enabled": module.prior_enabled,
        "passes": REFINEMENT_PASSES,
        "root_batch_size": REFINEMENT_ROOT_BATCH_SIZE,
        "root_count": n,
        "optimizer_steps": optimizer_steps,
        "learning_rate": REFINEMENT_LEARNING_RATE,
        "weight_decay": REFINEMENT_WEIGHT_DECAY,
        "clip_norm": REFINEMENT_CLIP_NORM,
        "loss": "unweighted_binary_cross_entropy_with_logits_all_five_actions",
        "trainable_parameter_count": int(
            sum(p.numel() for p in module.parameters())
        ),
        "pass_records": pass_records,
    }


def crossfit_arms(
    parent_outcome: nn.Module, ev: CALEvidence,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Cross-fit both arms; return OOF raw logit tables [fold, 3072, 5].

    H_retrain: base (2H) module retrained per fold.
    H_prior:   prior (3H) module retrained per fold.
    Each arm gets its own freshly grafted module per fold so the two folds
    train from the identical parent initialization.

    Each OOF slot is zero-initialized: only slot[f]'s fold-f eval rows are
    written; the complement half stays a defined 0.0 so whole-array
    byte-comparisons (the determinism check) never read uninitialized
    memory. (Trial #1, 2026-08-25, ran before this fix and its published
    ``determinism: false`` flag is a documented false negative — the eval
    rows were verified byte-exact by post-hoc re-derivation; see run report.)
    """
    oof_retrain = np.zeros((2, TOTAL_ROOTS, ACTION_COUNT), dtype=np.float64)
    oof_prior = np.zeros((2, TOTAL_ROOTS, ACTION_COUNT), dtype=np.float64)
    provenance: dict[str, object] = {"retrain": [], "prior": []}
    for fold in range(2):
        fit_rows = ev.fold_index != fold
        eval_rows = ev.fold_index == fold
        for arm in ("retrain", "prior"):
            module = (
                graft_base(parent_outcome)
                if arm == "retrain"
                else graft_prior(parent_outcome)
            )
            record = train_hazard_fold(
                module,
                ev.beliefs,
                ev.targets,
                None if arm == "retrain" else ev.prior_applied,
                fold_index=ev.fold_index,
                fold=fold,
            )
            provenance[arm].append(record)
            ids = torch.arange(ACTION_COUNT).unsqueeze(0).expand(
                int(eval_rows.sum()), -1
            )
            logits = module(
                torch.from_numpy(ev.beliefs[eval_rows]).to(torch.float32).detach(),
                ids,
                None if arm == "retrain" else torch.from_numpy(
                    ev.prior_applied[eval_rows]
                ).to(torch.long),
            )
            table = (
                oof_retrain if arm == "retrain" else oof_prior
            )[fold]
            table[eval_rows] = logits.detach().to(
                "cpu", dtype=torch.float64
            ).numpy()
    return oof_retrain, oof_prior, provenance


# ---------------------------------------------------------------------------
# Canonical AA cross-fit calibration (frozen V2.1i/PB21M contract)
# ---------------------------------------------------------------------------

L2 = 1.0e-6
MIN_SCALE = 1.0e-4
MAX_ITER = 100
TOL = 1.0e-10


def aa_crossfit_calibration(
    raw_oof: np.ndarray, ev: CALEvidence,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Fit per-action affine (s, b) on one fold, apply to the other fold.

    Uses the canonical deterministic Newton fitter
    (irene_brain.v2.hazard_calibration._fit_one_action) with the frozen
    PB21M/V2.1i constants. Returns the calibrated OOF logits [2, 3072, 5].
    """
    from irene_brain.v2 import hazard_calibration as hc
    calibrated = np.empty_like(raw_oof)
    provenance: list[dict[str, object]] = []
    for fold in range(2):
        fit_fold = 1 - fold
        # Canonical PB21M/V2.1i cross-fit (byte-verified in probe_aa_refit):
        # the calibrator for fold `fold` is fitted on the OTHER fold's OOF
        # logits + targets, then applied to fold `fold`'s OOF logits.
        fit_mask = ev.fold_index == fit_fold
        eval_rows = ev.fold_index == fold
        scales = np.empty(ACTION_COUNT, dtype=np.float64)
        biases = np.empty(ACTION_COUNT, dtype=np.float64)
        for act in range(ACTION_COUNT):
            s, b = hc._fit_one_action(
                torch.from_numpy(raw_oof[fit_fold][fit_mask, act]),
                torch.from_numpy(ev.targets[fit_mask, act]),
                l2_regularization=L2,
                minimum_scale=MIN_SCALE,
                max_iterations=MAX_ITER,
                tolerance=TOL,
                fit_mode="per_action_affine",
            )
            scales[act] = s
            biases[act] = b
        calibrated[fold][eval_rows] = (
            raw_oof[fold][eval_rows] * scales[None, :] + biases[None, :]
        )
        provenance.append({
            "cal_fold": fold,
            "fit_fold": fit_fold,
            "scales": scales.tolist(),
            "biases": biases.tolist(),
        })
    return calibrated, provenance


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Gate evaluation (frozen, prereg §5)
# ---------------------------------------------------------------------------

def _factual_mask(applied: np.ndarray) -> np.ndarray:
    ids = np.arange(ACTION_COUNT)[None, :]
    return ids == applied[:, None]


def _factual_metrics(probs: np.ndarray, targets: np.ndarray,
                     applied: np.ndarray) -> dict[str, object]:
    mask = _factual_mask(applied)
    agg = binary_probability_metrics(
        targets[mask], probs[mask], ece_bins=10
    ).as_dict()
    per_action = []
    for act in range(ACTION_COUNT):
        rows = mask[:, act]
        t = targets[rows, act]
        p = probs[rows, act]
        if int(rows.sum()) == 0 or not (0.0 < float(np.mean(t)) < 1.0):
            # No mixed outcomes -> canonical metric undefined; record only.
            per_action.append({"scorable": False, "count": int(rows.sum()),
                               "prevalence": (None if int(rows.sum()) == 0
                                              else float(np.mean(t)))})
            continue
        m = binary_probability_metrics(t, p, ece_bins=10).as_dict()
        per_action.append({
            "scorable": True,
            "calibration_bias": m["calibration_bias"],
            "ece_equal_mass": m["ece_equal_mass"],
            "bce": m["bce"],
            "roc_auc": m.get("roc_auc"),
        })
    return {"aggregate": agg, "per_action": per_action}


def _all_action_metrics(probs: np.ndarray, targets: np.ndarray) -> dict:
    return binary_probability_metrics(
        targets.reshape(-1), probs.reshape(-1), ece_bins=10
    ).as_dict()


def _factual_losses(targets: np.ndarray, probs: np.ndarray,
                    applied: np.ndarray) -> np.ndarray:
    """[n, 2] = (factual BCE, factual Brier) per root."""
    t = targets
    p = probs
    mask = _factual_mask(applied)
    p_safe = np.clip(p, 1e-12, 1 - 1e-12)
    bce = -(t * np.log(p_safe) + (1 - t) * np.log(1 - p_safe))
    brier = (p - t) ** 2
    count = mask.sum(axis=1)
    return np.stack(
        ((bce * mask).sum(axis=1) / count, (brier * mask).sum(axis=1) / count),
        axis=1,
    )


def _episode_cluster_bootstrap(
    per_root: np.ndarray, ordinal: np.ndarray, draws: int,
    *, seed: int,
) -> np.ndarray:
    """[draws, k] episode-clustered bootstrap of the mean per-root delta.

    ``ordinal`` values need not start at 0 (per-fold slices); the unique
    episode IDs present in the slice are used.
    """
    ids = np.unique(ordinal)
    ep_means = np.stack(
        [per_root[ordinal == g].mean(axis=0) for g in ids]
    )
    if any(int((ordinal == g).sum()) != ROOTS_PER_EPISODE for g in ids):
        raise ValueError("episode mean geometry drifted")
    rng = np.random.default_rng(seed)
    out = np.empty((draws, per_root.shape[1]), dtype=np.float64)
    for d in range(draws):
        sampled = rng.integers(0, len(ids), size=len(ids))
        out[d] = ep_means[sampled].mean(axis=0)
    return out


def evaluate_gates(
    h_prior: np.ndarray, h_retrain: np.ndarray, h_base: np.ndarray,
    ev: CALEvidence,
) -> dict[str, object]:
    probs_prior = _sigmoid(h_prior)
    probs_retrain = _sigmoid(h_retrain)
    probs_base = _sigmoid(h_base)

    # Combined OOF tables [3072, 5]: row r takes its own fold's eval value.
    # Used for G5 (structure on the full table, audit B3 analog).
    def combined(probs: np.ndarray) -> np.ndarray:
        out = np.empty_like(probs[0])
        for f in range(2):
            rows = ev.fold_index == f
            out[rows] = probs[f][rows]
        return out

    probs_prior_full = combined(probs_prior)

    # G1: factual absolute on H_prior, both OOF tables.
    g1_tables = []
    g1_passed = True
    for fold in range(2):
        rows = ev.fold_index == fold
        m = _factual_metrics(
            probs_prior[fold][rows], ev.targets[rows], ev.applied[rows]
        )
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

    # G2: paired H_prior vs H_base, factual, per root, episode-clustered.
    delta_2 = np.stack(
        [
            _factual_losses(
                ev.targets[ev.fold_index == f],
                probs_base[f][ev.fold_index == f],
                ev.applied[ev.fold_index == f],
            )
            - _factual_losses(
                ev.targets[ev.fold_index == f],
                probs_prior[f][ev.fold_index == f],
                ev.applied[ev.fold_index == f],
            )
            for f in range(2)
        ],
        axis=0,
    )  # [fold, n, 2]; positive = candidate better
    g2_cells = {}
    g2_passed = True
    for fold in range(2):
        point = delta_2[fold].mean(axis=0)
        draws = _episode_cluster_bootstrap(
            delta_2[fold], ev.episode_ordinal[ev.fold_index == fold],
            BOOTSTRAP_RESAMPLES, seed=TRIAL_SEED + 500 * fold + 1,
        )
        lower = np.quantile(
            draws, ONE_SIDED_ALPHA, axis=0, method=QUANTILE_METHOD
        )
        ok = bool(point[0] > 0.0 and lower[0] > 0.0 and point[1] > 0.0
                  and lower[1] > 0.0)
        g2_passed = g2_passed and ok
        g2_cells[f"fold{fold}"] = {
            "point": {"BCE": float(point[0]), "Brier": float(point[1])},
            "LCB_97.5": {"BCE": float(lower[0]), "Brier": float(lower[1])},
            "passed": bool(ok),
        }

    # G3: all-action control not degraded (H_prior vs H_base).
    g3_cells = []
    g3_passed = True
    for fold in range(2):
        rows = ev.fold_index == fold
        aa_prior = _all_action_metrics(
            probs_prior[fold][rows], ev.targets[rows]
        )
        aa_base = _all_action_metrics(
            probs_base[fold][rows], ev.targets[rows]
        )
        bias_delta = abs(float(aa_prior["calibration_bias"])) - abs(
            float(aa_base["calibration_bias"])
        )
        ece_delta = float(aa_prior["ece_equal_mass"]) - float(
            aa_base["ece_equal_mass"]
        )
        dom = _factual_mask(ev.applied[rows])
        auc_ok = True
        auc_worst_drop = 0.0
        for act in range(ACTION_COUNT):
            r = dom[:, act]
            if int(r.sum()) < 20:
                continue
            t_aa = ev.targets[rows][r, act]
            if not (0.0 < float(np.mean(t_aa)) < 1.0):
                # No mixed outcomes -> AUC undefined; record, do not gate.
                continue
            aa_auc = float(binary_probability_metrics(
                t_aa,
                probs_base[fold][rows][r, act], ece_bins=10
            ).as_dict()["roc_auc"])
            p_auc = float(binary_probability_metrics(
                t_aa,
                probs_prior[fold][rows][r, act], ece_bins=10
            ).as_dict()["roc_auc"])
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

    # G5: within-episode prior-label shuffle specificity (audit B3 analog),
    # on the combined OOF table.
    observed_biases: list[float] = []
    g5_skipped: list[dict[str, int]] = []
    for act in range(ACTION_COUNT):
        rows = ev.applied == act
        for pv in range(ACTION_COUNT):
            m = rows & (ev.prior_applied == pv)
            if int(m.sum()) < MIN_STRATUM_ROWS:
                continue
            t_stratum = ev.targets[m, act]
            if not (0.0 < float(np.mean(t_stratum)) < 1.0):
                # No mixed outcomes -> canonical metric undefined; record, skip.
                g5_skipped.append({"act": act, "prior": int(pv),
                                   "n": int(m.sum())})
                continue
            agg = binary_probability_metrics(
                t_stratum, probs_prior_full[m, act], ece_bins=10
            ).as_dict()
            observed_biases.append(float(agg["calibration_bias"]))
    observed_range = float(max(observed_biases) - min(observed_biases))
    permuted_ranges: list[float] = []
    rng = np.random.default_rng(G5_SHUFFLE_SEED)
    for rep in range(G5_SHUFFLE_PERMUTATIONS):
        permuted = np.empty_like(ev.prior_applied)
        for g in np.unique(ev.episode_ordinal):
            gm = ev.episode_ordinal == g
            permuted[gm] = rng.permutation(ev.prior_applied[gm])
        sb: list[float] = []
        for act in range(ACTION_COUNT):
            rows = ev.applied == act
            for pv in range(ACTION_COUNT):
                m = rows & (permuted == pv)
                if int(m.sum()) < MIN_STRATUM_ROWS:
                    continue
                t_stratum = ev.targets[m, act]
                if not (0.0 < float(np.mean(t_stratum)) < 1.0):
                    continue
                agg = binary_probability_metrics(
                    t_stratum, probs_prior_full[m, act], ece_bins=10
                ).as_dict()
                sb.append(float(agg["calibration_bias"]))
        permuted_ranges.append(float(max(sb) - min(sb)))
    permuted_p95 = float(np.quantile(permuted_ranges, 0.95))
    g5_passed = bool(observed_range > permuted_p95)

    # G6: mechanism isolation H_prior vs H_retrain (factual BCE LCB > 0).
    delta_6 = np.stack(
        [
            _factual_losses(
                ev.targets[ev.fold_index == f],
                probs_retrain[f][ev.fold_index == f],
                ev.applied[ev.fold_index == f],
            )
            - _factual_losses(
                ev.targets[ev.fold_index == f],
                probs_prior[f][ev.fold_index == f],
                ev.applied[ev.fold_index == f],
            )
            for f in range(2)
        ],
        axis=0,
    )
    g6_cells = {}
    g6_passed = True
    for fold in range(2):
        point = delta_6[fold].mean(axis=0)
        draws = _episode_cluster_bootstrap(
            delta_6[fold], ev.episode_ordinal[ev.fold_index == fold],
            BOOTSTRAP_RESAMPLES, seed=TRIAL_SEED + 600 * fold + 7,
        )
        lower = np.quantile(
            draws, ONE_SIDED_ALPHA, axis=0, method=QUANTILE_METHOD
        )
        ok = bool(point[0] > 0.0 and lower[0] > 0.0)
        g6_passed = g6_passed and ok
        g6_cells[f"fold{fold}"] = {
            "point": {"BCE": float(point[0]), "Brier": float(point[1])},
            "LCB_97.5": {"BCE": float(lower[0]), "Brier": float(lower[1])},
            "passed": bool(ok),
        }

    gates = {
        "G1_factual_absolute": {"tables": g1_tables, "passed": bool(g1_passed)},
        "G2_paired_vs_base": {"cells": g2_cells, "passed": bool(g2_passed)},
        "G3_all_action_controls": {"cells": g3_cells, "passed": bool(g3_passed)},
        "G5_shuffle_specificity": {
            "observed_range": observed_range,
            "stratum_bias_values": observed_biases,
            "skipped_unscorable_strata": g5_skipped,
            "permuted_p95": permuted_p95,
            "permuted_max": float(max(permuted_ranges)),
            "passed": g5_passed,
        },
        "G6_mechanism_isolation": {"cells": g6_cells, "passed": bool(g6_passed)},
    }
    gates["passed"] = bool(
        g1_passed and g2_passed and g3_passed and g5_passed and g6_passed
    )
    return gates


# ---------------------------------------------------------------------------
# Trial orchestration + create-only publish
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


def _provenance() -> dict[str, object]:
    base = run_provenance.provenance(
        deterministic=True, train_seed=TRIAL_SEED, eval_seed=None,
        bank_digest=None,
    )
    base["mode"] = MODE
    base["run_id"] = RUN_ID
    base["classification"] = CLASSIFICATION
    return base


def run_trial() -> dict[str, object]:
    started = time.time()
    provenance = _provenance()

    # 1. Parent integrity (byte-exact, read-only).
    model, result, parent_provenance = load_parent()
    outcome = model.world_model.outcome_model
    parent_state_sha_before = _state_dict_sha256(model.state_dict())
    if parent_state_sha_before != PARENT_STATE_SHA256:
        raise RuntimeError("parent state digest drifted at trial start")

    # 2. Graft + init identity (the prior channel must add ~nothing at init).
    base_init = graft_base(outcome)
    prior_init = graft_prior(outcome)
    init_identity = assert_prior_init_identity(base_init, prior_init)
    del base_init, prior_init

    # 3. Frozen upstream belief + parent raw hazard table on PB21N-CAL.
    ev = collect_cal_evidence(model)

    # 4. Cross-fit both arms (H_retrain control + H_prior candidate).
    oof_retrain, oof_prior, train_provenance = crossfit_arms(outcome, ev)

    # 5. Canonical AA cross-fit calibration of all three OOF tables.
    # H_base: the frozen parent's raw logit table (same for both folds),
    # AA-calibrated per the frozen V2.1i contract (fit fold 1-f -> fold f).
    base_raw_oof = np.stack([ev.base_raw_logits, ev.base_raw_logits])
    h_base, base_cal = aa_crossfit_calibration(base_raw_oof, ev)
    h_retrain_cal, retrain_cal = aa_crossfit_calibration(oof_retrain, ev)
    h_prior_cal, prior_cal = aa_crossfit_calibration(oof_prior, ev)

    # 6. G4: parent preservation (read-only check).
    parent_state_sha_after = _state_dict_sha256(model.state_dict())
    g4_passed = bool(parent_state_sha_after == PARENT_STATE_SHA256)
    g4 = {
        "parent_state_sha256_before": parent_state_sha_before,
        "parent_state_sha256_after": parent_state_sha_after,
        "parent_state_sha256_sealed": PARENT_STATE_SHA256,
        "passed": g4_passed,
    }

    # 7. Gates G1/G2/G3/G5/G6 on AA-calibrated OOF tables.
    gates = evaluate_gates(h_prior_cal, h_retrain_cal, h_base, ev)
    gates["G4_parent_preservation"] = g4
    gates["passed"] = bool(gates["passed"] and g4_passed)

    # 8. Determinism: re-derive the candidate OOF logits byte-exactly.
    oof_retrain_2, oof_prior_2, _ = crossfit_arms(outcome, ev)
    prior_rederived = bool(np.array_equal(oof_prior, oof_prior_2))
    retrain_rederived = bool(np.array_equal(oof_retrain, oof_retrain_2))
    determinism = bool(prior_rederived and retrain_rederived)
    del oof_retrain_2, oof_prior_2

    result_doc = {
        "mode": MODE,
        "run_id": RUN_ID,
        "classification": CLASSIFICATION,
        "status": "completed",
        "passed": bool(gates["passed"] and determinism),
        "determinism": {
            "prior_oof_logits_rederived_byte_equal": prior_rederived,
            "retrain_oof_logits_rederived_byte_equal": retrain_rederived,
            "crossfit_rederivation_byte_equal": determinism,
        },
        "parent_provenance": {
            "v3_result_sha256": V3_RESULT_SHA256,
            "parent_sha256": PARENT_SHA256,
            "parent_state_sha256": PARENT_STATE_SHA256,
            "provenance": _jsonable(parent_provenance),
        },
        "partition": {
            "label": "PB21N-CAL",
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
            "H_retrain": "parent 2H hazard path, retrained per OOF fold, AA cross-fit calibrated",
            "H_prior": "3H hazard path + prior-action embedding, retrained per OOF fold, AA cross-fit calibrated",
        },
        "gates": gates,
        "init_identity": init_identity,
        "oof_logit_sha256": {
            "H_prior": _sha256_bytes(np.ascontiguousarray(h_prior_cal).tobytes()),
            "H_retrain": _sha256_bytes(np.ascontiguousarray(h_retrain_cal).tobytes()),
            "H_base": _sha256_bytes(np.ascontiguousarray(h_base).tobytes()),
        },
        "aa_calibration_provenance": {
            "base": base_cal, "retrain": retrain_cal, "prior": prior_cal,
        },
        "training_provenance": train_provenance,
        "retry_allowed": False,
        "provenance": provenance,
        "wall_seconds": time.time() - started,
    }
    _publish(result_doc, ev, h_prior_cal, h_retrain_cal, h_base,
             oof_prior, oof_retrain)
    return result_doc


def _publish(
    result_doc: dict[str, object],
    ev: CALEvidence,
    h_prior_cal: np.ndarray,
    h_retrain_cal: np.ndarray,
    h_base: np.ndarray,
    oof_prior: np.ndarray,
    oof_retrain: np.ndarray,
) -> None:
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
        encoding="utf-8",
    )
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
        oof_H_prior_raw=oof_prior,
        oof_H_retrain_raw=oof_retrain,
        oof_H_prior_calibrated=h_prior_cal,
        oof_H_retrain_calibrated=h_retrain_cal,
        oof_H_base_calibrated=h_base,
        partition_manifest_sha256=ev.partition_manifest_sha256.encode(),
        episode_permutation_sha256=ev.episode_permutation_sha256.encode(),
    )
    reg = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "mode": MODE,
        "classification": CLASSIFICATION,
        "prereg": "brain/docs/preregistrations/"
                  "2026-08-25-pb21n-prior-action-hazard-conditioning-v1.md",
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
            "g5_permutations": G5_SHUFFLE_PERMUTATIONS,
            "g5_shuffle_seed": G5_SHUFFLE_SEED,
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
        "aa_calibration": {
            "L2": L2, "min_scale": MIN_SCALE, "max_iter": MAX_ITER,
            "tol": TOL,
        },
    }
    reg_path = EVIDENCE_DIR / f"{RUN_ID}.registration.json"
    if reg_path.exists():
        raise RuntimeError("create-only: registration already published")
    reg_path.write_text(
        json.dumps(_jsonable(reg), sort_keys=True, allow_nan=False,
                   separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


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
