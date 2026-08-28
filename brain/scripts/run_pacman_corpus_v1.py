"""T1-3: expert behavior-cloning corpus generator (preregistration section 1-6).

Preregistration: ``brain/docs/preregistrations/2026-08-28-pacman-corpus-v1.md``.

Runs the frozen pixel-only expert (``diagnostic.scripted_maze_chase_planner.v1``,
family-matched knobs) on the registered TRAIN seeds (50 per family,
``partition_seeds(f, "TRAIN")[:50]``) and stores, per native tick:

- ``frame``: the 16x16x3 uint8 ``Observation.rgb`` the teacher saw (model input);
- ``prev_action``: the teacher's applied action class on the previous tick
  (0 at tick 0) — model input context;
- ``target_action``: the teacher's applied action class on this tick — the
  5-class BC supervision label.

Evidence-only fields (never model input): reward, event flags,
terminated/truncated, per-episode M_P / pellet fraction / final state hash.

Outputs (gitignored artifact dir ``brain/datasets/embodied-corpus-v1/``):
  frames/ep_{fam}_{i}.npy   uint8 [T, 16, 16, 3]
  actions/ep_{fam}_{i}.npy  int8  [T, 2]      (prev_action, target_action)
  meta/ep_{fam}_{i}.npy     float32 [T, 4]    (reward, event_flag, terminated, truncated)
  manifest.json             per-episode records + corpus digest

Create-only acceptance report:
``brain/runs/embodied-corpus-v1/2026-08-28-corpus-construction-v1.json``.

Usage (from brain/):
  OMP_NUM_THREADS=1 py -3.11 scripts/run_pacman_corpus_v1.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from hashlib import sha256
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

_HERE = Path(__file__).resolve()
BRAIN_ROOT = _HERE.parent.parent
REPO_ROOT = BRAIN_ROOT.parent
sys.path.insert(0, str(BRAIN_ROOT / "src"))
sys.path.insert(0, str(BRAIN_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402  (provenance only; the generator is CPU-logical)

from run_provenance import apply_deterministic_mode, provenance  # noqa: E402

apply_deterministic_mode()

from irene_brain.environments.maze_chase import MazeChaseEnv  # noqa: E402
from irene_brain.environments.pacman_harness import (  # noqa: E402
    FAMILIES,
    partition_seeds,
)
from irene_brain.evaluation.diagnostic_policies import (  # noqa: E402
    ScriptedMazeChasePlannerPolicy,
)
from irene_brain.v2.trajectory_objective import control_action_class  # noqa: E402

# --- Frozen constants (preregistration 2026-08-28-pacman-corpus-v1.md) -----

EP_PER_FAMILY = 50  # section 2 (frozen; 50 < 524 available TRAIN seeds/family)
GRID = 16
CANDIDATE_PELLETS = 6  # section 1 frozen planner knobs
HORIZON = 24
INPUT_DELAY_TICKS = 0
RESULT_RELPATH = BRAIN_ROOT / "runs" / "embodied-corpus-v1" / "2026-08-28-corpus-construction-v1.json"
DATASET_RELPATH = BRAIN_ROOT / "datasets" / "embodied-corpus-v1"

# event flags (section 3): 0 none, 1 pellet_eaten, 2 caught, 3 cleared.
_EVENT_NONE = 0
_EVENT_PELLET = 1
_EVENT_CAUGHT = 2
_EVENT_CLEARED = 3


def _planner_for(family) -> ScriptedMazeChasePlannerPolicy:
    """Family-matched knobs (section 1 frozen)."""
    return ScriptedMazeChasePlannerPolicy(
        ghost_period=family.ghost_period,
        player_period=family.player_period,
        ghost_elroy=family.ghost_elroy,
        candidate_pellets=CANDIDATE_PELLETS,
        horizon=HORIZON,
        input_delay_ticks=INPUT_DELAY_TICKS,
    )


def _event_flag(events: tuple[str, ...]) -> int:
    if "cleared" in events:
        return _EVENT_CLEARED
    if "caught" in events:
        return _EVENT_CAUGHT
    if "pellet_eaten" in events:
        return _EVENT_PELLET
    return _EVENT_NONE


def run_episode(
    family_index: int,
    seed: int,
) -> dict:
    """Run one clean expert episode; return per-tick arrays + evidence.

    Model-input fields come ONLY from ``Observation.rgb`` and the teacher's
    own emitted control (section 5, C8).  Evidence fields read the public
    step outcome and the snapshot; they never feed the model.
    """
    fam = FAMILIES[family_index]
    kw = fam.env_kwargs()
    env = MazeChaseEnv(**kw)
    first_obs = env.reset(seed)
    assert first_obs is not None
    plan = _planner_for(fam)
    plan.reset(seed)

    total_pellets = 0
    for (x, y) in env._maze:  # noqa: SLF001 (evidence only: total pellet count)
        if env._has_pellet(x, y):  # noqa: SLF001 (evidence only)
            total_pellets += 1

    frames: list[bytes] = []
    prev_actions: list[int] = []
    targets: list[int] = []
    rewards: list[float] = []
    event_flags: list[int] = []
    terminated_flags: list[int] = []
    truncated_flags: list[int] = []

    prev_action = 0
    pellets = 0
    tick = 0
    while True:
        obs = env.current_observation
        # The teacher's action for this tick (pixel-only; its own FIFO model
        # of earlier presses is its private state, never simulator state).
        control = plan.act(obs)
        target = int(control_action_class(control))
        outcome = env.step(control)

        frames.append(obs.rgb.pixels)
        prev_actions.append(prev_action)
        targets.append(target)
        rewards.append(outcome.reward)
        event_flags.append(_event_flag(outcome.events))
        terminated_flags.append(int(outcome.terminated))
        truncated_flags.append(int(outcome.truncated))
        if "pellet_eaten" in outcome.events:
            pellets += 1

        prev_action = target
        tick += 1
        if outcome.terminated or outcome.truncated:
            break

    snapshot = env.snapshot()
    state_hash = sha256(snapshot).hexdigest()
    survival_ticks = tick
    m_p = 0.5 * (pellets / max(1, total_pellets)) + 0.5 * (survival_ticks / fam.max_ticks)
    m_p = min(1.0, max(0.0, m_p))

    return {
        "family": family_index,
        "seed": seed,
        "ticks": tick,
        "total_pellets": total_pellets,
        "pellets_eaten": pellets,
        "pellet_fraction": pellets / max(1, total_pellets),
        "survival_ticks": survival_ticks,
        "m_p": m_p,
        "final_state_hash": state_hash,
        "frames": np.frombuffer(b"".join(frames), dtype=np.uint8).reshape(tick, GRID, GRID, 3),
        "prev_actions": np.array(prev_actions, dtype=np.int8),
        "targets": np.array(targets, dtype=np.int8),
        "rewards": np.array(rewards, dtype=np.float32),
        "event_flags": np.array(event_flags, dtype=np.float32),
        "terminated": np.array(terminated_flags, dtype=np.float32),
        "truncated": np.array(truncated_flags, dtype=np.float32),
    }


def _episode_name(family_index: int, index: int) -> str:
    return f"ep_{family_index:02d}_{index:03d}"


def store_episode(ep: dict, index: int) -> dict:
    name = _episode_name(ep["family"], index)
    frames_dir = DATASET_RELPATH / "frames"
    actions_dir = DATASET_RELPATH / "actions"
    meta_dir = DATASET_RELPATH / "meta"
    for d in (frames_dir, actions_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    np.save(frames_dir / f"{name}.npy", ep["frames"])
    np.save(actions_dir / f"{name}.npy", np.stack([ep["prev_actions"], ep["targets"]], axis=1))
    np.save(
        meta_dir / f"{name}.npy",
        np.stack([ep["rewards"], ep["event_flags"], ep["terminated"], ep["truncated"]], axis=1),
    )
    return {
        "name": name,
        "family": ep["family"],
        "seed": ep["seed"],
        "ticks": ep["ticks"],
        "total_pellets": ep["total_pellets"],
        "pellets_eaten": ep["pellets_eaten"],
        "pellet_fraction": round(ep["pellet_fraction"], 6),
        "survival_ticks": ep["survival_ticks"],
        "m_p": round(ep["m_p"], 6),
        "final_state_hash": ep["final_state_hash"],
        "frames_sha256": sha256(ep["frames"].tobytes()).hexdigest(),
        "actions_sha256": sha256(
            np.stack([ep["prev_actions"], ep["targets"]], axis=1).tobytes()
        ).hexdigest(),
        "meta_sha256": sha256(
            np.stack([ep["rewards"], ep["event_flags"], ep["terminated"], ep["truncated"]], axis=1).tobytes()
        ).hexdigest(),
    }


def replay_digest(family_index: int, seed: int) -> str:
    """Re-run one episode and hash (frame stream, action streams, state hash).

    Used by gates C/D: two independent calls with the same (family, seed) MUST
    byte-match, and the stored arrays MUST equal the live replay byte-for-byte.
    """
    ep = run_episode(family_index, seed)
    h = sha256()
    h.update(ep["frames"].tobytes())
    h.update(ep["prev_actions"].tobytes())
    h.update(ep["targets"].tobytes())
    h.update(ep["final_state_hash"].encode())
    return h.hexdigest()


def main() -> int:
    started = time.time()
    records: list[dict] = []
    total_ticks = 0

    for fam_index in range(len(FAMILIES)):
        seeds = partition_seeds(fam_index, "TRAIN")[:EP_PER_FAMILY]
        assert len(seeds) == EP_PER_FAMILY
        for i, seed in enumerate(seeds):
            ep = run_episode(fam_index, seed)
            total_ticks += ep["ticks"]
            rec = store_episode(ep, i)
            records.append(rec)
            if i % 10 == 0:
                print(
                    f"F{fam_index} ep {i}/{EP_PER_FAMILY} seed {seed} "
                    f"ticks={ep['ticks']} M_P={ep['m_p']:.3f} "
                    f"pellet_frac={ep['pellet_fraction']:.3f}",
                    flush=True,
                )

    # --- manifest -----------------------------------------------------------
    manifest = {
        "mode": "pacman_corpus_v1",
        "preregistration": "brain/docs/preregistrations/2026-08-28-pacman-corpus-v1.md",
        "episodes_per_family": EP_PER_FAMILY,
        "total_episodes": len(records),
        "total_transitions": total_ticks,
        "episodes": records,
    }
    manifest["corpus_sha256"] = sha256(
        json.dumps(records, sort_keys=True).encode()
    ).hexdigest()
    manifest_path = DATASET_RELPATH / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    # --- Gate C: determinism (fixed 2-episode pair: one F1, one F3) ---------
    det_pair = [
        (1, partition_seeds(1, "TRAIN")[0]),
        (3, partition_seeds(3, "TRAIN")[0]),
    ]
    determinism = {}
    det_pass = True
    for fam, seed in det_pair:
        a = replay_digest(fam, seed)
        b = replay_digest(fam, seed)
        # Cross-check against the stored arrays (gate D byte check):
        name = _episode_name(fam, 0)
        stored_frames = np.load(DATASET_RELPATH / "frames" / f"{name}.npy")
        stored_actions = np.load(DATASET_RELPATH / "actions" / f"{name}.npy")
        live = run_episode(fam, seed)
        frames_match = (stored_frames == live["frames"]).all()
        actions_match = (stored_actions[:, 0] == live["prev_actions"]).all() and (
            stored_actions[:, 1] == live["targets"]
        ).all()
        deterministic = a == b
        det_pass = det_pass and deterministic and bool(frames_match) and bool(actions_match)
        determinism[f"{fam}|{seed}"] = {
            "run1_sha256": a,
            "run2_sha256": b,
            "independent_runs_identical": deterministic,
            "stored_frames_match_live_replay": bool(frames_match),
            "stored_actions_match_live_replay": bool(actions_match),
        }
    gate_c = {"passed": bool(det_pass), "pairs": determinism}

    # --- Gate D: causal boundary (C8) --------------------------------------
    # Model-input fields are built only from Observation.rgb + the teacher's
    # emitted control.  Static audit: the generator reads no MazeChaseEnv
    # attribute to build a model-input field; the only env attribute reads
    # are `_maze`/`_has_pellet` (total-pellet evidence) and the public
    # `current_observation` / `step` / `snapshot` API.  Record the audit:
    import inspect

    src = inspect.getsource(sys.modules[__name__])
    model_input_lines = [
        ln for ln in src.splitlines()
        if ("frames.append" in ln or "targets.append" in ln or "prev_actions.append" in ln)
    ]
    privileged = [
        ln for ln in model_input_lines
        if "env._" in ln or "env._has_pellet" in ln or "reward" in ln or "events" in ln
    ]
    gate_d = {
        "passed": not privileged and len(model_input_lines) >= 3,
        "model_input_construction_lines": model_input_lines,
        "privileged_reads_in_model_input_lines": privileged,
        "note": "frames come from obs.rgb only; actions come from the teacher's "
        "emitted control via control_action_class; evidence fields read the "
        "public StepOutcome + snapshot only.",
    }

    # --- Gate E: action-class validity --------------------------------------
    e_pass = True
    for rec in records:
        pass  # arrays validated at load time below
    for fam in range(len(FAMILIES)):
        for i in range(EP_PER_FAMILY):
            arr = np.load(DATASET_RELPATH / "actions" / f"{_episode_name(fam, i)}.npy")
            if arr.shape[1] != 2 or arr.min() < 0 or arr.max() > 4:
                e_pass = False
    gate_e = {"passed": e_pass, "action_domain": [0, 4]}

    # --- Gate F: scale ------------------------------------------------------
    f_pass = len(records) == 5 * EP_PER_FAMILY
    seed_sets_ok = True
    for fam in range(len(FAMILIES)):
        stored_seeds = [r["seed"] for r in records if r["family"] == fam]
        if stored_seeds != list(partition_seeds(fam, "TRAIN")[:EP_PER_FAMILY]):
            seed_sets_ok = False
    gate_f = {
        "passed": f_pass and seed_sets_ok,
        "total_episodes": len(records),
        "seed_sets_match_partition": seed_sets_ok,
    }

    # --- Gate G: expert competence floor ------------------------------------
    per_family: dict[int, dict] = {}
    for fam in range(len(FAMILIES)):
        fam_recs = [r for r in records if r["family"] == fam]
        per_family[fam] = {
            "mean_m_p": round(np.mean([r["m_p"] for r in fam_recs]), 6),
            "mean_pellet_fraction": round(np.mean([r["pellet_fraction"] for r in fam_recs]), 6),
        }
    floor_pellet = all(v["mean_pellet_fraction"] >= 0.50 for v in per_family.values())
    mp_pass_count = sum(1 for v in per_family.values() if v["mean_m_p"] >= 0.50)
    gate_g = {
        "passed": floor_pellet and mp_pass_count >= 4,
        "per_family": per_family,
        "floor": {"mean_pellet_fraction_min": 0.50, "families_with_mean_m_p_ge_0.50_min": 4},
    }

    gates = {
        "C_determinism": gate_c,
        "D_causal_boundary": gate_d,
        "E_action_class_validity": gate_e,
        "F_scale": gate_f,
        "G_expert_competence_floor": gate_g,
    }
    passed_all = all(g["passed"] for g in gates.values())

    result = {
        "mode": "pacman_corpus_v1_construction_acceptance",
        "preregistration": manifest["preregistration"],
        "started_unix": int(started),
        "wall_seconds": round(time.time() - started, 3),
        "total_episodes": len(records),
        "total_transitions": total_ticks,
        "corpus_sha256": manifest["corpus_sha256"],
        "gates": gates,
        "all_passed": passed_all,
        "provenance": {
            **provenance(deterministic=True),
            "note": "corpus generator (T1-3); no model, seeds are the registered TRAIN partition",
        },
    }
    RESULT_RELPATH.parent.mkdir(parents=True, exist_ok=True)
    if RESULT_RELPATH.exists():
        print(f"REFUSED: {RESULT_RELPATH} already exists (create-only).", file=sys.stderr)
        return 1
    RESULT_RELPATH.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    for name, gate in gates.items():
        print(f"[{'PASS' if gate['passed'] else 'FAIL'}] {name}")
    print(f"all_passed={passed_all} wall={result['wall_seconds']}s corpus_sha={manifest['corpus_sha256'][:16]}")
    return 0 if passed_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
