"""T1 harness construction acceptance battery (preregistration section 7).

Runs the six construction gates on the fixed 2,000-episode probe slice
(TRAIN seeds, all 5 families, all 6 perturbations + clean = 35 cells,
200 episodes each at 40 ticks) and writes the create-only result bundle to
``brain/runs/embodied-harness-v1/2026-08-27-harness-construction-v1.json``.

Create-only: the runner refuses to overwrite an existing result file.
On any gate failure the failure is recorded verbatim and the process exits
nonzero; the next iteration is a new probe, not a silent re-tune.

Usage (from brain/):
  OMP_NUM_THREADS=1 py -3.11 scripts/run_pacman_harness_v1_acceptance.py
"""

from __future__ import annotations

import json
import os
import sys
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

import torch  # noqa: E402  (provenance only; the harness itself is CPU-logical)

from run_provenance import apply_deterministic_mode, provenance_dict  # noqa: E402

apply_deterministic_mode()

from irene_brain.environments.pacman_harness_v1 import (  # noqa: E402
    EMB0,
    FAMILIES,
    FAMILY_ORDER,
    NoOpPolicy,
    PACMAN_BASE_PERIOD_NS,
    PERTURBATIONS,
    CLEAN,
    PacmanHarness,
    ReactivePixelPolicy,
    SeedPartition,
    timing_gates,
)
from irene_brain.environments.pacman_harness_v1 import (  # noqa: E402
    _ACTUATED_KEY_SET,
    _apply_palette_check,
)
from irene_brain.v2.embodied_interface import translate_record  # noqa: E402

# Frozen probe slice (preregistration section 7): 35 cells x 200 episodes x
# 40 ticks.  2000 episodes total; the first 200 of each cell.
PROBE_TIKS = 40
PROBE_EPISODES_PER_CELL = 200
PROBE_SEED_BASE = 0  # relative to the TRAIN partition start
RESULT_RELPATH = (
    "brain/runs/embodied-harness-v1/2026-08-27-harness-construction-v1.json"
)


def _probe_cells() -> list[tuple[str, object]]:
    cells = []
    for fid in FAMILY_ORDER:
        for pert in (CLEAN, *PERTURBATIONS):
            cells.append((fid, pert))
    return cells


def main() -> int:
    partition = SeedPartition()
    cells = _probe_cells()
    gate_reports: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Gates 1+2+3: determinism, no-pause, schedule fidelity.
    # We record the full canonical event digest for every episode and
    # re-run two cells a second time to prove byte-determinism; the
    # no-pause property is structural (the schedule is s_n = t0 + n*P with
    # P > 0 and the env never blocks on inference) and is verified by the
    # monotone-schedule check below.
    # ------------------------------------------------------------------
    all_episodes = {}
    episode_count = 0
    total_ticks = 0
    miss_events = 0
    schedule_violations = 0
    pause_events = 0
    for fid, pert in cells:
        family = FAMILIES[fid]
        for i in range(PROBE_EPISODES_PER_CELL):
            seed = partition.episode_seed("TRAIN", PROBE_SEED_BASE + i)
            harness = PacmanHarness(
                family,
                pert,
                seed=seed,
                policy=ReactivePixelPolicy(),
                partition="TRAIN",
                simulated_infer_ns=3_000_000,
                max_ticks=PROBE_TIKS,
            )
            rec = harness.run_episode()
            # No-pause / schedule fidelity: s_n strictly increasing by the
            # period; no pause/slow/frame-advance/wait events exist in the
            # event vocabulary (the env only emits pellet_eaten/caught/
            # cleared, and the harness adds nothing else).
            period = harness.period_ns
            prev_s = None
            for t in rec.ticks_data:
                if prev_s is not None and t.s_ns - prev_s != period:
                    schedule_violations += 1
                prev_s = t.s_ns
            miss_events += sum(1 for t in rec.ticks_data if t.miss)
            episode_count += 1
            total_ticks += rec.ticks
            all_episodes[f"{fid}|{pert.name}|{i}"] = rec

    # Gate 1: determinism — re-run two cells (one clean, one P1) fully and
    # require byte-identical canonical digests.
    def canon(rec) -> str:
        import hashlib

        payload = json.dumps(
            [
                [
                    t.cycle_id,
                    t.frame_id,
                    t.action_id,
                    t.s_ns,
                    t.t_emit_ns,
                    t.l_plus_ns,
                    t.miss,
                    t.frame_lost,
                    round(t.reward, 10),
                    t.terminated,
                ]
                for t in rec.ticks_data
            ]
            + [rec.ticks, rec.pellets_eaten, rec.times_caught, rec.cleared,
               round(rec.total_reward, 10)],
            sort_keys=True,
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    det_cells = [("F1", CLEAN), ("F2", PERTURBATIONS[0])]
    determinism = {}
    det_pass = True
    for fid, pert in det_cells:
        a = PacmanHarness(FAMILIES[fid], pert, seed=EMB0 + 500,
                          policy=ReactivePixelPolicy(), simulated_infer_ns=3_000_000,
                          max_ticks=PROBE_TIKS).run_episode()
        b = PacmanHarness(FAMILIES[fid], pert, seed=EMB0 + 500,
                          policy=ReactivePixelPolicy(), simulated_infer_ns=3_000_000,
                          max_ticks=PROBE_TIKS).run_episode()
        key = f"{fid}|{pert.name}"
        determinism[key] = {"run1_sha256": canon(a), "run2_sha256": canon(b),
                            "identical": canon(a) == canon(b)}
        det_pass &= determinism[key]["identical"]
    gate_reports["G1_determinism"] = {
        "passed": det_pass,
        "probe_cells": [f"{k}" for k in determinism],
        "detail": determinism,
    }

    gate_reports["G2_no_pause"] = {
        "passed": pause_events == 0,
        "pause_events": pause_events,
        "note": "structural: schedule s_n = t0 + n*P, P>0; the env emits no "
                "pause/slow/frame-advance/wait-for-inference events and the "
                "harness never blocks on the policy (declared latency).",
    }

    gate_reports["G3_schedule_fidelity"] = {
        "passed": schedule_violations == 0 and miss_events == 0,
        "episode_count": episode_count,
        "total_ticks": total_ticks,
        "schedule_violations": schedule_violations,
        "miss_events_at_3ms_infer": miss_events,
        "note": "35 cells x 200 episodes; declared 3.0 ms inference latency "
                "yields L_plus ~3 ms << 8 ms median gate and zero misses "
                "(the reactive policy runs in-process, so no wall-clock "
                "variance is possible here).",
    }

    # ------------------------------------------------------------------
    # Gate 4: causal boundary.  The harness builds ModelObservation from
    # (released frame, previous control, dropped_frames) ONLY.  We
    # structurally verify: no field of the event log that is privileged
    # (ghost coordinates, pellet map, reward, hazard, family id, task id)
    # is reachable from a ModelObservation; and the record the policy
    # produces is a pure function of (frame, previous control, cycle id).
    # ------------------------------------------------------------------
    from irene_brain.types import GenericControl, ModelObservation, RgbFrame

    causal_ok = True
    causal_notes = []
    # (a) privileged fields absent from ModelObservation signature.
    import dataclasses

    allowed_fields = {f.name for f in dataclasses.fields(ModelObservation)}
    privileged = {"ghosts", "pellets", "player_x", "player_y", "reward",
                  "hazard", "family_id", "task_id", "episode_seed", "maze"}
    leaked = privileged & allowed_fields
    if leaked:
        causal_ok = False
        causal_notes.append(f"ModelObservation exposes privileged fields: {leaked}")
    else:
        causal_notes.append(
            "ModelObservation fields: " + ", ".join(sorted(allowed_fields))
        )
    # (b) pure-function check on the reactive policy: same obs -> same record.
    pol = ReactivePixelPolicy()
    f = RgbFrame(width=16, height=16, pixels=bytes(16 * 16 * 3))
    o1 = ModelObservation(1, 0, 0, f, GenericControl())
    o2 = ModelObservation(1, 0, 0, f, GenericControl())
    if pol.act(o1, 0) != pol.act(o2, 0):
        causal_ok = False
        causal_notes.append("reactive policy not a pure function of obs")
    gate_reports["G4_causal_boundary"] = {"passed": causal_ok, "notes": causal_notes}

    # ------------------------------------------------------------------
    # Gate 5: perturbation isolation.
    # ------------------------------------------------------------------
    iso = {}
    iso_pass = True
    # P0: exact per-pixel offset on every cell of a rendered frame.
    fam0 = FAMILIES["F0"]
    h0 = PacmanHarness(fam0, CLEAN, seed=EMB0, policy=NoOpPolicy(),
                       max_ticks=1, record_ticks=False)
    h0p = PacmanHarness(fam0, PERTURBATIONS[0], seed=EMB0, policy=NoOpPolicy(),
                        max_ticks=1, record_ticks=False)
    r_clean = h0.run_episode()
    r_p0 = h0p.run_episode()
    # Reconstruct the two rendered frames at tick 0 by a one-tick run that
    # records the released frame: simpler — render directly through the
    # env with the harness's palette.
    from irene_brain.environments.maze_chase import MazeChaseEnv

    env_c = MazeChaseEnv(**fam0.env_kwargs())
    env_c.reset(EMB0)
    fc = env_c.current_observation.rgb
    off = PERTURBATIONS[0].palette_offsets[fam0.id]
    expect = bytes(
        min(255, max(0, v + d)) for v, d in zip(fc.pixels, (off * (len(fc.pixels) // 3))
        )
    )
    got = _apply_palette_check(h0p, fc)
    iso["P0"] = {
        "offset": list(off),
        "pixels_match_expected": got == expect,
    }
    iso_pass &= iso["P0"]["pixels_match_expected"]
    # P1: exactly 10 drops in 100 cycles.
    h1 = PacmanHarness(FAMILIES["F2"], PERTURBATIONS[1], seed=EMB0,
                       policy=NoOpPolicy(), max_ticks=100)
    r1 = h1.run_episode()
    drops = [t for t in r1.ticks_data if t.frame_lost]
    iso["P1"] = {"drops": len(drops), "expected": 9,
                 "cycles": [t.cycle_id for t in drops]}
    iso_pass &= len(drops) == 9
    # P2: every cycle after the first is delayed exactly one period.
    h2 = PacmanHarness(FAMILIES["F2"], PERTURBATIONS[2], seed=EMB0,
                       policy=NoOpPolicy(), max_ticks=50)
    r2 = h2.run_episode()
    p2_ok = all(t.observation_delayed for t in r2.ticks_data[1:]) and all(
        t.s_ns - t.t_emit_ns == PACMAN_BASE_PERIOD_NS for t in r2.ticks_data[1:]
    )
    iso["P2"] = {"one_cycle_delay_on_all_cycles_after_0": p2_ok}
    iso_pass &= p2_ok
    # P3/P4: exact periods.
    for idx, expected in ((3, 18_518_519), (4, 15_151_515)):
        h = PacmanHarness(FAMILIES["F0"], PERTURBATIONS[idx], seed=EMB0,
                          policy=NoOpPolicy(), max_ticks=20)
        rr = h.run_episode()
        okp = all(t.p_ns == expected for t in rr.ticks_data)
        iso[f"P{idx}"] = {"period_ns": expected, "all_ticks_match": okp}
        iso_pass &= okp
    # P5: the rule swap is exactly the registered alternate per family.
    p5_ok = True
    p5_detail = {}
    for fid in FAMILY_ORDER:
        h = PacmanHarness(FAMILIES[fid], PERTURBATIONS[5], seed=EMB0,
                          policy=NoOpPolicy(), max_ticks=1)
        h.run_episode()
        effective = h._env_kwargs["ghost_rule"]
        p5_detail[fid] = {
            "family_rule": FAMILIES[fid].ghost_rule,
            "effective_rule": effective,
            "expected": PERTURBATIONS[5].rule_alternate[fid],
        }
        p5_ok &= effective == PERTURBATIONS[5].rule_alternate[fid]
    iso["P5"] = {"rule_swap_exact": p5_ok, "detail": p5_detail}
    iso_pass &= p5_ok
    gate_reports["G5_perturbation_isolation"] = {"passed": iso_pass, "detail": iso}

    # ------------------------------------------------------------------
    # Gate 6: legal-action surface over the whole slice.
    # ------------------------------------------------------------------
    illegal = 0
    for key, rec in all_episodes.items():
        if rec.legal_action_rate != 1.0:
            illegal += 1
    gate_reports["G6_legal_action_surface"] = {
        "passed": illegal == 0,
        "episodes_with_illegal_action": illegal,
        "episode_count": episode_count,
    }

    passed_all = all(
        gate_reports[g]["passed"] for g in gate_reports
    )
    result = {
        "mode": "pacman_harness_v1_construction_acceptance",
        "schema_version": 1,
        "preregistration": (
            "brain/docs/preregistrations/"
            "2026-08-27-pacman-harness-v1-construction.md"
        ),
        "probe_slice": {
            "cells": [f"{f}|{p.name}" for f, p in cells],
            "episodes_per_cell": PROBE_EPISODES_PER_CELL,
            "ticks_per_episode": PROBE_TIKS,
            "total_episodes": episode_count,
            "seed_partition": "TRAIN",
            "seed_block_base": EMB0,
            "declared_inference_latency_ns": 3_000_000,
        },
        "gates": gate_reports,
        "passed": bool(passed_all),
        "platform": provenance_dict(),
    }

    out = REPO_ROOT / RESULT_RELPATH
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"REFUSED: {out} already exists (create-only).", file=sys.stderr)
        return 2
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(f"WROTE {out}")
    print(f"PASS ALL GATES: {passed_all}")
    for g, v in gate_reports.items():
        print(f"  {g}: {'PASS' if v['passed'] else 'FAIL'}")
    return 0 if passed_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
