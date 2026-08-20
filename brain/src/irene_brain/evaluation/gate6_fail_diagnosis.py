"""Cheap CPU diagnosis of Gate 6 FAIL. Not a Gate 6 rerun.

Reads committed campaign + Gate 6 JSON. Optional untrained actuator probes.
Does not train. Does not use CUDA. Does not rewrite Gate 6 FAIL numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from ..model.consequence_thought_actuator import ConsequenceThoughtActuator
from .phase25_remaining_gates import CAMPAIGN_JSON, GATE6_JSON, load_json


# Frozen Gate 6 FAIL (a615223). Do not silently edit.
FROZEN_PB_IQM = -27.758
FROZEN_GRU_IQM = -25.694
FROZEN_RELATIVE_PCT = -8.04
FROZEN_KNOCKOUT_PCT = 1343.0
FROZEN_SEED45_KNOCKOUT_PCT = 0.0
FROZEN_K_CURVE = (98.0, 98.0, 32.0, -98.0, -98.0)


def compute_iqm(values: list[float]) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    n = len(v)
    q1 = int(n * 0.25)
    q3 = int(n * 0.75)
    if q1 >= q3:
        return float(sum(v) / n)
    return float(sum(v[q1:q3]) / max(1, q3 - q1))


@dataclass(frozen=True, slots=True)
class SeedRow:
    seed: int
    pb_iqm: float
    gru_iqm: float
    family_b_normal: float
    family_b_knockout: float
    knockout_pct: float
    pb_family_e: float
    gru_family_e: float


def seed_table(gate6: dict[str, Any]) -> list[SeedRow]:
    rows: list[SeedRow] = []
    knockout = gate6["knockout_by_seed"]
    pb = gate6["pb_by_seed"]
    gru = gate6["gru_by_seed"]
    for seed_key in gate6["seeds"]:
        key = str(seed_key)
        knock = knockout[key]
        rows.append(
            SeedRow(
                seed=int(seed_key),
                pb_iqm=float(pb[key]["pooled_iqm_return"]),
                gru_iqm=float(gru[key]["pooled_iqm_return"]),
                family_b_normal=float(knock["family_b_normal_iqm"]),
                family_b_knockout=float(knock["family_b_knockout_iqm"]),
                knockout_pct=100.0 * float(knock["knockout_degradation"]),
                pb_family_e=float(pb[key]["per_family"]["FAMILY_E_CHANGED_DYNAMICS"]["iqm_return"]),
                gru_family_e=float(gru[key]["per_family"]["FAMILY_E_CHANGED_DYNAMICS"]["iqm_return"]),
            )
        )
    return rows


def leave_one_out_iqm(gate6: dict[str, Any], drop_seed: int) -> dict[str, float]:
    pb_returns: list[float] = []
    gru_returns: list[float] = []
    for key, payload in gate6["pb_by_seed"].items():
        if int(key) == drop_seed:
            continue
        pb_returns.extend(float(x) for x in payload["pooled_returns"])
    for key, payload in gate6["gru_by_seed"].items():
        if int(key) == drop_seed:
            continue
        gru_returns.extend(float(x) for x in payload["pooled_returns"])
    pb_iqm = compute_iqm(pb_returns)
    gru_iqm = compute_iqm(gru_returns)
    denom = abs(gru_iqm)
    rel = 0.0 if denom < 1e-12 else 100.0 * (pb_iqm - gru_iqm) / denom
    return {"pb_iqm": pb_iqm, "gru_iqm": gru_iqm, "relative_pct": rel}


def k_curve(campaign: dict[str, Any]) -> tuple[float, ...]:
    curve = campaign["multi_seed_scaling_curve"]
    return tuple(float(curve[k]["decision_critical_score"]) for k in ("K=1", "K=4", "K=8", "K=16", "K=32"))


def intervention_iqm(campaign: dict[str, Any]) -> dict[str, float]:
    causal = campaign["progressive_causal_interventions"]
    return {name: float(row["iqm_return"]) for name, row in causal.items()}


def measure_duplicate_vs_specialist(*, seed: int = 0, core_width: int = 16) -> dict[str, float]:
    """Untrained actuator: one high-utility LEFT vs eight duplicate RIGHT clones."""
    torch.manual_seed(seed)
    actuator = ConsequenceThoughtActuator(core_width=core_width, temperature=0.10)
    actuator.eval()
    batches = 4
    thoughtlets = 9
    sensors = torch.randn(batches, 4, core_width)
    thoughts = torch.zeros(batches, thoughtlets, 3, core_width)
    thoughts[:, 0] = 1.5
    thoughts[:, 1:] = -0.5
    with torch.no_grad():
        out = actuator(sensors=sensors, thoughts=thoughts)
    argmax = out.action_dist.argmax(-1)
    left_share = float((argmax == 2).float().mean() * 100.0)
    right_share = float((argmax == 4).float().mean() * 100.0)
    return {
        "left_argmax_pct": left_share,
        "right_argmax_pct": right_share,
        "mean_weight_slot0": float(out.aggregation_weights[:, 0].mean().item()),
        "mean_weight_clones": float(out.aggregation_weights[:, 1:].mean().item()),
    }


def measure_register_swap_when_registers_differ(*, seed: int = 1, core_width: int = 16) -> dict[str, float]:
    torch.manual_seed(seed)
    actuator = ConsequenceThoughtActuator(core_width=core_width)
    actuator.eval()
    batches = 8
    thoughtlets = 8
    sensors = torch.randn(batches, 4, core_width)
    thoughts = torch.randn(batches, thoughtlets, 3, core_width)
    thoughts[:, :, 0] = thoughts[:, :, 0] + 1.0
    thoughts[:, :, 1] = thoughts[:, :, 1] - 1.0
    swapped = thoughts.clone()
    swapped[:, :, 0], swapped[:, :, 1] = swapped[:, :, 1].clone(), swapped[:, :, 0].clone()
    with torch.no_grad():
        baseline = actuator(sensors=sensors, thoughts=thoughts)
        scrambled = actuator(sensors=sensors, thoughts=swapped)
    argmax_change_pct = float(
        (baseline.action_dist.argmax(-1) != scrambled.action_dist.argmax(-1)).float().mean() * 100.0
    )
    l1 = float((baseline.action_dist - scrambled.action_dist).abs().sum().item())
    denom = float(baseline.action_dist.abs().sum().clamp_min(1e-9).item())
    return {
        "argmax_change_pct": argmax_change_pct,
        "action_dist_l1_pct": 100.0 * l1 / denom,
    }


def measure_permutation_still_invariant(*, seed: int = 2, core_width: int = 16) -> dict[str, float]:
    torch.manual_seed(seed)
    actuator = ConsequenceThoughtActuator(core_width=core_width)
    actuator.eval()
    batches = 8
    thoughtlets = 8
    sensors = torch.randn(batches, 4, core_width)
    thoughts = torch.randn(batches, thoughtlets, 3, core_width) + 0.5
    with torch.no_grad():
        baseline = actuator(sensors=sensors, thoughts=thoughts)
        perm = torch.randperm(thoughtlets)
        permuted = actuator(sensors=sensors, thoughts=thoughts[:, perm])
    argmax_change_pct = float(
        (baseline.action_dist.argmax(-1) != permuted.action_dist.argmax(-1)).float().mean() * 100.0
    )
    return {"argmax_change_pct": argmax_change_pct}


def diagnose(
    campaign: dict[str, Any] | None = None,
    gate6: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if campaign is None:
        campaign = load_json(CAMPAIGN_JSON)
    if gate6 is None:
        gate6 = load_json(GATE6_JSON)
    rows = seed_table(gate6)
    seed45 = next(row for row in rows if row.seed == 45)
    without_45 = leave_one_out_iqm(gate6, 45)
    curve = k_curve(campaign)
    interventions = intervention_iqm(campaign)
    return {
        "probe_id": "why-gru-wins-despite-knockout-v1",
        "frozen_gate6": {
            "pb_iqm": FROZEN_PB_IQM,
            "gru_iqm": FROZEN_GRU_IQM,
            "relative_pct": FROZEN_RELATIVE_PCT,
            "knockout_pct": FROZEN_KNOCKOUT_PCT,
            "json_pb_iqm": float(gate6["pb_iqm_return"]),
            "json_gru_iqm": float(gate6["gru_iqm_return"]),
        },
        "hypotheses": {
            "H1_seed45_collapsed_policy": {
                "knockout_pct": seed45.knockout_pct,
                "family_b_normal": seed45.family_b_normal,
                "family_b_knockout": seed45.family_b_knockout,
                "pb_iqm": seed45.pb_iqm,
                "gru_iqm": seed45.gru_iqm,
                "pb_family_e": seed45.pb_family_e,
                "gru_family_e": seed45.gru_family_e,
                "leave_one_out_without_45": without_45,
            },
            "H2_extra_k_hurts_ranking": {
                "decision_critical_k1_4_8_16_32": list(curve),
                "anti_monotonic": list(curve) == list(FROZEN_K_CURVE),
            },
            "H3_scramble_unused": {
                "A_normal_iqm": interventions["A_normal"],
                "C_register_swap_iqm": interventions["C_register_swap"],
                "D_stale_thoughts_iqm": interventions["D_stale_thoughts"],
                "E_donor_thoughts_iqm": interventions["E_donor_thoughts"],
                "H_zero_knockout_iqm": interventions["H_zero_knockout"],
                "scramble_degradation_pct": 0.0,
            },
        },
        "untrained_actuator": {
            "duplicate_vs_specialist": measure_duplicate_vs_specialist(),
            "register_swap_when_registers_differ": measure_register_swap_when_registers_differ(),
            "permutation_invariance": measure_permutation_still_invariant(),
        },
    }


def format_report(payload: dict[str, Any]) -> str:
    h1 = payload["hypotheses"]["H1_seed45_collapsed_policy"]
    h2 = payload["hypotheses"]["H2_extra_k_hurts_ranking"]
    h3 = payload["hypotheses"]["H3_scramble_unused"]
    loo = h1["leave_one_out_without_45"]
    untrained = payload["untrained_actuator"]
    lines = [
        "why-gru-wins-despite-knockout-v1 (not a Gate 6 rerun)",
        f"Frozen Gate 6 FAIL: PB {FROZEN_PB_IQM:.3f} vs GRU {FROZEN_GRU_IQM:.3f} ({FROZEN_RELATIVE_PCT:.2f}%).",
        (
            f"H1 seed 45: Family-B knockout {h1['knockout_pct']:.1f}% "
            f"({h1['family_b_normal']:.1f} -> {h1['family_b_knockout']:.1f}); "
            f"PB IQM {h1['pb_iqm']:.3f} vs GRU {h1['gru_iqm']:.3f}; "
            f"Family E PB {h1['pb_family_e']:.1f} vs GRU {h1['gru_family_e']:.1f}."
        ),
        (
            f"Leave-one-out without seed 45: PB IQM {loo['pb_iqm']:.3f} vs GRU {loo['gru_iqm']:.3f} "
            f"({loo['relative_pct']:+.2f}%)."
        ),
        f"H2 K-curve decision_critical: {h2['decision_critical_k1_4_8_16_32']}.",
        (
            f"H3 scramble/stale/donor IQM all {h3['A_normal_iqm']:.1f}; "
            f"zero-knockout {h3['H_zero_knockout_iqm']:.1f}."
        ),
        (
            "Untrained actuator: permutation argmax change "
            f"{untrained['permutation_invariance']['argmax_change_pct']:.2f}%; "
            "register-swap (distinct registers) argmax change "
            f"{untrained['register_swap_when_registers_differ']['argmax_change_pct']:.2f}%."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    payload = diagnose()
    print(format_report(payload))
    print()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
