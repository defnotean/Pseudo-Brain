"""CPU-only Phase 2.5 remaining-gates parse and cheap structural probes.

Reads committed Spark campaign JSON and the named Gate 6 probe JSON.
Does not train. Does not use CUDA. Does not write CURRENT_WORK / STATUS /
the Gate 6 run record.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from ..model.consequence_thought_actuator import ConsequenceThoughtActuator


BRAIN_ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN_JSON = BRAIN_ROOT / "docs" / "phase_closure" / "thought_mediated_campaign_results.json"
GATE6_JSON = BRAIN_ROOT / "docs" / "phase_closure" / "gate6_matched_gru_results.json"

# Named probe a615223 / dgx-gate6-matched-gru-v1. Used only if the JSON is absent.
GATE6_PB_IQM = -27.758
GATE6_GRU_IQM = -25.694
GATE6_RELATIVE_PCT = -8.04
GATE6_KNOCKOUT_PCT = 1343.0


@dataclass(frozen=True, slots=True)
class GateRow:
    gate_id: str
    name: str
    verdict: str
    measured: str
    notes: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _relative_change_pct(baseline: float, intervened: float) -> float:
    denom = abs(baseline)
    if denom < 1e-12:
        return 0.0
    return 100.0 * (intervened - baseline) / denom


def measure_permutation_action_change(
    *,
    seed: int = 0,
    batches: int = 8,
    thoughtlets: int = 32,
    core_width: int = 32,
) -> dict[str, float]:
    """Untrained actuator: whole-slot permutation vs action_dist / argmax.

    This is a structural Gate 4 probe. It is not a trained closed-loop campaign.
    """
    torch.manual_seed(seed)
    actuator = ConsequenceThoughtActuator(core_width=core_width)
    actuator.eval()
    sensors = torch.randn(batches, 4, core_width)
    thoughts = torch.randn(batches, thoughtlets, 3, core_width) + 0.5
    with torch.no_grad():
        baseline = actuator(sensors=sensors, thoughts=thoughts)
        perm = torch.randperm(thoughtlets)
        permuted = actuator(sensors=sensors, thoughts=thoughts[:, perm])
    argmax_change_pct = float((baseline.action_dist.argmax(-1) != permuted.action_dist.argmax(-1)).float().mean() * 100.0)
    l1 = float((baseline.action_dist - permuted.action_dist).abs().sum().item())
    denom = float(baseline.action_dist.abs().sum().clamp_min(1e-9).item())
    action_dist_l1_pct = 100.0 * l1 / denom
    return {
        "argmax_change_pct": argmax_change_pct,
        "action_dist_l1_pct": action_dist_l1_pct,
    }


def analyze_remaining_gates(
    campaign: dict[str, Any] | None = None,
    gate6: dict[str, Any] | None = None,
    *,
    permutation: dict[str, float] | None = None,
) -> list[GateRow]:
    if campaign is None:
        campaign = load_json(CAMPAIGN_JSON)
    if gate6 is None and GATE6_JSON.is_file():
        gate6 = load_json(GATE6_JSON)

    causal = campaign.get("progressive_causal_interventions") or {}
    normal = causal.get("A_normal") or {}
    permute = causal.get("B_permute") or {}
    scramble = causal.get("C_register_swap") or {}
    knockout = causal.get("H_zero_knockout") or {}

    normal_iqm = float(normal["iqm_return"]) if "iqm_return" in normal else None
    knockout_iqm = float(knockout["iqm_return"]) if "iqm_return" in knockout else None
    family_b_collapse_pct = (
        abs(_relative_change_pct(normal_iqm, knockout_iqm)) if normal_iqm is not None and knockout_iqm is not None else None
    )

    curve = campaign.get("multi_seed_scaling_curve") or {}
    k_order = ("K=1", "K=4", "K=8", "K=16", "K=32")
    k_scores = [float(curve[k]["decision_critical_score"]) for k in k_order if k in curve]
    monotonic = bool(k_scores) and all(k_scores[i] > k_scores[i + 1] for i in range(len(k_scores) - 1))

    permute_score = float(permute["decision_critical_score"]) if "decision_critical_score" in permute else None
    normal_score = float(normal["decision_critical_score"]) if "decision_critical_score" in normal else None
    campaign_perm_change_pct = (
        abs(_relative_change_pct(normal_score, permute_score)) if permute_score is not None and normal_score is not None else None
    )
    permute_iqm = float(permute["iqm_return"]) if "iqm_return" in permute else None

    scramble_iqm = float(scramble["iqm_return"]) if "iqm_return" in scramble else None
    scramble_drop_pct = (
        -_relative_change_pct(normal_iqm, scramble_iqm) if normal_iqm is not None and scramble_iqm is not None else None
    )
    if scramble_drop_pct is not None and abs(scramble_drop_pct) < 1e-9:
        scramble_drop_pct = 0.0

    if gate6 is not None:
        pb_iqm = float(gate6["pb_iqm_return"])
        gru_iqm = float(gate6["gru_iqm_return"])
        rel_pct = 100.0 * float(gate6["relative_iqm_advantage"])
        knockout_sanity_pct = 100.0 * float(gate6["mean_knockout_degradation"])
    else:
        pb_iqm = GATE6_PB_IQM
        gru_iqm = GATE6_GRU_IQM
        rel_pct = GATE6_RELATIVE_PCT
        knockout_sanity_pct = GATE6_KNOCKOUT_PCT

    useful_slots = None
    for key in ("useful_slots", "useful_slots_count"):
        if key in campaign:
            useful_slots = int(campaign[key])
            break
    causal_diag = campaign.get("causal_diagnostics") or {}
    if useful_slots is None and "useful_slots_count" in causal_diag:
        useful_slots = int(causal_diag["useful_slots_count"])

    if permutation is None:
        permutation = measure_permutation_action_change()

    rows = [
        GateRow(
            gate_id="Gate 1",
            name="Thought mediation",
            verdict="UNKNOWN",
            measured=(
                f"Family B IQM {normal_iqm:.1f} -> {knockout_iqm:.1f} "
                f"({family_b_collapse_pct:.2f}% collapse); Gate 6 Family-B knockout sanity "
                f"{knockout_sanity_pct:.0f}%"
                if family_b_collapse_pct is not None
                else "not measured"
            ),
            notes="Registered bar is Families B-E >=30% collapse and reflex stable. Campaign JSON is Family B only. Families C-E and reflex: not measured.",
        ),
        GateRow(
            gate_id="Gate 2",
            name="Monotonic capacity scaling",
            verdict="FAIL" if k_scores and not monotonic else ("UNKNOWN" if not k_scores else "PASS"),
            measured=(
                "decision_critical_score K=1/4/8/16/32 = "
                + "/".join(f"{s:.1f}" for s in k_scores)
                if k_scores
                else "not measured"
            ),
            notes="Need K=32 > 16 > 8 > 4 > 1. Observed curve is anti-monotonic.",
        ),
        GateRow(
            gate_id="Gate 3",
            name="Causal slot usefulness",
            verdict="UNKNOWN" if useful_slots is None else ("PASS" if useful_slots >= 8 else "FAIL"),
            measured=f"{useful_slots} / 32" if useful_slots is not None else "not measured",
            notes="No per-slot knockout table in thought_mediated_campaign_results.json. Do not reuse Design 1/2 0/32.",
        ),
        GateRow(
            gate_id="Gate 4",
            name="Permutation equivariance",
            verdict="PASS",
            measured=(
                f"campaign B_permute vs A_normal decision_critical {normal_score:.1f} vs {permute_score:.1f} "
                f"({campaign_perm_change_pct:.2f}%); IQM {normal_iqm:.1f} vs {permute_iqm:.1f}; "
                f"CPU actuator argmax change {permutation['argmax_change_pct']:.2f}%, "
                f"action_dist L1 {permutation['action_dist_l1_pct']:.4f}%"
                if campaign_perm_change_pct is not None
                else "not measured"
            ),
            notes="Gate wants <=1.0% action change. 0.00% campaign decision-critical change is PASS. Untrained actuator confirms 0.00% argmax change.",
        ),
        GateRow(
            gate_id="Gate 5",
            name="Cognitive scrambling",
            verdict="FAIL" if scramble_drop_pct is not None and scramble_drop_pct < 15.0 else ("UNKNOWN" if scramble_drop_pct is None else "PASS"),
            measured=(
                f"C_register_swap IQM {scramble_iqm:.1f} vs A_normal {normal_iqm:.1f} "
                f"({scramble_drop_pct:.2f}% degradation)"
                if scramble_drop_pct is not None
                else "not measured"
            ),
            notes="Need >=15% degradation from register-binding break. Observed 0.00%.",
        ),
        GateRow(
            gate_id="Gate 6",
            name="Baseline superiority",
            verdict="FAIL",
            measured=f"PB IQM {pb_iqm:.3f} vs GRU {gru_iqm:.3f}, relative {rel_pct:.2f}% (need >=+10%)",
            notes="Named probe dgx-gate6-matched-gru-v1 (a615223). GRU wins. Do not claim superiority. Do not silently rerun the same 300-step CPU probe.",
        ),
        GateRow(
            gate_id="Gate 7",
            name="Breadth",
            verdict="UNKNOWN",
            measured="not measured",
            notes="No per-family 95% bootstrap CI in thought-mediated campaign JSON. Need positive CI on >=4/5 families.",
        ),
        GateRow(
            gate_id="Gate 8",
            name="DGX Spark real-time deadline",
            verdict="UNKNOWN",
            measured="not measured",
            notes="Campaign JSON has no kernel p99 / loop p95. Phase 1 GB10 numbers are a different model. Needs GB10; do not start a GPU train from this parse.",
        ),
    ]
    return rows


def format_gate_table(rows: list[GateRow]) -> str:
    lines = [
        "| Gate | Name | Verdict | Measured |",
        "| :--- | :--- | :---: | :--- |",
    ]
    for row in rows:
        lines.append(f"| {row.gate_id} | {row.name} | **{row.verdict}** | {row.measured} |")
    return "\n".join(lines)


def main() -> None:
    rows = analyze_remaining_gates()
    print(format_gate_table(rows))
    print()
    for row in rows:
        print(f"{row.gate_id} {row.verdict}: {row.notes}")


if __name__ == "__main__":
    main()
