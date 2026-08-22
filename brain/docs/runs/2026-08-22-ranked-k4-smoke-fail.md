# Ranked K=4 unmatched-suppress smoke — FAIL (local CPU, 2026-08-22)

**Probe:** `ranked-k8-unmatched-suppress-v1` follow-up, K=4 with the same losses (the
preregistration's prescribed "distinct second idea" after the K=8 FAIL).

**Run:** local CPU (DGX Spark hosting Qwen on GB10 — deliberate override), seeds {43, 45},
60 steps, 3 episodes, K=4, W=60 (831,741 params, +1.42% vs matched GRU 820,075).

## Results

| Seed | PB IQM | GRU IQM | PB-GRU diff | Family-B normal → knockout | %degr. |
|---|---|---|---|---|---|
| 43 | −168.875 | −30.750 | −138.125 | −443.0 → −37.5 | −91.5% |
| 45 | −147.500 | −21.625 | −125.875 | −626.0 → −42.0 | −93.3% |

- Pooled: PB IQM **−141.667** vs GRU IQM **−24.467** → PB **−479%** advantage (far below the
  ≥10% preregistered margin).
- Mean knockout degradation: **−92.4%** (median of the two per-seed degradations) — the
  knockout *improves* Family B on both seeds, so the ≥30% mediation sanity gate fails.

## Verdict

**STATUS = FAIL.** Proposal-GRU wins or ties; no architecture-superiority claim.

## Disposition (per preregistration)

- Smoke did not put PB ahead → `--full` is not launched (protocol: full only if smoke PB
  ahead).
- This closes the legacy `ranked-k8-unmatched-suppress-v1` branch: K=8 FAIL (IQM +1.11% < 10%),
  K=4 FAIL (IQM −479%). Both recorded; the branch is archived.
- Results: `C:\Users\Demon\AppData\Local\Temp\ranked_k4_smoke_results.json`.
- Behavioral results only — no GB10 latency comparison (Spark is hosting Qwen).
