# Phase 2.5: reliability test — statistical analysis of per-seed returns (2026-08-22)

**Data:** `runs/reliability10-dump-v1/per_seed_returns.json` (10 independent training
seeds × {GRU, K=32}, 3000 steps, 8-hypothesis escalation task, GB10).
Analysis script run locally (pure arithmetic on dumped returns; no training on CPU).

## Per-seed returns [MEASURED]

| Rank | GRU | PB K=32 |
|---|---|---|
| best | −43.0 | −20.7 |
| 2 | −63.9 | −65.6 |
| 3 | −482.8 | −93.0 |
| 4 | −518.2 | −315.95 |
| 5 | −521.8 | −389.05 |
| 6 | −532.9 | −455.35 |
| 7 | −535.0 | −475.7 |
| 8 | −535.0 | −487.35 |
| 9 | −535.0 | −508.45 |
| worst | −535.0 | −535.0 |

## Statistical tests

- **Catastrophic-seed count** (return ≤ −520, i.e. optimization collapse to floor):
  GRU **6/10**, K=32 **1/10**. Fisher exact one-sided p = **0.0274**.
- **Mann-Whitney U**: U_pb = 74 (max 100), permutation test (100k draws, seed 7,
  tie-corrected average ranks): one-sided p = **0.0363** favoring PB.
- Medians: GRU −527.35 vs K=32 −422.20.

## Interpretation [LABELS]

[MEASURED] On this task, at this budget, PB K=32's training-seed failure distribution
is significantly better than the matched GRU's: fewer catastrophic collapses (1/10 vs
6/10, p≈0.027) and stochastically larger per-seed returns (p≈0.036).

[INFERRED] The architectural advantage, when it exists here, is in **training
reliability** — resistance to optimization collapse — rather than in best-case return
(both architectures' top seeds reach similar peaks, −43 vs −21). Parallel persistent
hypothesis slots appear to keep learning alive where a monolithic state falls into the
catastrophic attractor.

## Required caveats before calling it an architecture result

1. Single task (8-hyp escalation), single budget (3000 steps), n=10. Needs replication:
   fresh 10 seeds, and the same reliability framing applied to the ephemeral-memory task
   (where GRU currently wins mean return — if the reliability advantage appears there too,
   the claim generalizes; if not, it is task-specific).
2. Derived scripts used (dump + reduced configs); base science code untouched and
   md5-matched earlier. Derivations documented inline.
3. Resource accounting (params/FLOPs/GB10 latency) for K=32 vs GRU not yet re-measured
   this session.

## Next actions

1. Ephemeral-memory reliability run (n=10, GRU vs K=32, per-seed dump).
2. Latency/params re-measurement table for the report.
3. If both hold up: persistence causal battery on surviving checkpoints → Phase 2 closure memo.
