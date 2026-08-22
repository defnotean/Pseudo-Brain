# Phase 2.6 Workstream G: Memory v1 — preregistered battery result (2026-08-22)

**Protocol:** frozen prereg `2026-08-22-memory-v1-prereg.md`. 16k steps
(12k t17/t18-only + 4k with fillers), write-gate sparsity pressure 0.05,
injection-gain sweep, generalization tests. Data: Spark `runs/memory_v1_results.json`.

## Battery [MEASURED]

| Condition | t17 | t18 | Prereg bar |
|---|---|---|---|
| normal | **0.150** | **0.200** | ≥0.50 / ≥0.65 |
| store disabled | 0.125 | 0.275 | Δ≥0.15 on t17 |
| unread | 0.125 (=disabled ✓) | 0.275 (=disabled ✓) | within 0.05 of #2 |
| G1 unseen seeds | 0.200 | — | — |
| G2 long delay | 0.175 | — | — |

Injection sweep (t17): correct/donor by gain —
0.0: .125/.125 · 0.25: .225/.325 · 0.5: .15/.45 · 1.0: .25/.175 · 2.0: .025/.00

Telemetry: gate still saturated (460k/476k writes committed ≈ 97%).

## Verdict against the preregistered criteria

**FAIL — cleanly and definitively.**

- Condition 1 (normal succeeds): failed by a wide margin (0.15 vs required 0.50).
- Conditions 2–3 passed structurally again (unread = disabled), confirming the
  read path is real but weak.
- Injection sweep is incoherent: donor sometimes *beats* correct (gain 0.5:
  .15 vs .45), and gain 2.0 collapses everything to zero. The decision head is
  not reading memory content in a stable, interpretable way.
- Generalization sits near chance.

## Per the preregistered failure condition

> "If after focused curriculum + 16k steps the battery remains near v0 levels,
> memory v0/v1 records FAIL and the workstream moves on."

**Memory v0/v1: FAIL recorded. Moving to Dynamic-K/lifecycle as directed.**
No further tuning. The store implementation and both experiment scripts remain
in-tree for a later revisit once the BrainCell itself is stronger — the owner's
directive anticipated exactly this possibility ("the memory interface may need to
be revisited later once the BrainCell itself is stronger").

## What was learned (worth keeping)

1. The read path is real (unread = disabled; normal > disabled in v0) — the failure
   is in *learning to use* retrieval, not in plumbing.
2. Write-gate sparsity pressure alone did not induce selectivity (97% open).
   Selection likely requires task pressure that this curriculum doesn't create.
3. Injection gains behave non-monotonically — evidence the persistent-channel
   injection interface (additive bias on slot state) is not a clean route for
   retrieved content in the current core. This feeds the BrainCell workstream:
   a proper *read* interface for memory into cognition may need to be part of
   the Core V1 contract rather than an external add-on.
