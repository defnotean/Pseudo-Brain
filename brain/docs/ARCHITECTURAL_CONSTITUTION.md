# Pseudo-Brain Architectural Constitution

**Status:** canonical invariants, v1 (2026-08-22). Every item encodes a lesson
from a real Phase 2 / Phase 2.5 failure. Future code changes must keep these
green; violations invalidate results and require a [DIAGNOSTIC ONLY] rerun note.

Each invariant lists: the contract, why it exists (the historical failure), and
how it is tested today.

| # | Invariant | Contract | Historical failure it prevents | Current test |
|---|---|---|---|---|
| C1 | **Whole-slot permutation invariance** | Permuting thought state + slot identity + associated metadata together must leave outputs ~unchanged (float32 noise). | Any slot-index semantics would make "which thought is #3" meaningful and destroy exchangeability. | Permutation ablation in every campaign; repeatedly ≈0 diff. |
| C2 | **No belief→action bypass** | Main action intent flows through thought-mediated proposals only. | Original unmediated design let thoughtlets be bypassed → thesis falsified (Phase 2 original). | Architecture review + mediation/knockout sanity gates. |
| C3 | **Reflex bounded** | Reflex correction stays within its small bound and must not flip discrete cognitive decisions. | A strong reflex path would quietly replace cognition. | Bounded-reflex unit tests; reflex-influence audits. |
| C4 | **No multiplicity vote-stuffing** | Duplicate/identical hypotheses must not gain utility through raw count. | RIGHT with 8 mediocre slots beating LEFT with 1 excellent slot. | Multiplicity-proof aggregator tests; duplicate-action invariance checks. |
| C5 | **Probability↔consequence binding is meaningful** | Branch probabilities attach to their own predicted outcomes; scrambling binding must degrade behavior on tasks needing it. | Mean-pooling register commutativity destroyed hypothesis identity. | Binding-scramble intervention in hostile batteries. |
| C6 | **Persistence is causal where required** | On tasks that need persistent state, reset-every-frame must collapse performance; stale-state injection must degrade gracefully with D. | Reactive cheating could score without memory. | Reset/stale interventions vs GRU-reset control. |
| C7 | **Inactive slots earn nothing from bias** | Zero/inactive thoughts must not generate useful policy via proposal-head bias alone. | Proposal-bias shortcut discovered in earlier phases. | Zero-content forced-active probes. |
| C8 | **No future leakage** | Simulator state, RNG, or future labels never enter inference inputs. | Any leakage invalidates all results. | Leakage audits; target-blind registration discipline. |
| C9 | **One-brain evaluation** | One checkpoint across all tasks/games of a campaign; no per-game adapters, heads, planners, or model swaps. | Specialization would trivially win benchmarks while proving nothing general. | Compliance tests; manifest identity checks. |
| C10 | **Resource honesty** | Claims of "matched" must report parameters AND approximate FLOPs AND measured latency. | Parameter-matched-but-3×-FLOPs comparisons are fake matches. | Resource tables in run records (e.g. GRU 908k @ 0.62 ms vs PB K=32 825k @ 1.11 ms). |
| C11 | **Training-seed honesty** | Independent training seeds are reported separately; best-checkpoint-only reporting is forbidden. | Checkpoint oscillation previously masked instability. | per-seed dumps + trajectory telemetry (`escalation_mechanism_discovery.py`). |
| C12 | **Negative results preserved** | Failed preregistrations close cleanly; no post-hoc retuning; no reopening closed branches. | Legacy ranked-K8 branch produced valid negatives that must stay archived. | CURRENT_WORK legacy blocks; git history. |
| C13 | **Claim labels** | Every finding carries [MEASURED] / [INFERRED] / [HYPOTHESIS] / [ASPIRATIONAL]. | Blurring aspiration into evidence destroys credibility. | Documentation review at each commit. |
| C14 | **Stale-document defense** | Agents reconstruct state from HEAD + CURRENT_WORK + newest logs before choosing work. | A prior agent followed an obsolete preregistration and time-traveled backward. | Agent startup sequence in MASTER_ROADMAP.md §7. |

## Amendment procedure

A new invariant requires: the failure that motivates it, the smallest test
that would have caught it, and a commit adding both. Invariants are never
removed; if obsolete, they are marked superseded with reasons.
