# Named follow-up: ranked K=8 unmatched-suppress vs matched Proposal-GRU

**Date**: 2026-08-20  
**Probe ID**: `ranked-k8-unmatched-suppress-v1`  
**Parent FAIL**: `dgx-gate6-matched-gru-v1` (`a615223`) — do not rewrite those numbers  
**Parent diagnosis**: [`2026-08-20-gate6-fail-diagnosis.md`](../runs/2026-08-20-gate6-fail-diagnosis.md)  
**Harness**: `brain/scripts/dgx_run_ranked_k8_unmatched_suppress_probe.py`  
**Status**: preregistered; smoke first

This is a newly named architectural/training change. It is not a retune of the
failed K=32 300-step probe. It is not RCQ. GB10 must not be stolen from Irene
sglang; Spark **CPU** docker is the default.

## Why this design (from the diagnosis)

1. Gate 2 is anti-monotonic: extra slots past K=8 invert decision-critical
   score. Gate 6 used K=32, the worst point.
2. Unmatched Hungarian leftovers still voted. Ranking did not suppress them.
3. Seed 45 knockout was 0% — thoughts can go inert with no collapse penalty.

## Exact pair

| Side | Constructor | K / hidden | Width | Cycles |
| :--- | :--- | ---: | :--- | ---: |
| Thought-mediated | `ThoughtMediatedBrainModel` | **K=8** | searched to GRU ±6% | 3 |
| Proposal-GRU | `ProposalGRUBaseline(hidden_dim=180)` | 1 | 180 | — |

Invariants kept: no belief→action bypass, shared proposal head, bounded reflex.

Training extras vs the FAIL probe:

- Unmatched-slot branch_prob/utility suppression in `MultiHypothesisBranchLoss`
- Multiplicity-proof Q(a) in `ConsequenceProposalAggregator`
- Collapse guard (active-fraction floor) and register-0/1 cosine penalty
- Thoughts passed into the branch loss

## Smoke then full

**Smoke** (`--smoke`, default): seeds `{43, 45}` (the only non-tie Gate 6 seeds),
**60** steps, **3** episodes/family. Documented as a smoke, not a Gate 6 pass.

**Full** (`--full`): seeds `{42, 43, 44, 45, 46}`, **300** steps, **5**
episodes/family — the Gate 6 envelope. Launch full **only if smoke IQM has PB
ahead of GRU**.

Pass on `--full` only if all of:

1. `(iqm_pb - iqm_gru) / abs(iqm_gru) ≥ 0.10`
2. Mean Family-B knockout degradation ≥ 30%
3. GRU does not tie or win IQM

If GRU still wins: record FAIL for this named design and move to a **distinct**
second idea (K=4 with the same losses). Do not retune this run.

Architecture superiority is claimed only on a full PASS.
