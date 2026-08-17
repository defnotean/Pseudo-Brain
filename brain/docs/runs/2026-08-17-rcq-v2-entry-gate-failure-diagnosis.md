# RCQ-v2 entry-gate failure diagnosis (2026-08-17)

Status: post-mortem analysis of the terminally failed `rcq_v2_reference_v2`
candidate. This document changes nothing about the outcome. The frozen
`rcq_v2_development_v1` gate at optimizer step 1,536 returned
`passed: false`; the candidate is closed, may not be resumed, tuned, or
retried under this registration, and sealed TEST stays untouched. Everything
below uses only the already-open development slice and the checkpoint-bound
run artifacts. It is not new evidence and not a new claim.

Terminal record:
[2026-08-16-rcq-v2-reference-v2-smoke-canary-train.md](./2026-08-16-rcq-v2-reference-v2-smoke-canary-train.md).

## Inputs and their identities

| Artifact | SHA-256 |
|---|---|
| `development-gate-step-00001536.json` | `6e1bcdec69949e1b1e43f271ebf300f07a39e25318790aa82de99c5b641e7724` |
| `metrics.jsonl` (31 rows) | `10d7faa18912fd89bef2ecab1622245247bd550acf863997755be6d83a96302e` |
| Terminal checkpoint `step-00001536.pt` | `56cbf6d34325582ca4d68a3b6620193effe37825cb59f67f057f9c656c770fb2` |

Local analysis copies were sha256-verified byte-identical to the remote
artifacts before any number below was computed. Analysis ran CPU-only on the
workstation (`brain/artifacts/rcq-v2-entry-gate-analysis/`, git-ignored).

## What failed

Two of ten frozen checks failed at step 1,536:

1. `opposite_conflicts`: 9 observed, limit 7 (9/1,536 = 0.0059 per decision
   vs the < 0.005 bound).
2. `continuous_outputs_in_deadzone`: 875 observed, limit 0 (875/16,896 =
   5.18% of continuous outputs, 0.57 per decision, exceeded |0.05|).

Every action-volume check passed with margin: movement exact 1,422/1,536
(0.926 vs 0.80), changed exact 402/467 (0.861 vs 0.50), recall 0.976 vs 0.90,
64 movement false positives vs 230 allowed, and zero violations on all four
off-support/target-zero checks.

## Trend across checkpoints (open development slice, logger rows)

| Step | Move exact | Recall | Move FP/dec | Opp. conflicts | Deadzone outs/dec | Value loss |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.247 | 0.330 | 0.009 | 0 | 7.57 | 0.388 |
| 512 | 0.721 | 0.860 | 0.066 | 14 | 3.70 | 0.379 |
| 768 | 0.842 | 0.928 | 0.034 | 12 | 3.22 | 0.385 |
| 1,024 | 0.842 | 0.988 | 0.156 | 62 | 3.71 | 0.368 |
| 1,280 | 0.900 | 0.954 | 0.019 | 1 | 7.08 | 0.370 |
| 1,536 | 0.926 | 0.976 | 0.042 | 9 | 0.57 | 0.293 |

(Opposite-conflict and deadzone columns are derived from the logged
per-decision rates multiplied by 1,536 decisions.)

## Mechanics of the two failures

**Quiescence (875 outputs).** All continuous targets are exactly zero on this
task (`continuous_target_nonzero_count = 0`), and the head's mean squared
magnitude is tiny (0.0069), yet a long tail — about one output in nineteen —
still lands outside ±0.05. The per-decision violation count was noisy all
run (7.6 → 3.2 → 7.1 → 0.57) and only collapsed in the final interval. The
gate requires exactly zero; a heavy-tailed continuous head trained with no
explicit deadzone penalty has no mechanism to guarantee that.

**Opposite conflicts (9).** Per-key development rates show the residual
conflicts concentrate in the W/S axis: W is over-predicted (+0.022 predicted
minus true positive rate) and S is over-predicted (+0.017), while A and D are
essentially calibrated (+0.003, +0.000). With positive logits near +6.6 and
inactive logits near −6.1, conflicts are rare borderline co-activations of
opposing keys, not a calibration collapse — key accuracy is 0.9997. The count
was non-monotonic (0 → 14 → 12 → 62 → 1 → 9), so the step-1,536 value is
partly luck of the stopping point in both directions: it was 1 at step 1,280.

## Context that does not change the result

- Movement exact was still improving at the stop (0.900 → 0.926) and sat
  0.23 above the copy-previous-W/A/S/D reference (flat 0.696), so the policy
  was genuinely state-conditioned on the open slice.
- Entry `value_loss` 0.2928 was already below the stage-2 absolute ceiling
  (0.3281), but that gate never ran and the entry report's value loss has no
  stage-1 pass/fail role.
- Per-exit movement exact was uniform (0.923–0.927 across exits 0–3); no
  single exit carried the failure.
- "One more interval might have cleaned it" is not actionable: the recipe,
  gate step, and thresholds are frozen, and extending the run would have been
  a protocol violation.

## What a future experiment may take from this

Any follow-up is a newly named, newly preregistered qualification with its
own release and pin — never a resume of this candidate. Observations worth
carrying into that preregistration discussion:

1. The frozen zero-tolerance quiescence check interacts badly with a
   continuous head whose targets are identically zero and whose loss does not
   penalize the deadzone boundary; a future recipe would need to address that
   explicitly at design time, not after a failure.
2. Opposite-key co-activation on W/S was the marginal action failure mode at
   this scale (9/1,536 against a limit of 7).
3. Development action metrics above every volume floor were not predictive of
   the two hard-zero checks; both failed checks were visible in the logger
   rows (deadzone outs/dec, conflict rate) well before the gate.

None of these observations justify reopening, extending, or warm-starting
this run. Sealed TEST ranges remain unopened; `sealed_test_examples_opened`
stays 0.
