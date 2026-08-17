# Stage-A constant-schedule control record

The registered 500-step constant-after-warmup control completed normally and
validly failed the frozen Stage-A continuation gate. It improved action
learning substantially over the equal-FLOP cosine control, but it did not meet
the preregistered policy or value criteria. The 1,000-step bootstrap and causal
thought ablation therefore remain blocked for this checkpoint.

## Immutable identity

- Release: `r20260816t203009z-3eef763b36ed`
- Release archive SHA-256:
  `3eef763b36ed1bd841d946ba1fb6c9f605237dac9d7071d0a6f969bb557ef9cf`
- Container image ID:
  `sha256:177a406d7cb2a11338bcd8c67ab7590b330799cdc5a3193ac0ca40728ea2501b`
- Run: `stagea-constant-r20260816t203009z`
- Config: `dgx-stagea-continuation-gate-b.toml`
- Raw config SHA-256:
  `f013e3e14dfb854da1131aa726dd141bbe22e5766db8a48ed7cdb1aff6b9a6bf`
- Canonical config SHA-256:
  `8e8e4cc12123f55ee22aace1d65517dbb7586465bbf704c77cc7fe45bb3fc1f9`
- Final checkpoint SHA-256:
  `fcf4e69c359d2570ee7a6467852ba32f7a4c6addfbf8722694fc3b6a939d24a9`

The run used the same model, seed, data order, optimizer budget, objective,
precision, determinism policy, and resource bounds as the cosine control. Its
only substantive training change was a 20-step warmup followed by a constant
`1e-4` learning rate. Applied-learning-rate telemetry was corrected before
this release and records the rate used by each update.

## Final registered gate

| Check | Observed | Requirement | Result |
| --- | ---: | ---: | --- |
| Exact movement sets | 72/96 | at least 80/96 | fail |
| Exact changed decisions | 11/26 | at least 13/26 | fail |
| Positive-key recall | 0.880208 | at least 0.90 | fail |
| Movement false positives | 8/96 | at most 12/96 | pass |
| Predicted active keys | 1.291667 | within 0.15 of 137/96 | pass |
| Opposite-direction conflicts | 0/96 | 0/96 | pass |
| Non-movement false positives | 0/96 | 0/96 | pass |
| Summary/register rank proxies | 0.804100 / 0.803318 | at least 0.125 each | pass |
| Effective movement-query slots | 29.4051 | at least 4 | pass |
| Value loss | 0.413696 | below 0.35806523 | fail |
| Validation world loss | 0.015177 | both registered sanity checks | pass |

The evaluator returned exit status 1: a valid scientific failure. Artifact
identity and the fixed 96-decision slice passed every structural check.

## Equal-compute interpretation

At step 500 the cosine control reached 51/96 exact decisions, 6/26 changed
decisions, 0.645833 positive-key recall, 22 movement false positives, and one
opposite-direction conflict. The constant control reached 72/96, 11/26,
0.880208, eight false positives, and no conflicts. This supports the narrower
claim that late cosine decay constrained action learning in the earlier run.

It does not establish general control, causal thought use, or a superior
architecture. The fixed validation slice has now informed schedule selection,
and geometric rank does not show that individual thoughtlets perform distinct
functions. Further work must use untouched seeds, matched monolithic and
modular baselines, multiple training seeds, and causal interventions.

## Decision

- Do not resume this checkpoint into the existing 1,000-step cosine recipe.
- Do not run the registered causal thought ablation, because the policy
  precondition failed.
- Preserve the checkpoint and metrics as immutable evidence.
- Build the matched-baseline and larger held-out evaluation framework before
  another scale-up.
