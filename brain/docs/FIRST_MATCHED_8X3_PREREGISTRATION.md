# First matched 8-seed × 3-suite campaign

Status: **blocked and non-launchable**.

This repository does not currently contain a campaign registration, per-seed
effective training configs, a comparison criterion, or a campaign execution
record for the first matched-architecture claim campaign. That absence is
intentional.

The frozen architecture manifest with internal digest
`52bba6a9b723dda51da18d2842c5eaf360453db011e7f97bbb066e4034905421`
is retained as design history. Its recipes stop at 500 optimizer steps. The
reference candidate did not pass the prerequisite competence gate, so treating
those recipes as the first 8-seed campaign would preregister a study that cannot
answer the intended architecture question.

Before a campaign may be registered:

1. Reference Candidate Qualification must pass under its own frozen gate.
2. A new versioned architecture manifest must bind the qualified source bundle.
3. That new manifest must register the same staged 2,048-step budget for the
   routed reference, isolated-slot control, and parameter-matched monolith.
4. The campaign generator must produce eight effective configs per variant by
   changing only `run.name` and `run.seed` from each registered template.
5. One common runtime fingerprint must be registered and externally pinned
   before training.
6. The multiseed evaluator must introduce `CampaignExecution` schema 2 and bind
   that runtime registration into execution evidence and observations.
7. The resulting registration must be externally pinned before any campaign
   training or suite outcome is inspected.

The mechanical helpers for exact TOML derivation, canonical hashing, strict
campaign loading, and post-run checkpoint binding live in
`scripts/build_first_matched_multiseed_campaign.py`. Its campaign-building entry
point fails closed while the blocker is active. The post-run binder is dormant:
it can create `CampaignExecution` only from a complete run matrix whose
checkpoints pass restricted CPU loading and match the registered config, data,
source, run identity (through the config hash), and optimizer cursor. No
checkpoint hashes are invented in advance. Checkpoint paths must be relative to
one explicit root and cannot escape it or traverse symlinks/reparse points; the
binder hashes each file before restricted loading and again afterward.

The current `CampaignExecution` schema does not carry runtime identity. The
dormant binder can enforce one externally pinned `RuntimeRegistration` across
all runs, but the current comparison report cannot independently prove that
binding. Therefore no result produced under the current evaluator may be called
runtime-matched. The final campaign remains blocked until the multiseed evaluator
supports `CampaignExecution` schema 2 and binds runtime identity into both
execution evidence and observations.

## Frozen held-out suites

The suite set is real and can be preregistered independently of model training:

- `moving-shapes-test-hazards-1-v1`
- `moving-shapes-test-hazards-3-v1`
- `moving-shapes-test-hazards-5-v1`

Each suite contains 256 deterministic length-8 sequences from one of these exact
local half-open ranges inside the collision-free MovingShapes `test` namespace:
`[2,097,152, 2,097,408)`, `[2,097,408, 2,097,664)`, and
`[2,097,664, 2,097,920)`. These ranges are reserved for the future architecture
campaign and are disjoint from RCQ recipients, RCQ donor-only data, the RCQ guard
band, and every opened historical Stage-A slice. The first two steps are burn-in,
leaving exactly 1,536 scored decisions per suite. The sole normalized score is
the existing final-exit `movement_exact_match`: a decision scores one only when
the predicted active W/A/S/D set exactly equals the target set, using strict
`button_logit > 0.0` activation. Suites have equal weight.

These are same-family held-out procedural control conditions at hazard counts 1,
3, and 5. They do not measure general game intelligence, causal thought
specialization, real-time performance, or human-like cognition.

The canonical suite definitions, sample-selection digests, evaluation-code
bundle, and self-digests are under
`configs/evaluation/moving-shapes-heldout-hazards-1-3-5-v1/`. The machine-readable
blocker is
`configs/campaigns/first-matched-8x3-v1/campaign-blocker.json`.

CPU-only integrity check:

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONPATH = 'brain/src'
C:\Users\Demon\AppData\Local\Programs\Python\Python311\python.exe `
  brain/scripts/build_first_matched_multiseed_campaign.py --check
```

`--write-static` is create-only: existing differing bytes are never replaced.
`--attempt-campaign` raises the registered blocker. Training, GPU access,
network access, and benchmark result inspection are outside this tool's static
generation path.
