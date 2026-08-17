# Baseline architecture-manifest freeze (2026-08-16)

Status: local source freeze only. This is not an RCQ-v2 result, not a matched
architecture claim, and not permission to launch the first 8-seed campaign.

## Why this freeze existed

Local play-safe tests stopped because the checked-in architecture manifest no
longer matched `src/irene_brain/training/objective.py`. The live file hash was
`8cec05a3c03d5905a954c5c44ae68217ed0e086957861a2c83dd468fde467815`. The hash
recorded in the previous manifest was
`b4bbc738ad9315c8a29e6e3257adab0cd60c2b9a7b436debefd5421aee3ab42f`.

That mismatch is expected after an intentional objective change. Silently
keeping the old digest would have made architecture-comparison tests lie about
which source they were identifying.

## What was regenerated

Play-safe wrapper, CPU-only, CUDA hidden, isolated Python:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\brain\scripts\Update-BaselineArchitectureManifest.ps1
```

The wrapper writes `configs/baseline-architecture-manifest.json` by instantiating
the four registered variants only to count parameters and bind recipe hashes.
It does not train, does not open TEST, and does not contact the DGX.

| Identity | Previous | This freeze |
|---|---|---|
| Manifest digest | `52bba6a9b723dda51da18d2842c5eaf360453db011e7f97bbb066e4034905421` | `30d4c119795a3c967e0f1251787cf5fb7effce8d02b814224b4383c40d783fc7` |
| Source bundle | `a7313109b6f99cbe57e5d23cef6e6444a224d348597e73e82da9f71c4a7e8448` | `7b8354f7df1e441a404540b72c97f62acfd1957e0a4fde7041158946831b2925` |
| `training/objective.py` | `b4bbc738ad9315c8a29e6e3257adab0cd60c2b9a7b436debefd5421aee3ab42f` | `8cec05a3c03d5905a954c5c44ae68217ed0e086957861a2c83dd468fde467815` |

The other seven implementation files in the bundle kept their previous hashes.

## What this freeze is not

The previous digest `52bba6a9…` remains the **historical** first-matched
campaign pin. That campaign stays blocked: its recipes are 500-step Stage A
templates, and RCQ-v2 has not passed. The live freeze is a new comparison
identity. It must not be copied into
`HISTORICAL_ARCHITECTURE_MANIFEST_SHA256` to make the old campaign look
current.

A later architecture campaign still requires, in order: RCQ-v2 pass, a
versioned manifest that binds the qualified source and the 2,048-step staged
budget, then a new externally pinned campaign registration. See
[FIRST_MATCHED_8X3_PREREGISTRATION.md](../FIRST_MATCHED_8X3_PREREGISTRATION.md).

## Local verification after this freeze

`run_play_safe_tests.ps1` then passed every isolated module: 294 tests, one
expected POSIX skip, about 47 seconds, CPU-only. The first-matched campaign
test now keeps `52bba6a9…` as a named historical constant and requires the live
digest to differ from it, so this freeze cannot silently reopen that campaign.

Generated `__pycache__` directories left by the previous test run were deleted
before any registration attempt. The play-safe runner now sets
`PYTHONDONTWRITEBYTECODE=1` so a later test pass cannot recreate them inside
`brain/src/irene_brain`.
