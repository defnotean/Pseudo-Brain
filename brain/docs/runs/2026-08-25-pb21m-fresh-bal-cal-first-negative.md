# PB21M fresh BAL-UPMIX CAL-first qualification: terminal negative

Date sealed: 2026-08-25

Canonical run: `2026-08-25-pb21m-fresh-bal-cal-first-v1`

Classification: `fresh_CAL_futility_negative_no_DEV`

## Outcome

PB21M is a valid scientific CAL-stage negative. The fresh, preregistered
BAL-UPMIX scorer did not pass the frozen cross-fitted TRAIN-CAL futility gate
against paired BASE. The failure is scientific, not an infrastructure failure:
the run completed all 18 fixed head fits, published and reloaded authoritative
CAL evidence, deterministically reproduced all 36 cross-fit and 18 final AA
calibrators, evaluated the frozen decision, and published a terminal result.

DEV remained sealed. No DEV-open receipt, DEV source, DEV evidence, checkpoint,
recipe nomination, scaling claim, full-model claim, or qualification claim was
created. The same namespaces may not be retried.

## Immutable artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| registration | 36,966 | `f2ca97558697e4597d57578261a82f13621808758cfd2ec5e1b8542ba516dee6` |
| attempt | 1,578 | `e3322c79cd979d6b3b486680360c1eeafb961ad528a194d4706e68d2434d8758` |
| CAL evidence | 111,262,688 | `5bdf26f0988a1bd79faa07b650eede2fadd08c37616a97b5b21ee739069ffe02` |
| CAL decision | 783,343 | `20bc265e17c02e25a78cb76d31ceb281e253b44cfb1478db88dd64bc912706df` |
| terminal result | 1,718,454 | `0833b6c848003f9639c7e69250724639948b6cda334f992b0ff32fb0abd4d18f` |

Registered source-bundle SHA-256:
`75078d29c254cb8636d3e56191040888739348fac8abf3726cb9d45906a8bde7`.
The CAL evidence key-set SHA-256 is
`20ca5fe47484c5265f93eb830ac9edcc61dc0bc1a0a792fe08d29249cf6d9494`;
its complete array-manifest SHA-256 is
`dd125f388146e8701100a7c780150b10cddf29238476bd8ba4f84feb807a3bcb`.

## Execution and integrity

PB21M used the three fresh FIT cohorts, paired BASE/BAL-UPMIX objectives, and
initialization seeds 91042, 92042, and 93042. Each of the 18 heads received 32
complete passes and 4,096 AdamW updates, for exactly 73,728 optimizer steps.
All nine FIT and two-fold CAL partitions were pairwise root-ID and
episode-group-ID disjoint.

Every cross-fit AA and every final AA accepted. Authoritative reload performed
36 cross-fit and 18 final deterministic refits; parameters and acceptance were
byte-identical. The frozen parent stayed byte-identical at the CAL terminal
boundary:
`2619b5b0a180bdd55296d3471d89725f4346110e74ef093f84c3068a32e6576d`.

One pre-attempt guard plus all eight result snapshots passed, for nine resource
checks total. CUDA was hidden, with one process and one intra/inter-op thread.
Peak recorded working set was
675,676,160 bytes, available physical RAM never fell below 9,374,375,936
bytes, committed memory stayed below 49.47%, and artifact free space stayed
above 65,554,550,784 bytes.

## Scientific result

BAL-UPMIX retained strong all-action and complement behavior but did not solve
the factual calibration problem:

- all nine BAL cells passed the absolute all-action domain;
- all nine passed the absolute nonselected-complement domain;
- zero of nine passed the absolute factual domain;
- factual aggregate ECE and positive calibration bias were identical within
  each cell and ranged from 0.0582808500638499 to 0.0713055505644124, above
  the frozen 0.05 aggregate limit;
- all AA acceptance, raw/calibrated ranking, prior-ratio, proper-score, support,
  and causal checks passed outside the factual calibration failures.

The paired direction is `BASE loss - BAL-UPMIX loss`. Grand factual proper
scores moved in the intended direction, but the gain was not stable across
fixed cells and cohorts:

| Domain | Passing cells | Passing cohorts | Grand BCE point / LCB | Grand Brier point / LCB |
|---|---:|---:|---:|---:|
| factual | 3/9 | 1/3 | 0.0042678333 / 0.0018268188 | 0.0015748206 / 0.0006946022 |
| all-action | 7/9 | 3/3 | 0.0003628481 / -0.0018075681 | -0.0001616093 / -0.0009037912 |
| nonselected complement | 6/9 | 3/3 | -0.0006133983 / -0.0030226279 | -0.0005957168 / -0.0014078748 |

Thus the useful part of the hypothesis survives only as a diagnostic: balanced
factual pressure can improve grand factual BCE/Brier while preserving the
other domains on average. It did not produce per-cell robustness or factual
calibration under the fixed standard AA recipe. PB21M nominates nothing, and
its negative cannot unlock scaling or DGX training.

## Next gate: PB21M post-hoc consumed mechanism audit

Before fresh qualification, PB21M gets two separate, explicitly post-hoc and
nonqualifying diagnostic arms using only its already consumed TRAIN-CAL data:

- fit one fixed MIX35 cross-fit calibrator on the sealed PB21M raw CAL logits;
  there is no weight sweep and no scorer training;
- stratify residuals by owned prior action using reconstructed consumed CAL
  tapes bound back to immutable PB21M identities; there is no training.

Both arms must remain separate and may never open DEV, CPU-QUAL, PLAY-QUAL, or
TEST. The audit may nominate at most one single change—MIX35 calibration or
prior-action conditioning—for a later fresh PB21N preregistration; it must
never combine them. No checkpoint, qualification, or scaling claim follows.
Until a fresh PB21N preregistration and its gates pass, DGX and full-model
training remain blocked.
