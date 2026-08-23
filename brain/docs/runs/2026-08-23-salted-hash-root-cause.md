# ROOT CAUSE FOUND: salted-hash episode banks — cross-process data nondeterminism (2026-08-23)

## Finding [MEASURED]

Every experiment script builds its training episode bank with:

```python
r = np.random.default_rng(seed + hash(fn.__name__) % 99991 + i)
```

Python's `hash()` on strings is **salted per process** (`PYTHONHASHSEED`
random by default). Verified locally: identical expressions return
(43441, 21891) in one interpreter and (82198, 33543) in another; only
`PYTHONHASHSEED=0` pins them.

**Consequence:** two processes claiming "identical config + seed" trained on
DIFFERENT episode banks. Eval is unaffected (`eval_lift` uses a fixed
base_seed stream), so evaluation was always comparable — training data was not.

Affected scripts (grep `hash(fn.__name__)`): adaptive_gate_experiment,
batched_equivalence_test, braincell_candidates_screening, dynamic_k_v0/v1,
episodic_v0_experiment, evidence_residual_confirmation,
multiseed_throughput_experiment, profile_training_pipeline,
stage_l_w_scaling, torture_regression_check, **train_torture_multitask**
(the canonical locked-baseline trainer).

## Which prior conclusions are affected

| Comparison type | Status |
|---|---|
| Arm-vs-arm WITHIN one process (gate-exp, dynk v0/v1, BC screening/confirmation, equivalence solo-vs-batched) | **STAND** — all arms shared the same per-seed bank inside each process |
| Locked absolute baselines (GRU −0.042, PB −0.215) | One draw each from a bank distribution; treat as reference points, not constants |
| Stage E torture regression ("reproduction") | Cross-process → different bank than original. Agreement at W120 (σ≈0.004 regime) plausible; claim downgraded to "consistent", not "reproduced" |
| Stage L vs L2 W480 "instability" | **CONFOUNDED** — different banks explain the fixed-seed divergence (s142 −0.009 vs −0.246) without invoking GPU numerics |

Coherence check: W120 spread across seeds/banks was tiny (±0.004) while
W480 spread was huge — consistent with bank sensitivity growing with width,
which is now the leading hypothesis for the entire W-scaling variance story.

## Actions

1. All future scripts use a stable task offset:
   `zlib.crc32(fn.__name__.encode()) % 99991`.
   Frozen science scripts are NOT edited (md5 discipline); new scripts carry
   the fix and import nothing from the salted path.
2. Determinism audit (this workstream) proceeds with the corrected bank:
   same-process repeat → deterministic-flags repeat → cross-process repeat
   → THEN true W480 seed variance.
3. The UNSTABLE verdict on W480 is suspended pending the corrected rerun.

## Classification per director's framing

- "Is W480 currently stable enough for scaling-law claims?" — NO (unchanged),
  but the reason has shifted: measurement-path nondeterminism identified and
  removed; true optimization stability not yet measured.
