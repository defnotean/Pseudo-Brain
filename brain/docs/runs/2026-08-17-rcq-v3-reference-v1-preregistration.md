# RCQ-v3 reference v1 preregistration machinery and delegated decisions (2026-08-17)

Status: machinery complete; **no v3 registration file, release, pin, smoke,
canary, or training run exists yet.** The v2 registration, pins, gates, and
ranges are untouched and terminal. Nothing in this change opens sealed data.

## Delegated decisions (owner veto window: before the registration ceremony)

The owner delegated the open preregistration decisions ("do it all for me").
The frozen choices, each reversible only by discarding the uncreated v3
registration and starting over:

| # | Decision | Frozen choice |
|---|---|---|
| D1 | Continuous-head quiescence | (c) both: deadzone hinge (`continuous_deadzone_hinge_weight = 0.5`, `continuous_deadzone_margin = 0.04`) **and** structural squash (`continuous_output_squash = "deadzone_tanh"`, bound 0.046875, exactly representable, strictly inside the frozen ±0.05 deadzone). |
| D2 | Opposite-key conflicts | (c) multi-label geometry kept; `opposite_key_pair_weight = 0.25` on W/S and A/D sigmoid pairs. D2(b) rejected as a geometry change. |
| D3 | Budget | Same 2,048 updates with the same 1,536/2,048 gate geometry. The v2 miss was small and close; no re-justified budget was needed. |
| D4 | Seed and slices | Seed 1702 reused (only the objective recipe changes relative to v2). Fresh sealed TEST family at the next unused 2^20-aligned offsets: recipient `[4194304, 4194816)`, donor `[4194816, 4195328)`, guard `[4195328, 4195840)`. The v1 retired ranges, the v2 `[3145728, 3147264)` family, and the matched-campaign reservation `[2097152, 2097920)` stay untouched forever. |
| D5 | Thresholds | Unchanged. Gate IDs `rcq_v2_development_v1` / `rcq_v2_value_development_v1` and every registered threshold are frozen; the recipe change lives in the v3 config hash, not in the gates. |

Frozen v3 identities:

- Qualification id `rcq_v3_reference_v1`; evaluator id `rcq_v3_final_v1`.
- Run id `dgx-rcq-v3-reference-seed-1702`; frozen config
  `brain/configs/training/dgx-rcq-v3-reference.toml` (config SHA-256
  `17f2c1c2e95e30e5bd50dbf4ed409d9c25904c40b89b3b5eff52676f7d658105`).
- Registration path `registrations/rcq-v3-reference-v1.json` (**not yet
  created**; create-once at ceremony time).
- Range-claim id `af51be662e789d49b6be1b4ff16bace3d976c3227acdc375b74701599d63b5ce`
  under the deliberately unchanged protocol string
  `irene_moving_shapes_test_range_retirement_v1`.
- Dispatcher action family `rcq_v3_*` so v2 (closed) and v3 authority coexist.

## What was built

1. **Evaluator lineage** (commit `da4debe`): `evaluation/rcq_v3_final.py`,
   `evaluation/rcq_v3_torch.py`, `evaluation/rcq_v3_registration.py`, derived
   from the frozen v2 modules by
   `brain/scripts/build_rcq_v3_lineage.py` with asserted exact-count
   substitutions. `data/moving_shapes_dataset.py` guards the new sealed v3
   test range; `tests/test_rcq_v3_lineage.py` pins the lineage (10 tests).
2. **Trusted dispatcher** (`scripts/dgx/_remote_dispatch.sh`): the v3
   qualification family (9 functions, 5 mechanical action cases) is derived
   by `build_rcq_v3_lineage.py --dispatcher` between marked sections; the
   structural edits (v3 constants block, parameterized canary config, extended
   smoke/train/resume case labels, both-config generic refusals, release-sync
   whitelist for `registrations/rcq-v3-reference-v1.json`) are hand-reviewed
   in version control. The derivation asserts those anchors and refuses to
   run twice. The shared qualification-neutral helpers keep their historical
   names (`resolve_canonical_rcq_v2_workspace`,
   `open_rcq_v2_range_authority_lock`) deliberately.
3. **Operator wrappers** (`scripts/dgx/`): `New-RcqV3Registration.ps1` (with
   `brain/scripts/run_rcq_v3_registration.py`), `New-DgxRcqV3PretrainingPin.ps1`,
   `Invoke-DgxRcqV3Smoke.ps1`, `Invoke-DgxRcqV3StagingCanary.ps1`,
   `Start-DgxRcqV3Reference.ps1`, `Resume-DgxRcqV3Reference.ps1`,
   `Invoke-DgxRcqV3Preclaim.ps1`, `New-DgxRcqV3FinalAuthorization.ps1`,
   `Invoke-DgxRcqV3FinalOnce.ps1`, `Test-DgxRcqV3FinalReceipt.ps1`. The generic
   launchers (`Start-DgxBrainTraining.ps1`, `Resume-DgxBrainTraining.ps1`,
   `Invoke-DgxBrainSmoke.ps1`) now also refuse the v3 reference and staging
   configs. `Sync-DgxBrainRelease.ps1` carries the v3 registration file once
   it exists (optional until then).
4. **Smoke-factory squash parity**: `build_smoke_model` now honors
   `continuous_output_squash = "deadzone_tanh"` exactly like
   `build_thesis_model`, so the v3 staging canary exercises the frozen v3
   readout. The default `none` path is byte-identical behavior.
5. **New staging canary config**
   `configs/training/dgx-rcq-v3-staging-canary.toml` (v2 canary geometry,
   run name `dgx-rcq-v3-staging-canary`, seed offset 1703, v3 recipe fields).

## Intentional source freeze

The smoke-factory parity edit touched a matched-baseline implementation file,
so the architecture manifest was regenerated deliberately:

- Previous live digest: `5decb402bc9ba68a8b8a80317d09f05458b4aa9584dab81a9d3b8575409f49e7`
- New live digest: `eb46988b178d593da6f98f6b99273fd2e791dd62e35f9bcef719c03958d4bb2a`
- `MANIFEST_IDENTITY=new_comparison`; source bundle
  `e99e789b1b9d19e6f8511ec417925c1c6b3c9e7010a1346482dfc4c1adadc0b7`

Historical Stage A/RCQ-v2 evidence stays bound to the old tree digests; no old
claim is rewritten.

## Verification

- `tests/test_rcq_v3_lineage.py` (10), `tests/test_rcq_v3_recipe.py` (12),
  `tests/test_matched_baselines.py` (6), `tests/test_multithought_core.py` (5):
  green.
- `tests/test_dgx_launch_contract.py`: 32 tests green, including bash `-n` on
  the dispatcher, PowerShell parser passes over every wrapper, the v3
  wrapper-to-action mapping, the derived-section markers, and the
  create-only/chronology pins.
- Full play-safe suite: green (runner exit 0, one expected POSIX skip).
- `bash -n` clean; the derived sections contain zero residual v2-identity
  tokens outside the two deliberately shared helper names.
- In-memory v3 registration build was exercised end to end (SHA is
  tree-state-dependent and deliberately not pinned; the real file is created
  only at ceremony time).

## Ceremony sequence (owner-run, in order)

Windows OpenSSH is unreliable from non-console processes in this harness, so
run these from a console-attached terminal. Read-only status polls work from
Git Bash ssh.

1. `powershell -File .\brain\scripts\dgx\New-RcqV3Registration.ps1` — creates
   `registrations/rcq-v3-reference-v1.json` (create-once). Record the printed
   `REGISTRATION_SHA256` off-repo.
2. Commit the registration file; push.
3. `Sync-DgxBrainRelease.ps1` — new immutable release carrying the v3
   registration (the remote whitelist accepts exactly that one added path and
   revalidates it against the derived v3 canonical checker).
4. `New-DgxRcqV3PretrainingPin.ps1` with the release id, archive SHA,
   registration SHA, and pinned container image reference/id.
5. `Invoke-DgxRcqV3Smoke.ps1`, then `Invoke-DgxRcqV3StagingCanary.ps1`
   (foreground, two-phase, invariance receipt).
6. `Start-DgxRcqV3Reference.ps1 -AcknowledgeDetached` — the pinned 2,048-update
   reference run; entry gate at 1,536, completion gate at 2,048, unchanged
   thresholds.
7. Only on pass: `Invoke-DgxRcqV3Preclaim.ps1`, independent review,
   `New-DgxRcqV3FinalAuthorization.ps1`, `Invoke-DgxRcqV3FinalOnce.ps1
   -AcknowledgePermanentTestRetirement`, `Test-DgxRcqV3FinalReceipt.ps1`.

On any gate failure: the same terminal discipline as v2 — no resume, no
tuning under this registration; diagnose, then decide whether the recipe
family deserves a newly preregistered v4.
