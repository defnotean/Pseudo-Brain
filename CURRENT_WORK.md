# CURRENT_WORK — ACTIVE FRONTIER

**Last updated:** 2026-08-28 (Tier-1 harness closeout: the working tree held TWO uncommitted generations of the Pac-Man-like harness. Stack A (`environments/pacman_harness.py`, P1–P6) implements the committed frozen T1-1 prereg (as amended for P2/P4/P5) and passes all H1–H7 gates; a divergent uncommitted "construction" stack (Stack B, `pacman_harness_v1.py`, P0–P5) was found to (a) deviate from the committed §1 family table, (b) overflow the registered seed block ~15×, and (c) rescale the control period to 54/66 Hz under P3/P4 — violating the frozen v3 §4.3 60 Hz native-control gate. Stack B RETIRED (moved, not deleted, to `brain/scratch/retired-pacman-harness-v1-construction/`). The missing P1–P6 test module `tests/test_pacman_harness.py` (cited as 21/21 in the T1-2 report but never on disk) was reconstructed: 25/25 OK. Frozen §4.3 timing gates now live in the harness layer (`TIMING_GATES`/`check_timing_gates`). H1–H7 re-run at frozen scale: all PASS, 3.419 s. Full regression 1169 tests: every non-green accounted for — 10 pre-existing/environmental (incl. the 6 T0-1 documented), 2 load artifacts (60 Hz latency test passes in isolation); none depends on harness modules. Full record: `brain/docs/runs/2026-08-28-t12-pacman-harness-stack-adjudication.md`. Next: T1-3 expert/scripted policy → BC corpus on TRAIN.)

**Prior update:** 2026-08-27 (Tier-0 closed: T0-1 full regression 1151 run / 6 structural non-green, none a behavioral regression; T0-2 V2-C four-seed confirmation closed AMBIGUOUS (sigma 0.0653 > 0.06 frozen tol); T0-3 local-CPU-canonical-compute decision record supersedes the DGX-Spark-primary designation. Tier-1: T1-1 harness prereg frozen (`2026-08-27-pacman-harness-v1.md`); T1-2 harness build ran H1–H7 at frozen scale — all PASS (report `brain/docs/runs/2026-08-27-t12-pacman-harness-gates.md`); the P4/P5 degenerate-rounding defect in the committed table was amended to channel-level world hold/extra-step rules in the working tree. The implementation was left uncommitted and a second divergent generation was started on top — resolved 2026-08-28, see above.)

**Prior update:** 2026-08-26 (hazard branch fully characterized at a clean boundary: L8 diagnostics close the calibration function-class family; L9 probe closes the prior-action window-length axis at K=1 (2-step factual AUC 0.6973 < 1-step 0.7060, no PB21Q trial); L10 probe closes the crude outcome-history feature (recent-hazard-count drops factual AUC 0.6706 → 0.6583, no PB21R trial); L11 probe closes the recurrent applied-trajectory hazard-history state at fresh-partition level (discovery +0.0307 AUC on PB21O-CAL does NOT generalize — −0.0188 on fresh PB21S-CAL, p=0.104; drafted PB21S prereg refuted before execution, no trial published). Binding constraint: factual-hazard ranking/signal density, AUC ~0.70 — no tested representation-level mechanism improves it on fresh data. Next step: realtime-embodied-qualification v3 battery for the integrated model. Stale PB21M runner crash triaged: sealed artifacts re-verified byte-exact, no re-execution.)
**Base HEAD:** `8c4c127` · **Tag:** `core-v1` @ `890e4d0`

## ACTIVE FRONTIER — V2.1i all-action causal outcomes

**PB21M status: TERMINAL SCIENTIFIC CAL NEGATIVE; DEV SEALED.** Registration
`f2ca97558697e4597d57578261a82f13621808758cfd2ec5e1b8542ba516dee6`,
attempt `e3322c79cd979d6b3b486680360c1eeafb961ad528a194d4706e68d2434d8758`,
CAL evidence `5bdf26f0988a1bd79faa07b650eede2fadd08c37616a97b5b21ee739069ffe02`,
CAL decision `20bc265e17c02e25a78cb76d31ceb281e253b44cfb1478db88dd64bc912706df`,
and result `0833b6c848003f9639c7e69250724639948b6cda334f992b0ff32fb0abd4d18f`
are terminal and retry is forbidden. All 18 heads completed 73,728 optimizer
steps; authoritative reload byte-reproduced all 36 cross-fit and 18 final AA
calibrators. No DEV-open receipt, DEV source/evidence, checkpoint, or nomination
exists; classification is `fresh_CAL_futility_negative_no_DEV`.

Scientifically, all nine BAL-UPMIX cells passed absolute all-action and
nonselected-complement gates, but zero of nine passed the factual domain.
Factual aggregate positive bias/ECE was 0.05828–0.07131 against the frozen
0.05 limit. Grand paired factual BCE/Brier improvements were positive with
positive 97.5% lower bounds, but only 3/9 cells and 1/3 cohorts passed both.
No checkpoint, scaling, full-model, qualification, or live-play claim follows.

**DGX status: BLOCKED.** PB21M nominated no recipe and cannot unlock scaling or
full-model training. The two-arm PB21M post-hoc mechanism audit is now
**SEALED (2026-08-25)**: `2026-08-25-pb21m-posthoc-mechanism-audit-v1`,
classification `post_hoc_consumed_data_nonqualifying`. Solver controls passed
byte-exact (AA 180/180 refits, worst diff 0.0; PB21K MIX w=0.5, 45/45, 0
acceptance mismatches). **Arm A (MIX35, focal w=0.35) FAILED on gate A3** —
all 18 tables degrade all-action |bias| by ≈0.020 against the 0.01 tolerance
(A1 factual absolute PASS: worst |bias| 0.04987 < 0.05; A2 paired factual
PASS: 9/9 positive LCBs; context variants w=0.5→0.0352, w=1.0→−0.0006 report
only). **Arm B (prior-action residual strata) PASSED B1+B2+B3** — all six
cohorts byte-identical data-only replay; 18/18 cells with ≥3 structured
strata (worst |bias| 0.140–0.174); B3 specificity observed range 0.3253 vs
permuted p95 0.3021 (permuted max 0.3284 — thin margin, recorded honestly).
**Nomination (frozen rule): Arm B single change** — "condition the hazard
representation on the prior applied action (one-step lag, burn-in-aware)" —
licensed for exactly one fresh PB21N preregistration in a fresh namespace;
the two changes may never be combined. Mechanism-signal only
[HYPOTHESIS-LEVEL OUTPUT]. Run report:
`brain/docs/runs/2026-08-25-pb21m-posthoc-mechanism-audit.md`. Artifacts under
`brain/runs/pb21m-posthoc-audit/` (result `0d0be6f7…`, evidence
`e5fd6c67…`, registration `f9c2d13e…`; completion provenance in
`…v1.artifact-completion.json`).

**Nomination executed and closed (2026-08-25):** PB21N trial #1 (fresh
namespace) → FROZEN NEGATIVE (G5/G6/G2/G4 pass; G1/G3 fail — mechanism
specific but not calibrating). Its licensed follow-up, PB21O (isotonic/PAV
calibrator on the same conditioned path, fresh namespace) → FROZEN
NEGATIVE (G5 function-class isolation fails: isotonic strictly worse than
affine; G1/G2 fail; G3/G4/G6 pass). The hazard post-processing family
(affine ± prior-action conditioning, isotonic/PAV) is therefore closed as
a route to the frozen factual gate; see
`brain/docs/ARCHITECTURE_IDEAS_LEDGER.md` L1/L4/L7 and
`brain/docs/runs/2026-08-25-pb21o-isotonic-hazard-calibration-trial1.md`.

The current mechanism combines the evidence-backed parts of prior hypotheses:

- one factual recurrent trajectory and one expert action label per state;
- five exact same-snapshot idle/W/A/S/D one-step outcome targets;
- a vectorized all-action outcome table with semantic action-ID joins;
- cosine next-latent targets and symlog two-hot reward distributions;
- fused latent/reward/hazard/surprise prediction-error feedback;
- a dedicated stop-gradient nonlinear hazard path, calibrated separately so it
  cannot alter next-state, reward, decision, belief, or thought parameters.

V2.1h is now frozen as exploratory evidence. Its replicated hazard ranking is
retained, but its 44,161-parameter "calibration" pass is rejected as a true
calibrator: seed 43 improved ranking while worsening BCE and underpredicting the
validation hazard rate by about 11.5 percentage points. V2.1i instead freezes the
raw scorer and fits only five action-specific logit biases on a disjoint
TRAIN-CAL episode range. The five scale buffers remain exactly one. This keeps
the useful state discrimination while making calibration incapable of hiding a
weak representation.

**Qualification infrastructure now implemented:** strict aggregate/per-action
BCE, Brier, ROC-AUC, PR-AUC, equal-mass ECE, calibration bias, TRAIN-fitted
branch and factual action priors, 20 deterministic cross-episode derangements,
10,000-resample episode-clustered confidence bounds, and latent non-collapse
metrics. A create-only namespace contract reserves disjoint TRAIN/DEV/CPU-QUAL
ranges, forbids TEST, binds the complete five-seed cohort and source bundle, and
prevents CPU-QUAL materialization until a canonical preregistration is sealed.
CPU-QUAL has not been materialized.

**Causal dataset evidence:** on held-out validation, action choice changed
caught/safe outcome in 197/384 states (51.30%) and some immediate outcome in
337/384 (87.76%). Historical intervention batch manifest remains exactly
`94470d5ffd38d7e1a5d275e36af28d6e6d40f05bde94e1f71e2954baeb1122ae`.

**Seed 42 integrated calibrated checkpoint (exploratory):** next-latent
`1.0212 -> 0.3869`; reward CE `4.3388 -> 2.9628`; all-action hazard
`0.6680 -> 0.6153`. Factual caught probability is `0.5255` versus safe
`0.3259`. Hazard-only held-out ROC-AUC is `0.6655`; state shuffle worsens BCE
from `0.6169` to `0.7263`; all five per-action AUCs exceed `0.63`. Exactly
seven dedicated hazard tensors changed during calibration and every other
checkpoint tensor is byte-identical.

**Seed 43 replication (exploratory):** next-latent `1.0014 -> 0.3850` and
reward CE `4.1276 -> 2.6506` remain strong after integrated calibration.
Factual caught probability is `0.3413` versus safe `0.2020`. Hazard-only
ROC-AUC is `0.6813`, but BCE `0.6402` misses the existing 1%-better-than-action-
prior gate (`0.6429`) even though Brier and all causal discrimination controls
improve. This is not yet confirmatory qualification.

**Live speed:** calibrated CPU pixels-to-control plus in-process simulator step
is 3.001 ms mean, 3.984 ms p99 over 200 ticks on one thread; capture, network,
physical HID, and display latency remain excluded.

**Verification:** the last pre-V2.1i full suite passed 809 tests in 156.862 s,
with 2 expected skips. The current V2.1i focused suites are green for the
counterfactual dataset/objective, integrated bias-only calibration, qualification
metrics, sealed namespace, provenance guards, and latent non-collapse checks;
a new full-suite run is still required after runner integration. Historical
manifest and sequence-content goldens are pinned. Source identity now covers the
entire `irene_brain` package plus exact pipeline scripts. Artifacts and
checkpoints are create-only, and failed qualification writes negative evidence
before exiting nonzero. No TEST split was opened by V2.1f/g/h/i work.

**Audit executed and closed (2026-08-25).** The PB21M post-hoc consumed
mechanism audit ran: Arm A (MIX35 cross-fit calibrator on sealed raw CAL
logits, no sweep or scorer training) FAILED gate A3 (all-action |bias|
degraded ≈0.020 vs the 0.01 tolerance; A1/A2 passed); Arm B (prior-action
residual strata from reconstructed consumed CAL tapes, no training)
PASSED B1+B2+B3. Both arms are permanently post-hoc/nonqualifying and never
opened DEV, CPU-QUAL, PLAY-QUAL, or TEST. The frozen nomination rule sent the
Arm B single change to a fresh PB21N preregistration (executed, FROZEN
NEGATIVE) and, as its licensed follow-up, PB21O isotonic/PAV calibration
(executed, FROZEN NEGATIVE). The hazard post-processing family is closed as a
route to the frozen factual gate. DGX and full-model training remain blocked.
The next unexplored mechanism is representation-level (new hazard-path
inputs) and is **not licensed** by any closed trial; it would need its own
fresh namespace + preregistration + control arm before any run.


The downstream real-time embodied qualification target is documented in
`brain/docs/preregistrations/2026-08-25-realtime-embodied-qualification-v3.md`.
It is a Q0--Q6 design contract only: it does not authorize a run, alter the
PB21M/PB21N sequence, or relax any component, integration, or DGX gate.

## PHASE 2.6: CLOSED — Core V1 frozen

Six mechanism candidates tested under preregistration; **zero promoted**:

| Mechanism | Verdict | Evidence |
|---|---|---|
| Episodic Memory v0/v1 | REJECTED | causal direction right, magnitude failed prereg |
| Dynamic-K v0/v1 | REJECTED | v0 gain failed replication; no true closure |
| BrainCell gated-GRU | REJECTED (screening) | worse lift, no causal intake |
| BrainCell evidence-residual | REJECTED (confirmation) | 1/5 causal seeds; small lift regression; σ-stabilization finding preserved |
| Adaptive halting | DESCOPED (audit) | telemetry-only, never skips cycles, `.item()` sync defect |
| Adaptive gate (wired) | REJECTED (prereg experiment) | memory-causal 0/3 under task gradient |

**Constitution CI added:** 15 contracts A–O, all green
(`brain/tests/test_constitution_ci.py`, runner `run_constitution_ci.py`).
Torture regression confirmed code-state changes caused zero behavioral drift.
GRU baseline note updated: its locked −0.042 was seed-optimistic (n=3 σ=0.14).

**Core V1 known limitations (explicit):**
1. evidence-intake deficit [MEASURED, 4 failed fixes]
2. weak multitask sample efficiency vs GRU-on-good-seeds
3. no adaptive compute (fixed C=3)
4. keep/accept decomposition stabilizes variance but doesn't fix intake

## PHASE 2.7: OPEN — throughput + scaling prep

| Item | Result |
|---|---|
| Training profile | GPU-bound; 80 ms/step; forward+backward 97.7%; recurrent serialization dominates |
| Episode-batched engine | equivalence PASS per prereg (3/3 seeds ≤0.03); **~12× per-episode throughput** |
| W-scaling sweep (v1) | mean lift ↑ with width but variance ↑↑ — **later found CONFOUNDED** |
| Salted-hash root cause | `hash()` episode banks differ per process; within-process comparisons stand, cross-process absolute numbers confounded [MEASURED] |
| Determinism audit B/C | **Outcome A**: normal-mode same-seed runs diverge from step 50 (1 ULP → amplified); deterministic flags make 3/3 reps bitwise identical (lift −0.2273 ×3) |

## Determinism protocol (now mandatory for all scaling runs)

```python
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
# banks: zlib.crc32(fn.__name__.encode()) % 99991  (NEVER hash())
```

## Next recommended work

1. ~~W=480 variance disambiguation~~ superseded by root-cause + audit work.
2. ~~Stage E corrected W-curve~~ **DONE 2026-08-23: width does NOT buy
   capability** (W120 −0.227±0.013 / W240 −0.232±0.039 / W480 −0.257±0.057).
   v1 sweep improvement was artifact; Core V1 stays W=120 on clean evidence.
3. ~~Stage 3a training-budget curve~~ **DONE 2026-08-23: MIXED by strict
   gate (Outcome C requires |Δ|≤0.03 in both arms; SCHEDULED |Δ|=0.039
   fails). Scientific read: strong train/eval decoupling / generalization
   plateau.** FIXED Δ(24k-6k)=−0.008, SCHEDULED Δ=−0.039. SCHEDULED
   overfits loss→0.0 by step 7k but generalizes worse (−0.233 vs −0.221);
   reduces σ(24k) 0.102→0.046 without improving mean. Core V1 not
   undertrained; cannot convert optimization on current data/objective into
   capability. All three capacity axes (W, K, budget) closed.
4. ~~Stage 3b Data / Generalization Audit~~ **DONE 2026-08-23: AMBIGUOUS by
   frozen gate (strongly MECHANISM-leaning).** DATA rejected at 6k (Δ=−0.008,
   not ≥+0.03). MECHANISM gate fails second limb: ARM B loss ≈1.85 did NOT
   collapse (ARM A ≈1.01). Two explanations alive: (A) learning
   mechanism/objective wrong; (B) fresh stream undertrained at 6k. ARM B 16×
   higher seed variance (σ 0.007→0.112), 13.6× wall cost (9976s vs 732s).
   Seed 142 (−0.046) is strong relative outlier, still negative, undecided
   noise-vs-basin. **Disambiguation:** train ARM B to matched training-loss
   criterion, then compare — requires Training Engine V2 first.
   **TE V2 profiler (2026-08-24) [MEASURED]:** ARM B 13.6× cost was NOT frame
   conversion or RNG — it is 14 serial B=1 GPU launches/step (Python/launch
   overhead dominates 2.5ms forwards). Batched padded B=14 single pass:
   **6.86× speedup** (1614.7→235.2 ms/step), zero numeric change; non-blocking
   H2D a wash. ARM B disambiguation now ~24min/seed instead of ~166min.
   `brain/scripts/te_v2_profiler.py`, results in `runs/tev2prof/`.
   b. ~~Stage 3c0 Learning Contract Audit~~ **SUPERSEDED by Core V2 build
      (2026-08-24):** the learning-contract mismatch hypothesis is now being
      addressed directly by building Core V2 with the DEPLOYED decision loss
      as the primary objective (`losses.deployed_decision_loss` trains the
      aggregated action_values path, not replicated per-slot logits).
      The V1 audit diagnostics remain available if V2 shows the same failure.
   c. **Core V2 (active):** explicit-timescale one-brain architecture at
      `brain/src/irene_brain/v2/`. **Stage V2.0 terminal GAMMA (2026-08-24):**
      both V2-A and V2-C produced mean lift −0.3567, sigma 0, delta −0.1437
      vs Core V1. Root cause is an exact zero-gradient contract defect:
      zero-initialized scalar consequence utility makes all deployed action
      values zero (`0/56` parameters with a non-zero gradient). The final JSON
      was lost to a root-owned output-dir permission error; aggregate container
      logs are preserved and the failure is not rerun. **Stage V2.0b smoke:**
      V2-A validated the direct set policy (lift −0.0100, delta +0.2030, three
      actions), but V2-C became NaN and stayed constant-policy; full V2.0b was
      correctly blocked. Diagnosis found two expansive recurrences: the five
      additive BrainCell gates exploded by update 9, and belief's unbounded
      residual reached 299k after thought normalization. **Stage V2.0c is
      frozen:** convex normalized thought updates plus GRU-style bounded belief.
      Full V2-C completed 120 CPU updates finite. **V2.0c Spark smoke stayed
      finite and moved both arms to near-chance lift, but failed:** V2-A used
      one majority action, V2-C used two, and fixed-probe learning missed the
      frozen 10% gate. The actual padded exposure counts are
      `[24,236,268,111,113]`. **V2.0d Spark smoke failed:** its nominal inverse
      weights were renormalized inside every length bucket, producing effective
      shares `[4.0%,29.0%,28.7%,21.5%,16.8%]`; both macro probes worsened and
      balanced accuracy stayed near chance (`0.2017`/`0.2131`). The 6k run was
      correctly blocked. **V2.0e is frozen:** fixed-exposure mean makes the
      aggregate loss exactly 20% per class while changing no other factor.
      Eight focused contracts and the final full repository regression suite pass.
      V2.0e's exact 20%-per-class correction also failed: probes worsened to
      `2.686/2.459` and balanced accuracy stayed `0.213/0.205`. Its 6k run is
      blocked. Class 0 occurs only at batch positions 35–36, while update 399
      stops 34 updates later. A bounded equal-exposure fixed-order versus
      deterministic-reshuffling diagnostic is active to test within-cycle
      interference before another preregistration. V2.0f measured fixed-order
      macro loss `1.878->3.542` versus reshuffled `1.878->1.413`, proving order
      is one defect, but rare classes 0/4 still had zero recall. A gradient-clip
      diagnostic measured that all 47 batches clip and nominal class-0 update
      share collapses from 20% to 0.56%. Full-cycle accumulation restored both
      rare pathways (classes 0 and 4 each reached 100% training recall) and
      raised held-out balanced accuracy to 0.238, but lr `5e-4` oscillated and
      ended overcommitted to action 0. A 32-cycle lower-LR diagnostic found the
      first non-collapsed policy: at `1e-4`, macro loss `1.878->1.196`, all five
      held-out actions, balanced accuracy 0.511, lift -0.051 (delta +0.162 vs
      V1), and stable state. **V2.0j preregistered smoke PASSED both arms:**
      V2-A macro `1.878->1.203`, balanced acc. 0.492, lift -0.066; V2-C macro
      `1.613->1.396`, balanced acc. 0.501, lift -0.087. Both emitted all five
      actions and stayed bounded. **V2-A four-seed confirmation is now ALPHA:**
      mean lift `-0.0457`, delta `+0.1673` versus Core V1, sigma `0.0208`;
      every seed retained all five actions. V2-C four-seed confirmation remains
      active on the unchanged frozen release. V2.1 predictive trajectory wiring
      is implemented and CPU-integrated: live same-tick world-model gradients,
      detached recurrent predictions, EMA target encoder, probability hazard,
      turn-weighted causal maze trajectories, and 30 focused contracts. The
      mini checkpoint completed a 32-tick closed loop with 32/32 submissions
      and no invalid controls. Single-thread CPU model latency is 2.27 ms mean,
      2.60 ms p99 over 200 recurrent ticks (inference only, not end-to-end).
      The extended pixel-to-control plus simulated-world loop is 2.58 ms mean,
      3.24 ms p99; physical capture/network/HID/display latency remains open.
      The final repository regression is green: 779 tests, 2 expected skips. A
      three-arm V2.1 smoke is frozen but will not compete with V2-C on the
      Spark. Then multi-tick prediction error plus live episodic memory (V2.2),
      and meta-learning across trials (V2.3).
      **Post-hoc mechanism audit (2026-08-25, SEALED):** the two-arm audit of
      the PB21M terminal negative ran fully local (CPU-only, single-thread,
      CUDA hidden, 126.0 s wall). Arm A MIX35 failed on all-action control A3
      (focal w=0.35 fixes the factual domain — worst factual |bias| 0.04987 —
      but costs ≈0.020 all-action |bias| vs the 0.01 tolerance; context
      bracket w=1.0 reaches factual bias −0.0006, so the two constraints
      collide at the focal point). Arm B prior-action strata passed: the
      factual miscalibration is organized by prior applied action (18/18
      cells structured; specificity 0.3253 vs 0.3021 p95, thin but frozen-
      rule-passing). Nomination: prior-action conditioning for a fresh PB21N.
      **PB21N trial #1 (2026-08-25, SEALED — FROZEN NEGATIVE):** the fresh-
      namespace single-change trial ran fully local (CPU-only, single-thread,
      CUDA hidden, 34.8 s wall; 256 train episodes @ offset 167,774,464,
      contiguous after PB21M C2B, zero overlap; upstream V2.1i parent frozen
      byte-exact, G4). The prior-action hazard path (3H trunk + prior-action
      embedding, retrained 8×64 steps/fold) is **confirmed specific but not
      calibrating**: G5 PASS (observed stratum-bias range 0.5206 vs permuted
      p95 0.4247 — mechanism specific, stronger than the audit's thin
      margin), G6 PASS (beats the pure-retrain control, fold-0 LCB +0.00045 /
      fold-1 +0.0245), G2 PASS (factual BCE/Brier improvement vs frozen base,
      both folds LCB>0), G4 PASS — but **G1 FAIL** (fold-1 factual bias/ECE
      0.0823 vs 0.05 limit) and **G3 FAIL** (all-action bias/ECE drift
      0.011–0.015 vs 0.01 no-degradation margin, both folds). The mechanism
      is a genuine hazard-representation effect that does not, as a *single*
      change, clear the frozen calibration gates — same two-constraint
      collision as MIX35 Arm A. Namespace closed (single-shot); DEV sealed;
      no retry. Run report:
      `brain/docs/runs/2026-08-25-pb21n-prior-action-hazard-conditioning-
      trial1.md`. Integrity note: the published `determinism: false` flag is
      a documented harness false-negative (whole-array byte check read
      uninitialized OOF complement half); eval-row OOF logits re-derived
      byte-exact post-hoc and all gates read eval rows only. Runner fixed
      post-trial (zero-init OOF slots) + regression test.
      **PB21O trial #1 (2026-08-25, SEALED — FROZEN NEGATIVE):** the
      licensed follow-up — isotonic (PAV) per-action hazard calibrator on
      the prior-action conditioned path, affine calibrator as control arm
      (fresh PB21O-CAL partition, seed offset 167,774,720, verified
      disjoint; PAV hand-rolled, sklearn cross-checked at 1e-16) — **FAILED**
      G1 factual absolute (fold0 bias/ECE 0.0842/0.0842; fold1 ECE 0.0625),
      G2 paired vs base, and G5 function-class isolation (isotonic strictly
      *worse* than affine on the identical path, factual BCE Δ −0.121/−0.109,
      LCB < 0); G3 all-action control, G4 parent preservation, and G6
      determinism passed. Report:
      `brain/docs/runs/2026-08-25-pb21o-isotonic-hazard-calibration-trial1.md`.
      **Cumulative conclusion [INFERRED]:** the factual-domain
      miscalibration of the V2.1i frozen hazard scorer is
      **representation-limited** — every post-processing remap of the
      current hazard-path logit (affine with/without prior-action
      conditioning, isotonic/PAV) fails the frozen factual gate while
      sometimes improving the all-action aggregate; the two-constraint
      collision persists across the whole family. The hazard post-
      processing lane is closed; the remaining unexplored mechanism is
      representation-level (new hazard-path inputs: 2-step prior window,
      outcome-history features, different context trunk) — each a new
      mechanism needing its own fresh namespace + prereg + control arm,
      **not licensed** by any closed trial.
      **Next-gate leads (architecture ideas ledger):** L4 isotonic →
      `[MEASURED negative]` (PB21O); L7 combined lead fully refuted (both
      function classes); L3 pre-prune calibration, L1 representation-level
      extensions, L5/L6 roadmap leads remain open.
      **Test-suite status (2026-08-25, local):** full module suite is green
      (103/104 modules incl. new `test_v21n_prior_action_hazard_conditioning_v1.py`
      19/19 and `test_v21o_isotonic_hazard_calibration_v1.py` 11/11) except
      the two pre-existing environmental failures in
      `test_dgx_launch_contract.py` (POSIX `/tmp`/`realpath` bash contracts
      that only resolve on the Linux DGX host; Windows git-bash target).
      `test_v21_artifact_verifier.py` shipped in `5ead7b5` without a
      `sys.path` shim for `brain/scripts` and was fixed test-only. The new
      audit module `test_v21m_posthoc_mechanism_audit_v1.py` (22 contracts:
      frozen constants, MIX35 weight algebra, weighted-solver guards +
      determinism + zero-weight context case, burn-in prior replay semantics,
      nomination rule, create-only publish, parent-integrity/OOF
      reconstruction) passes 22/22. All V2.1 workstreams remain
      preregistered; the audit is post-hoc and non-qualifying.
      **L8 hazard diagnostics (2026-08-26, READ-ONLY on sealed PB21O
      evidence — no partition, no publish, no frozen-state change):**
      two probes closed the last open calibration hypothesis and diagnosed
      the binding constraint. (a) Information-ceiling probe: factual
      aggregate AUC **0.631 (frozen base) → 0.708 (prior-conditioned
      OOF)**; per-action 0.649–0.746 → 0.679–0.746; in-sample PAV factual
      ECE **0.0039 (fold0) / 0.0082 (fold1)** vs the trial's cross-fit
      PAV 0.0842/0.0625; the hazard logit beats a frozen linear belief
      readout in 7/10 factual cells. Factual positive rates per applied
      action are sparse (0.067–0.375; n 413–747). (b) Block-cap PAV sweep:
      cross-fitting a frozen block-count cap K ∈ {2,4,6,8,16, full} on the
      sealed OOF logits gives factual ECE (fold0/fold1) affine
      0.0834/0.0644, K=2 0.0936/0.0873, K=4 0.0836/0.0663, K=6
      0.0800/0.0668, **K=8 0.0798/0.0625 (best)**, K=16 0.0837/0.0621,
      full PAV 0.0842/0.0625 — **no member clears the 0.05 G1 limit on
      either fold**. Conclusion [INFERRED on measured anchors]: the entire
      monotone-calibration spectrum saturates at ~0.06–0.08 factual ECE;
      the factual failure is **ranking/signal-density-limited** (AUC 0.71),
      not calibrator-limited; a PB21P block-cap calibration trial is
      **refuted at probe level and will not be run**. The hazard
      post-processing family (L1/L4/L7 + L8) is now closed. Ledger entry:
      `brain/docs/ARCHITECTURE_IDEAS_LEDGER.md` **L8**; probes:
      `brain/scratch/probe_hazard_info_ceiling.py`,
      `brain/scratch/probe_pb21p_blockcap_pav.py` (both re-run 2026-08-26,
      deterministic, reproduce these numbers exactly). Next mechanism, if
      any, is representation-level (new hazard-path inputs) and — because
      ranking is the binding constraint — should carry a **ranking gate**
      (factual AUC over the 0.708 ceiling), not only a calibration gate;
      it needs its own fresh namespace + prereg + control arm and is
      **not licensed** by any closed trial.
      **L9 2-step prior-action window probe (2026-08-26, READ-ONLY, no
      publish):** the single-change representation-level extension of the
      measured 1-step prior-action mechanism (PB21N) was tested as a
      signal probe on the PB21O-CAL partition (deterministic replay of the
      frozen parent; three zero-extended hazard arms retrained per OOF
      fold: base 2H / 1-step 3H / 2-step 4H; 33 s wall). Factual aggregate
      AUC: base **0.6706**, 1-step **0.7060**, 2-step **0.6973** — the
      2-step window **fails to clear the 1-step ceiling** and is worse on
      actions 1 (0.664 vs 0.711) and 2 (0.638 vs 0.674); its in-sample PAV
      ECE ceiling also degrades on the binding actions (fold0 a0 0.0550 vs
      0.0416; fold1 a0 0.0593 vs 0.0562). Conclusion [INFERRED on
      measured anchors]: a longer window adds trunk capacity / fit
      variance, **not** ranking signal; the prior-action **window-length
      axis is closed at K=1**; **no PB21Q trial is warranted** (refuted at
      probe level, same discipline as PB21P). Ledger: L9; probe:
      `brain/scratch/probe_pb21q_2step_window_signal.py`. The remaining
      unexplored representation-level directions are orthogonal to window
      length (different context trunk, outcome-history features,
      richer candidate-action encoding, different upstream representation);
      none is motivated by this probe and each would need its own fresh
      namespace + prereg + control arm.
      **L10 outcome-history hazard features (2026-08-26, READ-ONLY, no
      publish):** a representation-level direction orthogonal to the L9
      window-length axis. Motivated by a first measured [MEASURED]: the
      factual hazard target is **not exchangeable over ticks** — on the
      sealed PB21O evidence (`brain/scratch/probe_outcome_history_signal.\
py`), P(h_t=1 | h_{t-k}=1) = 0.193/0.182/0.213/0.231 for k=1/2/3/4
      (lift 1.46/1.38/1.62/1.76× vs 0.1315 marginal), any-hazard-in-last-K
      lift 1.36–1.53× (K=1–4): persistent multi-tick hazard
      autocorrelation. The signal probe
      (`brain/scratch/probe_pb21r_outcome_history_signal.py`; deterministic
      replay of the frozen parent on PB21O-CAL; two zero-extended arms
      retrained per OOF fold — base 2H vs hist 2H+1-dim raw scalar =
      recent-hazard count in last K=4 applied-ticks / 4, mean 0.0898, 28.5%
      nonzero; init-identity max|Δ|=0.0; 31 s wall) tested the
      outcome-history mechanism **alone** against base. Factual aggregate
      AUC: base **0.6706** (exactly reproduces L9's base — harness
      self-validation passes), hist **0.6583** (*below* base; per-action
      0.623–0.702 vs 0.634–0.704). In-sample PAV ECE ceiling: hist
      fold0 0.0188/0.0175/0.0328/0.0121/0.0248 vs base
      0.0437/0.0149/0.0312/0.0219/0.0223, fold1 0.0740/0.0097/0.0397/0.0338/
      0.0214 vs base 0.0404/0.0173/0.0325/0.0359/0.0243 (fold1 a0 0.0740
      is the worst single cell in this program). Conclusion [INFERRED on
      measured anchors]: although the target has genuine autocorrelation, a
      *coarse recent-hazard count* adds trunk fit variance, **not**
      ranking signal — the hist arm sits below base and far below the
      1-step action ceiling (0.7060); consistent with L8's sufficiency
      finding (the belief context already encodes recent state/outcome
      information). **No PB21R trial is warranted** (refuted at probe
      level). The target's autocorrelation is preserved as a genuine
      [MEASURED] fact; finer-grained history mechanisms (per-action
      recency-weighted, belief-residual, a dedicated recurrent
      hazard-history state) remain untested but are not motivated by this
      probe. Ledger: L10.
      **L11 recurrent applied-trajectory hazard-history state (2026-08-26,
      READ-ONLY, no publish — REFUTED AT FRESH-PARTITION LEVEL):** a
      representation-level direction motivated by L10's [MEASURED]
      multi-tick hazard autocorrelation, and *non-redundant* with L9
      (window length) and L10 (feedforward scalar): a 1-layer GRU
      (hidden 128, reset to zero at tick 0) over the applied
      (action-emb 32 + hazard-event 1) sequence, whose **causal** state
      through tick t−1 (never including tick t's own event — the label)
      is concatenated into the hazard outcome trunk input
      (`cat([state 120, action 120, hist 128]) = 368 → 256 → head`).
      Probes: `brain/scratch/probe_pb21s_recurrent_history_signal.py`
      (two zero-extended arms retrained per OOF fold, base 2H vs rec
      2H+128, init-identity max|Δ|=4.77e-07), `probe_pb21s_stability_
      check.py` (multi-seed), `probe_pb21s_specificity_perm.py` (1000
      episode-permutations of the applied (action, hazard) sequence,
      OOF-trained modules, no retraining). **Discovery partition
      (PB21O-CAL 167,774,720):** base 0.6499, rec 0.6806 (**+0.0307**;
      multi-seed mean +0.023, all 4 seeds positive); specificity perm
      p = 0.0000 (shuffled-rec 0.6472 ≈ base → gain specifically from
      the sequence, not trunk width). **Fresh disjoint partition
      (PB21S-CAL 167,774,976, contiguous after PB21O-CAL):** base
      0.6711, rec 0.6523 (**−0.0188** — the rec arm is *worse* than
      base on unseen data); specificity perm p = 0.104. Conclusion
      [INFERRED on measured anchors]: the discovery gain **does not
      generalize** — it is partition-specific noise consistent with the
      sparse factual hazard rates (6.7–37.5% per action). A
      publish-once trial must not be tuned against its own test
      partition, so the fresh-partition check is the correct pre-trial
      gate, and it fails. **No PB21S trial is warranted**; the drafted
      prereg (`2026-08-26-pb21s-recurrent-hazard-history-v1.md`) is
      marked REFUTED-BEFORE-EXECUTION. **Determinism lesson:** the
      builtin `hash(str)` is process-salted (PYTHONHASHSEED) and MUST
      NOT seed parameter init — it made the rec arm's GRU init
      non-deterministic across processes (base arm identical, rec arm
      drifted 0.6846 → 0.6479 between two runs of the same partition);
      fixed with `zlib.crc32(name)`-stable seeds. Ledger: L11.
      **Stale PB21M runner crash (2026-08-26, triaged — NO re-execution):**
      a background notification reported a crash of
      `brain/scripts/v21m_posthoc_mechanism_audit_v1.py` inside Arm A's
      MIX35 fit (`ValueError: targets must be binary and weights
      positive`, line 286/498/530/995). Attributed to a **stale pre-commit
      working-tree process**: the committed runner (`561f5ae`, the sealed
      one) raises "weights **non-negative**" at the same guard and is
      1077 lines (crash line numbers and message text do not match it;
      "weights positive" appears in no git revision). The crash was in
      `arm_a`, *before* any publish, and the create-only `_publish` guard
      (`result already published`) would have blocked any overwrite. All
      five sealed PB21M artifacts re-hashed 2026-08-26 and match the run
      report byte-exact (result `0d0be6f7…`, attempt `ec2dcf6f…`, evidence
      `e5fd6c67…`, registration `f9c2d13e…`, artifact-completion
      `3f2fbc2e…`; sizes 143,408 / 368 / 8,848,530 / 2,075 / 629). The
      audit remains sealed and non-retryable; nothing was modified or
      regenerated.
   d. **Intake-mechanism program** (only path that ever moved memory causality:
      ideal-evidence probe in Phase 2) — folded into V2 roadmap.
   e. Optimizer stability workstream deferred — σ(24k)=0.102 at FIXED but no
      mean improvement; SCHEDULED reduces σ to 0.046 without mean gain.
5. Provenance helper (`run_provenance.py`) mandatory in all new scripts:
   bank digest · init param digest · seeds · det flag · torch/CUDA versions.
6. Multi-model ensemble batching (deferred prereg) — motivation weakened;
   re-evaluate only if a future axis shows budget-bound instability.
7. Post-V1 memory program gated on an update rule that passes ideal-evidence probe.

## Session totals (2026-08-22 full day → 2026-08-24)

~62 commits pushed. All experiments preregistered or audit-only; every number
traceable to Spark run artifacts under `runs/`.
