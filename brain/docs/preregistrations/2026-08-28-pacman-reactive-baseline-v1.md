# T1-4 — Reactive baseline (never weakened) trained on TRAIN (preregistration)

**Date:** 2026-08-28
**Status:** FROZEN (design contract; run artifacts go to `brain/runs/embodied-reactive-baseline-v1/`)
**Mode:** local CPU only, single-thread where determinism needs it, CUDA hidden.
**Upstream contract:**
- `brain/docs/preregistrations/2026-08-27-pacman-harness-v1.md` §8
  ("Reactive baseline (never weakened): a fixed-depth, pixel-only reactive
  policy: a small CNN/MLP on the last 1–4 rendered frames that greedily
  maximizes immediate pellet gain while minimizing measured ghost proximity
  (BFS-free, pixel-derived). It is a genuine baseline, trained on the same
  TRAIN partition under the same budget class as the candidate, and is the
  Q2/Q3 comparison arm. It must not be weakened to make the candidate look
  better; its own competence is reported.");
- `brain/docs/preregistrations/2026-08-28-pacman-corpus-v1.md` (the shared
  TRAIN corpus, `corpus_sha256=59c41b6dc71f…`);
- `brain/docs/preregistrations/2026-08-25-realtime-embodied-qualification-v3.md`
  §7 Q2 (the candidate-minus-reactive paired comparison).

This preregistration authorizes **training** the frozen `ReactiveBaselineV1`
on the shared TRAIN corpus under the candidate's budget class, and reporting
its own competence. It is the Q2/Q3 comparison arm. It does **not** open
DEV/CAL/TEST, does not train or evaluate the candidate, and does not publish
a checkpoint claim. **Never weakened:** the baseline is a real, non-
privileged, pixel-reactive model given the full 4-frame temporal context
(the strongest value in the registered 1–4 range) and the candidate's full
training budget. If it scores below random, that is a recorded datum, not a
reason to lower it.

## 1. The model (frozen): `ReactiveBaselineV1`

`brain/src/irene_brain/v2/reactive_baseline.py`. A small, **non-RECURRENT**
CNN/MLP:
- input: a stack of the last `N=4` rendered 16×16 frames (nearest-
  upsampled to 32×32, exactly as the candidate's deployed maze policy
  encodes pixels) concatenated with a one-hot of the previous applied action
  (5 classes). So the sensory boundary is the last 4 frames + the previous
  action; no clocks, frame IDs, reward, hazard, or privileged state (C8);
- architecture: `Conv2d(12→24,s2)→BN→MaxPool→Conv2d(24→48,s2)→BN→
  Conv2d(48→48,s1)→BN→flatten→Linear(+prev_action one-hot)→128→ReLU→
  Linear(5)` action logits;
- **133,773 parameters** (frozen, measured). It is intentionally far smaller
  than the ~1M-param recurrent Core V2 candidate — that param-count and
  recurrence difference is exactly the "no world-model / no thoughtlet
  field / no belief / no episodic memory" distinction this baseline exists
  to isolate. It is **not** a weakened strawman: it has the full temporal
  context and the full training budget.

## 2. Supervision (frozen)

Same target as the candidate's BC: 5-class cross-entropy on the teacher's
applied action class (`trajectory_objective.control_action_class` → 0..4).
The same corpus, same target distribution. The objective is plain
cross-entropy on the aggregated 5-class head (the reactive baseline has no
per-slot/per-cycle structure to mirror — the candidate's `action_values` is
itself the 5-class aggregate that `deployed_decision_loss` trains; matching
that 5-class target is the fair apples-to-apples objective).

**Window construction (frozen):** for each corpus episode, for tick
`t >= 1`, the training sample is:
- input frames: the last 4 rendered frames up to and including tick `t`,
  right-aligned; ticks with fewer than 4 history frames are left-padded with
  the episode's first frame (the reset frame) so every sample is exactly
  `N=4` frames;
- input `prev_action`: `target_action[t-1]` (the applied action at tick
  `t-1`; the corpus stores it as `prev_action[t]`);
- label: `target_action[t]` (the teacher's applied action at tick `t`).
This is the same "act on the current frame given recent history" task the
candidate learns; the reactive baseline receives the history explicitly
(last 4 frames) instead of via recurrence. Tick 0 has no `prev_action`
context, so samples start at `t=1`.

## 3. Budget class (frozen — matched to the candidate's T2-1 reference)

Mirrors `stage_v20_core_v2_deployed_loss_baseline.py`'s optimizer contract:
- optimizer: `AdamW(lr=5e-4, weight_decay=1e-4)`;
- gradient clip: `max_norm=1.0`;
- batch size: 16;
- **total optimizer steps: 6000** (the V2.0j `TOTAL_STEPS` budget);
- **confirmatory seed cohort: `[42, 142, 242, 342]`** (the same 4-seed cohort
  used for the V2-A/V2-C confirmations);
- one full deterministic pass over the corpus per seed (no early-stopping on
  an unseen split — there is no DEV/CAL here; the 6000-step cap is the
  budget, matching the candidate's).

## 4. Determinism protocol (mandatory)

```
torch.use_deterministic_algorithms(True)
CUBLAS_WORKSPACE_CONFIG=:4096:8
cudnn.deterministic=True, benchmark=False
tf32 off both paths
seed banks via zlib.crc32(name) — NEVER builtin hash()
```
Model init uses a `torch.Generator` seeded from `zlib.crc32(f"reactive.<seed>")`.
Two runs with the same seed MUST byte-match every checkpoint parameter tensor
and every per-step loss. A determinism check (§6, gate R) verifies this on a
fixed seed.

## 5. Causal-integrity invariant (C8, binding)

The model input is built only from the corpus `frame` tensors (rendered
16×16 pixels) and the previous-applied-action one-hot. No reward, hazard
event, cleared/caught flag, maze map, task label, environment token, or
clock/frame ID is read. The corpus itself was already proven causally clean
(T1-3 gate D). A static audit (§6, gate R) proves the training script builds
no model-input tensor from any evidence-only field.

## 6. Acceptance gates for THIS build (run-local)

A reactive-baseline build is accepted only if all hold:

1. **R — Determinism:** two runs at seed 42 byte-match the final checkpoint
   parameter tensors and the per-step loss curve.
2. **S — Causal boundary (C8):** the static audit passes — the training
   script's input tensors are built only from `frame` + previous-action;
   zero reads of reward/event/terminated/truncated/meta fields into the
   input path.
3. **T — Budget fidelity:** exactly 6000 optimizer steps, batch 16,
   AdamW(5e-4, 1e-4), grad-clip 1.0, seeds [42,142,242,342].
4. **U — Non-degenerate baseline (not a strawman):** over the 4 seeds, the
   model's per-class recall on its own training corpus (a self-evaluation,
   **not** a qualification claim) shows all 5 action classes emitted
   (per-class recall > 0 for every class in the aggregate), and training
   loss decreases from its initial to final value. This proves the baseline
   actually learned a reactive policy rather than collapsing to one action.
   (Its *competence against the candidate* is the separate Q2/Tier-3
   comparison, under its own preregistration — this gate only proves the
   baseline is a live, non-degenerate model.)

Gate pass is recorded in
`brain/runs/embodied-reactive-baseline-v1/
2026-08-28-reactive-baseline-v1.json` (create-only). On any gate failure,
the build is NOT accepted, the failure is recorded verbatim, and the next
iteration is a new probe — no silent re-tune of §1/§2/§3.

## 7. What this preregistration does NOT do

- No candidate training, no candidate evaluation, no CPU-QUAL, no TEST open.
- No DEV/CAL/TEST seed used (TRAIN corpus only). No re-baselining of any
  sealed pin. No DGX/Spark dispatch (local CPU only, per the 2026-08-26
  decision record).
- This gate set does NOT establish the candidate beats the baseline — that
  is the Q2 paired one-sided 97.5% lower-bound test, run under the Tier-2/3
  preregistrations on the sealed TEST layouts. The baseline's own numbers
  are reported honestly either way.

## 8. Claim boundary

Passing §6 permits only: "a non-degenerate, causally-clean, deterministic
reactive baseline (ReactiveBaselineV1, 133,773 params, 4-frame + prev-action
context) was trained on the shared TRAIN corpus under the candidate's
budget class." It does not permit any "the candidate beats the baseline"
claim — that requires the Tier-2/3 paired comparison on TEST.

## 9. Next (queue)

1. **T1-4** (this) — train the reactive baseline; record the checkpoints +
   acceptance.
2. **T2-1** — integrate Core V2 + `EmbodiedInterfaceV1` heads under the
   deployed decision loss on the same corpus; multi-seed; then **T2-2**
   CPU-QUAL prereg; then the Tier-3 Q2 candidate-vs-baseline comparison.
