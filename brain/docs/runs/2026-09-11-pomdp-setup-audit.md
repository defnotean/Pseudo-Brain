# POMDP setup audit and compensated-state candidate

Status: investigation and candidate training in progress. Frontier-level capability
has not been demonstrated. The original champion and historical receipts are preserved.

## Colab and independent reasoning update

The owner selected their existing Google Colab setup instead of DGX Spark. The
authenticated WSL CLI allocated a dedicated NVIDIA A100-SXM4-40GB session. The
initial native preflight failed before training: a float64 Triton tile requested
262,144 bytes of shared memory against the A100 limit of 166,912 bytes. A separate
source revision now sizes the tile by element size. Native preflight then passed:
182-token checkpoint logit delta 5.684341886080802e-14; worst of batched 31/257/513
token probes 1.687538997430238e-13; 2,048-token scan forward delta 3.55e-15 and
gradient delta 2.13e-14. All remain below the unchanged 1e-6 tolerance. Native
runtime evidence is saved separately from earlier CPU/interpreter tests.

The bounded Colab candidate uses the current corrected feedback pipeline and
supervised-position readout. It is not an exact resume of the older CPU experiment.
The recurrent scan is parallel, but the pointer head retains a Python time loop;
do not describe the complete pointer-enabled training path as O(log T).

An independent, fixed 16-question GSM8K diagnostic of the unchanged historical
checkpoint scored **0/16**. Raw outputs generally attempted coding-tool actions
instead of answering ordinary arithmetic questions. Labels were excluded from
model input, with reset before each question; every failed output was retained.
This is a historical public-data sample, not the complete official benchmark or
a decontaminated frontier comparison. Dataset revision:
`3101c7d5072418e28b9008a6636bde82a006892c`, test-file SHA-256:
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
See `brain/runs/gsm8k-original-20260911/` and
[the upstream dataset](https://github.com/openai/grade-school-math).

## Measured baseline

The actual POMDP checkpoint has 34,372,860 trainable parameters. Its fast working
tensor is 16 x 64 float32, exactly 4,096 bytes. Total persistent tensors in the
audited state occupy 41,360 bytes, before Python objects, immutable prompt tokens,
model weights, transient activations and execution logs. The 4 KB figure is not
total inference memory.

The 182-token pointer-enabled, gated-skip streaming/parallel audit fails the
original float32 checkpoint at 1.9073486328125e-05 absolute logit difference versus
the owner's strict <1e-6 requirement. Original tests did not cover those heads;
the sequential forward wrapper also omitted their inputs. That wrapper is fixed.

Perturbing the active slot changes next-token logits by 13.82 in the bounded
causal probe. Perturbing other working slots or episodic memory changes logits by
exactly zero. The software agent's default path disables routing and discards the
retrieved episodic context. The allocated deep recurrent stack, cognitive input
gate and router receive no gradient from the audited language objective.
Non-language head gradients are not expected from language-only loss, but these
results refute claims that the present coding policy uses deep multi-slot memory.

The open training bank contains 120 tasks but only 27 normalized reference-code
templates and 17 duplicate task IDs. Eight of ten Level B tasks reuse training
templates after renaming the target symbol. Level B therefore measures resampling
within open templates, not unseen algorithms. Level C's local templates do not
overlap this bank, but the broader streaming corpus already includes BFS and LRU
algorithms; using it would invalidate a blanket claim that those families remain
unseen. No broad streaming pretraining was performed in this experiment.

## Runtime evaluation with unchanged weights

The new receipt is `brain/runs/pomdp-repair-20260911-runtime-audit/three_tier_benchmark_receipt.json`.
It uses the old checkpoint, with corrected protocol boundaries and pointer source
isolation. It predates the later compact exception summary and compensated-state
candidate and must not be attributed to the latter.

| Metric | A | B | C |
| --- | ---: | ---: | ---: |
| Task completion | 1/10 | 2/10 | 0/10 |
| Executable action rate, legacy classifier | 94.7% | 88.9% | 50.0% |
| Task relevance, legacy classifier | 73.7% | 88.9% | 45.0% |
| Verified autonomous repair episodes | 0 | 0 | 0 |

Correct targeting and parseability improved; actual completion did not. An action
containing the right filename is not evidence of algorithmic competence. The
legacy relevance/progress classifier uses strings and does not measure partial
test success. Milestone acceptance is false.

## Implemented corrections

- Shared task/observation formatting in `agent/pomdp_protocol.py`; one observation
  wrapper and one consumed action terminator, including truncation boundaries.
- Pointer source remains the initial task specification. Observation tokens update
  recurrent state and never extend the pointer source.
- Response-only pointer history in training agrees with inference resets.
- Correct next-token loss boundaries: train prediction of EOS, not the first token
  of the following environment observation. Deliberately broken first actions are
  teacher-forced context with zero imitation loss.
- Pointer targets are derived from exact token spans rather than fixed vocabulary
  IDs; class names are covered and target indices equal actual next-token labels.
- Training trajectories execute open reference tasks in the environment. Compact
  exception summaries preserve the error type and normalize temporary paths.
- Repairs require failure, a repair phase ingested into recurrent state, subsequent
  code mutation, and hidden-test-approved FINISH. Scripted plans are excluded.
  Receipts retain actions, observations, phase transitions, return codes and source.
- Checkpoint loading is strict; training performs an invariant audit before updates.
  Run directories are create-only; candidates do not overwrite champions. The run
  records seeds, checkpoint hashes, source hashes and the exact training bank hash.
- The repository test dispatcher now uses pytest, collecting both pytest functions
  and unittest cases. Previously unittest-only discovery could report a module as
  passed while executing none of its pytest tests. Discovery finds 1,576 tests.
- The old transformer script now labels its randomly initialized models and
  handwritten repair heuristic explicitly. Its cache prefill uses the reported
  context length instead of silently capping at 32 tokens. It remains a systems
  microbenchmark and cannot establish competence or frontier equivalence.

## Registered numerical candidate

Registration: `registrations/2026-09-11-pomdp-compensated-state-v1.md`.
The opt-in `compensated_state` mode uses float64 arithmetic and paired float32
high/residual vectors inside the original physical working allocation. There are
eight logical thought slots and eight residual slots. This is a capacity tradeoff,
not sixteen independent thoughts. Routing is explicitly rejected in this mode.
Weights, episodic memory and temporary arithmetic use more memory; speed and GPU
parity have not been qualified.

The actual-checkpoint 182-token preflight passes at 9.592326932761353e-14. Batched
probes at 31, 257 and 513 tokens pass <1e-6, while the physical fast tensor remains
4,096 bytes float32. Parameters remain 34,372,860. No logits are rescaled and no
tolerance is relaxed. A bounded 350-step open-task candidate is being trained in
`brain/runs/pomdp-compensated-20260911-candidate-v1/`; its exact 177-file source
snapshot is preserved under `source/`. No final capability claim is available yet.

## Verification and remaining blockers

Preflight: 58 model/agent tests pass, including the requested original 32; another
50 streaming/memory/scan tests and five subtests pass. Full-suite execution is in
progress; the result will be recorded separately. CPU-only tests and Triton
compilation/interpreter checks are not evidence of A100 runtime parity.

The current environment validates Python syntax and contains actuator paths, but
subprocess-generated Python is not OS-isolated. It must not be described as a
secure sandbox for arbitrary untrusted code. The RUN_TESTS path also executes
files as scripts; pytest-style functions need real test discovery before that
tool can serve as reliable general software-engineering feedback.

Frontier comparisons require independently held-out coding and reasoning tasks,
matched tool access, measured latency/compute, checkpoint/data provenance, and an
actual named frontier baseline. None exists in the repository's current reports.
The larger 1B artifacts are outside this task's <36M parameter requirement and
are not silently substituted for the POMDP model.
