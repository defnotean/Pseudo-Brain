# Frontier capability evaluation contract

Owner target: a general-purpose assistant competitive with frontier transformer
models across coding and reasoning. This document defines evidence requirements;
it does not assert that the current checkpoint meets them.

## Separate four kinds of evidence

1. Implementation checks: protocol parsing, real test execution, reset isolation,
   checkpoint identity, numerical parity, physical fast-state and parameter limits.
2. Development capability: the existing procedural tasks, with known template
   overlap disclosed. These are useful for diagnosing regressions.
3. Independent capability: fresh, versioned external coding, reasoning, instruction
   following and language tasks whose solutions are excluded from model inputs and
   training selection. Pin question IDs, release, scoring code and source hashes.
4. Frontier comparison: actual outputs from a named, versioned frontier model on
   exactly the same questions and scoring harness, with tool access, sampling,
   generated-token/action budgets and latency/cost disclosed. A randomly
   initialized transformer or a published score on a different task set is not
   such a comparison.

## External evaluation candidates

LiveCodeBench supplies coding generation, execution, test-output prediction and
self-repair evaluation, with time-window/version controls. Use pinned questions
and its official scorer, rather than the local filename-relevance proxy.
Source: [official repository](https://github.com/LiveCodeBench/LiveCodeBench).

LiveBench covers reasoning, mathematics, coding, language, data analysis and
instruction following, and provides objective ground-truth scoring. Its own
documentation notes that local-model inference is unmaintained and agentic
coding needs Docker. The repository's adapter therefore emits raw model text
without manufacturing action wrappers or forwarding ground-truth fields; an
official-format/scoring integration still needs to be completed and pinned.
Source: [official repository](https://github.com/LiveBench/LiveBench).

SWE-bench Verified must not be the sole frontier gate. OpenAI's February 2026
audit reports contamination and test-quality problems and recommends SWE-bench
Pro instead. Repository-level engineering should use a versioned, independently
audited task set, with isolation and complete failing/passing test evidence.
Source: [OpenAI benchmark audit](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/).

## Reporting and stopping

- Preserve per-question outputs, failures and refusal/timeout outcomes. No
  selectively omitted failures, oracle answer injection, heuristic patch credit,
  or counting correct filenames as solved tasks.
- Report each domain separately, not only a favorable aggregate. Compare paired
  results and uncertainty; a thirty-task toy score cannot establish generality.
- Report training-data exposure separately from reset isolation. Resetting state
  does not remove answers or algorithm templates learned into weights.
- Keep instruction-following, self-repair, factual correctness and long-context
  retention as separate measurements. Stored-vector cosine similarity is not
  a question-answering result.
- Measure full memory (weights, all persistent tensors, input storage and transient
  activations) alongside the 4,096-byte physical fast-state requirement.
- A candidate must pass the user's strict numerical and size gates before training
  or promotion. The compensated candidate has eight logical vectors plus eight
  numerical residual vectors and must disclose that tradeoff.
- Do not substitute the repository's 1B checkpoints for the required <36M model.

## Current gaps

No actual frontier-model baseline has been run. The current research candidate
is `parallel_depth_v1`, with22,387,458 parameters and a4096-byte physical state.
That state packs eight logical recurrent vectors plus eight numerical residual
vectors; it does not implement16 independently role-labelled thought vectors.
The paired22,970,882-parameter transformer is a locally trained research control,
not a frontier assistant. It retains a full prefix for the tool-policy comparison
and does not meet the recurrent state bound.

The last completed V4 comparison scored0/32 HumanEval for both models and1/32
GSM8K for the recurrent model versus0/32 for the transformer. These are already
observed development tasks. V5 stopped at its registered training time limit
after57,559 recurrent updates; the transformer arm never started. Its partial
checkpoint has been preserved, but no completed V5 comparison or independent
V5 capability evaluation exists. Training loss and parameter count cannot fill
those evidence gaps.

New isolated tool sessions, executed demonstrations and paired policy evaluation
are implemented. CPU infrastructure controls include scripted repairs, which
are not learned behavior. Native carried-state, multi-turn, corrected action-loss
gradient, and policy training preflight checks passed. The completed V4 tool
baseline recorded48/48 cases per model with zero completions/repairs and no
missing cases. Policy fine-tuning completed3520 updates/model; policy development
loss improved while all three foundation-domain development losses worsened for
both models. The actual post-training tool evaluation also completed with0/48
task successes/repairs per model, despite more recognized tool actions. Coding/math
retention evaluation completed with0/32HumanEval and0/32GSM8K per model, no missing
responses. Read-only diagnostics show near-perfect sampled-training token accuracy
but poor development code-body generalization; the pointer helps on fixed teacher
prefixes. The training failure bank contains only RuntimeError observations,
unlike the syntax/import errors seen in actual behavior. Consult CURRENT_WORK.md for authoritative
process/evidence locations.
Retrieval coverage, broader real-project tasks, fresh general-purpose evaluation
and a named frontier-model comparison remain missing.

The legacy default policy still has the separately audited discarded episodic
readout and unused recurrent-stack problems. The new candidate uses an opt-in
path rather than silently treating those legacy components as functional.
Local model tests and training remain prohibited while the owner is gaming;
all execution uses the authorized Colab or an identified VPS.

The handoff's original A/B/C thresholds,90% executable-action and module-binding
targets, and a model-generated repair receipt remain separate outstanding
milestone requirements. The new48-case policy bank does not silently replace
those criteria. Legacy regression counts alone cannot establish them:
`test_triton_and_35m.py` allows up to40M parameters, and portions of
`test_agent_heldout_capability.py` validate telemetry or a historical zero-score
receipt. The active candidate's drivers enforce <36M and its capability must
be measured directly. Preserve historical regression evidence while requiring
fresh qualification of the final candidate.

Parallelizable recurrent models are a defensible research direction, but their
published results do not transfer automatically to this implementation. For
example, minGRU/minLSTM language experiments use multi-layer architectures with
additional temporal and feedforward components. The candidate's own capacity,
numerical constraints and training regime require direct capability evidence;
neither the legacy single-slot policy nor the newer candidate inherits published
results merely by using a recurrent mechanism.
Source: [Were RNNs All We Needed?, architecture appendix](https://arxiv.org/html/2410.01201v2).
