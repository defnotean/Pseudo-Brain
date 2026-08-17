# Real-time evidence protocol

Pseudo-Brain must not claim human-speed or real-time control from simulator
throughput alone. Timing evidence is registered before collection, preserves
every measured attempt, and keeps three claim scopes distinct:

1. `simulated` establishes only timing inside the registered simulator and
   clock domain. It is not desktop or physical latency evidence.
2. `desktop` measures **capture start through control submission** on one
   registered machine and software stack. It includes capture work, queueing,
   inference, and submission. It does not include display stimulus latency,
   actuator latency, a visible effect, or proof that the chosen action worked.
3. `physical` measures a registered observable stimulus through the first
   registered visible effect, as well as capture-start-to-submit. It requires
   an independently defined effect detector and deadline.

`irene_brain.evaluation.latency_evidence` is a pure offline validator. It does
not capture a screen, read a clock, inject input, start threads, access a
network, or use an accelerator.

## Preregister the run

Create a `LatencyRunRegistration` and archive its canonical JSON plus SHA-256
before collecting measured attempts. Pass that archived digest separately as
`expected_registration_sha256` to `summarize_latency`. Rewriting and re-hashing
the registration after seeing results is not acceptable and will not match the
external pin.

The registration binds all of the following:

- campaign and run IDs, measurement scope, environment, and architecture
  variant;
- architecture identity and architecture-manifest digests;
- checkpoint, model configuration, evaluation code, and workload digests;
- the exact measured-attempt schedule as nanosecond offsets from the first
  scheduled measured attempt, plus its recomputed canonical digest;
- device and runtime manifest digests plus precision mode;
- clock source, clock domain, and synchronization method;
- capture path and resolution, action transport, and process priority;
- warm-up count, the first measured attempt ID, exact measured-attempt count,
  and submission deadline;
- for physical runs, the separately registered visible-effect deadline.

All digest fields are lowercase SHA-256 values. The device and runtime
manifests must contain the details needed to reproduce the measurement, such
as hardware, OS/driver versions, runtime/library versions, power mode, clock
policy, and relevant process placement. The workload manifest must be frozen
independently; putting mutable labels behind its digest invalidates the
evidence.

`measured_attempt_schedule_offsets_ns` contains one nonnegative offset per
measured attempt, begins at zero, and is strictly increasing. The validator
recomputes `attempt_schedule_sha256` from schema version, warm-up count, first
measured ID, measured count, and those offsets. At collection time the absolute
monotonic clock origin may vary, but each recorded `scheduled_ns` minus the
first recorded `scheduled_ns` must exactly equal the registered tuple. Thus
changing an attempt's relative scheduled offset after preregistration is
rejected even when IDs and counts still match.

Warm-up attempts are excluded according to the preregistered count. Every
registered *measured* attempt is mandatory, in exact contiguous ID and relative
schedule order. A shorter stream, duplicate, reordered attempt, rescheduled
attempt, or extra attempt is rejected.

## Attempt record

Each `LatencyAttempt` records a scheduled time, mandatory capture-start time,
and one explicit terminal status:

- `capture_failed`: capture began but produced no observation;
- `inference_timeout`: capture completed and inference began, but the
  registered inference deadline expired;
- `submission_failed`: inference completed but action transport failed;
- `submitted`: action submission completed; valid only outside physical scope;
- `effect_observed`: physical submission produced a timestamped visible
  effect;
- `effect_timeout`: physical submission completed but no effect was detected
  by its registered timeout.

Milestones are optional only when their status makes them impossible. The
validator rejects holes and contradictions, such as a submitted status without
a submission timestamp, a capture failure with an observation timestamp, or
an effect observation without a visible-effect timestamp. Failure statuses
also carry a nonempty reason. All pipeline timestamps use the registered
monotonic clock and are nondecreasing.

A physical attempt additionally requires `stimulus_presented_ns`. Every
submitted physical attempt must end as `effect_observed` or `effect_timeout`;
plain `submitted` is rejected. Capture, inference, and submission failures are
still complete terminal outcomes and count as physical effect-deadline misses.

## Fail-closed denominators and reported values

Submission latency is `control_submitted_ns - capture_started_ns`. Missing
submission is always a submission-deadline miss. Exact-deadline submission is
on time; only a value strictly greater than the deadline is late. The miss rate
denominator is every registered measured attempt, not only successful calls.

For physical scope, effect latency is
`visible_effect_ns - stimulus_presented_ns`. A missing effect for any reason,
including an earlier pipeline failure, is an effect-deadline miss. An observed
effect exactly on its deadline is on time. The physical effect miss-rate
denominator is again every registered measured attempt.

Successful phase samples report sample count, minimum, nearest-rank p50, p95,
p99, maximum, and mean for:

- capture start to observation ready;
- observation ready to inference start;
- inference execution;
- inference finish to control submission;
- capture start to control submission;
- for physical runs, stimulus to visible effect.

Each phase's sample count is explicit because failed attempts may not reach
that phase. Status counts, total attempts, successful submissions, deadline
misses, and miss rates prevent those partial distributions from hiding
failures.

The report contains the complete raw attempt stream, a canonical raw-trace
SHA-256, and a canonical full-report SHA-256. `LatencyEvidence` recomputes all
counts, distributions, and both hash relationships during construction. It
also requires the canonical built-in numeric types, so JSON-distinct but
value-equal substitutions such as an integer count changed to a float are
rejected. An internally inconsistent report therefore cannot be instantiated.
These hashes provide integrity and identity, not authenticity: registrations,
raw traces, and reports still need independent archival, access control, and
external digest pins (or signatures) for a consequential claim.

## Benchmark controls and claim gate

- Use the registered warm-up count and exact relative measured-attempt schedule.
- Compare architectures under the same workload, capture path, resolution,
  precision, process priority, device/runtime state, and action transport.
- Report parameter-matched/equal-FLOP and measured-latency regimes separately.
- Treat p99 and all-attempt deadline-miss rate as primary timing results;
  average latency alone is insufficient.
- Report accuracy or task return, power, memory, and latency together. Fast
  ineffective actions are not a controller win.
- Measure clock-domain synchronization error and include it in the registered
  method. Do not combine timestamps from unsynchronized clocks.
- Run desktop or physical collection only in a separately armed performance
  window, never while the owner is gaming.

The initial engineering screen is a 60 Hz capture-start-to-submit deadline of
16,666,667 ns, with a separately preregistered physical stimulus-to-effect
budget. Passing either screen is evidence only for that exact registered stack
and workload. It is not by itself evidence of general real-time performance,
human-like cognition, useful game play, or a scientific breakthrough.
