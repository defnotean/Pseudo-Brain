# Closed-loop play latency evidence (2026-08-17)

Status: implementation only. Closes the ROADMAP §6 item-6 leftover:
`evaluation/latency_evidence.py` now extends from benchmarks to play. No
live-play (real capture/HID) measurement is claimed; every record here is
`measurement_kind="simulated"` on the manual-clock evaluator.

## What was added

**`evaluation/play_latency.py`** — a fail-closed bridge from closed-loop
play to the latency evidence contract:

- `play_decision_schedule_offsets_ns` — the analytic decision schedule.
  Decision *k* starts at `k * max(decision_interval_ns,
  inference_latency_ns)`: the manual clock advances by the declared
  inference latency and then jumps to the next decision-interval boundary,
  so the schedule is policy-independent and pinnable before measurement.
- `measure_policy_play_latency` — plays one episode through
  `_run_episode_core` with a timing sink and returns the episode report
  plus a validated `LatencyEvidence`. The registration must be pinned
  before the run (simulated kind, `first_measured_attempt_id` equal to
  `warmup + 1`, schedule equal to the analytic one); the episode must
  survive every registered measured decision — a shorter episode is an
  error, never a silently truncated stream. `summarize_latency` then
  re-validates the whole stream against the externally pinned digest.
- `DecisionTiming` + an optional `attempt_sink` on `_run_episode_core` in
  `evaluation/closed_loop_play.py`. The sink is None by default and does
  not alter the episode or its report (pinned by an equality test against
  the plain policy path).

Simulated capture and queueing are free, so those legs are exactly zero;
inference legs equal the declared latency exactly; a rejected envelope
becomes a `submission_failed` attempt carrying the driver's reason string
(for example a stale source frame when latency exceeds the tick period).

## Verification

New `tests/test_play_latency.py` (10 tests, torch-free): analytic schedule
under interval- and latency-bound configs, validated clean-run evidence
(zero capture/queue, exact 4 ms inference legs, attempt IDs following
warmup), sink transparency (the measured episode report equals the plain
run's), stale-frame rejections surfacing as `submission_failed` with the
driver's reason, wrong-schedule and wrong-first-ID registrations rejected
before any run, pinned-digest mismatch failing closed at summarization,
short episodes rejected, and non-simulated registrations refused. The full
play-safe suite (472 tests) exits 0.

## Remaining gap

Physical live-play records (Phase 0B harness integration: real capture,
HID transport, visible-effect deadlines) remain blocked on the armed
capture/control harness, ROADMAP §6 item 5.
