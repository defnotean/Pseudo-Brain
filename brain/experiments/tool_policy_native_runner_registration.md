# Native execution details for tool-policy evaluation v1

Apply the existing tool_policy_evaluation_registration.md without changing its
prompts, limits, checkpoints, scoring or fixed48-task denominator. Preserve
request-file order, and alternate from_scratch then repair for each request.
Run the recurrent baseline first, followed by transformer. Each child has a
hard1800-second subprocess deadline including imports and setup; completed case
artifacts are preserved. The controller independently recomputes the48-task
scores from those artifacts even if the child is killed before final reporting.
Missing cases stay explicit failures, and the run is labelled incomplete.

Use seed198 for Python, NumPy and Torch. Each child applies the already validated
strict deterministic mode. Freeze all imported project sources, both runners,
registrations, and provenance helper before native launch. Verify their hashes
again in the launcher and controller. No model operations run in the controller.

Before launch, all existing owned GPU jobs must be terminal, both recurrent
ingestion and multi-turn parity gates must have passed, and the frozen procedural
bank integrity report must have passed. The native action-gradient gate also
precedes this queued baseline, because its outcome determines whether the later
training protocol is usable. No optimizer updates are part of this evaluation.

CPU preflight must check CLI imports and the fixed-denominator behavior under
interruption, duplicate/unplanned-result rejection and impossible repair credit.
These tests run on Colab with CUDA hidden and one CPU thread.
