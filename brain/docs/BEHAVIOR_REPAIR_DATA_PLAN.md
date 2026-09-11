# Behavior repair data and next training stage

Current status: the codec and actual two-action model-prefix collection are
complete. There are 58 verified teacher continuations from 64 preserved actor
attempts; six token-cap attempts are excluded. All selected cases are audited
MBPP training cases. Neither actor created the expected target module within its
two-action prefix. A fixed eight-case prompt-order probe failed to improve that
binding (0/8 before and after for each actor). Keep the original audited prompts.

The clean-demonstration collection completed READ, canonical WRITE, public
RUN_TESTS and externally verified FINISH on all 357 audited MBPP training cases.
See `experiments/mbpp_clean_trajectories_v1_registration.md`. All saved records
passed the separate provenance/encoding audit: 298099 input tokens and 56763
supervised tokens, no failed or replaced cases. These examples complement
the 58 behavior-prefix records and 220 varied-fault demonstrations. The new matched
protocol is registered in `experiments/broad_policy_v2_registration.md`: two passes
over all987 policy records, with one distinct code/reasoning/language foundation
example after each policy update. Both actors start from completed policy-r2
checkpoints. This gives7896 updates,5922 unique foundation rows and75% replay by
update count. CPU preparation passed in24.96 seconds:4579904 input/1835388
supervised tokens,1492810 supervised tokens from foundation replay (81.33%).
Schedule SHA256:16cbc46c6fe8b380024634520cfc210e2879737278ce8d1b2a5eeed3087cd949.
Both native preflights passed: eight discarded updates per actor, finite losses
and gradients; recurrent state4096B and parity error at most6.56e-14. The full
matched run is now live on Colab at `/content/pb-broad-policy-training-v2-r0/training`.
Frozen training source SHA256:
442afc835cef0b4ae9f269f48ef353c16401e0ad1e5511fae9587ccdb28baf3d.
Foundation retention and autonomous completion remain empirical requirements.

The following preserves the implemented codec/collection rationale.

This is an implementation plan, not a frozen collection/training registration.
The versioned codec and its remote preflight are now complete. Implementation:
`src/irene_brain/data/behavior_repair_trajectory.py`. The eligible frozen bundle is
`/content/pb-behavior-codec-v2-r2`, source manifest
`8a675c4ed40db80cee711739a82060639b49ab9ed979e75da67ff042bd81c9be`.
All22 boundary/masking/actual-repair checks passed in5.31s, and all596 previous
v1 records matched the independent frozen encoder and observation-frame function
byte-for-byte. No model-generated data or weight updates occurred in that gate.
All test records are explicitly codec fixtures and are rejected by the ordinary
training encoder. The real model collection described below has since completed.

Resolve and hash the completed corrected MBPP case bank first. The procedural
pilot demonstrated fault coverage, but its scripted errors differ from model
behavior. Use the broader audited training cases to address that mismatch.

Add an explicit versioned record format for mixed model/teacher trajectories.
Keep all v1 scripted records and their encodings unchanged. Distinguish masked
model actions from deliberately injected faults; never relabel generated mistakes
as scripted injections. Record actor architecture, checkpoint SHA256, tokenizer,
initial case hash, fixed decoding settings, actual action token IDs and actual
environment feedback. Supervise only teacher repair actions and their EOS.

Generated token IDs must be preserved: decoding and re-encoding arbitrary BPE
output can change segmentation. The new encoder must reconstruct the exact
consumed behavior prefix, including RESP, one actual terminal EOS per completed
action, and the original observation frame. It must not invent EOS after a token
cap or replay/replace original feedback. A capped unfinished action should be
preserved as an excluded collection attempt, rather than silently truncated into
a different state. Reject ambiguous control tokens according to a fixed rule.

Collection uses fresh model state and the unchanged initial prompt per task.
The recurrent model must retain only its existing state and immutable prompt;
controller-side audit logs must not become model-accessible history or a cache.
Continue the same real environment after the actor prefix. Do not reset files
between behavior and teacher repair. Exclude and report attempts that already
verified completion or altered protected initial tests, under preregistered rules.
Do not modify tests to obtain a pass. Keep every attempted prefix and exclusion.

The teacher should inspect the public test and current target as needed, restore
the canonical target program, execute the real public tests, and obtain FINISH
from the original private validator. Store actual outcomes. Private checks must
never enter the pointer prompt or public observations. Failed teacher completion,
oversized frames and infrastructure failures must remain explicit in receipts.

Before any model collection, verify the v2 codec on remote fixtures for malformed
completed actions, noncanonical BPE segmentation, rejected token caps, immutable
prompts, one task reset, loss masking and exact v1 encoding compatibility. Freeze
a small fixed training-case selection and both actors' checkpoint hashes before
generation. Do not use development examples for collection or select cases from
model scores. Record all matched actor attempts and exclusions; this is training
data generation, not an independent benchmark.

Only then freeze a training mixture that includes broader foundation coverage,
with a matched transformer control and independent generated-task evaluation.
The prior eight-epoch policy run memorized training data and degraded foundation
loss in all measured domains despite 20% replay updates. More teacher loss
improvement is insufficient evidence of better coding, reasoning or repair.
