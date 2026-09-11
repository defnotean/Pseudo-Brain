# Paired tool-policy pilot v1

This registration precedes baseline policy outputs and any policy optimizer
updates. Start both models from the fixed V4 checkpoints used in the baseline.
The active V5 foundation run is a separate experiment and is not substituted.
No frontier capability claim follows from this small procedural pilot.

Use all352 accepted policy training trajectories for eight epochs, with a new
deterministic permutation per epoch seeded1986+epoch. Mask observations, prompt,
and injected-fault actions exactly as encoded_executed_trajectory specifies.
Supervise only teacher actions and EOS. A trajectory is one update, with one
task-start reset and the unchanged initial prompt as pointer source.

After every four policy updates, insert one distinct foundation training example.
Shuffle all69632 frozen V5 training-row indices using Python Random(1986), then
take704 in that order. No foundation development row is used for replay. This
gives2816 policy and704 replay updates,3520 total. Save the complete hash-bound
schedule before training either model; both models must use identical entries,
whole-document encodings and update order. Foundation replay mitigates forgetting;
its benefit must be measured and is not presumed. Procedural development family
isolation does not establish global semantic novelty against foundation data.

Use action_trajectory_loss for both kinds: selected-target mean CE, chunk256,
recurrent-only256-token padding, no cross-example state. Strict deterministic
settings; Python/NumPy/Torch seed1986. AdamW lr1e-4, betas(.9,.95), eps1e-8,
weight_decay.1, gradient norm clip1 with nonfinite errors. Warmup32 updates,
then cosine decay to1e-5 at update3520. Train all parameters. No sampling,
reinforcement learning, reward shaping, curriculum changes or checkpoint selection.

Evaluate all24 policy development demonstrations before and after training using
token-weighted action NLL. This is teacher-forced loss, separate from autonomous
completion and repair. Evaluate all1536 frozen V5 foundation development rows
before and after to expose forgetting. Preserve final checkpoint and bounded
progress checkpoints; do not auto-resume or extend a failed/time-limited run.
Registered training budget900seconds/model, measured over updates/checkpointing;
outer process budget1800seconds/model includes encoding and development scoring.
Partial checkpoints must be labelled partial and cannot replace the final model.

Require the native ingestion, multi-turn parity and action-gradient gates, and
the prepared V4 policy baseline to finish before launching this pilot. Before
training, verify native one-update loss/gradient finiteness for both model routes
and frozen mixed-data selection. Required post-training recurrent parity must
remain finite, <1e-6 across parallel/serial logits and exactly4096 state bytes.
No broader rollout if the architecture's correctness gate fails.

After both complete, run the same48-task procedural conditions with identified
final checkpoint hashes and unchanged prompts/scoring. Compare with baseline;
separately run the existing32-code/32-math bank and report that these are already
observed development results. Use new independent tasks later for confirmation.
No claim that a loss reduction or scripted repairs demonstrate frontier parity.
