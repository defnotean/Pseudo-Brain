# Stage-A causal thought ablation

This is a narrow checkpoint gate for one question: on the registered
96-decision validation slice, do final W/A/S/D decisions beneficially depend on
the sample-specific thought tensor supplied to the final actuator?

It does not certify human-like thought, multiple world models, independent or
specialized slots, recurrent benefit, or general game ability.

## Registered evaluation

The evaluator reconstructs validation sequence indices 0 through 15 at epoch
zero. Each sequence has eight transitions and the first two are burn-in, giving
exactly 96 evaluated decisions. It fails closed unless the slice has all of
these invariants:

- 137 active movement targets;
- 26 targets changed from the preceding applied movement control;
- 70 targets exactly matched that preceding movement control;
- validation dataset manifest
  `a9cdc763e85e9bb13d21fb2ecf1098af11912d51bc3c8b9801f03d125347c1e5`.

For every timestep, all 16 sequences first run separately with batch size one,
exactly matching training-time validation. A temporary hook captures the
arguments of each model's final actuator call. The hook is removed immediately.
The captured unmodified actuator call is replayed and must reproduce the
original logits bit-for-bit before either intervention is measured.

The two preregistered interventions are:

1. `zeroed`: replace the complete final thought tensor with zeros.
2. `batch_shuffled`: for receiver sequence `i`, use the complete thought tensor
   from sequence `(i + 1) mod 16` at the same timestep.

The cyclic donor mapping is fixed before checkpoint results are observed:
`[1, 2, ..., 15, 0]`. It is a derangement. Receiver and donor movement targets
happen to coincide on 15 of 96 decisions and on 3 of the 26 receiver-changed
decisions; those overlap counts are also fail-closed slice invariants.

Only the full pass advances belief, memory, thoughts, ages, and every other
recurrent field. The interventions replay the final actuator only. Sensors,
belief, working memory, retrieved memory, and goal context are the exact tensor
objects captured from the full pass.

## Pass rule

A pass requires all six preregistered integer-count checks:

- full movement exact-set accuracy is at least 80/96;
- full accuracy on changed targets is at least 13/26;
- full movement exact count exceeds the zeroed count by at least 5/96;
- full movement exact count exceeds the shuffled count by at least 5/96;
- full changed-target exact count exceeds the zeroed count by at least 3/26;
- full changed-target exact count exceeds the shuffled count by at least 3/26.

The report retains paired correctness discordances (full-correct/intervention-
wrong and the reverse), decision-change counts, and all outcomes per sequence.
The effect-size thresholds are engineering screens, not significance tests. In
the most favorable paired case, five wins and no losses still gives a two-sided
exact McNemar p-value of 0.0625; three wins and no losses gives 0.25. Decisions
within a trajectory are correlated, and this validation slice was used during
training-time model selection.

## Identity and invocation

Evaluation accepts no `latest` pointer. It requires the exact checkpoint hash
and only accepts the registered cursor `(epoch=0, next_batch=4000,
optimizer_step=500)`. It accepts exactly these two preregistered recipe
identities; canonical/raw hashes cannot be mixed between rows:

| Recipe ID | Config | Schema / scheduler | Canonical config SHA-256 | Raw TOML SHA-256 |
| --- | --- | --- | --- | --- |
| `stagea_continuation_cosine_a_schema1` | `dgx-stagea-continuation-gate.toml` | 1 / `cosine_after_warmup` | `d6f8c8ba3caaaab4643c430ff75747c0025f866363ddfbbb2114fc80969c8fcd` | `47ef30ba199f0b83b5719caa938e3eb77dfd1695dd5e10cda615ab503dc4a162` |
| `stagea_continuation_constant_lr_b_schema2` | `dgx-stagea-continuation-gate-b.toml` | 2 / `constant_after_warmup` | `8e8e4cc12123f55ee22aace1d65517dbb7586465bbf704c77cc7fe45bb3fc1f9` | `f013e3e14dfb854da1131aa726dd141bbe22e5766db8a48ed7cdb1aff6b9a6bf` |

Both recipes require and verify the batch-source manifest SHA-256
`9dbc93218e26e5b522c0d1f2bddd9172499bf66ebdc2855ef51570fce26de70e`.
The evaluator also requires and verifies:

- the original training source-tree SHA-256;
- the evaluator file SHA-256;
- the checkpoint's runtime fingerprint and strict full system state.

The canonical JSON artifact reports the selected registry entry under
`registered_recipe`, including its recipe ID, filename, schema, scheduler,
three identity hashes, and required cursor.

For future experiments, put the evaluator in the immutable release before
training. For a checkpoint from an earlier immutable release, execute this file
directly while `PYTHONPATH` points to that original release. The evaluator
verifies that every imported `irene_brain` module comes from the named release
and that its complete source tree matches the checkpoint's expected source
hash. The external evaluator is separately bound by its required SHA-256.

Example Linux/DGX form for recipe A (for B, use the B config and the exact B
canonical/raw hashes from the table; placeholders must be replaced by recorded
exact values):

~~~bash
PYTHONPATH="$TRAINING_RELEASE/brain/src" python3 "$EVALUATOR_FILE" \
  --training-release-root "$TRAINING_RELEASE" \
  --config "$TRAINING_RELEASE/brain/configs/training/dgx-stagea-continuation-gate.toml" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --config-sha256 d6f8c8ba3caaaab4643c430ff75747c0025f866363ddfbbb2114fc80969c8fcd \
  --config-file-sha256 47ef30ba199f0b83b5719caa938e3eb77dfd1695dd5e10cda615ab503dc4a162 \
  --data-sha256 9dbc93218e26e5b522c0d1f2bddd9172499bf66ebdc2855ef51570fce26de70e \
  --training-code-sha256 "$TRAINING_CODE_SHA256" \
  --evaluator-sha256 "$EVALUATOR_SHA256" \
  --optimizer-step 500
~~~

Exit status is 0 for a pass, 1 for a valid scientific failure, and 2 for any
identity, input, runtime, nonfinite-output, or evaluator-integrity failure. The
command is read-only and emits one canonical JSON object to standard output.

## Interpretation limits

The shuffled control is stronger than zeroing: a zero tensor becomes constant
tokens after normalization and is not literal token removal. A pass can still
reflect a redundant actuator dependency; it does not prove the thought tensor
contains information unavailable from sensors or belief, nor that this model
beats a matched no-thought architecture. One fixed cyclic derangement can also
be permutation-sensitive. Confirmatory work should preregister multiple
derangements, use untouched test seeds, and compare equal-compute reactive and
recurrent baselines.
