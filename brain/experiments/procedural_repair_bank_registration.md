# Executed procedural tool-policy development bank v1

This is a small tool-protocol learning/development bank, not a frontier benchmark
or globally unseen task set. Source algorithms already exist in the repository
and may be familiar from earlier models or foundation pretraining. Do not claim
randomized names make an algorithm novel. Do not read the sealed benchmark's
task/solution bank to select this data.

Generate2048 candidates from existing ProceduralTrainingGenerator(seed=1985),
diversify_indices=False. Group by AST of reference solution with the target
function/class and references renamed to TARGET. Sort family SHA256 keys;
every fifth family (rank%5==0) is development, others training. Select first16
training or4 development candidates per family sorted by module/function name.
Require at least10 families, full quotas, and disjoint family/solution keys.
Selection occurs before any execution outcome and is never changed to hide a
failure. Family grouping does not certify semantic paraphrase isolation.

Expose the source test prefix through its first top-level assertion as one public
test. Keep the complete original assertion sequence and setup behind the external
validator, including the repeated public prefix so state-changing assertions
(e.g. queue pop) retain their semantics and full contract checking. Require at
least two assertions. Hidden source/tests/reference are
stored separately from development requests and never placed in the agent's
initial prompt or ordinary workspace. Same-interpreter grader limitations remain.

Alternate ordinary write and injected-fault repair demonstrations within each
family. Every trace first reads public tests. Repair traces write an explicit
injected RuntimeError (masked from targets), run failing tests, inspect the file,
apply an exact edit to the reference, run passing public tests, and FINISH.
Ordinary traces write the reference, run tests, and FINISH. These are scripted
demonstrations with real outcomes, not generated model-policy results. Retrieval
is not exercised in this first bank and requires separate coverage later.

Before accepting each trace, isolated positive reference and negative fault
controls must pass/fail both public and private checks as expected. Execute all
actions through the new isolated environment and require external final success;
repair traces must exhibit an actual public-test failure followed by success.
Encode with the frozen BPE and validated causal encoder, max8192 whole-trajectory
tokens, max4096 prompt tokens, max1024 tokens per action and4096 per observation.
Reject oversized/failed records with explicit evidence; do not silently replace
them. No training is authorized until the entire registered bank passes.

Run preparation on Colab CPU only, CUDA hidden/one thread, bounded900seconds.
Record source/tokenizer/task/partition/artifact hashes and run_provenance. Freeze
train/development records separately plus request-only development inputs and
private grader records. Preserve failures and publish incomplete status if any
quota/control/token-bound fails. Existing V5 run and benchmark banks unchanged.
