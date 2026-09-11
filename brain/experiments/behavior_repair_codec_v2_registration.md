# Mixed behavior/teacher trajectory codec v2 preflight

This is a Colab CPU-only infrastructure experiment, no model generation, optimizer
updates, training-case selection or capability score. Freeze sources before
execution, use the existing tokenizer and sandbox, and impose a180-second outer
deadline on all checks. Keep all historical v1 code/data and model artifacts.

Create a separate v2 format for exact model-token prefixes followed by actual
teacher execution. Model token IDs must not be reconstructed from decoded text.
Only observed EOS-terminated, executed, observation-consumed actor prefixes with
a cycle-limit stop may receive a teacher continuation. Reject capped, completed,
changed-prompt, invalid-state, malformed-token and changed-workspace prefixes.
Check deterministic file operations against inert initial text; never rerun tests
or replace original actor observations. Preserve original trace and prefix hash.

Keep actor provenance explicit: architecture, checkpoint/tokenizer/source/case
hashes and collection protocol. Synthetic codec fixtures must identify themselves
as fixtures and be rejected by the ordinary training encoder. Real collection
must later bind nonzero hashes to actual files. The codec alone is not proof that
a claimed checkpoint generated a trace.

Teacher actions execute in the same environment and need final external FINISH.
Only teacher action/EOS tokens receive labels. Initial prompt, original actor
tokens, actor EOS, observations and observation transitions remain masked. Use
one reset and only the immutable initial prompt for pointer input. Preserve
8,192 whole-record,1,024 action+EOS,4,096 prompt/observation limits without
truncation. Verify exact stored observation frames and raw actor-trace coherence.

Preflight uses the real frozen BPE tokenizer, deliberately noncanonical actor
token segmentation, actual sandbox failure/repair controls, explicit malformed
and immediate-EOS behavior, rejected token/observation/context caps, altered
workspaces, record corruption, invalid metadata, and loss-mask/one-reset checks.
Exercise both architecture metadata variants without pretending fixtures are
model-generated samples. Include an underscore-free small repair fixture whose
public and separate private assertions both run through the existing sandbox.

Compare the dispatch encoder byte-for-byte with the preserved v1 encoder on all
352 training+24 development procedural v1 records and all220 varied repair pilot
records, with frozen gzip hashes. No development record becomes training data;
this is only encoding compatibility. Save counts, tensor digest, raw fixture
feedback, source manifest, logs and receipts; archive and hash-verify download.

Only after this gate passes may a separate small, fixed, training-only model
collection be registered. No codec fixture enters that collection or training.
