# Real model-prefix repair collection v1

Register before generation. Use only the completed corrected MBPP training-case
bank (gzip SHA256 e7645a3c98749815098ec66293e583087180b06c6bfd89249d6b1bb96f04d5d0).
Verify all357 case hashes and official training IDs. Sort by case_sha256, take
the first32 once, and save the exact selection before either actor runs. No
adaptive selection, retries, replacement cases or development examples.

Use both unchanged completed3520-update policy-r2 actors, resolved against
training report SHA256
1b5cf6883fcc3ad536c96ac0c06bcac354c1c5c96622ad8426953106b605a0b4.
Recurrent checkpoint0b3edca04de150068c81fb8f71f05cd0aad8f42e423b9989bc789b919ffb85b7;
transformer346c9d4c9d334490cb2e93c54f4a1f0f66020a3833b4e7565f015b6d08c44dfe.
The frozen resolver checks complete paired training, source, schedule and parity
receipts, final checkpoint names and actual file hashes. Keep all weights fixed.

Run on Colab A100, one actor child at a time, inference only, strict deterministic
settings, seed198. Use the frozen evaluated agent/session implementation and
the completed v2 codec source8a675c4ed40db80cee711739a82060639b49ab9ed979e75da67ff042bd81c9be.
Fresh state and immutable initial prompt per case. Actor prefix:2cycles,
512 action tokens,4096 prompt/observation limits, transformer32768 full-prefix
limit. Recurrent state4096bytes, no retained replay prefix, <36M parameters.
Internal deadline900seconds/model; controller outer limit960seconds/model.

Save every raw actor attempt, emitted token IDs, actual feedback and resulting
workspace before attempting a teacher continuation. Raw incomplete/capped or
already-completed episodes are explicit exclusions, not repaired or truncated.
For eligible prefixes, continue the same environment with exactly six teacher
actions: READ public test, RUN_TESTS, READ target, WRITE unchanged canonical
target program, RUN_TESTS, FINISH. The first teacher test captures failures caused
by the actual model implementation or absent target. Tests/reads may fail before
repair. Original private validator controls FINISH. Never restore or alter tests
to obtain a pass; exclude protected-initial-file changes.

Record source/case/checkpoint/tokenizer hashes with provenance_kind=model_checkpoint.
Preserve noncanonical actor BPE IDs and actual EOS/observation boundaries. Mask
all actor tokens and observations; supervise only teacher actions and EOS. Keep
8192 whole-record/1024 teacher-action/4096 observation limits without truncation.
Save every teacher attempt before encoding/admission. Reject explicit caps,
protected-file changes, ambiguous actor control tokens, unverified teacher
completion, and encoded size limits with per-attempt reasons. Unexpected trace,
state, workspace, hash or framing inconsistencies fail the experiment rather
than quietly excluding a case. No acceptance quota; zero usable records is a
possible complete collection result requiring diagnosis before further work.

All64 raw attempts must be accounted for on successful completion. Report actor
stops/actions separately from teacher completions, failed-then-passed tests,
accepted examples and token totals. These are training-data collection results,
not independent capability scores. Archive source, selection, raw attempts,
teacher records, exclusions and receipts; verify checkpoint hashes again at
the end. No model training or checkpoint promotion follows from this registration.
