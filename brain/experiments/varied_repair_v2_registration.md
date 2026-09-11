# Varied executed repair demonstrations v2 — pilot registration

Motivation: the completed r2 policy pair solved zero of 48 tool tasks each.
Training exposed only RuntimeError failures, while rollouts frequently produced
syntax/import errors, wrong paths, and WRITE/EDIT format confusion. A fixed-prefix
ablation showed substantial pointer benefit. Keep the architecture and pointer.

This is a CPU-only data experiment on the existing Colab, with no optimizer,
model generation, model scoring, checkpoint changes, or local verification.
Its success establishes usable scripted data only, never autonomous competence.

Reconstruct all 352 original training cases from seed 1985, 2048 candidates,
diversify_indices=False and the original family partition, using the frozen v1
source. Verify the compressed training corpus hash, every record digest, prompt,
initial workspace, source ID, family, repair flag, private validator SHA256 and
canonical final workspace digest against the recorded v1 artifacts. Do not read
development observations or grade new model outputs. Preserve all old artifacts.

Pilot selection: first case in stored training order from each of the 22 training
families, crossed with exactly ten scenarios in this order: clean, syntax,
import_symbol, missing_module, write_edit_body, missing_write_body, invalid_header,
edit_mismatch, runtime, module_assertion. Total 220 attempted trajectories. No
adaptive selection, retries, filtering, or substitution. Any failed gate makes
the pilot incomplete; preserve all attempted traces and failure details.

All scenarios begin by reading the actual public test. Injected action errors
are masked using the existing injected_fault kind. Syntax, import_symbol,
write_edit_body, runtime, and module_assertion write faulty target contents;
missing_module writes the reference to a different inert path; missing_write_body
omits the required newline; invalid_header uses an unrecognized action header;
edit_mismatch writes a valid stub, then requests an absent exact edit target.
For every nonclean case, the teacher runs tests and reads the intended target,
then repairs by a full WRITE or an exact EDIT of the observed target contents
(fixed by scenario, not outcome). RUN_TESTS and FINISH must then pass using
the original public and private validators. Module assertion is a module-load
AssertionError, not a semantic unit-test mutation. No claim of natural model
rollout provenance, reflection quality, or comprehensive fault coverage.

For each case, run canonical public/private positive controls and a RuntimeError
public/private negative control. Require the intended exception class for each
nonclean pre-repair RUN_TESTS; require rejection of the three deliberately
malformed/unmatched actions. Actual tests and feedback use the frozen isolated
environment and Bubblewrap. All teacher file operations must succeed except a
READ of an actually missing target. Store complete raw actual feedback, step
outcomes, provenance, and hashes before encoding. Teacher targets remain only
action tokens and EOS; faults and observations are masked; one reset per task;
pointer input is only the immutable initial prompt. No truncation: whole record
<=8192 tokens, initial prompt/each observation <=4096, each action+EOS <=1024.

Per-run hard wall limit 900 seconds, per sandbox test the existing fixed limits.
Snapshot all source and registration files before launch and hash the snapshot.
Record every case/scenario token count, observed exception, total label count,
completion status, and digest. Archive all source, logs, attempt records and
receipts. A completed pilot allows planning a larger corpus, but does not
authorize silent expansion or another model training run under this protocol.
