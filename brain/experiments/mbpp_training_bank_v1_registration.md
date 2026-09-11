# MBPP training-bank v1: isolation and executable-reference audit

This collection broadens code tasks beyond the 22 procedural training families.
It does not train/evaluate a model or establish frontier competence. Execute
preparation only on Colab CPU, with a 900-second hard deadline, frozen sources,
and the existing Bubblewrap workspace/test limits. Preserve all prior artifacts.

Source: google-research/google-research revision
`08a8d6736475776f42ffac23b2c13111a28e5795`, `mbpp/mbpp.jsonl`, SHA256
`ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f`.
The full source contains 974 rows. Only official training IDs 601–974 (374 rows)
are candidates, processed in ascending ID. The extracted training JSONL SHA256
is `97f660b820f3d75a99a424ec249b9bce04e33517789584c05d6e2a0e84995d97`.
Preserve pinned README, repository LICENSE, source URLs and digests. The source
was downloaded as inert data before this registration; zero programs executed.

Isolation: form connected components using normalized question/documentation,
normalized code, and AST code fingerprints that consistently rename top-level
function/class identifiers and remove docstrings. All nontraining MBPP rows
are quarantine seeds, never candidate demonstrations or model inputs. Also seed
the prior 192-, 384-, and 1536-row development banks, the frozen 64 question-side
HumanEval/GSM requests, and the 24 procedural development canonical solutions.
Use their frozen hashes. A component touching a quarantine seed is excluded in
full, including bridges through training rows. Keep only the lowest task ID in
each surviving training component, logging every duplicate. No model-output
selection. These checks detect exact normalized/structural matches, not semantic
paraphrase contamination. Existing development sets stay out of training.

Adapter: preserve each reference program unchanged except surrounding whitespace.
Target module is `mbpp_<task_id>.py`. Initial files contain only `test_public.py`,
which imports that module's exports, applies the source setup, and executes the
first official assertion in a discovered test function. The private validator
imports the same module and runs all three original assertions, retaining the
first for possible shared-state effects. Setup and statements execute in source
order. No challenge tests exist among the 374 pinned training rows; assert this.
Reject unsupported/unparseable references/tests or ambiguous reserved tokens.
Do not silently fix a source solution, test, import, or failed expected value.

The initial prompt contains the six-action protocol, natural-language task,
target module, and body-free public interface declarations derived from top-level
functions/classes. Include class method signatures, but never implementation
bodies, docstrings, private tests or their results. Disclose this interface
conditioning: the task should name the required API. Public setup/tests remain
in the initial file. The pointer receives this immutable prompt only.

For every surviving supported candidate, execute canonical public and private
positive controls, then public/private negative controls with the target replaced
by `raise RuntimeError("Injected training fault")`. Both positives must pass;
both negatives must fail with that fault. Save all actual control outputs,
including failures. Reference/control failures are explicit dataset exclusions,
not pilot crashes or erased attempts. Do not replace excluded tasks or target a
quota. Report the disposition of every one of 374 candidates and exact eligible
count. An infrastructure exception or deadline makes the run incomplete.

Eligible examples must fit the frozen tokenizer and existing execution protocol:
prompt <=4096 tokens, canonical WRITE action plus EOS <=1024, public READ feedback
<=4096. No truncation. Save all cases (including private validator separately from
prompt fields), source/family/isolation identities, test results, raw/compressed
hashes, and per-case size records. The output is an audited training-case bank;
recording actual repair trajectories and any model training require separate
registrations with fixed limits. Archive source, provenance, controls, exclusions,
and receipts and verify the archive after download.
