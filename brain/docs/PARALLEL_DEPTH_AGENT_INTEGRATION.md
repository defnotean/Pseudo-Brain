# Parallel-depth agent integration status

Real model-prefix repair collection v1 completed on Colab A100 at
`/content/pb-behavior-collection-v1-r0`, source manifest
`e70ad80c5558b1623b43fb2784b374cbcd6649e1f8ff5ad3c76dc9d3a8ef71e5`.
The fixed selection is the first32 audited MBPP training cases sorted by case
hash, used once by each unchanged policy-r2 actor. Selection digest:
`f10ec916bd5251969ef7e779ebe8efe84ee5c9d74be574e188d02568a281f086`.
All selected cases and both completed checkpoints passed the CPU identity checks.
Two actor actions precede up to six teacher actions in the same workspace;
raw capped/excluded attempts remain saved. No weights are updated, and teacher
completion is not an autonomous model score. See
`experiments/behavior_repair_collection_v1_registration.md`.
All64 attempts were recorded in165.161 seconds. Recurrent27/32 and
transformer31/32 yielded verified teacher continuations; six token-cap exclusions
were preserved. All58 admitted records failed the first teacher test because a
module was missing, then passed after the teacher wrote the canonical target.
The post-collection audit re-encoded every admitted record, checked original raw
attempt hashes/trace equality, actor identities, masks and token counts. RNN
records contain39,331 input/5,176 supervised tokens; transformer45,651/5,897.
The downloaded, hash-verified archive is under
`runs/behavior-collection-20260911-v1/collection-r0-evidence`, SHA256
`1026b71e1fe11ffd03b3a7efbdf4cb838d2d6ba41784a2a778b13d3fe8679875`.
A read-only filename audit found0/32 expected target files present for either
actor after the two-action prefix. RNN attempted27 writes, only2 accepted by the
environment, all to wrong targets; transformer11 writes,1 accepted, also wrong.
This is not a full-episode capability metric. A fixed eight-case training-only
prompt-order probe moved the existing Target line after the interfaces. Both
actors remained at 0/8 expected target files, matching their original baseline.
All 16 new attempts and token-cap outcomes were preserved; no weights changed.
The probe completed in 36.50 seconds on Colab. Its downloaded, hash-verified
archive is under `runs/target-order-probe-20260911-v1/probe-r0-evidence`, SHA256
`40153299afdde371c88affdaf21c9748755bbff78d953ac70c9965a405b4846e`.
Original audited prompts remain unchanged. All357 clean MBPP demonstrations
completed under `experiments/mbpp_clean_trajectories_v1_registration.md` in97.70s
on Colab CPU. Every READ, canonical WRITE, public RUN and external FINISH passed;
all actual attempts were saved and re-encoded in a separate provenance audit.
The corpus has298099 input/56763 supervised tokens, max1249 per record. No model
updates occurred. Its downloaded, hash-verified archive is under
`runs/mbpp-clean-20260911-v1/collection-r0-evidence`, SHA256
`83082a228f61a1c32f57eebde25a52e498b28821bf493a8e5307577f0331b3c4`.
The new broader mixed schedule and both native preflights have now passed.
See `experiments/broad_policy_v2_registration.md`:987policy records, two passes,
three distinct foundation examples per policy update,7896updates/actor. Replay
is75% of updates and81.33% of supervised tokens. Both eight-update native checks
were discarded; recurrent parity max6.56e-14 and4096B. Full training is live at
`/content/pb-broad-policy-training-v2-r0/training`, reloading completed policy-r2
weights. Frozen source442afc835cef0b4ae9f269f48ef353c16401e0ad1e5511fae9587ccdb28baf3d;
selection16cbc46c6fe8b380024634520cfc210e2879737278ce8d1b2a5eeed3087cd949.
Downloaded preflight archive SHA256:
3181132ffcf4f6b3768561e922c37fc7252528da6c8ec3a80dbc9ae26fc9e7f6.
No new autonomous capability result is available yet.

The post-v2 evaluation preparation has also passed on Colab at
`/content/pb-broad-policy-evaluation-v2-r0`. Original rollout, generator and scorer
sources match their prior frozen versions. Controller changes are constrained to
new checkpoint/source binding and training provenance. CPU imports passed, and
the resolver rejected old completed weights and native-preflight-only weights.
Full final-checkpoint binding remains pending until both training arms finish.
See `experiments/broad_policy_v2_evaluation_registration.md`. Evaluation source:
1881c08707ee9bdc23a161e06bd2eabd3f2915d19fe94356c1254a86d5f38f40.
Downloaded preflight archive SHA256:
3a0a6fd9312e6ac7c3b5c5ccd346de43a8b421c661c6c1c675059b676b7da51c.

The mixed behavior/teacher codec v2 passed its Colab preflight:22 checks and
byte-identical v1 encoding on596 frozen records, including independent historical
observation framing. It retains exact emitted actor token IDs, masks behavior,
rejects incomplete prefixes and altered workspaces, and supervises only actual
teacher continuations ending in external verification. Codec fixtures cannot
enter ordinary training. No model generation or updates occurred in this check.
Successful source/evidence archive SHA256:
`83460a75aadd367baed88c520a1a1caf6b0c77ef3e26903200ba73f440c4fffb`,
preserved under `runs/behavior-codec-20260911-v2/r2-preflight`.
Two failed dependency-packaging attempts are preserved separately. The eligible
source combines frozen evaluated episode/session code with unchanged v1 codec
code; it does not modify historical runs. Real model-prefix collection is complete
as described above; see `docs/BEHAVIOR_REPAIR_DATA_PLAN.md` for training constraints.

The MBPP training-bank v1 audit completed on Colab CPU at
`/content/pb-mbpp-training-bank-v1-r2`, source manifest
`45e890c9d66e8b6fd74753e6cf5ee8e0750266043236ff2d8ccc4341fa0d3795`.
It pins the upstream revision, considers only official training IDs 601–974,
quarantines connected duplicates of held-out/development material, and executes
public/private positive and negative reference controls. Five adapter checks
passed remotely, including body-free interface conditioning, retained sequential
tests/setup, structural duplicate keys, and rejected nontraining IDs. All frozen
source/isolation inputs passed hash/count checks. Twelve actual sandbox controls
also passed for underscore API, class and setup cases. Two preparation defects
were corrected: module-docstring-sensitive structural numbering and wildcard
imports omitting a required `_sum` function. Superseded r0 completed; r1 was
explicitly stopped and archived with its partial controls. Both were downloaded
and hash-verified, and neither is eligible for downstream training. See
`experiments/mbpp_training_bank_v1_registration.md` and amendments 1–2.
The corrected audit accounted for all 374 candidates in 195.556 seconds:
357 eligible, 11 quarantined, 6 duplicate components, zero remaining reference
or negative-control failures. Maximum prompt/WRITE/public-observation sizes were
268/358/441 tokens. Every eligible case digest and every candidate disposition
was verified before archiving. The downloaded, hash-verified archive is under
`runs/mbpp-training-bank-20260911-v1/bank-r2-evidence`, SHA256
`f5285c4503c74e5587d133be2db2504e2beceb5dbc401c12039111f432fdd540`.
The case gzip SHA256 is
`e7645a3c98749815098ec66293e583087180b06c6bfd89249d6b1bb96f04d5d0`.
See `docs/BEHAVIOR_REPAIR_DATA_PLAN.md` for the next mixed actor/teacher format.
This creates cases only;
it does not train a model or record new autonomous capability.

The separately registered varied-repair v2 pilot completed on Colab CPU under
`/content/pb-varied-repair-v2-pilot-r0`. Its frozen source manifest is
`6a6f927cd0dcc427a5cd0b83ce4329c251b0d19b4121493da5f604aa5ed91b89`.
All 352 original training cases reconstructed exactly, including private-validator
and final-workspace hashes; identity-list digest
`48deb163c70d7d2c16c6dcdde96fe6ac2adad396a6595af8a538263154970ba2`.
The pilot crosses the first case from each of 22 training families with ten fixed
clean/fault scenarios (220 attempts). It checks actual public/private tests,
rejected malformed actions, masking, one reset, and existing token limits.
All 220/220 trajectories passed in 99.903 seconds: 22 clean and 198 injected-fault
repairs, 315,200 input tokens and 42,113 supervised tokens, maximum 1,931 tokens
per record. All serialized records exactly match raw attempts and verified
digests. The source/raw-feedback/evidence archive is preserved locally under
`runs/varied-repair-20260911-v2/pilot-r0-evidence`; archive SHA256
`421f41fba86680abeceae710e03d043523f3f58f82aa183ab18656df0352d04b`.
No model was trained or evaluated by this data experiment. See
`experiments/varied_repair_v2_registration.md`; preserve attempted failures and
do not enlarge the run or treat scripted teacher completions as model success.

Both trained policies'48-case evaluations are complete:0task completions or
repairs. The recurrent model issued384recognized actions (162successful
environment operations); the transformer359 (179successful operations).
Coding/math retention evaluation also finished:both0/32HumanEval and0/32GSM8K,
with no missing responses. The first inspected recurrent failure
included invalid function syntax, a later write to the wrong filename, and EDIT
content emitted under WRITE_FILE. These are unresolved model failures.

The read-only copy-readout diagnostic completed after those fixed evaluations.
It compared the additive pointer contribution at the same teacher-prefix features
on32fixed training examples and24development examples. The pointer helped1491
recurrent development token predictions and hurt35 (transformer1568/14). Sampled
training accuracy was99.75%/100%, while development body accuracy was71.02%/71.82%.
This supports a generalization gap, not disabling the pointer. No weights were
updated or counterfactual trajectories generated. All56records/model completed.
See `experiments/policy_copy_diagnostic_registration.md` and the preserved evidence
in `runs/policy-copy-diagnostic-20260911-v1/native-evidence`.

The completed feedback audit found176failed training RUN_TESTS observations,
all RuntimeError, while actual rollouts mainly encountered syntax/import errors.
New training should address those observed failure states and broader code
generalization; the present bank and all completed evaluations remain frozen.

Latest native results: carried-state ingestion through 32,768 tokens, multi-turn
parity, and both models' corrected action-loss gradients passed on Colab. The
V4 tool baseline completed all 48 cases per model with zero completions/repairs.
Both models then passed the native two-update policy/foundation preflight; those
weights were discarded. The registered 3520-update policy training pair completed
on Colab from the original V4 checkpoints, and all eight checkpoint boundaries
are preserved in a verified archive. Post-training tool evaluation is complete.
Numerical correctness and
training loss do not establish learned agency; the unchanged task evaluation
must run on the eligible final weights.

The post-training coding/math retention evaluator is prepared under
`/content/pb-policy-retention-v1-r1`. It binds final weights to the completed r2
policy training pair and supplies the exact training tokenizer separately because
policy checkpoints contain no embedded tokenizer. Three Colab CPU controls and
both original-checkpoint tokenizer comparisons passed. All 187 original evaluator
source files are unchanged; an AST check limits the separate generator's changes
to tokenizer resolution. The registered run keeps the original 64 observed tasks,
greedy decoding, serial initial ingestion, output limits, and scoring rules. It
has not run yet. See `experiments/policy_retention_v1_registration.md` and the
source/evidence in `runs/policy-retention-20260911-v1/preflight-r1`.

The following records describe the integration and its earlier validation stages.

The experimental language core now has an opt-in `ParallelDepthSession` in
`irene_brain.agent.parallel_depth_session`. It is distinct from the legacy
`RecurrentSoftwareAgent` and does not expose semantic slot addressing.

The session holds eight logical recurrent layers packed into a 16×64 float32
state (4,096 bytes), plus the immutable initial prompt permitted for copying.
The model weights are shared by reference. It caches no logits, alignment
vectors, generated history or observation history. Chunk features and returned
token IDs are transient workspace/output, not future replay input.

The token API accepts nonempty `[1,T]` long tensors on the model's device:

- `start(prompt_tokens)` initializes a fresh task and freezes the pointer input.
- `ingest(new_tokens)` updates state with parallel scans and returns final logits.
- `generate(prefix_tokens, eos_id=..., max_new_tokens=...)` ingests the explicit
  response prefix and greedily emits tokens through ordinary recurrent steps.
- `state` returns an audit copy; `state_bytes` reports the carried-state size.

Generated EOS enters state exactly once and is omitted from returned content.
A `token_limit` result consumes its final emitted token but does not invent EOS.
Callers must distinguish truncated output before executing any action. The
session never repairs output, inserts a repair phase, executes tools, or declares
task success. Responses and observations are supplied by the caller; there is
no tokenizer fallback or automatic prompt reconstruction.

Six Colab CPU tests passed in 4.07 seconds with CUDA hidden and one thread.
Actual reduced-width models with pointer on/off matched serial state through
multiple action/observation exchanges, EOS and truncation. Fresh tasks and
external mutations were isolated. Full-vocabulary logit tolerance was 1e-6 and
decoded-state tolerance 1e-12. Evidence archive SHA256:
`a321d848612cf463a1b21c59f5488d0e5697407a131267b5aa36308af4bea88a`.
These checks establish adapter correctness on CPU, not trained native behavior
or learned agency. The separate native ingestion gate must run after V5 training;
a trained multi-turn session gate remains to be prepared.

The V5 corpus audit found exactly one response/EOS per document in all 69,632
training and 1,536 development records. There are zero literal observation
markers or response lines for the six `ACTION:` actuator verbs. Thus V5 supplies
no direct examples of this tool protocol. Its foundation capability evaluation
remains useful, but a separate corpus of executed, verified multi-turn traces
is needed for direct tool-policy supervision. Existing handwritten trajectories
and their scripted success strings are not execution evidence.

An opt-in `IsolatedSoftwareEnvironment` now implements the six actuator verbs
against a virtual text workspace. `WRITE_FILE` and unique-target `EDIT_FILE`
change inert strings; `READ_FILE` returns them. `RETRIEVE_MEMORY` uses an explicit
provider. `FINISH` requires an external validator and exposes a distinct
`verified_completion` flag; passing public tests cannot establish completion.
Validator callbacks are trusted harness code and must isolate candidate execution.

`RUN_TESTS` snapshots those files into a read-only mount inside the existing
Bubblewrap boundary. The initial backend supports zero-argument test functions
and unittest.TestCase, not pytest fixtures/plugins or arbitrary dependencies.
Its limits are 64 files, 64 KiB/file and 1 MiB total. It rejects unsafe paths,
file/directory conflicts, ambiguous edits, empty test discovery, early exits,
unexecuted async/generator tests and skipped tests. The same-interpreter harness
remains explicitly not adversarially tamper-proof.

Twenty scripted infrastructure controls passed on Colab CPU in 3.31 seconds,
including wrong code, exact repair, public tests and separate hidden-function
validation. Initial positive-control failures exposed a post-seccomp chdir;
moving directory setup into Bubblewrap fixed it without adding syscalls. Both
attempts are archived. Passing evidence SHA256:
`70636073e5bcbb79bba829fc2d278de1fc472513a6f45214a4395efca466a08f`.

`ParallelDepthSoftwareAgent` in `agent.parallel_depth_episode` now connects
the session and environment. `execute_episode(initial_prompt, environment, ...)`
creates a fresh session per task, preserves raw greedy output, and ingests only
new feedback frames. It rejects truncated actions before execution and stops
incomplete on oversized observations. Limits cover prompt tokens, action tokens,
observation tokens and cycles. Completion requires a successful FINISH with
`verified_completion=True`. Its full audit trace is returned output, never replay
context. JSON data brackets are escaped to keep literal control tokens out of
observation data; this is framing, not prompt-injection certification.

Seven Colab CPU controls passed in 2.40 seconds: scripted repair through the
actual isolated tools/validator, event ordering, invalid and truncated output,
limits and observation roundtripping. An actual reduced-width model separately
passed fresh-task repeatability. Archive SHA256:
`79cfdf68b43aa6c838034900c34747a20b615b7d92d9724e5f414c6d6dc2aa2f`.

Native trained validation and direct action-policy supervision remain pending.
The new mechanical validator in `evaluation.multiturn_parity` checks all token
logits, state at action/observation boundaries and three generated continuations.
Its two Colab CPU tests passed, including rejection of NaN outputs. The native
runner is frozen and imports successfully; it has not run. It must wait for V5
training and the preceding native ingestion gate. The real executed control is
an already observed diagnostic input, not a new capability benchmark.
The new `data.executed_trajectory` module supplies a tested path for the latter:
`record_executed_trajectory` executes explicitly tagged teacher/fault actions
through the isolated tools and records real feedback and external completion.
`encode_executed_trajectory` uses the same separately tokenized frames as the
episode loop. It teaches only teacher action tokens and EOS, masks injected
faults and observations, resets once at task start, and uses only the initial
prompt for copying. Records are hash-checked and incomplete/ambiguous/overlong
traces are rejected. Scripted demonstrations remain explicitly labelled.

Six Colab CPU controls passed in 2.35 seconds, including actual sandbox repair
feedback and the frozen BPE tokenizer. The raw executed control is preserved in
the evidence archive, SHA256:
`ab0a80bd8d95519ddf2a7d988d05bb8de8e443540d5cc6307034ec4a460a5239`.
This is data infrastructure, not a trained policy.

A separately registered procedural bank build completed on Colab CPU. Its
2048 source candidates group into 28 normalized solution families, preselected
as 352 training records from 22 families and 24 development records from six
families. It alternates ordinary write and injected-fault repair demonstrations.
All references and faults must pass positive/negative isolated controls before
their executed traces can be accepted. The public workspace contains only the
test prefix through its first assertion. The external validator retains the full
suite, including the public prefix to preserve state-changing assertions.

The four family-partition/helper controls passed in 1.97 seconds. All 376 bank
records then passed in 322.10 seconds, including 188 repair demonstrations.
Training contains 402,615 input and 61,340 supervised tokens; development contains
27,997 input and 4,183 supervised tokens. An independent integrity pass checked
all record hashes, control outcomes, disjoint families and request-only fields.
The downloaded source/data archive hash is
`2d3834f9fc2d41272bddc9d90735c6f988c1d1ae56480b55ccab2f2a22f708ab`.
No model has been trained on this bank yet. Its algorithms come from existing procedural
sources; family separation is useful for tool-policy development but does not
make this a frontier benchmark or certify globally unseen algorithms. Retrieval
coverage and broader real-project tasks remain missing.

The shared `training.action_trajectory_loss` path now has three passing Colab CPU
checks. Both actual model architectures match explicit full-logit action-target
cross entropy and every participating parameter gradient. The recurrent route
retains its previously validated256-token padding; the transformer stays
unpadded. Interior resets and changes to the initial pointer prompt are rejected.
The CPU evidence archive hash is
`3a21cc1a587c994fce02ec6b3b847942efcebe6786c9b4cd830ff6fab8a124dd`.
The native paired gradient gate is frozen/import-checked but has not run. It
must pass on the two fixed V4 checkpoints before this fine-tuning path is used.

The same episode loop now supports `TransformerSoftwareAgent` through a separate
`TransformerPrefixSession`. That comparison retains all action/observation tokens
and recomputes the full prefix, with an explicit32768-token cap. It reports
`state_bytes=None` and its retained prefix token count; it does not satisfy the
recurrent4KB constraint and is not an optimized generation-speed baseline.
Both routes preserve the initial pointer prompt, consume EOS once, reject
truncated actions and use the same environment and completion validator.

All ten Colab CPU checks passed in 2.66 seconds, including the seven existing
recurrent episode controls and exact transformer-prefix comparisons across
multiple turns. The evidence archive hash is
`15e06ef00bbdb8ef8483ba41fb3823b1e8e20a79b77eda400b7bc30846038e9c`.
No native tool-policy comparison or policy fine-tuning has run yet.

The paired policy scorer now binds public request fields to private grading
records without adding answers to agent input. Seven Colab CPU controls passed
in2.65 seconds. Repair credit requires observed execution failure, an actual
target-file hash change through successful WRITE/EDIT, and verified completion.
The archive hash is
`b01995e8fb22c04f1f42a0a4177e916014aff21f04b02fec993e9859b3087d3d`.

A frozen V4 evaluation runner is prepared for24 development requests in both
from-scratch and injected-fault conditions:48 tasks per model. Each model gets
1800 seconds including setup, and missing tasks remain explicit failures in
the fixed denominator. Atomic case artifacts permit independent scoring after
interruption. The missing/duplicate/unplanned-result controls and CLI imports
passed on Colab CPU. Native preflight archive hash is
`132f53b087969ba4de606f3b5ff25f78f3eb575befd521187676418057deef7b`.
Launch remains gated on current GPU work and native integration checks. This
procedural comparison will measure initial tool behavior, not frontier parity.

The first paired policy training protocol is now prepared. It uses eight epochs
of352 executed demonstrations, interleaving one of704 distinct foundation
training examples after every four policy updates. The complete3520-update
schedule contains3,557,955 input and707,728 supervised tokens. A Colab CPU check
passed in11.679 seconds, including exact agreement of all704 replay encodings
with the existing encoder and complete policy epoch/family partition checks.
Selection hash:
`5933901753e5416e75ae2fac765e3d21746de29d3e7437415a0fd12420ca98cd`.

The driver starts from the fixed V4 pair, uses identical updates/data order, and
measures both policy loss and all1536 foundation development examples before
and after. It has a900-second training budget and1800-second outer limit per
model. Neither policy training nor its native preflight has run. The native
preflight will discard its two diagnostic updates per model; pilot training
will reload the unchanged V4 checkpoints. Successful training additionally
requires the recurrent numerical/state gate before model-policy evaluation.

The initial remote bundle omitted an existing parity-check dependency, which
the CPU import preflight caught. The corrected r1 bundle passed both imports;
the failed r0 evidence remains preserved. Corrected source/data-preflight
archive hash:
`b8f9d7f18efdb809137027e3a8117ea07ddd34d2a656b191c0030164ed53d4cb`.
The r1 bundle is preserved on Colab; the current eligible training bundle is
`/content/pb-policy-training-v1-r2`, amended after the native gradient diagnosis
below. The complete evaluation baseline and native integration gates must precede
training. Broader competence and autonomous repair remain to be demonstrated.

The native carried-state ingestion gate now passed all eight cases in476.73
seconds, including32768 tokens, uneven chunks and batched resets. It preserves
4096 bytes per stream; maximum boundary/continuation logit error was below
3.4e-14. Downloaded native evidence archive:
`397df5debdc3077a0b37696618f2630b74401a084ad44eb90cf3d3e2a2842c76`.
The native multi-turn gate passed in27.58 seconds across854 control tokens,
with full-position logit error4.44e-14 and identical generated continuations.
Archive:
`6a4baede9cc5dc02414732eeada4bceafc510919d1694247709242050040b91a`.
These are numerical integration results, not learned repair scores.

The paired native action-gradient gate then failed on the transformer control's
embedding gradient. Both recurrent records passed every participating parameter
gradient, with errors below1.3e-15. The failed gate is preserved in archive
`ad7943c34ee0edf3d2abb91c2378c28319e4c3700690593b8a1f5fb2719be53f`.
The read-only diagnostic found bit-exact repeats and bit-exact checkpointed versus
uncheckpointed chunks. Changing readout batching produced the discrepancy. The
corrected transformer training route uses full readout/ignore-index mean CE;
the recurrent route retains chunk256 and its bucket padding. Both native cases
then passed the original tolerances: all55 transformer parameter gradients were
bit-exact to reference, and all95 recurrent gradients differed by at most1.25e-15.
An8192-token transformer forward/backward/AdamW fixture passed with4.62GB peak
allocated memory; its single update was discarded. Corrected native evidence:
`f9b8b692d28e81a7f501b0bf8ad45a36dc42b88ad7e2adabf1fd43357117de75`.

The policy training r2 amendment changes only that transformer loss route and
records it in driver metadata. All data, updates, seeds, optimizer settings,
limits and initialization checkpoints remain fixed. Its frozen source manifest
is `3be24612bfb31fac5376bb5c33ca33b5514906a5d0e1186659cea82616183005`.
The baseline evaluation is now running on unchanged V4 inference/scoring code.
No policy fine-tuning has started.

Post-training evaluation is prepared with the same frozen requests, tool loop
and scoring as the baseline. It accepts only the completed3520-update policy
pair and verifies final checkpoint hashes. The Colab CPU binding/import checks
passed; source/preflight archive:
`8d8a499fcef30e0fa74fba6238cd0652b3aa533d841ce9e4d164dc048ae78fbc`.
The r2 adapter binds to the corrected training source. Its binding/import checks
passed; archive:
`558a14b79dcd5626e12f7904decfbc212c0a4f54c8799f541cecf5fba164b0e1`.
No post-training model evaluation has run yet.

The copy head already computes independent query rows in parallel. An isolated
readout profile measured prompt-key recomputation at prompt lengths31/4096/32768
and one/32 query rows. All full logits matched within3.20e-14. For32 independent
rows, parallel medians were1.806/1.945/2.895ms versus53.710/59.484/55.445ms
for serial rows:29.73/30.58/19.15x isolated row-batching speedups. A single full
readout took1.750/1.773/2.084ms, including1.451/1.488/1.810ms pointer overhead.
These measurements cover readout rows; greedy token decisions remain sequential.
No key cache or additional persistent state was introduced. Native archive:
`387745fe7ce9de905f76ab059d17a86447b20e92173bbd7f435e7e0d07a7ce9c`.

Preserve task/solution isolation and distinguish these scripted controls from
independently generated policy actions. The legacy environment's
plain pytest subprocess is not an OS sandbox and is not used by this new backend.

Neither the session nor the latest model meets the general-purpose frontier
assistant objective. Do not change the success criteria to these small checks.
