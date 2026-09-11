# Broader policy v2: unchanged autonomous evaluation conditions

The training registration already requires complete48-tool and32-code/32-math
evaluation after both arms complete. This adapter registration fixes binding and
execution details before any post-v2 generation. Do not select weights from scores.

Only the final7896-update pair is eligible, source manifest
442afc835cef0b4ae9f269f48ef353c16401e0ad1e5511fae9587ccdb28baf3d and schedule
16cbc46c6fe8b380024634520cfc210e2879737278ce8d1b2a5eeed3087cd949. The resolver verifies
complete pair/child reports, both actual checkpoint hashes, initial policy-r2
identities, exact group counts, final-boundary filenames, frozen source integrity
and all three recurrent parity lengths with finite errors<1e-6 and4096B. Reject
old completed runs, native-preflight weights, partial arms or changed bindings.
Check those rejections on Colab CPU and resolve both actual eligible checkpoints
before generation. Evaluation never trains, repairs output, retries a task or
relaxes the scorer. No local tests.

Tool evaluation reuses evaluate_trained_tool_policy.py and its complete frozen
src/ from the prior eligible evaluator. The thin wrapper changes only the
checkpoint resolver, training selection and provenance train_seed1987. The paired
controller changes only the wrapper name, resolver, manifest and train seed.
All24procedural requests x2modes stay ordered, with original48denominator,
max8cycles/512action tokens, memory accounting, environment/scoring, EOS and
token-cap behavior. 1800seconds internal and outer/model as before. Raw missing
cases remain failures in aggregates; unfinished evaluation remains incomplete.
Run recurrent then transformer, preserving a failed arm and continuing the other.

Retention reuses the previously completed retention-v1-r1 brain/ tree byte for
byte (source manifest8143d4ab610ba1ec0fad4f8a0fadb7e304cb9373a6935c2eeeae4a3e7b2a9390).
Its generator receives requests only and the same hash-bound external tokenizer.
Keep original serial initial recurrent ingestion, greedy512tokens, max32768
transformer prefix, fresh4KBrecurrent state and source parity checks. The new
retention controller changes only manifest/resolver binding. Use the original
32HumanEval/32GSM8Kbank, hashes1afbb0d6b1f8c3ba4f78a4b50d1040c67ee0701d97660ef96b267728f913642a
(requests) andacd9144610d3b1fddd80e956aede28d90a7f266b82e9f2a85d2226ff8b89b63b(graders).
Preserve actual generator/scorer execution separation, complete-function code
requirement, strict math #### parser, missing-as-failure handling and original
sandbox. No grader data reaches generation. Keep internal1800/outer1900generation
and180scoring seconds/model. Wait for tool evaluation to become terminal before
retention GPU execution. Keep64planned responses per actor and report missing
outputs explicitly. The same-interpreter grader is not adversarially tamperproof.

Freeze all adapter, controller, model and scorer source before execution. Preserve
all raw outputs, receipts, training-report/checkpoint/tokenizer/source/task hashes.
The banks are already-observed development evidence. Compare v2 with both prior
policy-r2 and retained V4 results; neither teacher NLL nor these small controls
proves frontier competence or independent generalization.
