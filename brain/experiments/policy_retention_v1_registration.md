# Policy-training retention evaluation v1

Registered before policy training and its post-training generation results.
This is a retention diagnostic on an already observed development bank, not a
fresh frontier qualification or a new model-selection opportunity.

Evaluate both final checkpoints from the completed 3520-update r2 policy pair.
Use the existing checkpoint resolver from the frozen post-training tool evaluator
(source manifest 98b85f02412d4b8bf02f8cea4b50322fe4815087b8eed8ec17d4f467653d9cee).
It binds the final weights to the complete pair, training source manifest
3be24612bfb31fac5376bb5c33ca33b5514906a5d0e1186659cea82616183005, fixed training
selection, and all required recurrent parity records. Reject partial, diagnostic,
or retrospectively selected checkpoints. Wait for the registered post-training
tool evaluation to become terminal before starting this GPU run.

Use the same 32 HumanEval and 32 GSM8K requests, ordering, prompts, serial initial
recurrent ingestion, greedy 512-token generation, EOS behavior, and scorers used
for the V4 foundation checkpoint diagnostic. Requests SHA256:
1afbb0d6b1f8c3ba4f78a4b50d1040c67ee0701d97660ef96b267728f913642a.
Graders SHA256: acd9144610d3b1fddd80e956aede28d90a7f266b82e9f2a85d2226ff8b89b63b.
Generation receives request-only inputs. Scoring is a later separate process;
the existing sandbox, positive/negative grader controls, complete-function code
requirement, strict math parser, and missing-as-failure denominator remain intact.
No output cleanup, task retry, prompt adaptation, or model selection is allowed.

The policy checkpoints intentionally contain architecture/configuration/weights,
without embedded tokenizer JSON. The generator therefore accepts the separately
hash-verified training tokenizer (SHA256
760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e).
If a checkpoint also embeds a tokenizer, its parsed JSON must match the external
one. This adds no weights or training metadata to the checkpoint. Before GPU
execution, verify tokenizer IDs against both original V4 embedded tokenizers on
Colab CPU, reject absent/mismatched bindings, and verify all model, decoding, and
scoring source files are unchanged from the frozen V4 evaluator. The generator
change is limited to tokenizer resolution and its receipt fields.

Preserve the original 1800-second internal generation bound, 1900-second outer
generation bound, and 180-second scoring bound per model. Continue to the other
model after a failed or timed-out arm. Report incomplete arms explicitly and
retain planned denominators of 32 per benchmark. Do not treat missing responses
as completed evaluation, or silently invent zero-score measurements when scoring
did not run. The actual V4 reference is recurrent 0/32 HumanEval and 1/32 GSM8K;
transformer 0/32 on both, with no missing responses.

Record source, training-report, tokenizer, checkpoint, response, and grader hashes,
raw generation and scoring receipts, subprocess outcomes and elapsed times.
The recurrent state remains 4096 bytes; the transformer retains a full prefix.
No optimized generation-speed or frontier-competence claim follows from this run.
