# Independent pilot diagnostic v1

Freeze this protocol before generating any answers. Use the already frozen
64 requests,32 HumanEval and32 GSM8K, selected by the archived preparation
script. Requests SHA256:
1afbb0d6b1f8c3ba4f78a4b50d1040c67ee0701d97660ef96b267728f913642a.
Upstream revisions, exact overlap checks and their limits are in its manifest.

Use each explicitly hashed final fixed-step checkpoint if training completes.
Any evaluation of a partial checkpoint is a separately labeled diagnostic;
it cannot supply the completed or equal-token transformer comparison.
No outcome-based checkpoint selection or prompt edits. Greedy generation,
512 new tokens per task,32768 total-context ceiling,1800 seconds per checkpoint.
Record early EOS and truncation. A failure preserves an incomplete receipt.

Before recurrent generation, validate the selected trained weights at lengths
31,257,2049 with an internal reset and immutable initial prompt. Require all
parallel/streaming logits and states finite, physical state16x64 float32, and
maximum absolute logit difference<1e-6. This checks both scan paths. A failure
aborts generation. Include this preflight within the1800-second evaluation
limit; it does not establish full32768-token parity.

Use the training-style prefix `[THREAD:0]User: {question} [RESP]`. Every task
starts fresh. Recurrent generation retains only16x64 float32 fast state and
the immutable initial prompt before RESP; generated outputs are stored by the
evaluator, never fed back as history except the current generated token.
The transformer control uses its complete growing prefix. Neither model
loads reference answers, canonical solutions or hidden tests during generation.

GSM8K: strict numerical equality on a final standalone `#### number` line,
normalizing ordinary thousands commas and decimal trailing zeroes. Invalid
format, truncation without a valid final answer, and wrong numbers fail.
Report format validity separately; do not extract incidental numbers.
HumanEval correctness requires a verified sandbox and canonical/incorrect
grader controls before any generated code execution. Until then it is unscored.
All tasks, including unsuccessful ones, remain in denominators. A subset or
interrupted generation must be labeled incomplete. These small diagnostics
cannot establish frontier parity, even with perfect scores.
