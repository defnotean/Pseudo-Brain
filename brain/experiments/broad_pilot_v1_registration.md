# Broad learning pilot v1 — frozen protocol before outcomes

Purpose: test whether the parallel-depth candidate learns broader conditional
language on fixed data, compared with a small causal transformer. This is not a
frontier benchmark or proof of general-purpose competence.

Data: immutable UltraChat, CodeSearchNet Python and OpenR1 Math revisions listed
in build_broad_pilot_corpus.py. Shuffle each with seed 198, buffer 4096; inspect
4096 rows per source. Exclude unusable supervision, duplicate text and complete
documents over 32768 tokens, recording reasons. Do not truncate answers.
Join related records transitively by normalized question, repository, normalized
exact code and text. Hash components into train/development (80/20), then select
1024 train and 64 development examples per domain by text hash. Freeze the bank
and tokenizer hashes before learning. No post-outcome split changes.

Models: parallel_depth_v1 (8 recurrent layers, 384-wide, 22,387,458 parameters,
float64, paired float32 fast state) and comparison_transformer_v1 (6 causal
attention layers, width384, 6 heads, RoPE, FF expansion4, float32 parameters and
bfloat16 Flash attention). Both use the same 32K tokenizer, tied readout and
parallel initial-prompt copy head. Parameter counts must differ by less than 5%.
Transformer attention has full prefix access; the recurrent candidate does not.
Precision and compute are not matched; report elapsed time and peak memory.

Training: seed198, fresh initialization, one pass over the identical 3072-example
order. One complete document per update, response-only mean next-token loss,
immutable pointer prompt strictly before the first response marker, chunk256
vocabulary loss. AdamW lr3e-4, betas(.9,.95), eps1e-8, weight_decay.1, grad norm
clip1.0. Linear warmup64 updates then cosine to3e-5. Each example has equal weight
per update; report token-weighted development NLL separately by domain. No
checkpoint selection using development outcomes: evaluate initialization and
the final fixed-step checkpoint. Limit training to1800s/model; timeout/failure
is incomplete, not an equivalent-token comparison. Save partial evidence.

Gates before training: prior candidate architecture/long-context gates passed;
split isolation tests and transformer causality/chunked-loss tests pass on Colab;
full-size parameter counts below36M and within5%. Abort on nonfinite loss or
gradients, OOM, or malformed corpus. Do not silently shrink context/batch/data.
After training: save checkpoints, exact source/tokenizer/bank hashes, full logs,
development NLL and diagnostic greedy responses to fixed new prompts. Recheck
recurrent streaming parity using trained weights. Do not promote a checkpoint.

Follow-up requires fresh independent coding/reasoning correctness evaluations,
long-length parity and versioned POMDP integration. Development NLL, readable
samples or outperforming this small transformer do not establish frontier parity.
Sources: https://huggingface.co/docs/datasets/stream and PyTorch official SDPA
documentation for the baseline attention implementation.
