# Foundation corpus v4: data preparation protocol

Registered2026-09-11 after the completed v3 comparison scored0/32 coding and
0/32 math for both models. This authorizes a distinct data-preparation experiment;
a training schedule must be registered before training starts. It does not
extend or rewrite v3, and it is not a frontier capability claim.

Rationale: v3 has only1024 documents per domain, and99.8% of its math responses
exceed512 tokens, median5905.5. The completed model and independent gradient
checks have not identified a scan-gradient failure. Broader coverage of shorter,
complete answers is the next hypothesis, not an established causal remedy.

## Sources and candidates

Use the existing32K tokenizer, SHA256:
760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e.

- Language: HuggingFaceH4/ultrachat_200k default/train_sft,
  revision8049631c405ae6576f93f445c6b8166f76f5505a.
- Code: code-search-net/code_search_net python/train,
  revisionbd0cf261e357a3eb5c8fba490d23ec1a1cd59555.
- Reasoning: official GSM8K training data at revision
  3101c7d5072418e28b9008a6636bde82a006892c, raw train JSONL SHA256:
  17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465.

For each Hugging Face source, shuffle seed198 with buffer4096 and inspect32768
raw records. Inspect all7473 GSM8K training records. Never use test answers as
training examples. Preserve source metadata and all exclusion counts.
Bound preparation to1800 seconds; timeout or source/quota failure is incomplete.

Retain only the initial complete user/assistant exchange for language. Preserve
the full function and documentation for code. For math, preserve the complete
answer and final `####` line; remove only `<<...>>` calculator annotations,
checking the final answer is unchanged. The official source documents this
annotation removal. Format all three as `User: ... [RESP]... [EOS]`, with the
same thread marker and tokenizer boundaries as the current model interface.

Select complete response lengths<=1024 tokens including EOS and complete
document lengths<=4096. Exclude oversized examples; never truncate solutions.
Reject ambiguous special-token boundaries, missing questions/answers, duplicates,
and invalid structured records. Do not synthesize missing data or use fallback.

## Isolation and frozen selection

Join related records transitively using normalized question, repository,
normalized exact function source, and text hash, as in the existing corpus.
Include prior v1 development records as quarantined seeds; exclude any expanded
component connected to those records from both new partitions. Conservatively
exclude registered benchmark questions and matching code documentation, including
all related components. Preserve exclusion evidence without putting grader data
into training text. Exact normalization does not certify semantic disjointness.

Assign remaining components with the existing `pb-broad-v1` component-hash split
rule. Sort by text hash. Select8192 language,8192 code and4096 reasoning training
examples; select128 development examples per domain. Shuffle selected training
records with seed198. If any quota is unavailable, preparation is incomplete;
report actual availability rather than silently changing quotas or gates.

Verify all selected examples through SequencePacker's whole-document next-token
and response-only labels; verify component/key/text isolation; freeze raw/gzip
hashes, exact token totals, response-length distributions and manifest before any
training. Preserve v1/v3 banks unchanged. A future training protocol must explicitly
state initialization, optimizer/state persistence, budget and evaluation policy.

Future GSM8K results will be in-domain benchmark diagnostics after exposure to
its training split, not evidence of fresh frontier reasoning. Keep final frontier
qualification separate and unobserved.

Sources: https://github.com/openai/grade-school-math and
https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k.
