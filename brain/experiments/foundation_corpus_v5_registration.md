# Foundation corpus v5: broader coverage and explicit code interfaces

Registered before collection or training. V4's observed capability remains
0/32 coding and1/32 math for the recurrent candidate,0/32 each for the transformer.
The static code audit found6870/8104 parseable training functions lacked their
target name in the question. V5 addresses that known conditioning defect and
broadens unique training coverage; it is not a controlled architecture ablation.

Reuse the pinned v4 UltraChat and Python CodeSearchNet revisions, shuffle seed198
and4096-row buffer. Scan exactly131072 candidate rows from each, plus all7473
pinned official GSM8K training rows. Language/reasoning formatting is unchanged;
code uses the verified signature_conditioned_exchange formatter. Include the
original declaration and documentation, preserve the full answer unchanged, and
retain all old isolation keys. Reject unparseable or ambiguous functions.

Re-tokenize every candidate using the same hashed32000-token BPE. Keep complete
responses of at most1024 supervised tokens and documents of at most4096 input
tokens; reject rather than truncate. Keep the established special-token and
response-boundary checks. No samples are selected using model outputs or scores.

Quarantine connected components matching registered benchmark question/docstring
keys, the original192-document development bank, or the frozen v4 development
bank384 documents (gzip SHA256f95f6a2f491db9df37f8682283a49605af68da62e5d0274e5d33a7e7815388b5).
Retain component-level repository/code/text/question isolation. This remains exact
normalized matching, not semantic decontamination or fresh frontier qualification.

Select by sorted text hash within the existing deterministic component partition:
train32768 language +32768 code +4096 reasoning =69632 documents; development512
per domain =1536. Shuffle each selected partition with seed198. A missed quota,
changed source/hash, parsing/label check failure or timeout is incomplete; never
silently lower quotas or repeat records. Recheck whole-document labels, EOS,
no packer remainder and all split isolation over the selected bank. Save raw and
compressed digests, counts, token totals, exclusions and source/runtime provenance.

CPU-only preparation on Colab,1800-second limit. Preserve all earlier banks.
Do not start model training from this registration alone: a separate fixed
training/resume protocol and completed frozen manifest are required first.
