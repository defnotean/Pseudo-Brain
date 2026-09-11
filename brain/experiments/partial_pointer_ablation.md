# Partial-checkpoint repetition diagnosis

After the fixed64-task diagnostic produced0/32 coding and0/32 math, with all
responses capped and median four distinct tokens per512-token response, test
whether the immutable-prompt pointer contributes to the observed loops.

Use only checkpoint SHA256
85fe27c37d9eaa6c0140a188091e7ced9d0049ce5246dae1f55ff7a73529f228,
the933-update incomplete run. Compare unchanged pointer-enabled inference to
pointer-disabled readout with identical frozen weights and fresh states.
Use the six original PROMPTS in frozen train_broad_pilot.py,128 greedy new
tokens per prompt, identical tokenizer and prompt format. Record full responses,
EOS/limit stops, unique token counts and repeated4-gram fractions. No optimizer,
checkpoint overwrite, promotion, prompt tuning or extra benchmark claims.

This is a causal component ablation of a failed partial checkpoint, not evidence
that disabling the pointer yields a capable model. The64 observed benchmark
tasks remain excluded from training; future frontier qualification requires
fresh held-out evidence rather than tuning to these diagnostics.
