# Policy copy-readout diagnostic v1

Motivation: the first inspected completed case from the trained recurrent model
issued recognized actions but wrote invalid function headers, later changed an
unrequested filename and put EDIT-style syntax in a WRITE_FILE body. The fixed
48-case/model evaluation continues unchanged. The cause is not yet established.

After the registered tool and coding/math retention evaluations finish, diagnose
both final r2 policy checkpoints, without updates, generation, output repair or
changes to either completed evaluation. Recurrent checkpoint SHA256
0b3edca04de150068c81fb8f71f05cd0aad8f42e423b9989bc789b919ffb85b7;
transformer346c9d4c9d334490cb2e93c54f4a1f0f66020a3833b4e7565f015b6d08c44dfe.
Require complete registered training report SHA256
1b5cf6883fcc3ad536c96ac0c06bcac354c1c5c96622ad8426953106b605a0b4.

Use policy train indices0,11,...,341 (32 evenly spaced records) and all24
development records, in stored order. Pin the existing policy gzip hashes and
tokenizer in the diagnostic source. This is a fixed subset for diagnosis, not a
new benchmark score or a selection of favorable outputs. Publish every selected
record hash and all results. Use the unchanged whole-trajectory encoder, masked
faults/observations, one initial reset and immutable initial pointer prompt.

For each full teacher-forced trajectory, compute trunk features once. For each
256-position readout chunk, compare ordinary logits with the same model's readout
when prompt_tokens=None. This removes only the additive pointer contribution at
those fixed features/prefixes. It does not approximate a no-pointer generated
trajectory or establish that disabling copying improves task performance.

Report token-summed cross-entropy, top-1 correctness, changed argmax count,
pointer-helped and pointer-hurt counts, target presence in the prompt, and gate
sum for: all supervised tokens, action header, action body, EOS, targets present
in prompt and targets absent from prompt. Region assignment uses the real action
token offsets, checked against the unchanged shifted labels. Tokens touching the
first newline belong to the header; injected-fault labels remain excluded.
Publish per-record and train/development aggregate counts, not cherry-picked
examples. No pass-rate claim follows from these teacher-prefix statistics.

Strict deterministic native execution, seed1988, both unmodified architectures,
zero optimizer updates, no checkpoint writes,300 seconds/model internal and
330 seconds/model subprocess limit. CPU controls first validate metric counting
and all56 real encoded layouts with CUDA hidden. Preserve failures; no automatic
retry, gate relaxation, or changes to the running evaluation/training pipeline.
