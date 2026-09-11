# Executed multi-turn supervision preparation

Add an opt-in recorder that executes an explicit teacher plan through the
isolated environment and saves actual feedback, provenance IDs, initial file
snapshot and final file digest. Label its policy source scripted_teacher.
External completion validation is mandatory. Incomplete records remain evidence
but cannot enter supervised training. Do not infer successful observations from
templates or promote scripted success to an autonomous capability result.

Mark each action as teacher or injected_fault before execution. Fault actions
enter the recurrent context but produce no supervised targets. Teacher actions
may reveal failures, such as RUN_TESTS on buggy code; those valid choices remain
supervised. Malformed actuator output is not a valid teacher target.

Encode prompt, RESP, action, EOS, and each JSON observation as separate segments
matching inference exactly. Predict teacher action tokens and EOS only; mask
initial prompt, all observations, EOS-to-observation transitions, and injected
actions. Keep one reset at task start and the pointer source exactly the initial
prompt. Hash-check records, reject ambiguous action control tokens and overlong
complete trajectories rather than truncating them. Do not modify frozen V5.

Colab CPU controls, CUDA hidden/one thread: real isolated incorrect-program/test/
edit/test/validated-finish trace, exact segment concatenation and label positions,
fault masking, EOS target, observation masking, one task reset, immutable pointer
input, record-tamper rejection, incomplete-record rejection and input validation.
Include a check with the frozen32000-token BPE tokenizer and preserve the raw
executed control record in the evidence archive. Reject an embedded EOS token
inside an action instead of teaching an inference-incompatible boundary.
This verifies data infrastructure only. No new training bank, optimizer updates,
held-out evaluation or frontier claim is authorized by these controls alone.
