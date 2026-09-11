# Paired tool-policy development evaluation v1

Freeze evaluation before baseline outputs or policy training. Use the24 accepted
development requests (six solution families absent from procedural policy train)
in two conditions,48 tasks/model: original from_scratch and repair. Repair adds
an explicit RuntimeError stub to the public target module and a fixed repair
instruction; derive the target from the public prompt and verify grader binding.
Do not change original frozen requests or use private answers in model prompts.
These familiar procedural algorithms are not a frontier or globally unseen bank.

Both models use identical raw action/observation logic,8 cycles,1024 action tokens,
4096 prompt/observation tokens, fresh task state and external FINISH validation.
Recurrent retains4096-byte state; transformer retains full prefix with32768 cap,
reports that separately and is not an optimized generation-speed baseline.
Use strict deterministic settings and current fixed V4 checkpoints for baseline.
Later checkpoints must be separately identified and pass required native gates.

Run positive reference and negative RuntimeError controls on public/private
checks before each task. Scores: completion over48, observed repair over48 plus
condition-specific counts, recognized action verbs, successful environment actions
and stop reasons.
Repair credit requires an actual sandbox test attempt or external validation
failure, a later successful WRITE/EDIT that changes the target file hash, and
subsequent independently verified FINISH. Unknown actions, missing tests,
malformed paths, unchanged rewrites and unverified finish receive no repair
credit. Preserve raw generations, actual action audit and final virtual files.
Same-interpreter grader limitations remain explicit.

Each model run is bounded1800seconds, records all missing tasks as not completed
and reports incomplete status on interruption/control failure. Denominator48 is
fixed. No retries, output repairs, altered prompts or adaptive hyperparameters
based on evaluation outcomes. Do not conflate infrastructure/scripted controls
with learned policy performance. No model training under this registration.

Requests SHA256c306d8e0399531c1157fed1abb09314013f577cf4be0ca564aaeebaa05f561a1;
graders SHA25628d7ed1677ace53899058c56047253ee109523594e0e1c42328979fc480c468d.
Native launch must wait for active GPU jobs and passing ingestion/multi-turn gates.
First verify scoring logic and an explicitly scripted isolated repair control
on Colab CPU with CUDA hidden and one thread.
