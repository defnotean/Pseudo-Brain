# Multiscale recurrent retention diagnostic, v3

Registered before v3 execution. The parallel-pointer v1 training-trajectory
audit showed only 0–4 of 64 channels preserving greater than 1% first-response
state sensitivity at later responses. Mean-gate regularization does not prevent
individual tokens from erasing nearly every channel.

Add an opt-in retention profile with half the channels unchanged, one quarter
having a 0.99 retention floor, and the remaining quarter a 0.999 floor:
`a = floor + (1-floor)*sigmoid(raw_gate)`. Keep `b=(1-a)*tanh(candidate)`.
These transitions remain input-dependent affine recurrences, so parallel scan
and streaming use the same equations. This changes no trainable parameter shape
and does not expand fast working state. Static gate floors are model constants.
Default legacy behavior and historical checkpoint loading must remain intact.

Risk: slowly changing channels may acquire new information too slowly and
degrade existing predictions. Retention is not sufficient for useful memory.

Before training, require remote tests for legacy identity, minimum retention
over 1,000 steps, nonzero finite gate/candidate gradients, checkpoint profile
round trip, 4,096-byte physical state, <36M parameters, and actual-checkpoint
streaming/parallel parity below 1e-6 on native Colab CUDA, including resets and
pointer masking. Record failure as evidence; do not bypass the gate.

If all gates pass, allow one fresh warm start from the original champion,
parallel_predecessor pointer, token_mass supervision, compensated state, seed
1337, at most 350 optimizer steps and one hour on the owned Colab A100. Preserve
the same corrected procedural bank and all prior candidates. Evaluate the fixed
30-task development suite with four actions and task resets. Compare completion,
binding and verified repair, plus retention and token loss. No promotion from
retention or training loss alone. These repeated development tasks are not final
frontier or newly sealed qualification evidence.
