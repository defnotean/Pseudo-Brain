# GPU length-bucket gate before another learning pilot

Do not interrupt or modify broad pilot v1. Run this only when its owned process
is confirmed terminal. Purpose: determine whether trailing ignored padding
preserves numerical behavior and reduces repeated native-kernel compilation.

Full-width recurrent and transformer models, fresh seed909, lengths31,257,2049,
vocabulary32000, same immutable initial seven-token pointer prompt, chunk256.
Pad to multiples of256 with all new targets ignored. Require recurrent valid
logits, loss and all gradients to agree within1e-8. For the bfloat16-attention
control require loss difference<1e-5, valid-logit maximum difference<1e-3 and
whole-gradient relative L2 error<bf16 epsilon0.0078125. Baseline tolerances
reflect its recorded precision, not a relaxation of recurrent streaming parity.
This explicitly exercises the native scan and the >2048 fallback.

Timing: recurrent full-width forward/backward, no optimizer, lengths
[1025,1031,1057,1111,1173,1259,1281,1347], fresh seed909 for each arm.
Use separate new Triton cache directories for exact and bucketed arms, each
with a cold pass followed by the same warm pass. Record every loss, synchronized
elapsed time, peak memory and compiled cubin count. Do not call this a training
throughput benchmark or compare numerical losses across different random inputs.
Require the bucketed cold total to be lower and warm total no more than1.25x
exact warm total before adopting it in a future fixed-order learning pilot.
Otherwise preserve failure and diagnose; do not silently weaken the gate.

No model promotion follows from this gate. It does not establish learned
capability, trained-weight parity or full32768-token streaming equivalence.
