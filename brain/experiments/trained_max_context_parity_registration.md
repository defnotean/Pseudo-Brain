# Trained maximum-context parity check

Registered before execution on 2026-09-11. Frozen v3 recurrent checkpoint SHA256:
723bf4dcf2db108a17f6864a21e623c02f27ce9f2f62b999c5921eb9aeec13eb.
No weight updates, architecture changes, or relaxed numerical bounds.

On the owned Colab A100, after the active independent evaluation and registered
gradient tests are terminal, compare every vocabulary logit at every position
of one 32768-token input. Seed 1063; random nonzero vocabulary IDs; replace
positions 4096:8192 with one repeated token. Reset immediately before positions
0 and 16384. The immutable pointer prompt is the first31 input tokens throughout.

Compute full-sequence parallel features once, then read out chunks of256 for
workspace economy. Compare against sequential `step` calls, retaining exactly
the returned16x64 float32 state between tokens. Do not replay prior tokens or
save them in the state. Check state size/dtype throughout and all finite logits.
Require global maximum absolute logit difference strictly below1e-6. The same
frozen trained checkpoint and model source must be used on both paths.

Limit to900 seconds. Any timeout, OOM, nonfinite output or bound failure is an
incomplete/failed result, never a pass. Record errors and partial position count.
Record wall time, GPU/torch, source/checkpoint hashes, peak allocated memory,
all chunk maxima and final outcome. This validates one reproducible long input,
not every possible context or semantic capability. No champion promotion.
