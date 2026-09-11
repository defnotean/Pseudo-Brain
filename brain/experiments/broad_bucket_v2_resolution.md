# Conditional v2 not authorized by its gate

The registered GPU padding gate failed for the transformer control. At lengths
31/257/2049, valid-logit maximum differences were0.0030812/0.0023087/0.0023871;
the31-token loss difference was4.1008e-5. These exceed the frozen thresholds.
Recurrent logits, losses and gradients passed all three lengths (largest logit
difference4.21885e-15). Whole-gradient relative errors for the bfloat16 control
were below epsilon, but that does not excuse its failed logit/loss conditions.

The controller stopped before timing. Do not launch train_broad_bucket_v2.py,
relabel this gate as passed, or relax its thresholds after observing outcomes.
Its source, protocol and failures remain preserved. No v2 training occurred.

A reasonable distinct follow-up is recurrent-only padding with the transformer
kept entirely on its original exact-length input path. The measured compilation
bottleneck is in the recurrent native scan, so changing transformer input shapes
is unnecessary. Such a follow-up requires its own registration and the pending
cold/warm recurrent timing evidence; no launch is implied by this resolution.

Separately, the933-update v1 partial checkpoint scored0/32 coding and0/32 math,
with severe repetition. Disabling its pointer did not remove repetition on the
six frozen pilot prompts. Completing a training pass may inform the diagnosis,
but these results supply no positive capability claim or model promotion.
