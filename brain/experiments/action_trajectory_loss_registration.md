# Paired multi-turn loss preparation

Provide a shared action-trajectory objective for the existing recurrent and
transformer models. Use complete encoded trajectories, their causal action-only
labels, and the immutable initial prompt as pointer input. Require one reset at
task start and reject any interior reset. Preserve the previously validated
recurrent-only256-token bucket padding; transformer inputs remain unpadded.
Maximum trajectory length8192. Use existing chunked language_loss and no changed
weights, architecture or optimizer semantics.

Colab CPU controls, CUDA hidden/one thread, seed1987: actual reduced-width64/
vocab128 models on81-token inputs with three action spans and an entirely masked
first injected-fault action. Compare loss and every participating parameter's
gradient against explicit full-logit cross entropy selected only at teacher
targets. Require finite gradients, participating recurrent/attention layers and
pointer parameters. RNN tolerance atol1e-10/rtol1e-8; FP32 transformer
atol1e-6/rtol1e-4. Reject interior resets and changed pointer sources.

These checks validate the objective and padding/masking, not learning success.
No optimizer updates or new native training are started by this preparation.
Native paired gradient verification and a registered training/evaluation protocol
remain necessary before using it for a model comparison. V5 remains frozen.
