# Post-training tool-policy evaluation v1

Freeze this adapter before baseline or trained-policy outputs are observed.
Use the same frozen24 requests, two conditions, exact request-file order,
interleaved from_scratch/repair, initial prompts, tool environment, scoring,
8 cycles/1024 action/4096 prompt+observation limits, and1800-second outer budget
per model as tool_policy_evaluation_registration.md and its native runner details.
Use the same code modules for sessions, episode execution, scoring and fixed
denominator aggregation; do not modify the frozen V4 baseline bundle.

The only eligible weights are checkpoint-3520-trained.pt for each model in the
completed policy_training_v1 pair. Bind to its authoritative complete controller
report hash, selection5933901753e5416e75ae2fac765e3d21746de29d3e7437415a0fd12420ca98cd,
and training source manifest
e31323d90a9e2eed393119e7d0a4d0863c937109a50b275f0cbfff6bcfdfb559.
Both child reports must be complete with3520 updates and the registered
recurrent parity evidence; verify the final checkpoint SHA256 before loading.
No partial checkpoints, native-preflight weights or checkpoint selection.

Each model evaluates with strict deterministic seed198, matching the baseline;
report training seed1986 separately. Retain raw actions, actual tool events,
final files and all48 tasks including failures/missing cases. Compare completion
and observed repair with V4 baseline per model and condition; do not infer
frontier competence or generality from these procedural development tasks.

Before any native evaluation, run CPU-hidden import and checkpoint-binding
controls on Colab. These controls use inert fixture bytes and do not establish
model quality or replace actual native evaluation. Launch only when the training
controller is terminal and complete, all other owned GPU jobs are terminal,
and source/data/checkpoint hashes match the frozen evidence.
