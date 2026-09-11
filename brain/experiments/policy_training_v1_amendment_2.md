# Policy training v1 amendment2: validated transformer readout

This amendment precedes all policy baseline outputs and policy training. It
supersedes only the transformer chunk256 readout requirement in the original
policy_training_v1_registration.md. Use the corrected action_trajectory_loss v2:
recurrent chunk256 with recurrent-only256 padding, transformer full readout with
ignore-index mean CE. Model architecture, inference, checkpoint initializations,
data,3520-update schedule, optimizer, LR, seeds, budgets and evaluations remain
as originally registered. The driver reports these loss paths and amendment ID.

Original v1 native failure and diagnosis remain preserved. Repeated/chunked
control gradients were stable, but readout batching changed transformer gradients.
The full transformer route now matches all55 parameter gradients bit-exactly on
both original records; recurrent all95 gradients remain within1.3e-15. Original
tolerances are unchanged. The corrected native gate passed in6.136 seconds.

Maximum8192-token transformer forward/backward/AdamW fixture passed in2.532
seconds with4,619,615,232 allocated and4,733,272,064 reserved CUDA bytes on the
42,405,855,232-byte A100. Its one diagnostic update was discarded. This resource
result supports the larger transient readout memory; it is not policy training.
Evidence archive SHA256:
f9b8b692d28e81a7f501b0bf8ad45a36dc42b88ad7e2adabf1fd43357117de75.

Freeze a separate r2 source bundle; retain r0/r1 and their hashes. The queued
baseline uses its unchanged frozen inference/scoring source. The two discarded
mixed-data native preflight updates per model must use r2 before full training.
Post-training evaluation must bind to the r2 training source digest and completed
final checkpoints, not the superseded r1 source. No capability-driven tuning or
tolerance relaxation is authorized by this amendment.
