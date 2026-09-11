# Parallel observation ingestion preparation

The legacy POMDP agent expects UnifiedCognitiveState and role-labelled slots;
the newer language core has a different8-layer logical state packed into16x64
float32 values. Do not disguise those layers as independent goal/perception slots
or silently reuse the legacy agent's extra sequential pointer state.

Prepare an isolated opt-in primitive that consumes only newly arriving token
chunks plus the existing4096-byte state. Decode each layer's high/low state and
pass it as h0 to the existing parallel affine scan. Keep the initial task prompt
immutable for the pointer. Return only final logits and the ordinary fixed-size
state; no earlier generated tokens or observation history may be replayed.
Transient chunk features are computation workspace, not persistent model state.

Leave the established initial-prefill helper, current decoder and frozen V5
training bundle unchanged. First validate on Colab CPU with reduced width/vocab:
nonzero incoming states, batch2, lengths1/31/257, pointer on/off, per-stream resets,
ordinary generated-token continuation, different chunk boundaries, input-state
immutability and rejection of malformed/nonfinite state. Require logit error<1e-6,
reconstructed state error<1e-12 and4096 bytes per stream. Native trained-checkpoint
validation must follow after current GPU work; CPU checks alone do not qualify
the new path for native rollout or establish autonomous software competence.
