# Native multi-turn training/inference alignment gate

Prepare a mechanical gate, not a capability benchmark. Require the preceding
native carried-state ingestion gate to pass and all current GPU work to stop
before launch. Use the unchanged full-width V4 recurrent checkpoint SHA256
7feee0e9688dd4fb9014abedf677d09e071986c57ea4acdd8b8a97b61a2c28bf,
the frozen BPE tokenizer, and the archived executed-trajectory infrastructure
control record. This record is already observed and is not a held-out task.

Encode the complete real trace with the validated causal encoder. Compare every
position's full-vocabulary parallel logits to serial stepping (finite,absolute
error<1e-6). Compare actual session state and returned logits at prompt/action/
observation boundaries (decoded-state error<1e-12,4096-byte float32 state).
Start fresh and perform three greedy16-token diagnostic generations, ingesting
the first three recorded feedback frames between them. Require identical token
IDs/stopping and state parity with an independent serial reference. These fixed
observations are diagnostic inputs, not claimed outcomes of the generated actions.

Use strict deterministic settings through run_provenance and
CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python. Record source/checkpoint/record/
tokenizer/runtime hashes. No updates or changed model/decoder semantics.
Native gate bound300seconds. First validate the validator on Colab CPU with
reduced width32/vocab128 synthetic81-token/three-action inputs and explicit NaN
rejection. A CPU pass does not qualify the trained native path by itself.
