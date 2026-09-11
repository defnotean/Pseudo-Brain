# Native paired action objective gate

After existing GPU work and multi-turn parity gates finish, check the first two
records of the accepted procedural training bank (ordinary write then repair).
Use both unchanged V4 trained checkpoints, full width/vocabulary, strict
deterministic mode and CUBLAS_WORKSPACE_CONFIG=:4096:8 before Python.
Use actual frozen tokenizer/records and causal encoder; no development records.

Compare explicit full-logit, selected-action-target cross entropy and every
participating parameter gradient against action_trajectory_loss with its normal
256-token readout chunks and recurrent-only bucket padding. Require finite full
logits/losses/gradients, equal participating parameter sets, and per-element
torch.assert_close with RNN atol1e-10/rtol1e-8 and transformer atol1e-6/rtol1e-4.
Report maximal gradient differences, all input/record/checkpoint hashes and
run_provenance. A failed gate must remain failed and blocks fine-tuning under
this protocol. No optimizer updates. Bound the entire paired check to300seconds.

Training gzip SHA256896a39d478aaa674903c27eb9c7af0d96017bc721843c860a3d56daa056d73a0.
V4 recurrent checkpoint7feee0e9688dd4fb9014abedf677d09e071986c57ea4acdd8b8a97b61a2c28bf;
V4 transformer checkpointca080bd8c6562656073dc90b64341cd624c931c1972c07c19ea9a2b4a59d1160.
This gate verifies gradients/objective, not capability or learning effectiveness.
