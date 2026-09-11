# Post-training evaluation amendment2: corrected training-source binding

Supersede only the eligible training-source hash in
trained_tool_policy_evaluation_registration.md. Accept the completed policy v1
r2 bundle with amendment2 and source SHA256:
3be24612bfb31fac5376bb5c33ca33b5514906a5d0e1186659cea82616183005.

This is the bundle with validated full transformer training readout and unchanged
recurrent chunking. Data/schedule,3520 updates, seeds, initialization checkpoints,
optimizer, inference, requests, tool rules, scoring and evaluation limits remain
as originally registered. Final checkpoint filenames and byte-hash checks remain
mandatory. Preserve the earlier evaluation bundle and source identity.

Copy unchanged baseline session/scoring modules into a separate r2 evaluation
bundle. Verify the CPU binding control and CLI imports for this new identity.
No policy score, baseline output or development outcome is used to change the
evaluation protocol or select a model. This amendment follows numerical training
validation only and introduces no retry or tolerance change.
