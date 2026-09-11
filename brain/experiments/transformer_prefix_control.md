# Transformer restart discrepancy: repeated uninterrupted prefix control

The native restart gate failed exact transformer suffix losses, while a separate
audit found every restored model/optimizer/RNG tensor identical to the saved
boundary. Before changing checkpoint or numerical code, test whether independent
uninterrupted prefixes themselves differ.

Run the frozen native restart runner twice in fresh subprocesses, transformer
`interrupted` phase only. Each performs exactly three updates from seed198 on the
existing seed965 tensor bank and saves its optimizer boundary. Reuse the same
verified source manifest, architecture, precision, optimizer, LR and RNG sentinels.
No restore occurs in either training subprocess. Compare all model/optimizer
state and RNG against each other and against the original interrupted boundary.
Report exact equality, changed tensor counts and maximum absolute differences;
preserve losses, provenance and all mismatches. No pass threshold is relaxed.

Use Colab only, after previous GPU jobs are terminal, with180 seconds per prefix.
This is a six-update diagnostic, not longer capability training or a retry of the
failed gate. If uninterrupted states differ, determinism becomes a supported
explanation to investigate; it does not identify a particular kernel by itself.
Any later deterministic-mode experiment requires its own registration and keeps
the original failure intact. If states match, investigate other restart effects.
