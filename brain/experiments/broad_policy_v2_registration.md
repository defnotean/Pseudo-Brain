# Broader matched policy training v2

Motivation: the completed narrow policy run memorized demonstrations but solved
zero full tool/code/math tasks and increased foundation development NLL in all
three domains. The audited MBPP and actual behavior data now support a broader
experiment. No frontier equivalence follows from this small model comparison.

Start from completed policy-r2 checkpoints: recurrent
0b3edca04de150068c81fb8f71f05cd0aad8f42e423b9989bc789b919ffb85b7 and transformer
346c9d4c9d334490cb2e93c54f4a1f0f66020a3833b4e7565f015b6d08c44dfe.
They share the same completed prior schedule; retain learned actuator grammar.
Do not substitute partial V5 weights or select a checkpoint from new scores.

Use these hash-bound records in stored order before deterministic shuffling:
352 original procedural, 220 varied procedural, 357 clean MBPP, 27 recurrent
behavior-prefix continuations, 31 transformer behavior-prefix continuations.
Both actors train on the union of all 987 records, including both actor sources.
Repeated underlying cases across clean/fault/behavior versions are deliberate;
987 records are not 987 independent tasks. All records are training partition.
The existing codecs preserve exact actor IDs and mask all actor/injected-fault
actions, observations and prompt. Only verified teacher actions/EOS are targets.
Keep one reset per record and initial-prompt-only pointer source, no truncation.

Two policy epochs, shuffle global record indices with Random(1987+epoch).
After each policy record insert one distinct foundation code, reasoning and
language row in that fixed order. For each domain, shuffle its stored training
indices with Random(1987+domain_index), domain order code/reasoning/language.
Consume each domain's first1974 indices without replacement across both epochs.
This gives1974 policy plus5922 foundation updates,7896 total; replay is75% of
updates. Report actual supervised/input token shares as well. All replay comes
from frozen V5 train; no development rows. This is a new hypothesis, not a
guarantee against forgetting. Freeze the full schedule and digest before training.

Use unchanged proven action_trajectory_loss: recurrent chunk256/bucket256,
transformer full reference readout, whole examples, no cross-example state.
All parameters train. Strict deterministic seed1987 for Python/NumPy/Torch.
AdamW betas(.9,.95), eps1e-8, decay.1; finite gradient norm clip1.
Learning rate warms linearly to3e-5 over64 updates, then cosine to3e-6 at7896.
No adaptive sampling, reinforcement learning, output correction or early model
selection. Internal update/checkpoint budget1200seconds/model, outer budget
1800seconds/model. This newly bounded protocol does not resume failed old runs.
Save inference checkpoints after each epoch (3948,7896); do not auto-resume or
promote partial checkpoints. All model execution occurs on Colab A100.

Before optimizer training, CPU preparation (300-second deadline) verifies every input hash, record
identity, partition and family overlap against procedural development, encoding,
exact mask/reset and action/observation/prompt limits, foundation uniqueness,
domain counts and complete schedule. Existing upstream MBPP decontamination
remains structural, not a proof of semantic novelty. Native preflight exercises
the longest record from each of the five policy sources and each of the three
foundation domains: eight DISCARDED updates per actor. Require finite positive
gradients/parameters and recurrent streaming parity <1e-6,4096B,<36M parameters.
Both preflights must pass before either full training actor starts.

Before and after training, evaluate all24 frozen procedural development records
and all1536 frozen V5 foundation development rows using token-weighted NLL by
domain. No development gradient updates. Final recurrent parity must pass.
After both complete, run unchanged full48-tool and32-code/32-math conditions with
the final checkpoint identities and all attempts recorded. Compare with policy-r2
and retained V4 results. These are already-observed development banks; new held-out
capability evidence will still be required. No extra model-generated MBPP prefixes
are collected adaptively during this run. Do not promote based on teacher loss.
