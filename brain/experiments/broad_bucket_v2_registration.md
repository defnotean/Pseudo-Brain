# Broad pilot v2 — identical data with verified length buckets

Conditional follow-up: launch only after broad pilot v1 is terminal and its
checkpoint/evidence are preserved, and every registered bucket GPU equivalence
and timing gate has passed. Do not extend/restart or relabel v1.

Use the exact frozen3072 training/192 development documents, tokenizer, seed198,
architectures, initialization, full input/target tokens, example order, per-example
loss weighting, optimizer, schedule,3072 updates and1800-second per-model training
limit from `broad_pilot_v1_registration.md`. Both models start fresh. No additional
data, training budget, context truncation or capability-based selection.

The only numerical input change is trailing padding to multiples of256, with
every new target ignored, for both training and development. Original prompts
remain unchanged; padded final states are never used for generation. Generation
and parity use actual unpadded inputs. Preserve original token counts in reports;
padding is workspace, not additional training data. The v2 runner is a separate
file, `train_broad_bucket_v2.py`; original v1 source/artifacts stay unchanged.

Save a separate output directory and source/registration hashes. Abort on the
same nonfinite/OOM/parity/time gates. Record v1's terminal status and GPU gate
receipts with the v2 launch provenance. Partial runs remain incomplete.

Compare final development NLL only after both arms complete; run the frozen
independent coding/math protocol against explicit final checkpoint hashes.
Neither passing this small comparison nor improved runtime proves frontier
competence, trained long-context parity, or POMDP integration.
