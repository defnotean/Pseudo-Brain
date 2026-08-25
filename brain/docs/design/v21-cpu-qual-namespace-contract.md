# V2.1 sealed CPU-QUAL namespace contract

Status: implemented contract; CPU-QUAL remains unmaterialized.

The qualification lifecycle is deliberately one-way:

1. Reserve ordered `TRAIN`, `DEV`, and `CPU-QUAL` virtual-dataset ranges.
   Their local seed intervals must be disjoint. `TRAIN` must use the train
   namespace; `DEV` and `CPU-QUAL` must use validation. TEST is forbidden.
2. Freeze the exact dataset manifests, metric/gate protocol, and sorted source
   file bundle in a canonical preregistration. The protocol must name every
   candidate and require all of them to pass. Publish it create-only.
3. Only a canonical on-disk preregistration whose source hashes still match may
   authorize CPU-QUAL generation. Building a draft in memory is insufficient.
4. After authorized deterministic generation, freeze every ordered sequence
   content SHA-256 in the registered create-only content-manifest path.
5. Before evaluating any model, publish a once-only opening receipt binding the
   content manifest to the full candidate checkpoint cohort. The opening call
   refuses to run if content is absent, source changed, or the receipt exists.

This contract does not generate a maze, import Torch, train a model, or expose
TEST. The future qualification runner must accept the authorization objects from
`irene_brain.evaluation.v21_cpu_qual_namespace` instead of constructing a raw
CPU-QUAL dataset source directly.

Recommended fresh, human-readable local ranges for the first preregistration
are `16_777_216+` for TRAIN, `25_165_824+` for DEV, and `33_554_432+` for
CPU-QUAL. None of these ranges is materialized or registered by this document.
