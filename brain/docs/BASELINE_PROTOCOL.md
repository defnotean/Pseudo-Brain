# Matched architecture baseline protocol

This protocol tests a narrow claim: whether Pseudo-Brain's routed persistent thought
field improves learned control relative to simpler recurrent organizations
under controlled resources. It does **not** treat slot count, slot rank,
cosine diversity, attention breadth, or visualizations as evidence of separate
thoughts or human-like cognition.

The machine-readable source of truth is
`configs/baseline-architecture-manifest.json`. Its digest covers architecture
identities, exact parameter counts, persistent-state sizes, recipe hashes,
implementation-source hashes, fairness rules, and the explicit claim boundary.
Regenerate it only as an intentional freeze, using the play-safe wrapper:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\brain\scripts\Update-BaselineArchitectureManifest.ps1
```

The wrapper calls `scripts/build_baseline_architecture_manifest.py` with
isolated CPU-only Python. A changed digest defines a different comparison. Do
not rewrite the file to keep an old claim current.

## Frozen variants

| Variant | Purpose | Trainable parameters | BF16 persistent state, batch 1 |
|---|---|---:|---:|
| `irene.thought_field.routed.v1` | Reference: 32 slots, three registers, sparse peer routing and thought writes to working memory | 29,674,318 | 129,024 bytes |
| `irene.thought_field.isolated_slots.v1` | Communication ablation: same slots, but no peer routing and no thought-to-workspace writes | 29,674,318 | 129,024 bytes |
| `irene.thought_field.reset_slots.v1` | Persistence ablation: incoming thought state is discarded and slots reseed from sensors and belief every step | 29,674,318 | 129,024 bytes |
| `irene.thought_field.dense_routing.v1` | Dense-communication ablation: unrestricted all-to-all softmax routing instead of sparse top-k | 29,674,318 | 129,024 bytes |
| `irene.thought_field.reactive.v1` | Reactive control: no state crosses steps; belief, working memory, and thoughts reseed every forward | 29,674,318 | 129,024 bytes |
| `irene.monolithic_gru.same_width.v1` | Same-width pooled GRU control for hardware-calibrated latency/FLOP comparisons | 27,890,250 | 56,064 bytes |
| `irene.monolithic_gru.parameter_matched.v1` | Width-396 pooled GRU control selected by nearest parameter count | 29,643,900 | 57,816 bytes |

The parameter-matched monolith is 30,418 parameters below the reference, an
absolute difference of 0.102506%. Widths 396 and 397 straddle the reference
count; their absolute deltas are 30,418 and 115,433 respectively, so 396 is the
nearest integer width. Six is the largest valid attention-head count below the
reference's eight because 396 is not divisible by eight. PyTorch multi-head
attention projection capacity does not depend on head count, but head
granularity remains a documented architectural difference.

The isolated-slot model retains routing and utility parameters so allocated
parameter count is exact, but those parameters are disconnected and receive no
task gradient. The manifest therefore reports both 1,771,012 architecturally
disconnected and 27,903,306 connected trainable parameters for this control.
This makes it a clean communication-path ablation, not proof that active
capacity is equal. Actuator queries still inspect every slot;
action readout is an observer of slots, not a channel by which slots update one
another.

The reset-slot model likewise retains the lifecycle head (1,155 parameters,
allocated but disconnected) so allocated parameter count stays exact while
incoming thought state is provably uninfluential. The dense-routing and
reactive models have zero disconnected parameters: dense routing spends the
same projections on an all-to-all softmax instead of the sparse top-k, and
the reactive control reseeds every state field at the forward boundary. All
three slot ablations are exact-allocated-parameter controls by construction;
none is part of the registered parameter-matched comparison pair, which
remains the isolated-slot and width-396 monolithic controls.

The monolithic models are intentionally conventional bottlenecks: external
sources are pooled into one GRU latent while perception, belief, memory, world,
value, and actuator interfaces remain compatible with the training objective.
All variants retain 64 aggregate retrieved-memory tokens (32 x 2 for slots,
1 x 64 for monoliths), so the control does not silently lose retrieval bandwidth.
They represent one defensible monolithic family, not every possible RNN,
state-space model, transformer, or world-model baseline.

## Regime A: allocated-parameter matched

Compare the reference against both the isolated-slot and width-396 monolithic
controls. Before launching any run, freeze a multi-seed set and use identical:

- dataset generator version, bytes, split identities, ordering, and burn-in;
- optimizer, learning-rate schedule, gradient accumulation, and step count;
- objective and all loss weights;
- precision, determinism policy, hardware class, and evaluation decisions;
- checkpoint selection rule and stopping rule.

The supplied Stage-A TOMLs are single-seed templates matching the current
constant-learning-rate reference recipe. They are not registered continuation
gates and cannot be evaluated with the historical A/B gate as if their hashes
were interchangeable. A result-comparison protocol must aggregate every frozen
seed, report all failures, and avoid choosing seeds after seeing outcomes.

Primary task comparisons should include exact action-set accuracy and
changed-action accuracy. Also report per-direction recall and precision,
false-positive actions, conflicts, value/world losses, sample efficiency, and
the complete anytime-exit curve. Report effect sizes and uncertainty across
seeds; one successful run is exploratory evidence only.

## Regime B: measured latency or FLOP matched

The same-width monolith is 6.012% smaller than the reference and is therefore
not a parameter-matched control. Its purpose is a second question: what can a
conventional bottleneck accomplish within the same measured reaction budget?

This regime is deliberately marked `verified: false` in the manifest until a
device-specific calibration artifact records:

- immutable model/checkpoint and device/runtime identities;
- batch size and cognitive-cycle count;
- median and p95 end-to-end decision latency after synchronized warm-up;
- measured or profiler-derived FLOPs per decision;
- peak accelerator memory and precision;
- the predeclared matching tolerance (currently 5% for median latency).

Choose cycle count or another declared budget variable using calibration data
only, then freeze it before outcome training/evaluation. Do not tune the budget
on validation accuracy. Latency and FLOPs answer different questions, so label
which one is matched and report the unmatched metric too. Persistent-state bytes
in the manifest are tensor-layout accounting, not peak runtime memory.

## Evidence required for the breakthrough claim

A defensible positive result requires consistent multi-seed superiority over
both fairness regimes, not merely parity with one weak control. It must also
survive causal thought interventions, unseen-world generalization, and
end-to-end real-time evaluation. Slot geometry or non-collapse can support a
mechanistic investigation but cannot establish functional specialization.

Negative, null, or mixed results remain informative and must be retained. If a
control exposes a shortcut, fix the shared evaluation first and rerun every
architecture under a newly identified protocol; never patch only the losing
variant or reuse old metrics under the new manifest identity.
