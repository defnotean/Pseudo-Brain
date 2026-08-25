# V2.1i production PLAY-QUAL contract

This is the final closed-loop gate for a post-DGX V2.1i candidate. It cannot
qualify a development checkpoint and it does not train, open CPU-QUAL, or touch
TEST. The only accepted checkpoint is the schema-2 bounded-DGX envelope loaded
on CPU with CUDA hidden and one Torch intra/inter-op thread.

## Required lineage

PLAY-QUAL is causally downstream of all earlier gates:

1. The schema-2 checkpoint records the bounded DGX authorization, schedule,
   run artifact, exact source bundle, model state, configuration, feature flags,
   and accepted bias-only hazard calibration.
2. A canonical external post-DGX release receipt binds that exact checkpoint to
   the 5/5 CPU-QUAL cohort (`44..48`), its preregistration, opening receipt, and
   evaluation artifact. It also requires a separate post-DGX verification
   artifact with finite, calibration, causal-shuffle, hazard-quality,
   non-collapse, and preservation gates all passing. TEST remains unopened.
3. A real wall-clock artifact binds the same checkpoint, release receipt,
   source, and PLAY preregistration. It measures three repetitions of 100 warmup
   plus 5,000 measured ticks and must pass the frozen 8 ms model p99, 12 ms loop
   p99, 16.667 ms loop p99.9, and `<0.1%` miss-rate limits.
4. Only then may the production runner open the sealed PLAY-QUAL seed bank.

The post-DGX receipt is external because putting its digest into the checkpoint
that it qualifies would create a circular file-hash dependency.

## Sealed cohort

The create-only PLAY preregistration freezes:

- exactly 64 environment seeds, `0xc000000000000000` through
  `0xc00000000000003f`;
- the unassigned uint64 namespace with top bits `11`, disjoint from TRAIN
  (`00`), validation/DEV/CPU-QUAL (`01`), and TEST (`10`);
- one environment-seed bootstrap cluster per seed;
- 600 maximum ticks in the exact registered MazeChase configuration;
- one model execution plus eight uniform idle/W/A/S/D random streams per seed;
- one deterministic replay of every execution, used only as a determinism
  audit; and
- 10,000 paired cluster-bootstrap resamples at 95% confidence.

The random comparison is computed per environment as model minus the mean of
its eight random streams. The 512 random executions are never treated as 512
independent bootstrap clusters. A passing report requires strictly positive
lower confidence bounds for reward advantage and catch reduction, at least one
model pellet in every episode, legal controls, deterministic replay equality,
the exact simulated 60 Hz control schedule, and the separate real wall-clock
receipt.

## Production commands

After the post-DGX checkpoint and release receipt exist, seal PLAY-QUAL without
running an episode:

```powershell
python brain/scripts/v21i_live_qualification_runner.py preregister `
  --checkpoint <post-dgx-checkpoint.pt> `
  --post-dgx-release <post-dgx-release.json> `
  --post-dgx-release-sha256 <release-file-sha256> `
  --output <play-preregistration.json>
```

The wall-clock qualifier must then measure the same checkpoint using
`--checkpoint-envelope post_dgx_v1` and the same release receipt. Finally run:

```powershell
python brain/scripts/v21i_live_qualification_runner.py qualify `
  --checkpoint <post-dgx-checkpoint.pt> `
  --post-dgx-release <post-dgx-release.json> `
  --preregistration <play-preregistration.json> `
  --wallclock-receipt <wallclock-artifact.json> `
  --output <play-qualification.json>
```

Both the preregistration and final result are canonical create-only JSON. A
preflight error writes a create-only failure receipt when the result path is
still available. Existing files are never replaced. The production call has no
controller, policy, environment, configuration, or episode-runner injection
surface; it constructs `OutcomeAwareCoreV2MazePolicy` and exact MazeChase
internally.

The current repository work has only exercised synthetic contract tests. No
PLAY-QUAL cohort, CPU-QUAL data, gameplay, DGX training, or GPU context was
opened while implementing this gate.
