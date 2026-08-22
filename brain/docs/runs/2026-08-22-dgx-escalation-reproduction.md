# Phase 2.5: 8-hypothesis escalation reproduced on DGX GB10 (2026-08-22)

**Machine:** gx10-db18, NVIDIA GB10, CUDA 13.0, torch 2.13.0+cu130 (container
`vllm/vllm-openai:nightly-aarch64` @ `177a406d7cb2`, network-none, GPU enabled).
**Git HEAD at launch:** 25b16e3 (Windows workstation; scripts byte-identical on Spark).
**Script:** `dgx_phase2_definitive_escalation.py` (unmodified). Console:
`~/projects/pseudo-brain/runs/phase25-definitive-escalation-gpu-v1/console.log`.
**Qwen eviction:** container `vllm-qwen38-mtp` (vLLM Qwen3.8-27B-NVFP4, PID 899431,
111,335 MiB) stopped cleanly via `docker stop`; unified memory freed 117 GiB → 2.5 GiB used.

## Design

- Models: Proposal-GRU baseline + Pseudo-Brain K ∈ {1, 8, 16, 32} (W=120, C=3)
- **5 independent TRAINING seeds** {42, 142, 242, 342, 442}, 1200 steps each
- Held-out EVAL seeds {1000..5000}, delay=15
- Hostile ablations on seed-0 checkpoints: frame reset / prob scramble / binding scramble / permutation

## Results [MEASURED]

| Model | Mean Return (5 train seeds) | Median | S1 Surv | S2 Surv | Acc |
|---|---|---|---|---|---|
| Proposal-GRU | −421.66 ± 198.61 | −531.70 | 20.0% | 23.4% | 13.0% |
| PB K=1 | −482.06 ± 16.31 | −485.10 | 4.0% | 25.2% | 11.6% |
| PB K=8 | −396.52 ± 87.95 | −436.30 | 16.2% | 27.4% | 6.8% |
| PB K=16 | **−391.20 ± 81.11** | −437.80 | 22.6% | 27.0% | **14.2%** |
| PB K=32 | −396.76 ± 117.86 | −463.90 | 11.4% | 45.0% | 10.8% |

Hostile ablations (seed-0 checkpoint):
- Frame Reset hurts every model with real persistent state (K=16: −347.25 → −521.20;
  K=32: −463.90 → −530.50; GRU: −531.70 → −535.00).
- Prob/binding scramble: small effects (K=16 −347.25 → −324.75 actually *improves*;
  within seed noise).
- Whole-slot permutation: harmless (K=32: −463.90 → −464.10 ≈ noise) — invariance holds.

## Interpretation

1. The CPU preliminary (K=8 −352.80 vs GRU −437.66) did **NOT reproduce**: on DGX with
   5 independent training seeds the ordering is K=16 (−391) > K=32/K=8 (−396) > GRU (−422),
   and all differences are **within 1–2σ of between-seed variance** (GRU σ=198, PB σ≈81–118).
   [INFERRED] No architecture-superiority claim is supportable from this run.
2. What DOES replicate across CPU and DGX: **frame-reset causal sensitivity of the PB thought
   field** (reset collapses returns toward the catastrophic floor) while permutation stays
   harmless. Persistence is causally load-bearing for PB.
3. K=32's Stage-2 survival jump (45.0% vs ~23–27% elsewhere) is the only outlier signal
   worth following up — single-checkpoint (seed 0) though, so [HYPOTHESIS].
4. GRU median (−531.70) is much worse than its mean — heavy left tail from failed seeds.
   High-K PB has tighter medians. [INFERRED] PB may be more *stable* even where means tie.

## Next actions

- The 8-hypothesis benchmark does not separate architectures beyond noise at 1200 steps.
- Move to Priority 7 (ephemeral-memory campaign on DGX) and consider longer training /
  harder tasks before any superiority claim.
