# Run record: Phase 2.5 Gate 6 matched Proposal-GRU probe

**Date**: 2026-08-20  
**Probe ID**: `dgx-gate6-matched-gru-v1`  
**Gate**: Phase 2.5 Gate 6 (≥10% IQM/return over matched Proposal-GRU)  
**Status**: **FAIL** (Proposal-GRU wins pooled IQM; architecture superiority not claimed)

## Inventory before launch

| Item | Finding |
|---|---|
| Git HEAD | `97ffe62579c90744595960fc7d342b21e3e490ef` |
| Spark alias `defnotean` | NVIDIA Sync/Tailscale proxy down; LAN `192.168.0.176` (`gx10-db18`) used for this poll |
| GB10 `nvidia-smi` | Occupied: `sglang.launch_server` Qwen + `sglang::scheduler` (~1.2 GiB + 19.5 GiB). Not a Pseudo-Brain train. |
| Existing campaign JSON | `thought_mediated_campaign_results.json` @ 2026-08-20 06:35:40 UTC, `cuda:0` |
| Gate 6 in that JSON | **Absent.** No thought-mediated vs Proposal-GRU closed-loop IQM. |
| Stochastic proxy (not Gate 6) | Proposal-GRU expected utility **-37.74**; K=32 **-38.71**. GRU better. Do not treat as Gate 6. |
| Knockout sanity already on file | Family B `A_normal` IQM **-32.0** vs `H_zero_knockout` **-486.0** (thoughts still on the action path). |
| Latest Spark releases | `r20260820t175216z-db65efc48763`, `r20260820t165031z-6aab8657a55c`. Campaign SHA-256 `5953b4e1…` **does not match** HEAD campaign SHA-256 `f9efa4a7…`. Probe uses a read-only **HEAD snapshot**, not those releases, and not dirty working-tree model diffs. |

## Config pair

Documented in
[docs/preregistrations/2026-08-20-gate6-matched-proposal-gru-probe.md](../preregistrations/2026-08-20-gate6-matched-proposal-gru-probe.md).

- Thought-mediated K=32, width **60**, 3 cycles (runtime match; HEAD map W=32 was not used).
- Proposal-GRU hidden 180.
- 300 steps, seeds 42–46, five families, 5 episodes/family.

## Launch

Queued **behind** the GB10 occupant. Named Spark CPU docker job
`dgx-gate6-matched-gru-v1` **exited** after writing results (tmux + docker,
`CUDA_VISIBLE_DEVICES=-1`; container auto-removed):

- Image: `sha256:177a406d7cb2a11338bcd8c67ab7590b330799cdc5a3193ac0ca40728ea2501b`
  (`vllm/vllm-openai:nightly-aarch64`)
- 12 CPUs, 32 GiB, `--network none`
- HEAD snapshot tar SHA-256 `0eafbcb0e030dc8ab066900f695966bd10a082163f327f29cb78a2ca8a4c7945`
- Probe SHA-256 `631f85061f6abca0c5b427aabacdca573afc1caf64f4da3fffaf0e1370b37472`
- Runtime pair (printed on launch): **PB K=32 W=60 C=3 = 831,680 params** vs **GRU H=180 = 787,314 params** (**+5.64%**, inside the 6% envelope). HEAD width map W=32 was 270,112 and was **not** used.
- Output: `~/projects/pseudo-brain/runs/dgx-gate6-matched-gru-v1/gate6_results.json`

NVIDIA GB10 remains on Irene sglang Qwen. This probe did not start a second GPU train.
Sibling ARM job `play-competence-cpu-20260820-v1` was left running.

## Result

`gate6_results.json` written 2026-08-20 20:41:10 UTC on Spark ARM CPU
(`cuda_available: false`). Artifact copy:
[artifacts/gate6-matched-proposal-gru/gate6_results.json](artifacts/gate6-matched-proposal-gru/gate6_results.json)
(SHA-256 `a3dc94dcbfebd6d914ed3c7caaf4725db9e5d7ef1bd2f47e7772311b73e9454c`).
Container `dgx-gate6-matched-gru-v1` is gone (auto-removed). GB10 remained on
Irene sglang Qwen for the whole probe. No second GPU train.

| Metric | Thought-mediated K=32 W=60 C=3 | Proposal-GRU H=180 |
|---|---|---|
| Parameters | 831,680 | 787,314 (+5.64% PB) |
| Pooled IQM return | **−27.758** | **−25.694** |
| Pooled mean return | −106.504 | −99.280 |
| Relative IQM advantage | **−8.04%** (GRU wins) | — |
| Relative mean advantage | −7.28% (GRU wins) | — |

Gate 6 needs `(iqm_pb - iqm_gru) / abs(iqm_gru) ≥ 0.10` **and** GRU must not
tie or win. Observed advantage is **−8.04%**. **FAIL.**
`architecture_superiority_claimed` is **false**.

Knockout sanity (Family B, mean over seeds 42–46): degradation **13.43**
(1343% ≫ 30%). Thoughts still sit on the action path on four of five seeds
(seed 45 was 0%: normal IQM −21.0 = knockout −21.0). Sanity **passed**; it does
not rescue Gate 6.

| Seed | PB IQM | GRU IQM | Family-B knockout normal → zero |
|---|---|---|---|
| 42 | −31.000 | −30.417 | −32.0 → −526.0 |
| 43 | −28.417 | −31.417 | −20.5 → −477.0 |
| 44 | −27.500 | −27.500 | −27.5 → −681.0 |
| 45 | −26.917 | −19.333 | −21.0 → −21.0 |
| 46 | −30.833 | −30.833 | −27.0 → −181.0 |

Five families (A–E), 300 steps, 5 episodes/family. Probe reason:
“Proposal-GRU ties or wins IQM; do not claim thought-mediated architecture
superiority.” This is a failed **resource-matched 300-step** comparison, not a
claim that GRU is a better architecture in general.

## Hashes (pre-result)

| Object | Identity |
|---|---|
| Git HEAD | `97ffe62579c90744595960fc7d342b21e3e490ef` |
| HEAD `dgx_run_thought_mediated_campaign.py` SHA-256 | `f9efa4a7cc1e4c0610a0811668953bfad228b5980667695e439d43b0df0afd3a` |
| Probe script SHA-256 | `631f85061f6abca0c5b427aabacdca573afc1caf64f4da3fffaf0e1370b37472` |
| HEAD brain snapshot tar SHA-256 | `0eafbcb0e030dc8ab066900f695966bd10a082163f327f29cb78a2ca8a4c7945` |
| `gate6_results.json` SHA-256 | `a3dc94dcbfebd6d914ed3c7caaf4725db9e5d7ef1bd2f47e7772311b73e9454c` |
