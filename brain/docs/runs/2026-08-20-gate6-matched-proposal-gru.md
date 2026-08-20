# Run record: Phase 2.5 Gate 6 matched Proposal-GRU probe

**Date**: 2026-08-20  
**Probe ID**: `dgx-gate6-matched-gru-v1`  
**Gate**: Phase 2.5 Gate 6 (≥10% IQM/return over matched Proposal-GRU)  
**Status**: **INCOMPLETE** (measurement started; GB10 was occupied)

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

- Thought-mediated K=32, width 32, 3 cycles (committed width map).
- Proposal-GRU hidden 180.
- 300 steps, seeds 42–46, five families, 5 episodes/family.

## Launch

Queued **behind** the GB10 occupant. Named Spark CPU docker job
`dgx-gate6-matched-gru-v1` **is running** (tmux + docker, `CUDA_VISIBLE_DEVICES=-1`):

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

Not yet written. This record will be updated with IQM/return, knockout sanity,
hashes, and PASS/FAIL when `gate6_results.json` lands. Architecture superiority
will not be claimed if GRU ties or wins.

## Hashes (pre-result)

| Object | Identity |
|---|---|
| Git HEAD | `97ffe62579c90744595960fc7d342b21e3e490ef` |
| HEAD `dgx_run_thought_mediated_campaign.py` SHA-256 | `f9efa4a7cc1e4c0610a0811668953bfad228b5980667695e439d43b0df0afd3a` |
| Probe script SHA-256 | `631f85061f6abca0c5b427aabacdca573afc1caf64f4da3fffaf0e1370b37472` |
| HEAD brain snapshot tar SHA-256 | `0eafbcb0e030dc8ab066900f695966bd10a082163f327f29cb78a2ca8a4c7945` |
