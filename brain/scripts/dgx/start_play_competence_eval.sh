#!/usr/bin/env bash
# CPU-only maze-chase play competence vs planner. Do not pass --gpus. Do not chmod.
set -euo pipefail

workspace="${1:-$HOME/projects/pseudo-brain}"
release_id="${2:-r20260818t195814z-872818a4fa68}"
image="${3:-177a406d7cb2}"
name='play-competence-cpu-20260820-v1'
release="$workspace/releases/$release_id"
farm="$workspace/runs/play-gated-cpu-farm-scripts"
ckpt="$workspace/runs/dgx-play-maze-chase-distill-turn-weighted-v1"
run_dir="$workspace/runs/$name"
log_path="$workspace/logs/${name}.log"
uidgid="$(id -u):$(id -g)"

[[ -d "$release" ]] || { echo "missing release $release" >&2; exit 1; }
[[ -f "$farm/eval_play_competence.py" ]] || { echo "missing eval script" >&2; exit 1; }
[[ -f "$farm/render_maze_chase_neural_replay.py" ]] || { echo "missing renderer" >&2; exit 1; }
[[ -f "$ckpt/checkpoints/step-00000032.pt" ]] || { echo "missing checkpoint" >&2; exit 1; }

if tmux has-session -t "$name" 2>/dev/null; then
  echo "tmux $name already running" >&2
  exit 0
fi
if docker inspect "$name" >/dev/null 2>&1; then
  echo "container $name already running" >&2
  exit 1
fi

mkdir -p "$run_dir" "$workspace/logs"
echo "LAUNCH $name $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$log_path"

tmux new-session -d -s "$name" -- bash -lc "
  set -o pipefail
  docker run --rm --pull never \
    --name '$name' \
    --network none \
    --cpus 4 \
    --memory 12g \
    --pids-limit 256 \
    --user '$uidgid' \
    -e HOME=/workspace/run \
    -e PATH=/usr/sbin:/usr/bin \
    -e CUDA_VISIBLE_DEVICES=-1 \
    -e OMP_NUM_THREADS=4 \
    -e IRENE_BRAIN_SRC=/workspace/repo/brain/src \
    -v '$release:/workspace/repo:ro' \
    -v '$farm:/workspace/farm:ro' \
    -v '$run_dir:/workspace/run' \
    -v '$ckpt:/workspace/ckpt:ro' \
    --entrypoint /bin/bash \
    '$image' --noprofile --norc -c \
    'python3 -I /workspace/farm/eval_play_competence.py --config /workspace/repo/brain/configs/training/dgx-play-maze-chase-distill-turn-weighted.toml --checkpoint /workspace/ckpt/checkpoints/step-00000032.pt --out-dir /workspace/run --agent-id turn-weighted-v1-champion && python3 -I /workspace/farm/render_maze_chase_neural_replay.py --config /workspace/repo/brain/configs/training/dgx-play-maze-chase-distill-turn-weighted.toml --checkpoint /workspace/ckpt/checkpoints/step-00000032.pt --out-dir /workspace/run --seeds 5,9 --stem play-competence-20260820-turn-weighted-champion' \
    2>&1 | tee -a '$log_path'
  echo EXIT:\${PIPESTATUS[0]} | tee -a '$log_path'
"
echo "STARTED tmux $name"
