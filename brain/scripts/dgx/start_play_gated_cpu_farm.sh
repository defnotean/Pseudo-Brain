#!/usr/bin/env bash
# Launch named CPU-only play-gated campaign jobs. No --gpus. Do not chmod.
set -euo pipefail

workspace="${1:-$HOME/projects/pseudo-brain}"
release_id="${2:-r20260818t195814z-872818a4fa68}"
image="${3:-177a406d7cb2}"
release="$workspace/releases/$release_id"
farm="$workspace/runs/play-gated-cpu-farm-scripts"
uidgid="$(id -u):$(id -g)"
mkdir -p "$workspace/logs" "$farm"

[[ -d "$release" ]] || { echo "missing release $release" >&2; exit 1; }
[[ -f "$farm/play_gated_cpu_farm.py" ]] || { echo "missing farm script" >&2; exit 1; }

launch_cpu() {
  local name="$1"
  local cpus="$2"
  local mem="$3"
  local job="$4"
  local extra="$5"
  local run_dir="$workspace/runs/$name"
  local log_path="$workspace/logs/${name}.log"
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "tmux $name already running"
    return 0
  fi
  if docker inspect "$name" >/dev/null 2>&1; then
    echo "container $name already running"
    return 0
  fi
  mkdir -p "$run_dir"
  tmux new-session -d -s "$name" -- bash -lc "
    set -o pipefail
    docker run --rm --pull never \
      --name '$name' \
      --network none \
      --cpus '$cpus' \
      --memory '${mem}g' \
      --pids-limit 256 \
      --user '$uidgid' \
      -e HOME=/workspace/run \
      -e PATH=/usr/sbin:/usr/bin \
      -e CUDA_VISIBLE_DEVICES=-1 \
      -e OMP_NUM_THREADS='$cpus' \
      -e IRENE_BRAIN_SRC=/workspace/repo/brain/src \
      -v '$release:/workspace/repo:ro' \
      -v '$farm:/workspace/farm:ro' \
      -v '$run_dir:/workspace/run' \
      --entrypoint /bin/bash \
      '$image' --noprofile --norc -c \
      'python3 -I /workspace/farm/play_gated_cpu_farm.py --job $job --out-dir /workspace/run $extra' \
      2>&1 | tee '$log_path'
    echo EXIT:\${PIPESTATUS[0]} | tee -a '$log_path'
  "
  echo "started $name cpus=$cpus mem=${mem}g job=$job"
}

launch_cpu play-gated-cpu-planner-seeds-v3 8 16 planner-seeds '--seeds 132-147'
launch_cpu play-gated-cpu-tiled-hist-v1 4 8 tiled-hist ''
launch_cpu play-gated-cpu-multi-episode-coverage-v1 4 12 multi-episode-coverage ''
launch_cpu play-gated-cpu-offpolicy-teacher-v1 4 8 offpolicy-teacher ''
launch_cpu play-gated-cpu-window-majority-v1 4 12 window-majority ''
launch_cpu play-gated-cpu-turn-hold-v1 4 12 turn-hold ''

# Open-loop thoughtlets for the last GPU checkpoint (multi-episode mixed W/A/D).
# Uses the live release tree plus a read-only mount of that run's weights.
thoughtlets_name='play-gated-cpu-thoughtlets-multi-episode-v1'
thoughtlets_run="$workspace/runs/$thoughtlets_name"
thoughtlets_log="$workspace/logs/${thoughtlets_name}.log"
ckpt_run="$workspace/runs/dgx-play-maze-chase-distill-multi-episode-v1"
if tmux has-session -t "$thoughtlets_name" 2>/dev/null; then
  echo "tmux $thoughtlets_name already running"
elif docker inspect "$thoughtlets_name" >/dev/null 2>&1; then
  echo "container $thoughtlets_name already running"
elif [[ ! -f "$ckpt_run/checkpoints/step-00000032.pt" ]]; then
  echo "missing multi-episode checkpoint; skip $thoughtlets_name" >&2
else
  mkdir -p "$thoughtlets_run"
  tmux new-session -d -s "$thoughtlets_name" -- bash -lc "
    set -o pipefail
    docker run --rm --pull never \
      --name '$thoughtlets_name' \
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
      -v '$thoughtlets_run:/workspace/run' \
      -v '$ckpt_run:/workspace/ckpt:ro' \
      --entrypoint /bin/bash \
      '$image' --noprofile --norc -c \
      'python3 -I /workspace/farm/play_gated_cpu_farm.py --job thoughtlets --out-dir /workspace/run --config /workspace/repo/brain/configs/training/dgx-play-maze-chase-distill-multi-episode.toml --checkpoint /workspace/ckpt/checkpoints/step-00000032.pt --ticks 32 --seeds 5,9' \
      2>&1 | tee '$thoughtlets_log'
    echo EXIT:\${PIPESTATUS[0]} | tee -a '$thoughtlets_log'
  "
  echo "started $thoughtlets_name cpus=4 mem=12g job=thoughtlets"
fi

echo '===TMUX==='
tmux ls
echo '===DOCKER==='
docker ps
