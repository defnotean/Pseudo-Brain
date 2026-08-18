#!/usr/bin/env bash
# Launch named CPU-only play-gated campaign jobs. No --gpus. Do not chmod.
set -euo pipefail

workspace="${1:-$HOME/projects/pseudo-brain}"
release_id="${2:-r20260818t175216z-c804bcd101c1}"
image="${3:-177a406d7cb2}"
release="$workspace/releases/$release_id"
farm="$workspace/runs/play-gated-cpu-farm-scripts"
ckpt_run="$workspace/runs/dgx-play-maze-chase-distill-exclusive-ce-v1"
uidgid="$(id -u):$(id -g)"
mkdir -p "$workspace/logs" "$farm"

[[ -d "$release" ]] || { echo "missing release $release" >&2; exit 1; }
[[ -f "$farm/play_gated_cpu_farm.py" ]] || { echo "missing farm script" >&2; exit 1; }
[[ -f "$ckpt_run/checkpoints/step-00000032.pt" ]] || { echo "missing exclusive-ce checkpoint" >&2; exit 1; }

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
      -v '$ckpt_run:/workspace/ckpt-run:ro' \
      -v '$run_dir:/workspace/run' \
      --entrypoint /bin/bash \
      '$image' --noprofile --norc -c \
      'python3 -I /workspace/farm/play_gated_cpu_farm.py --job $job --out-dir /workspace/run $extra' \
      2>&1 | tee '$log_path'
    echo EXIT:\${PIPESTATUS[0]} | tee -a '$log_path'
  "
  echo "started $name cpus=$cpus mem=${mem}g job=$job"
}

launch_cpu play-gated-cpu-planner-seeds-v1 8 16 planner-seeds '--seeds 100-115'
launch_cpu play-gated-cpu-coverage-v1 4 8 coverage ''
launch_cpu play-gated-cpu-thoughtlets-long-v1 4 12 thoughtlets '--config /workspace/repo/brain/configs/training/dgx-play-maze-chase-distill-exclusive-ce.toml --checkpoint /workspace/ckpt-run/checkpoints/step-00000032.pt --ticks 32 --seeds 5,9'

echo '===TMUX==='
tmux ls
echo '===DOCKER==='
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'
