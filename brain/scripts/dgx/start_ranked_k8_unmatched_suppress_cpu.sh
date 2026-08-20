#!/usr/bin/env bash
# Spark CPU docker for ranked-k8-unmatched-suppress-v1. Do not pass --gpus.
# Do not stop Irene sglang. Do not chmod.
set -euo pipefail

workspace="${1:-$HOME/projects/pseudo-brain}"
image="${2:-177a406d7cb2}"
mode="${3:-smoke}"
name="ranked-k8-unmatched-suppress-v1-${mode}"
repo="$workspace"
run_dir="$workspace/runs/$name"
log_path="$workspace/logs/${name}.log"
uidgid="$(id -u):$(id -g)"
flag="--smoke"
if [[ "$mode" == "full" ]]; then
  flag="--full"
fi

if tmux has-session -t "$name" 2>/dev/null; then
  echo "tmux $name already running" >&2
  exit 0
fi
if docker inspect "$name" >/dev/null 2>&1; then
  echo "container $name already running" >&2
  exit 1
fi

mkdir -p "$run_dir" "$workspace/logs"
echo "LAUNCH $name $(date -u +%Y-%m-%dT%H:%M:%SZ) flag=$flag" | tee "$log_path"

tmux new-session -d -s "$name" -- bash -lc "
  set -o pipefail
  docker run --rm --pull never \
    --name '$name' \
    --network none \
    --cpus 12 \
    --memory 32g \
    --pids-limit 512 \
    --user '$uidgid' \
    -e HOME=/workspace/run \
    -e PATH=/usr/sbin:/usr/bin \
    -e CUDA_VISIBLE_DEVICES=-1 \
    -e OMP_NUM_THREADS=1 \
    -e MKL_NUM_THREADS=1 \
    -e IRENE_BRAIN_SRC=/workspace/repo/brain/src \
    -v '$repo:/workspace/repo:ro' \
    -v '$run_dir:/workspace/run' \
    --entrypoint /bin/bash \
    '$image' --noprofile --norc -c \
    'python3 -I /workspace/repo/brain/scripts/dgx_run_ranked_k8_unmatched_suppress_probe.py $flag --device cpu --output-json /workspace/run/ranked_k8_results.json' \
    2>&1 | tee -a '$log_path'
  echo EXIT:\${PIPESTATUS[0]} | tee -a '$log_path'
"
echo "STARTED tmux $name"
