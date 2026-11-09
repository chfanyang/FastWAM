#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${FASTWAM_TMPDIR:-$PWD/video_a800/mftmp}"
mkdir -p "$TMPDIR"
export HTTP_PROXY="${FASTWAM_HTTP_PROXY:-http://127.0.0.1:7890}" HTTPS_PROXY="${FASTWAM_HTTP_PROXY:-http://127.0.0.1:7890}"
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY" ALL_PROXY="$HTTP_PROXY" all_proxy="$HTTP_PROXY"
run_id="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${NPROC:-8}" --main_process_port "${MASTER_PORT:-29795}" \
  scripts/train.py task=libero_goal_axis_angle_constant_wan21 \
  "output_dir=./runs/libero_goal_axis_angle_constant_wan21/$run_id" "$@"
