#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$PWD/video_a800/mftmp"
mkdir -p "$TMPDIR"
export HTTP_PROXY="${FASTWAM_HTTP_PROXY:-http://127.0.0.1:7890}"
export HTTPS_PROXY="$HTTP_PROXY" http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY" ALL_PROXY="$HTTP_PROXY" all_proxy="$HTTP_PROXY"
unset WANDB_RUN_ID WANDB_RESUME
run_id="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
exec accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 --main_process_port "${MASTER_PORT:-29777}" \
  scripts/train_matched_fastwam.py task=libero_goal_matched_fastwam_ratio20_wan21_8gpu \
  "output_dir=./runs/libero_goal_matched_fastwam_ratio20_wan21_8gpu/$run_id" "$@"
