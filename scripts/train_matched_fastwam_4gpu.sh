#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
run_id="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 4 --main_process_port "${MASTER_PORT:-29775}" \
  scripts/train_matched_fastwam.py task=libero_goal_matched_fastwam_wan21 \
  "output_dir=./runs/libero_goal_matched_fastwam_wan21/$run_id" "$@"
