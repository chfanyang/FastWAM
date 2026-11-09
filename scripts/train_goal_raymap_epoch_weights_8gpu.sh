#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
run_id="${RUN_ID:-$(date -u +%Y-%m-%d_%H-%M-%S)}"
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 --main_process_port "${MASTER_PORT:-29785}" \
  scripts/train.py --config-name libero_goal_rothko_absolute_ray0_wan21_epoch_weights_8gpu \
  "output_dir=./runs/libero_goal_rothko_absolute_ray0_wan21_epoch_weights_8gpu/$run_id" "$@"
