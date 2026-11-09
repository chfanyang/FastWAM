#!/usr/bin/env bash
# Evaluation template only; explicitly forward defaults to the shared workers.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CKPT:?Set CKPT to a completed axis-angle checkpoint}"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
python_bin="${PYTHON_BIN:-/mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin/python}"
output_dir="${OUTPUT_DIR:-$PWD/evaluate_results/libero/closed_loop/goal_axis_angle_constant_wan21/$(date -u +%Y%m%d_%H%M%S)}"
if [[ -e "$output_dir" ]]; then
  printf 'Use a new output directory: %s\n' "$output_dir" >&2
  exit 2
fi
"$python_bin" experiments/libero/run_libero_manager.py --config-name sim_libero_goal_axis_angle \
  "ckpt=$CKPT" "EVALUATION.output_dir=$output_dir" \
  'MULTIRUN.task_suite_names=[libero_goal]' MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=2 \
  MULTIRUN.worker_timeout_seconds=43200 EVALUATION.num_trials=50 EVALUATION.replan_steps=8 \
  EVALUATION.action_horizon=16 EVALUATION.num_inference_steps=20 \
  EVALUATION.dataset_stats_path=./data/libero_mujoco3.3.2/dataset_stats.json \
  EVALUATION.use_action_ensembler=false EVALUATION.save_prediction_videos=false \
  EVALUATION.visualize_future_video=false EVALUATION.binarize_gripper=true seed=42 "$@"
