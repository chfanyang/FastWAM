#!/usr/bin/env bash
# Explicit foreground invocation only; no polling or automatic launch.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT"
export PLUS_ROLE=${1:?Usage: run.sh ROLE check|run}
MODE=${2:?Specify check or run}
case "$PLUS_ROLE" in
  3090_object) PYTHON_DEFAULT=/mnt/data/cfy/anaconda3/envs/fastwam_libero_plus/bin/python; SUITES='[libero_object,libero_spatial]'; WORKERS=1 ;;
  3090_goal) PYTHON_DEFAULT=/mnt/data/cfy/miniconda3/envs/fastwam_libero_plus/bin/python; SUITES='[libero_goal,libero_spatial]'; WORKERS=1 ;;
  a800_libero10) PYTHON_DEFAULT=/mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin/python; SUITES='[libero_10]'; WORKERS=4 ;;
  *) echo 'Unknown role' >&2; exit 2 ;;
esac
case "$MODE" in check) CREATE_ONLY=true ;; run) CREATE_ONLY=false ;; *) exit 2 ;; esac
export PYTHONPATH="$ROOT/src:$ROOT:$ROOT/third_party/LIBERO-plus${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$ROOT/data/libero_plus/.libero_config"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec "${PYTHON:-$PYTHON_DEFAULT}" experiments/libero_plus/run_absolute_ray0_three_host.py \
  task=libero_all4_rothko_absolute_ray0_2cam224_full_wan21_1_3b_1e-4 \
  "ckpt=$ROOT/artifacts/dit_step021700.pt" \
  "model.vae_safetensors_path=$ROOT/artifacts/vae_step008264.safetensors" \
  model.allow_vae_mismatch=true model.rothko_decode_mode=legacy model.rothko_decode_anchor_alpha=0 \
  "EVALUATION.dataset_stats_path=$ROOT/artifacts/dataset_stats.json" \
  "EVALUATION.output_dir=$ROOT/evaluate_results/plus_dit21700_vae8264/$PLUS_ROLE" \
  seed=42 EVALUATION.num_trials=1 EVALUATION.replan_steps=8 EVALUATION.num_inference_steps=20 \
  EVALUATION.use_action_ensembler=false EVALUATION.save_rollout_videos=false EVALUATION.save_prediction_videos=false \
  "MULTIRUN.task_suite_names=$SUITES" MULTIRUN.num_gpus=8 'MULTIRUN.gpu_ids=[0,1,2,3,4,5,6,7]' \
  "MULTIRUN.workers_per_gpu=$WORKERS" MULTIRUN.render_on_worker_gpu=true \
  "MULTIRUN.create_only=$CREATE_ONLY"
