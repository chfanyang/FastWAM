#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/cfy/FastWAM
case "${1:-}" in
  manipulation) PY=/mnt/data/cfy/anaconda3/envs/fastwam_libero_plus/bin/python ;;
  nav) PY=/mnt/data/cfy/miniconda3/envs/fastwam_libero_plus/bin/python ;;
  *) exit 2 ;;
esac
export PAIRED_HOST="$1"
cd "$ROOT"
OUT=${PAIRED_OUTPUT_DIR:-$ROOT/evaluate_results/libero_plus/libero_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-4/ckpt021700_vaeDecoderStep7498_replan8_ensembleOff_robustJointAnchor0_a800Matched2757_${PAIRED_HOST}_20260913}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 OMP_WAIT_POLICY=PASSIVE
export DIFFSYNTH_MODEL_BASE_PATH=$ROOT/checkpoints LIBERO_CONFIG_PATH=$ROOT/data/libero_plus/.libero_config
args=(
  task=libero_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-4
  ckpt=$ROOT/runs/libero_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-4/2026-08-28_17-06-22/checkpoints/weights/step_021700.pt
  model.vae_safetensors_path=$ROOT/runs/libero_rothko_vae_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2/Wan2.1_VAE_libero_rothko_step007498.safetensors
  model.allow_vae_mismatch=true model.rothko_decode_mode=robust_joint model.rothko_decode_anchor_alpha=0.0 model.rothko_decode_block_grid=4
  EVALUATION.dataset_stats_path=$ROOT/runs/libero_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-4/2026-08-28_17-06-22/dataset_stats.json
  EVALUATION.output_dir=$OUT seed=42 eval_random_seed=42 eval_num_inference_steps=20 EVALUATION.num_inference_steps=20
  EVALUATION.num_trials=1 EVALUATION.replan_steps=8 EVALUATION.use_action_ensembler=false EVALUATION.tiled=false
  EVALUATION.save_rollout_videos=false EVALUATION.save_prediction_videos=false EVALUATION.save_control_trace=false
  'MULTIRUN.task_suite_names=[libero_spatial,libero_object,libero_goal,libero_10]'
  'MULTIRUN.gpu_ids=[0,1,2,3,4,5,6,7]' MULTIRUN.num_gpus=8 MULTIRUN.workers_per_gpu=1 MULTIRUN.render_on_worker_gpu=true
  MULTIRUN.assignment_mode=round_robin MULTIRUN.resume=true MULTIRUN.task_start=0 MULTIRUN.task_end=null MULTIRUN.max_tasks_per_suite=null
)
if [ "${2:-}" = --check ]; then
  exec "$PY" -u experiments/libero_plus/run_a800_paired_3090_20260913.py "${args[@]}" MULTIRUN.create_only=true
fi
mkdir -p "$OUT"
exec "$PY" -u experiments/libero_plus/run_a800_paired_3090_20260913.py "${args[@]}" > "$OUT/manager.log" 2>&1
