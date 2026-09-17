#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/tmp
export LIBERO_CONFIG_PATH=/mnt/hwdata/cfy/FastWAM/data/libero_plus/.libero_config
RUN=/mnt/hwdata/cfy/FastWAM/runs/libero_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-4/2026-08-28_17-06-22
VAE=/mnt/hwdata/cfy/FastWAM/runs/libero_rothko_vae_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2/Wan2.1_VAE_libero_rothko_step007498.safetensors
OUT=/mnt/hwdata/cfy/FastWAM/evaluate_results/libero_plus/libero_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-4/ckpt021700_vaeDecoderStep7498_replan8_ensembleOff_robustJointAnchor0_stratified10_seed42
exec /mnt/hwdata/cfy/miniconda3/envs/fastwam_libero_plus/bin/python -u \
  scripts/eval_libero_plus_stratified10.py \
  task=libero_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-4 \
  "ckpt=$RUN/checkpoints/weights/step_021700.pt" \
  "model.vae_safetensors_path=$VAE" model.allow_vae_mismatch=true \
  model.rothko_decode_mode=robust_joint model.rothko_decode_anchor_alpha=0 \
  model.rothko_decode_block_grid=4 \
  "EVALUATION.output_dir=$OUT" "EVALUATION.dataset_stats_path=$RUN/dataset_stats.json" \
  EVALUATION.replan_steps=8 EVALUATION.use_action_ensembler=false \
  EVALUATION.save_rollout_videos=false EVALUATION.save_prediction_videos=false \
  'MULTIRUN.task_suite_names=[libero_spatial,libero_object,libero_goal,libero_10]' \
  'MULTIRUN.gpu_ids=[4,5,6,7]' MULTIRUN.num_gpus=4 MULTIRUN.workers_per_gpu=4 \
  MULTIRUN.assignment_mode=round_robin MULTIRUN.render_on_worker_gpu=true \
  MULTIRUN.resume=true "$@"
