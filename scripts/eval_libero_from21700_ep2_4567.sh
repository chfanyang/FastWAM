#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PATH=/mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin:$PATH
export CUDA_VISIBLE_DEVICES=${EVAL_GPUS:-4,5,6,7}
IFS=',' read -r -a GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/tmp
OUT=/mnt/hwdata/cfy/FastWAM/evaluate_results/libero/libero_all4_rothko_2cam224_full_from21700_ep2_1e-4/ckpt004340_vaeDecoderRerunFixedStep7498_replan8_ensembleOff_robustJointAnchor0
for SUITES in '[libero_spatial,libero_object,libero_goal,libero_10]'; do
  python experiments/libero/run_libero_manager.py \
    task=libero_all4_rothko_2cam224_full_from21700_ep2_1e-4 \
    ckpt=/mnt/hwdata/cfy/FastWAM/runs/libero_all4_rothko_2cam224_full_from21700_ep2_1e-4/2026-09-11_17-33-16/checkpoints/weights/step_004340.pt \
    model.vae_safetensors_path=/mnt/hwdata/cfy/FastWAM/runs/libero_rothko_vae_decoder_all4_h16_bs2_ga8_lr1e-5_ep2_rerun_fixed/Wan2.2_VAE_libero_rothko_step007498.safetensors \
    model.allow_vae_mismatch=true model.rothko_decode_mode=robust_joint \
    model.rothko_decode_anchor_alpha=0 model.rothko_decode_block_grid=4 \
    EVALUATION.output_dir="$OUT" \
    EVALUATION.dataset_stats_path=/mnt/hwdata/cfy/FastWAM/runs/libero_all4_rothko_2cam224_full_from21700_ep2_1e-4/2026-09-11_17-33-16/dataset_stats.json \
    EVALUATION.num_trials=50 EVALUATION.replan_steps=8 EVALUATION.use_action_ensembler=false \
    "MULTIRUN.task_suite_names=$SUITES" "MULTIRUN.num_gpus=${#GPU_LIST[@]}" MULTIRUN.max_tasks_per_gpu=3 \
    MULTIRUN.worker_timeout_seconds=604800
done
