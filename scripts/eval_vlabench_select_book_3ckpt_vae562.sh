#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PYTHONPATH=src:third_party/VLABench
export TMPDIR=/tmp MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
step=${1:?checkpoint step required}
case "$step" in 000843|001686|002529) ;; *) exit 2 ;; esac
exec /mnt/hwdata/cfy/miniconda3/envs/fastwam_vlabench/bin/python -u \
  experiments/vlabench/run_select_book_manager.py \
  --checkpoint "runs/vlabench_select_book_rothko_3cam192_wan21/2026-09-15_17-33-31/checkpoints/weights/step_${step}.pt" \
  --output-dir "evaluate_results/vlabench/vlabench_select_book_rothko_3cam192_wan21/ckpt${step}_vaeDecoderStep562_replan8_legacy_ep50" \
  --vae-safetensors-path runs/vlabench_select_book_vae_wan21_3cam192_bs2_ga4_lr1e-5_ep1/Wan2.1_VAE_vlabench_select_book_step000562.safetensors \
  --allow-vae-mismatch --gpu-ids 0 1 2 3 4 5 6 7 --workers-per-gpu 1 \
  --episodes 50 --replan-steps 8 --gripper-threshold 0.5 --threads 2
