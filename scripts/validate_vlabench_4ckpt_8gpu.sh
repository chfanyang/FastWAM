#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PYTHONPATH=src:. TMPDIR=/tmp TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
out=evaluate_results/vlabench/vlabench_select_book_rothko_3cam192_wan21/heldout382_4ckpt_vae562
mkdir -p "$out"
gpu=0
pids=()
for step in 843 1405 1686 2529; do
  for rank in 0 1; do
    CUDA_VISIBLE_DEVICES=$gpu /mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python -u scripts/validate_vlabench_checkpoints.py \
      --run runs/vlabench_select_book_rothko_3cam192_wan21/2026-09-15_17-33-31 \
      --vae runs/vlabench_select_book_vae_wan21_3cam192_bs2_ga4_lr1e-5_ep1/Wan2.1_VAE_vlabench_select_book_step000562.safetensors \
      --output "$out" --step "$step" --rank "$rank" > "$out/step${step}_rank${rank}.log" 2>&1 &
    pids+=("$!")
    gpu=$((gpu+1))
  done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
