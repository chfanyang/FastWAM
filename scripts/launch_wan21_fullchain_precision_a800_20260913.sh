#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=evaluate_results/libero_decoder_offline/wan21_fullchain_precision_a800_20260913
mkdir -p "$OUT"
/mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin/python -u scripts/audit_wan21_precision.py \
  --root "$PWD" --input /mnt/hwdata/cfy/tmp/wan21_precision_heldout800.pt \
  --output "$OUT/full800" --full-chain > "$OUT/full800.log" 2>&1
