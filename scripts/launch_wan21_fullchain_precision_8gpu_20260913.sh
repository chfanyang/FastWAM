#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=evaluate_results/libero_decoder_offline/wan21_fullchain_precision_a800_20260913
mkdir -p "$OUT/eight_gpu"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" /mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin/python -u scripts/audit_wan21_precision.py \
    --root "$PWD" --input /mnt/hwdata/cfy/tmp/wan21_precision_heldout800.pt \
    --output "$OUT/eight_gpu/shard$gpu" --full-chain --shard "$gpu" --num-shards 8 \
    --previous "$OUT/full800" > "$OUT/eight_gpu/shard$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [ "$failed" -ne 0 ]; then exit 1; fi
/mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin/python scripts/summarize_wan21_fullchain_8gpu.py > "$OUT/eight_gpu/merge.log" 2>&1
