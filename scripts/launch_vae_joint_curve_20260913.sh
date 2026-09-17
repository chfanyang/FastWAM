#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
OUT=evaluate_results/libero_decoder_offline/wan21_checkpoint_curve_joint_anchor0_heldout800_20260913
test ! -e "$OUT"
mkdir -p "$OUT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PIDS=()
for GPU in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$GPU /mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin/python -u scripts/compare_wan21_wan22_vae_joint_heldout.py --shard "$GPU" --output "$OUT" --wan21-steps 4000 4800 5600 6400 7200 > "$OUT/shard$GPU.log" 2>&1 &
  PIDS+=("$!")
done
RC=0
for PID in "${PIDS[@]}"; do wait "$PID" || RC=1; done
echo "All shards exited; status=$RC"
exit "$RC"
