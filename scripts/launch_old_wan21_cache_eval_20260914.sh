#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin/python -u -m torch.distributed.run \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29847 \
  scripts/eval_old_wan21_on_training_cache.py \
  --output evaluate_results/libero_decoder_offline/wan21_old_checkpoints_same_training_cache_20260914
