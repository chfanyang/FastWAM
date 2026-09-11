#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PATH="/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
export PYTHONPATH="/mnt/hwdata/cfy/FastWAM/src${PYTHONPATH:+:$PYTHONPATH}"
export MASTER_PORT=29563
exec bash scripts/train_zero1.sh 8 \
  task=libero_goal_rothko_absolute_ray0_2cam224_full_wan22_5b_1e-4
