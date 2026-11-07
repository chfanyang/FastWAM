#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
mode=${1:?decoupled or future_bidirectional}
case "$mode" in decoupled|future_bidirectional) ;; *) exit 2;; esac
export PATH=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export DIFFSYNTH_MODEL_BASE_PATH=$PWD/checkpoints
export MASTER_PORT=29624
exec bash scripts/train_zero1.sh 8 "task=libero_goal_absolute_ray0_relative_mask_${mode}_wan21"
