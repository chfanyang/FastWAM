#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PATH=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29573
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR=/tmp
bash scripts/train_zero1.sh 4 task=libero_all4_rothko_2cam224_full_state21700_ep2_1e-5 "$@"
