#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PATH=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin:$PATH
export PYTHONPATH=$PWD/src:$PWD
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TMPDIR=$PWD/evaluate_results/libero/ray0tmp
mkdir -p "$TMPDIR"
# No --execute: prints the command only. GPU training must be explicitly requested.
exec torchrun --standalone --nproc_per_node=8 scripts/prepare_all4_absolute_ray0_decoder.py train "$@"
