#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PATH=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin:$PATH
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TMPDIR=/tmp TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
exec bash scripts/train_zero1.sh 8 task=vlabench_select_book_rothko_full_wan21_1_3b
