#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PYTHONPATH=src:. TMPDIR=/tmp TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python
config=configs/vae/vlabench_select_book_3cam192_wan21_bs2_ga4_lr1e-5_ep1.json
mkdir -p logs/vlabench_select_book_vae
"$python" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_vlabench_vae_cached.py --config "$config" --smoke \
  > logs/vlabench_select_book_vae/smoke.log 2>&1
exec "$python" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/train_vlabench_vae_cached.py --config "$config" \
  > logs/vlabench_select_book_vae/train.log 2>&1
