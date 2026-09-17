#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PYTHONPATH=src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python
"$python" scripts/prepare_vlabench_three_camera_stats.py
# Each task is a complete independent cache. Pass other task names later to append.
if [ "$#" -eq 0 ]; then set -- select_book; fi
for task in "$@"; do
  "$python" scripts/precompute_vlabench_task_latents.py \
    --task-name "$task" \
    --output "data/vlabench_train99_rothko_3cam192_centerfrac05_wan21_bf16_h16_latents/$task" \
    --gpus 0,1,2,3,4,5,6,7 --processes-per-gpu 4 \
    --batch-size 2 --num-workers 1 --shard-size 256
done
