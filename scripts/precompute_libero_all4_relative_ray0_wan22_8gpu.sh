#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
export TMPDIR=/tmp
export PYTHONPATH="/mnt/hwdata/cfy/FastWAM/src${PYTHONPATH:+:$PYTHONPATH}"
# Same data/geometry/VAE as the current weights-only continuation. The
# precompute program encodes data only; init_weights/optimizer are not used.
exec /mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/torchrun \
  --standalone --nproc_per_node=8 scripts/precompute_visual_action_latents.py \
  --task libero_all4_rothko_2cam224_full_from21700_ep2_1e-5 \
  --output-dir /mnt/hwdata/cfy/FastWAM/data/libero_all4_rothko_centerfrac05_wan22_bf16_h16_latents \
  --batch-size 8 --num-workers 4 --samples-per-shard 1024 --log-every 20
