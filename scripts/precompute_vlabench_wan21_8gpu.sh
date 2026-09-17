#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false TMPDIR=/tmp
export PYTHONPATH=src
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
python=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/torchrun
args=(--task vlabench_rothko_2cam224_full_wan21_1_3b_1e-4
      --batch-size 8 --num-workers 4 --log-every 20)
# First prove online/cache latent and training-loss equivalence with 32 windows.
probe="runs/vlabench_latent_probe/$(date -u +%Y-%m-%d_%H-%M-%S)"
"$python" --standalone --nproc_per_node=8 scripts/precompute_visual_action_latents.py \
  "${args[@]}" --output-dir "$probe" --benchmark-max-samples 32 --samples-per-shard 4
"$python" --standalone --nproc_per_node=8 scripts/precompute_visual_action_latents.py \
  "${args[@]}" --samples-per-shard 1024 \
  --output-dir data/vlabench_train99_rothko_centerfrac05_wan21_bf16_h16_latents
