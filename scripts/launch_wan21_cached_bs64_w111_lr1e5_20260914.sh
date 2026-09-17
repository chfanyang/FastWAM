#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WANDB_RUN_ID WANDB_RESUME
out=runs/libero_wan21_vae_cached_bs64_w111_lr1e5_cosine4_hold1_gradlog_20260914
if [ -e "$out" ]; then
  echo "Refusing to overwrite existing experiment: $out" >&2
  exit 2
fi
exec /mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin/python -u -m torch.distributed.run \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29863 \
  scripts/train_wan21_decoder_cached.py \
  --config configs/vae/libero_rothko_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2.json \
  --output-dir "$out" --epochs 5 --accelerated --no-auto-resume \
  --batch64-cosine4-hold --cosine-peak-lr 1e-5 --micro-batch 2 \
  --center-loss-weight 1 --direction-loss-weight 1 --gripper-loss-weight 1 \
  --wandb --wandb-mode online --wandb-name libero_wan21_cached_bs64_w111_lr1e5_cosine4_hold1_gradlog_20260914
