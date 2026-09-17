#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset WANDB_RUN_ID WANDB_RESUME
out=runs/libero_wan21_vae_cached_bs128_from3000_const3e6_ep3_20260914
if [ -e "$out" ]; then
  echo "Refusing to overwrite existing experiment: $out" >&2
  exit 2
fi
exec /mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin/python -m torch.distributed.run \
  --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29853 \
  scripts/train_wan21_decoder_cached.py \
  --config configs/vae/libero_rothko_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2.json \
  --output-dir "$out" --epochs 3 --accelerated --no-auto-resume \
  --resume runs/libero_wan21_vae_decoder_cached_bs128_lr1e5_const_w100_ep2_fast_20260913/checkpoint_step003000_full.pt \
  --resume-constant-lr 3e-6 \
  --wandb --wandb-mode online --wandb-name libero_wan21_cached_from3000_const3e6_ep3_eval300_full500_20260914
