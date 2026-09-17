#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mode="${1:-train}"
args=()
epochs=2
port=29841
if [ "$mode" = smoke ]; then
  out=runs/libero_wan21_cached_fresh_smoke_20260913
  epochs=1
  args+=(--cached-smoke --no-wandb --no-auto-resume)
elif [ "$mode" = smoke_resume ]; then
  out=runs/libero_wan21_cached_fresh_smoke_20260913
  args+=(--cached-smoke --no-wandb --resume "$out/checkpoint_latest.pt")
elif [ "$mode" = smoke_micro4 ]; then
  out=runs/libero_wan21_cached_micro4_smoke_20260913
  epochs=1
  args+=(--cached-smoke --micro-batch 4 --no-wandb --no-auto-resume)
elif [ "$mode" = train ]; then
  out=runs/libero_wan21_vae_decoder_cached_bs128_lr1e5_const_w100_ep2_20260913
  args+=(--wandb --wandb-mode online --wandb-name libero_wan21_vae_cached_bs128_lr1e5_const_w100_ep2_20260913)
elif [ "$mode" = fast_smoke ] || [ "$mode" = fast_train ]; then
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=4 NUMBA_NUM_THREADS=4
  args+=(--accelerated)
  if [ "$mode" = fast_smoke ]; then
    out=runs/libero_wan21_cached_fast_smoke_20260913
    epochs=1
    args+=(--cached-smoke --no-wandb --no-auto-resume)
  else
    out=runs/libero_wan21_vae_decoder_cached_bs128_lr1e5_const_w100_ep2_fast_20260913
    args+=(--no-auto-resume --wandb --wandb-mode online --wandb-name libero_wan21_vae_cached_bs128_const1e5_eval300_full500_20260913)
  fi
else
  echo "Unknown mode $mode" >&2
  exit 2
fi
if [ "$mode" != smoke_resume ] && [ -e "$out" ]; then
  echo "Refusing to overwrite existing experiment: $out" >&2
  exit 2
fi
exec /mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port="$port" \
  scripts/train_wan21_decoder_cached.py \
  --config configs/vae/libero_rothko_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2.json \
  --output-dir "$out" --epochs "$epochs" "${args[@]}"
