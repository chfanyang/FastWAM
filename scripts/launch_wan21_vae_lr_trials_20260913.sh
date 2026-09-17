#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
PY=/mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin/python
CONFIG=configs/vae/libero_rothko_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2.json
SOURCE=runs/libero_rothko_vae_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2/checkpoint_latest.pt
BASE=runs/libero_wan21_vae7498_joint_lr_trials_20260913
mkdir -p "$BASE"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODE=${1:-smoke}
case "$MODE" in
  smoke) GPUS=0,1,2,3; LR=3e-6; STEPS=2; NAME=smoke2; PORT=29831; EXTRA=(--smoke);;
  lr1e6) GPUS=0,1,2,3; LR=1e-6; STEPS=1000; NAME=constant1e-6_warmup50_steps1000; PORT=29832; EXTRA=();;
  lr3e6) GPUS=4,5,6,7; LR=3e-6; STEPS=1000; NAME=constant3e-6_warmup50_steps1000; PORT=29833; EXTRA=();;
  *) exit 2;;
esac
test ! -e "$BASE/$NAME"
export CUDA_VISIBLE_DEVICES=$GPUS
exec "$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port="$PORT" \
  scripts/continue_wan21_vae_joint_trial.py --config "$CONFIG" --resume "$SOURCE" \
  --output-dir "$BASE/$NAME" --lr "$LR" --lr-scheduler constant --trial-steps "$STEPS" \
  --batch-size 2 --grad-accum-steps 8 --no-wandb --no-progress-bar "${EXTRA[@]}" \
  > "$BASE/$NAME.launch.log" 2>&1
