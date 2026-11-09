#!/usr/bin/env bash
# Run only after an explicit launch instruction; creation does not start training.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export TMPDIR="$PWD/video_a800/mftmp"
mkdir -p "$TMPDIR"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export HTTP_PROXY="${FASTWAM_HTTP_PROXY:-http://127.0.0.1:7890}"
export HTTPS_PROXY="$HTTP_PROXY" http_proxy="$HTTP_PROXY" https_proxy="$HTTP_PROXY" ALL_PROXY="$HTTP_PROXY" all_proxy="$HTTP_PROXY"
export WANDB_RUN_ID=7ypn8sop WANDB_RESUME=must
export PYTHONUNBUFFERED=1
exec accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 --main_process_port "${MASTER_PORT:-29785}" \
  scripts/train.py --config-name vlabench_all10_resume18000_state1000 "$@"
