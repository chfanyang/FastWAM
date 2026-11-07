#!/usr/bin/env bash
# Explicit launch only, after check_task_training_ready.py succeeds.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
task="${1:?task required}"
stage="${2:?stage must be dit or vae}"
case "$task" in insert_pen|stack_bowls_three) epochs=8;; white_board) epochs=5;; *) exit 2;; esac
export PATH=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin:$PATH
export PYTHONPATH="$PWD/src:$PWD"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TMPDIR=/mnt/hwdata/cfy/FastWAM/video_a800/rwt
mkdir -p "$TMPDIR"
python scripts/realworld/check_task_training_ready.py --task "$task"
case "$stage" in
 dit) MASTER_PORT="${MASTER_PORT:-29655}" bash scripts/train_zero1.sh 8 "task=realworld_${task}_ray0_absolute_wan21_q9995_ep${epochs}";;
 vae) torchrun --nnodes=1 --nproc_per_node=8 --master_port="${MASTER_PORT:-29665}" scripts/realworld/finetune_task_vae.py --config "configs/vae/realworld_${task}_q9995_decoder_ep1.json" --execute;;
 *) exit 2;;
esac
