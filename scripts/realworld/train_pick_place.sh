#!/usr/bin/env bash
# Explicit invocation only. Never launch automatically after cache completion.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${REALWORLD_TMPDIR:-/tmp}"
REALWORLD_PYTHON="${REALWORLD_PYTHON:-/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python}"
REALWORLD_ARTIFACTS=data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_q9995_v1
test -f "$REALWORLD_ARTIFACTS/latents_wan21_bf16/_SUCCESS"
CUDA_VISIBLE_DEVICES='' "$REALWORLD_PYTHON" scripts/realworld/prepare_pick_place.py
CUDA_VISIBLE_DEVICES='' "$REALWORLD_PYTHON" scripts/realworld/preflight.py \
  --artifacts "$REALWORLD_ARTIFACTS" --require-text --require-latents
export PATH="$(dirname "$REALWORLD_PYTHON"):$PATH"
export MASTER_PORT="${MASTER_PORT:-29625}"
bash scripts/train_zero1.sh "${NPROC_PER_NODE:-8}" \
  task=realworld_pick_place_ray0_absolute_wan21_q9995_ep8 "$@"
