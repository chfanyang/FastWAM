#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# Process-local IPC sockets must live on local storage, not an inherited NFS TMPDIR.
export TMPDIR="${REALWORLD_TMPDIR:-/tmp}"
REALWORLD_PYTHON="${REALWORLD_PYTHON:-/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python}"
REALWORLD_TASK=realworld_all4_ray0_absolute_wan21_ep8
REALWORLD_ARTIFACTS=./data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_v2
REALWORLD_MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$REALWORLD_MODE" in
  prepare)
    CUDA_VISIBLE_DEVICES='' "$REALWORLD_PYTHON" scripts/realworld/prepare_data.py "$@"
    ;;
  verify)
    CUDA_VISIBLE_DEVICES='' "$REALWORLD_PYTHON" -m unittest discover -s tests/realworld -v "$@"
    ;;
  text)
    CUDA_VISIBLE_DEVICES='' "$REALWORLD_PYTHON" scripts/realworld/preflight.py
    "$REALWORLD_PYTHON" scripts/realworld/prepare_text.py "$@"
    ;;
  cache)
    CUDA_VISIBLE_DEVICES='' "$REALWORLD_PYTHON" scripts/realworld/preflight.py --require-text
    "$REALWORLD_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
      scripts/precompute_visual_action_latents.py --task "$REALWORLD_TASK" \
      --output-dir "$REALWORLD_ARTIFACTS/latents_wan21_bf16" \
      --batch-size 2 --num-workers 4 "$@"
    ;;
  train)
    # Explicit command only: this workflow never starts training from prepare/verify.
    test -f "$REALWORLD_ARTIFACTS/text/_SUCCESS"
    test -f "$REALWORLD_ARTIFACTS/latents_wan21_bf16/_SUCCESS"
    CUDA_VISIBLE_DEVICES='' "$REALWORLD_PYTHON" scripts/realworld/preflight.py --require-text --require-latents
    export PATH="$(dirname "$REALWORLD_PYTHON"):$PATH"
    export MASTER_PORT="${MASTER_PORT:-29623}"
    bash scripts/train_zero1.sh "${NPROC_PER_NODE:-1}" task="$REALWORLD_TASK" \
      data.train.latent_cache_dir="$REALWORLD_ARTIFACTS/latents_wan21_bf16" \
      data.train.latent_cache_only=true "$@"
    ;;
  *)
    echo 'Usage: bash scripts/realworld/workflow.sh {prepare|verify|text|cache|train}'
    echo 'prepare/verify are CPU-only. text/cache/train use GPUs ONLY when explicitly invoked.'
    ;;
esac
