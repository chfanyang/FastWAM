#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${REALWORLD_TMPDIR:-/tmp}"
REALWORLD_PYTHON="${REALWORLD_PYTHON:-/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python}"
REALWORLD_MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$REALWORLD_MODE" in
  prepare)
    CUDA_VISIBLE_DEVICES='' "$REALWORLD_PYTHON" scripts/realworld/prepare_baseline.py "$@"
    ;;
  verify)
    CUDA_VISIBLE_DEVICES='' "$REALWORLD_PYTHON" -m unittest discover -s tests/realworld -p test_baseline.py -v "$@"
    ;;
  text)
    bash scripts/realworld/workflow.sh text "$@"
    ;;
  train)
    REALWORLD_TASK="${REALWORLD_TASK:-realworld_all4_original_fastwam_wan22_5b_ep8}"
    CUDA_VISIBLE_DEVICES='' "$REALWORLD_PYTHON" scripts/realworld/preflight.py --require-text
    CUDA_VISIBLE_DEVICES='' "$REALWORLD_PYTHON" scripts/realworld/preflight_baseline.py --task "$REALWORLD_TASK"
    export PATH="$(dirname "$REALWORLD_PYTHON"):$PATH"
    REALWORLD_RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
    accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
      --num_processes "${NPROC_PER_NODE:-8}" --main_process_port "${MASTER_PORT:-29624}" \
      scripts/realworld/train_baseline.py "task=$REALWORLD_TASK" \
      "output_dir=./runs/$REALWORLD_TASK/$REALWORLD_RUN_ID" "$@"
    ;;
  *)
    echo 'Usage: bash scripts/realworld/workflow_baseline.sh {prepare|verify|text|train}'
    echo 'Set REALWORLD_TASK to a task config name. Training uses original Wan2.2 VAE online; no Rothko latent cache.'
    ;;
esac
