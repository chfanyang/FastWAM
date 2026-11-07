#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
exec "${REALWORLD_PYTHON:-/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python}" \
  scripts/realworld/serve_baseline.py "$@"
