#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/cfy/FastWAM
if [ -x /mnt/data/cfy/anaconda3/envs/fastwam_libero/bin/python ]; then
  PY=/mnt/data/cfy/anaconda3/envs/fastwam_libero/bin/python
else
  PY=/mnt/data/cfy/miniconda3/envs/fastwam_libero/bin/python
fi
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_MODEL_BASE_PATH=$ROOT/checkpoints
export LIBERO_CONFIG_PATH=$ROOT/data/libero/.libero_config
export NUMBA_CACHE_DIR=$ROOT/data/libero/.numba_cache MPLCONFIGDIR=$ROOT/data/libero/.matplotlib_cache
exec "$PY" "$@"
