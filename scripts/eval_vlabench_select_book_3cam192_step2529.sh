#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PYTHONPATH=src:third_party/VLABench
export TMPDIR=/tmp MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec /mnt/hwdata/cfy/miniconda3/envs/fastwam_vlabench/bin/python \
  experiments/vlabench/run_select_book_manager.py \
  --checkpoint runs/vlabench_select_book_rothko_3cam192_wan21/2026-09-15_17-33-31/checkpoints/weights/step_002529.pt \
  --output-dir evaluate_results/vlabench/vlabench_select_book_rothko_3cam192_wan21/ckpt002529_vaeOriginal_replan8_legacy_ep32 \
  --gpu-ids 0 1 2 3 4 5 6 7 --workers-per-gpu 1 \
  --episodes 32 --replan-steps 8 --gripper-threshold 0.5 --threads 2
