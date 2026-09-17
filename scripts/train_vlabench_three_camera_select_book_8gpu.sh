#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PATH=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin:$PATH
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TMPDIR=/tmp TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
task=vlabench_select_book_rothko_3cam192_wan21
stamp=$(date -u +%Y-%m-%d_%H-%M-%S)
mkdir -p logs/vlabench_3cam192_train
# One real update with the formal batch shape, then full 20-step validation.
# No smoke checkpoints, no smoke W&B run, no resume test.
bash scripts/train_zero1.sh 8 "task=$task" \
  "output_dir=./runs/vlabench_3cam192_train_probe/$stamp" \
  max_steps=1 save_every_epochs=null state_save_every_epochs=null \
  save_every=0 state_save_every=0 save_at_end=false eval_every=1 \
  wandb.enabled=false log_every=1 \
  > "logs/vlabench_3cam192_train/probe_$stamp.log" 2>&1
echo "Probe passed; starting fresh formal training."
exec bash scripts/train_zero1.sh 8 "task=$task" \
  > "logs/vlabench_3cam192_train/formal_$stamp.log" 2>&1
