#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PATH=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin:$PATH
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
# Only multiprocessing scratch lives locally; outputs remain under runs.
export TMPDIR=/tmp
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
run="runs/vlabench_wan21_bs8_ga4_benchmark/$(date -u +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$run"
nvidia-smi -i 0,1,2,3 --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 2 > "$run/gpu.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
echo "BENCH_START $run"
bash scripts/train_zero1.sh 4 \
  task=vlabench_rothko_2cam224_full_wan21_1_3b_1e-4 \
  "output_dir=$run" batch_size=8 gradient_accumulation_steps=4 num_workers=8 \
  max_steps=6 log_every=1 warmup_ratio=0 \
  save_every_epochs=null state_save_every_epochs=null \
  save_every=1000 state_save_every=1000 save_at_end=false \
  eval_every=5 eval_num_inference_steps=20 wandb.enabled=false \
  > "$run/train.log" 2>&1
echo "BENCH_COMPLETE $run"
