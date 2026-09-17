#!/usr/bin/env bash
set -euo pipefail
cd /mnt/hwdata/cfy/FastWAM
export PATH=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin:$PATH
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
variant="${1:-wan22_5b}"
case "$variant" in
  wan22_5b|wan21_1_3b) ;;
  *) echo "Unsupported variant: $variant" >&2; exit 2 ;;
esac
run="runs/vlabench_${variant}_smoke/$(date -u +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$run"
args=("task=vlabench_rothko_2cam224_full_${variant}_1e-4"
  "output_dir=$run" batch_size=1 gradient_accumulation_steps=1 num_workers=1
  max_steps=3 log_every=1 warmup_ratio=0
  save_every_epochs=null state_save_every_epochs=null
  save_every=2 state_save_every=2 save_at_end=false
  eval_every=2 eval_sample_manifest=null eval_num_samples=1 eval_num_inference_steps=2
  wandb.enabled=false)
# Retain the same three-step schedule in both launches. Reload step 2 and
# repeat step 3, testing full state restoration without changing the budget.
bash scripts/train_zero1.sh 4 "${args[@]}" > "$run/train.log" 2>&1
bash scripts/train_zero1.sh 4 "${args[@]}" \
  "resume=$run/checkpoints/state/step_000002" > "$run/resume.log" 2>&1
echo "SMOKE_COMPLETE $run"
