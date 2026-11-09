#!/usr/bin/env bash
# Independent cache only. Resume completed shards without modifying baseline data.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python_bin="${PYTHON_BIN:-/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python}"
cache_dir="${CACHE_DIR:-$PWD/data/libero_goal_axis_angle_constant_wan21_bf16_h16_224x448_20261108}"
log_dir="${LOG_DIR:-$PWD/evaluate_results/libero/preparation/axis_angle_pipeline_20261108/cache_logs}"
IFS=',' read -r -a gpu_ids <<< "${GPU_IDS:-0,1,2,3,4,5,6,7}"
workers_per_gpu="${WORKERS_PER_GPU:-4}"
[[ "$workers_per_gpu" =~ ^[1-9][0-9]*$ ]] || exit 2
world=$((${#gpu_ids[@]}*workers_per_gpu))
mkdir -p "$log_dir"
if [[ ! -e "$cache_dir" ]]; then
  "$python_bin" scripts/prepare_goal_axis_angle_cache.py initialize --output "$cache_dir" > "$log_dir/initialize.log" 2>&1
fi
pids=()
cleanup() {
  # Only this invocation's worker processes; no process-name matching.
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
stamp="$(date -u +%Y%m%d_%H%M%S)_$$"
for ((rank=0; rank<world; rank++)); do
  gpu="${gpu_ids[rank/workers_per_gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u scripts/prepare_goal_axis_angle_cache.py worker \
    --output "$cache_dir" --rank "$rank" --world "$world" \
    > "$log_dir/${stamp}_worker${rank}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
printf 'Cache workers started: %s; output: %s; logs: %s\n' "$world" "$cache_dir" "$log_dir"
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" == 0 ]] || exit "$status"
"$python_bin" scripts/prepare_goal_axis_angle_cache.py finalize --output "$cache_dir" > "$log_dir/${stamp}_finalize.log" 2>&1
CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "$python_bin" scripts/prepare_goal_axis_angle_cache.py check \
  --output "$cache_dir" --online > "$log_dir/${stamp}_online_check.log" 2>&1
printf 'Cache complete and online-verified: %s\n' "$cache_dir"
