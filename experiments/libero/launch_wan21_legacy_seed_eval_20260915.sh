#!/usr/bin/env bash
set -euo pipefail
export ROOT_DIR=/mnt/data/cfy/FastWAM
case "${1:-}" in manipulation) infer_seed=43 ;; nav) infer_seed=44 ;; *) exit 2 ;; esac
export RUN_ID=wan21_legacy_env42_infer${infer_seed}_${1}_20260915
export SESSION_NAME=$RUN_ID EXP_NAME=$RUN_ID
export OUTPUT_DIR=$ROOT_DIR/evaluate_results/libero/libero_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-4/ckpt021700_vaeDecoderStep7498_replan8_ensembleOff_legacy_envSeed42_inferSeed${infer_seed}_${1}_20260915
export CONFIG=libero_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-4
export CKPT=$ROOT_DIR/runs/$CONFIG/2026-08-28_17-06-22/checkpoints/weights/step_021700.pt
export PYTHON_EXECUTABLE=$ROOT_DIR/experiments/libero/python_original_seed_eval.sh
export NUM_GPUS=8 NUM_TRIALS=50 MAX_TASKS_PER_GPU=1 WORKER_TIMEOUT_SECONDS=86400
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EXTRA_ARGS="model.vae_safetensors_path=$ROOT_DIR/runs/libero_rothko_vae_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2/Wan2.1_VAE_libero_rothko_step007498.safetensors model.allow_vae_mismatch=true model.rothko_decode_mode=legacy model.rothko_decode_anchor_alpha=0.0 EVALUATION.dataset_stats_path=$ROOT_DIR/runs/$CONFIG/2026-08-28_17-06-22/dataset_stats.json EVALUATION.replan_steps=8 EVALUATION.use_action_ensembler=false EVALUATION.save_prediction_videos=null EVALUATION.save_control_trace=false EVALUATION.tiled=false EVALUATION.device=cuda:0 seed=42 eval_random_seed=42 +EVALUATION.inference_seed=$infer_seed eval_num_inference_steps=20 EVALUATION.num_inference_steps=20"
cd "$ROOT_DIR"
EXTRA_ARGS+=" MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=1 MULTIRUN.worker_timeout_seconds=86400"
if [ "${2:-}" = --check ]; then
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_EXECUTABLE" experiments/libero/eval_libero_single.py task="$CONFIG" ckpt="$CKPT" EVALUATION.num_trials=50 $EXTRA_ARGS --cfg job --resolve
  exit
fi
test ! -e "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
cp "$0" "$OUTPUT_DIR/launch.sh"
cp experiments/libero/run_libero_seed_parallel_20260915.sh "$OUTPUT_DIR/scheduler.sh"
cp experiments/libero/tasks_original40_seed_eval.txt "$OUTPUT_DIR/tasks.txt"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_EXECUTABLE" experiments/libero/eval_libero_single.py task="$CONFIG" ckpt="$CKPT" EVALUATION.num_trials=50 EVALUATION.output_dir="$OUTPUT_DIR" $EXTRA_ARGS --cfg job --resolve > "$OUTPUT_DIR/resolved_config.yaml"
exec bash "$OUTPUT_DIR/scheduler.sh" "$OUTPUT_DIR/tasks.txt" > "$OUTPUT_DIR/manager.log" 2>&1
