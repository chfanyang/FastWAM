# VLABench Track-1 select_book evaluation

Use the independent `fastwam_vlabench` conda environment. Single-worker entry:
`eval_select_book.py`; multi-GPU entry: `run_select_book_manager.py`.

```bash
conda activate fastwam_vlabench
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
python experiments/vlabench/run_select_book_manager.py \
  --checkpoint /absolute/path/to/checkpoints/weights/step_002810.pt \
  --output-dir /absolute/path/to/new/evaluation_directory \
  --gpu-ids 0 1 2 3 4 5 6 7 \
  --workers-per-gpu 2 \
  --episodes 50 --replan-steps 8 --gripper-threshold 0.5 --threads 2
```

- `--dry-run` prints assignments and commands without loading a model or creating outputs.
- GPU IDs are physical IDs. Each child sees one CUDA GPU; its EGL renderer is assigned the same physical GPU. Each worker holds its own model, so more workers increase VRAM usage. Actual capacity needs a model rollout test.
- All workers receive disjoint **global** Track-1 episode IDs. A scene's seed remains `42 + global_episode_id`; policy diffusion seed remains 42, identical to the single-process entry. Worker rank does not change seeds.
- No distributed training/collectives or tmux dependency. Can be run inside user-created tmux.
- Default is one worker per GPU and two CPU threads per worker. There is no wall-clock evaluation timeout.
- Results and official videos are isolated under `workerN_gpuG/`; logs are `workerN_gpuG.log`. Global results are in `summary.json`, including worker paths.
- Aggregate success is computed across episodes, not a mean of worker success rates. Incomplete runs have `success_rate: null`, a separate completed-only rate, and missing episode IDs.
- Ctrl-C/termination stops this manager's workers only and writes the available summary. A worker runtime error stops the run instead of silently shrinking the denominator. Nonempty output directories are refused; automatic resume is not implemented.
- This entry currently supports **only Track 1 select_book**. Original VAE, legacy decoder, no ensemble; joint future RGB denoising is retained but unused future RGB pixel decoding is skipped. Actual rollout videos are saved; predicted RGB videos are not.

CPU checks: `python -m unittest discover -s tests -p test_vlabench_eval_sharding.py`.


## Persistent multi-task Track1 workers

`run_track1_persistent.py` keeps one model and policy per process. The shared
queue contains `(task, episode)` jobs; eight GPUs with four slots each means
32 persistent processes. A process starts the next episode immediately after
finishing its previous one, without reloading DiT, VAE, or the text encoder.
Each episode still resets the policy/RNG and creates a fresh official evaluator
and environment. The official loop closes the environment at episode end.

This entry defaults to the original VAE; pass `--vae-safetensors-path PATH --allow-vae-mismatch` for an explicitly selected finetuned decoder. Completed-result reuse also requires the same VAE path. It uses replan8, 20 denoising steps,
legacy decoder, seed42 and max_substeps1 by default. Each worker limits numerical
library threads to two. Worker failures stop the queue and are not scored as
ordinary unsuccessful episodes; no automatic model reload/retry is performed.

Prepare a plan without launching any worker:

```bash
python experiments/vlabench/run_track1_persistent.py   --checkpoint /absolute/path/to/checkpoints/weights/step_015000.pt   --output-dir /absolute/path/to/new_result_directory   --completed-from /absolute/path/to/old_dynamic_queue_results   --episodes 32 --gpu-ids 0 1 2 3 4 5 6 7 --workers-per-gpu 4   --threads 2 --replan-steps 8 --gripper-threshold 0.5 --seed 42   --dry-run
```

`--completed-from` reads the old queue's `task/episode_NNN/episode_NNN.json`
layout, validates checkpoint and protocol identity, and skips only completed
scenes. It does not overwrite or move old results. Stop/drain the old queue
before launching the new queue; do not run them on the same GPUs concurrently.
Use a fresh result directory and the same VLABench environment, PYTHONPATH,
DIFFSYNTH_MODEL_BASE_PATH, MUJOCO_GL and TMPDIR as the established launcher.
Remove `--dry-run` only when execution is authorized.

CPU checks cover exact task/episode coverage, four slots per GPU, runtime reuse
across tasks, error handling, protocol identity and reuse of completed results.
Actual GPU multi-episode reuse still needs a runtime check on first launch.
