# Three-camera cache (2026-09-15)

- New opt-in task: `vlabench_select_book_rothko_3cam192_wan21`.
- RGB: `[image | second_image | wrist_image]`, each 192x192, bilinear,
  antialias=True, align_corners=False, normalized to [-1,1].
- Raymap: one 192x192 single-arm Rothko copied three times horizontally;
  relative RAY0, 17 consecutive observations and 16 future targets including
  existing edge-repeat padding. No new absolute EE condition.
- Stats: reuse ALL 4,950 training episodes' exact all-window Q99.95 physical
  bounds; only rebuild region masks at 192x576, never resize the old map or
  refit on select_book. New stats file is independent of old 224x448 stats.
- Cache uses original Wan2.1 VAE, BF16 and the existing per-sample encode loop.
  RGB and raymap are encoded separately. Each latent is [16,5,24,72].
- Eight GPUs, four independent VAE processes per GPU, one loader worker per
  process, two samples per loader batch (NOT a new batched VAE implementation).
  Encoding processes do not load DiT or UMT5.
  Data-loading workers use one PyTorch thread and one Arrow CPU/I/O thread;
  OMP/MKL/OpenBLAS/NumExpr are limited to one by this launcher to avoid CPU
  oversubscription. VAE processes retain the original two PyTorch CPU threads;
  GPU encoding itself is unchanged.

## Launch and resume

```bash
bash scripts/cache_vlabench_three_camera_8gpu.sh select_book
# Later add any other task(s), without rebuilding select_book:
bash scripts/cache_vlabench_three_camera_8gpu.sh select_drink select_fruit
```

Root: `data/vlabench_train99_rothko_3cam192_centerfrac05_wan21_bf16_h16_latents/`.
Each task has its own local ordered episode/window indices, dataset contract,
VAE checksum, metadata and 256-window binary shards. A single task is usable
as soon as its `_SUCCESS` exists; other tasks need not be complete.

Run the same command to resume. It checks completed shard checksums and only
recomputes unfinished/corrupt shards. `.partial` files are not completed data.
Shards are disjoint across processes; a parent file lock blocks duplicate
coordinators. Each shard's first/last latent is compared bit-exactly with a
fresh single-sample encode before marking it complete. The GPU/process count
can change on resume, but geometry, VAE, source ordering and shard size cannot.

To read select_book during training, the new task's dataset config already
selects task-local indexing, points `data.train.latent_cache_dir` to the
`.../select_book` directory and sets `data.train.latent_cache_only=true`.
This step only prepares one task, not a combined all-task training configuration.
Later tasks use identical independent units and can be composed without
rewriting their binary shards.

## Verification

- `tests/test_vlabench_three_camera.py`: camera order, tensor/NumPy consistency,
  legacy metadata unchanged, all three tiles including gripper borders equal,
  pose/gripper round trip.
- `runs/vlabench_3cam192_cache_online_probe/metadata.json`: actual online
  training loss and cache loss bit-exact with fixed seed123456; latent indices
  0,2,3 bit-exact. Four samples only, NOT a complete training cache.
- `runs/vlabench_3cam192_multiprocess_probe/`: separate two-process smoke cache,
  not used for training. Both workers passed endpoint checks; all first four
  samples matched the single-process probe bit-exactly. Relaunch reused both
  shards with zero pending work. The old two-camera full-cache reader still
  validated and returned 35,951 select_book windows.
- 12 VLABench tests passed. The first multiprocess probe emitted an NFS
  multiprocessing temporary-directory cleanup warning on exit (both workers
  returned 0); the formal launcher explicitly sets TMPDIR=/tmp.

Formal tmux: `vlabench_3cam192_cache`.
Coordinator log: `logs/vlabench_3cam192_cache/select_book.log`.
Per-process logs are inside the task cache directory (`worker00.log` etc.).

Existing LIBERO/Plus, old two-camera VLABench configs, caches and weights are
not overwritten. New three-camera deployment must use the shared
`build_vlabench_rgb_canvas`; the old evaluation camera selection is not silently
changed as part of cache preparation.

## Formal select_book training

`scripts/train_vlabench_three_camera_select_book_8gpu.sh` runs a one-update
backward/full-validation probe (no weights/state saved, W&B disabled), then
starts a fresh formal run automatically only if the probe succeeds.
Formal settings match the old select_book experiment: 8 GPUs, batch4, GA4,
effective batch128, 10 epochs, full DiT fine-tuning from original Wan2.1 1.3B,
original frozen VAE, gradient checkpointing off, LR1e-4 cosine/warmup5%,
weight decay0.01, loss RGB1/Raymap5. W&B on, log every10, validation every500
with the existing select_book fixed validation-window manifest. Weights every
3 epochs and final; full state every5 epochs and final. Old runs untouched.
Tmux: `vlabench_select_book_3cam192_train`; logs: `logs/vlabench_3cam192_train/`.
