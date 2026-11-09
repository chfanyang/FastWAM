# Goal constant axis-angle image ablation

This experiment keeps the absolute current EE pose in RAY0 and expresses the
16 future target poses relative to the observed current EE, exactly as the Goal
Action Raymap baseline. Only the rotation region changes: every pixel outside
the translation center and gripper border stores the same three-component
shortest rotation vector divided by pi. RAY0 rotation is absolute; future
rotation is `conj(q_current) * q_target`. Two identical 224x224 tiles form a
224x448 canvas. Translation and gripper pixels remain bit-identical to baseline.

Rotation recovery averages corresponding pixels of the two decoded tiles,
takes the component-wise median over the baseline safe direction mask, multiplies
by pi, and converts the rotation vector to a quaternion. Future poses are restored
using the observed EE. This reader does not fit rays or apply an anchor correction.

## Isolation and data

- Codec: `src/fastwam/representations/libero_axis_angle.py`.
- Dataset/model: `AxisAngleDataset` / `FastWAMAxisAngle`, independent factory
  `fastwam.axis_angle_runtime.create_axis_angle_raymap`.
- Dataset: Goal, 52895 frame starts, edge-repeat input padding; the inherited
  DiT loss ignores groups containing only padded future frames.
- Frozen original Wan2.1 VAE, SHA256
  `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`.
- Existing relative translation stats and absolute RAY0 bounds remain unchanged;
  stats fingerprint
  `fb75ea4665b40a822f5e19c4d54a4439384fe8865db7ca90aa6cc7350405a21c`.
- RGB latents are copied bit-exactly from
  `data/libero_goal_rothko_absolute_ray0_wan21_bf16_h16_latents_20260921`.
  Action latents are newly encoded; text remains the existing BF16 batch1 cache.
- Independent output cache:
  `data/libero_goal_axis_angle_constant_wan21_bf16_h16_224x448_20261108`.
  Size: 26540595200 bytes (26.54 GB / 24.72 GiB).
- Metadata authenticates codec conversion, geometry, dataset, stats and VAE.
  Baseline/new checkpoints reject each other. Diagnostic caches cannot be used
  for formal training. Existing shared code and old caches are not modified.

The data configuration retains the base `libero_rothko` pose-preprocessing flag
so existing RGB construction and pose-to-OSC control are reused. The dataset's
actual codec, latent-cache contract and model/checkpoint representation are
`libero_axis_angle_constant`. They must match before training starts.

## Preparation and diagnostics

Artifacts: `evaluate_results/libero/preparation/axis_angle_pipeline_20261108/`.

As checked on 2026-11-08:

- Five codec tests and three pipeline tests passed, including exact pi rotations,
  quaternion sign invariance, representation rejection, actual small DiT
  forward/backward, portable checkpoint and optimizer restore.
- 256-window diagnostic cache passed per-shard hashes, bit-exact RGB checks and
  sampled online action checks. Original action encode matches the old cache;
  batch1/batch2 new action encoding is bit-exact. VAE worker peak: 1.09 GiB.
- Real online dataset matched raw parquet independently for indices
  `[0,122,137,255,52894]`, including two tails with 15 padded future actions.
- Full Wan2.1, real 17-frame data, one GPU / batch1 forward-backward passed;
  all 825 gradient-bearing parameter tensors were finite and VAE remained frozen.
  Peak allocated memory was 19.39--19.52 GiB. No optimizer was created or updated:
  this is not a measurement of formal batch4 + AdamW/ZeRO training memory.
- Full17-frame encode's first latent was bit-exact to single-frame encode, for
  both RGB and the new action image. Cached and online inputs matched exactly.
- The actual shared trainer sample adapter and 20-step validation routine passed,
  including video export and decoded pose/gripper metrics. A tail validation
  loss was finite. This used pretrained DiT, so its action errors measure no
  trained task performance.
- Full cache generation completed before the user authorized formal training:
  all 52895 windows / 207 shards passed hashes, every RGB copy is bit-exact,
  all 32 workers passed their encoding probes, and the final online check at
  indices `[0,122,137,26447,52894]` passed. `metadata.json:complete`, `_SUCCESS`
  and `check_online.json` confirm completion. The cache tmux workflow launches
  no training command.

The user subsequently authorized a separate 8-GPU training launch at
2026-11-08 17:31:45 UTC. Run:
`runs/libero_goal_axis_angle_constant_wan21/2026-11-08_17-31-45`;
tmux `goal_axis_angle_train_20261108`; W&B run `tjtcojxo`.
Launch record and source snapshots:
`evaluate_results/libero/launch_records/goal_axis_angle_8gpu_2026-11-08_17-31-45/`.

Cache launch (defaults: 8 GPUs, 4 workers/GPU, BF16 batch2 action encoding):

```bash
GPU_IDS=0,1,2,3,4,5,6,7 WORKERS_PER_GPU=4 \
  bash scripts/cache_goal_axis_angle.sh
```

The launcher finishes with all-shard hash verification and sampled online VAE
checks. To resume with another worker count, first end only the old cache job,
then rerun the same command with the new GPU/worker settings. Completed shards
are validated and reused. Incomplete uniquely named partial files are not
accepted as completed results. Do not change the codec while building a cache.

## Formal training, after preparation

Task: `configs/task/libero_goal_axis_angle_constant_wan21.yaml`.
Wan2.1 1.3B full DiT, original frozen VAE; baseline attention mask;
RGB/action loss 1:5. Eight GPUs, batch4/GPU, GA4, effective128; 10 epochs,
414 optimizer steps/epoch, 4140 total. AdamW LR1e-4, weight decay .01,
cosine schedule, warmup5% (207 steps), min LR1e-6, gradient clipping1,
BF16, seed42, gradient checkpointing disabled. Validation every500 steps
on the same four training probes. These are not a held-out validation split.
Portable weights every epoch, complete resumable state every2000 and at end.
Save frequency is the only operational change relative to the historical
reference; it matches the new per-epoch baseline rerun.

Do not start until the complete action cache is verified. The following is
documented for a later explicit launch:

```bash
PATH=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin:$PATH \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/train_goal_axis_angle_8gpu.sh
```

The launch script uses a unique new run path. W&B has its own run/group;
`FASTWAM_HTTP_PROXY` can override the local default proxy. Old run paths are not
used for output. No baseline checkpoint is loaded as an axis-angle checkpoint.

## Evaluation

Independent config: `configs/sim_libero_goal_axis_angle.yaml`.
Entry: `experiments/libero/eval_axis_angle.py`. Same shared Goal control loop,
50 trials/task (500 total), replan8, horizon16, denoise20, seed42, no action
ensemble, original VAE, threshold .5 and existing pose-to-OSC conversion.
This codec requires its median reader (`legacy` / anchor0); applying the
ray-field `robust_joint` SVD reader is not meaningful for constant rotation pixels.

The final step4140 is the preset comparison checkpoint. Do not select epochs
using closed-loop success. Example single-task worker command, only after a
checkpoint exists and an evaluation launch is requested:

```bash
PYTHONPATH=src:. /mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin/python \
  experiments/libero/eval_axis_angle.py \
  ckpt=/absolute/path/to/new/step_004140.pt \
  EVALUATION.task_id=0
```

The example above is a template, not a launched evaluation.
