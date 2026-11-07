# LIBERO / realworld alignment audit (2026-09-24)

Reference: `runs/libero_goal_rothko_absolute_ray0_2cam224_full_wan21_1_3b_1e-4/2026-09-21_18-42-23/config.yaml`.

The modified realworld model retains Wan2.1 T2V 1.3B, frozen original VAE,
full DiT finetuning, absolute frame-zero RAY0, 16 future poses relative to
the same current EE pose, RGB/ray loss weights 1/5, block-causal attention,
BF16, LR 1e-4, and legacy ray decoding. Images are left then wrist.

Remaining data differences:

- LIBERO is 20 Hz; realworld is 10 Hz. Sixteen future steps cover 0.8 s
  and 1.6 s respectively. Realworld execution uses 0.1 s target spacing.
- LIBERO uses OSC target poses; realworld labels are future measured EE
  poses. The original teleoperation targets were not recorded.
- Realworld gripper labels use measured width >= 0.08 m as open, not
  recorded commands. This can lag commands, particularly during contact.
- Realworld images retain the collection preprocessing: RGB, aspect-ratio
  preserving resize and black padding to 224 square for each camera.
- Realworld windows require all 16 future samples; they do not pad episode
  tails. FR3 poses use the collection's base / fr3_link8 convention.
- Same-index camera/state data are approximately synchronized; exposure
  timestamps were not saved. Closed-loop inference latency is unmeasured.

Future translation normalization now uses symmetric q99.95 with 100000
histogram bins over ALL valid training windows, instead of LIBERO's sampled
windows. Bounds in meters: [0.09253, 0.09474, 0.08849]. Absolute RAY0 bounds
retain train min/max plus a 1 cm margin. Validation data do not fit stats.
Pick_place shares these all-four-task training statistics and cache.

The explicit single-task entry is `scripts/realworld/train_pick_place.sh`:
8 GPUs, batch 4/GPU, accumulation 4, effective batch 128, 8 epochs, gradient
checkpointing OFF. Its separate task config selects 19042 train windows
from 99 episodes and 494 validation windows from 2 episodes. There is one
fixed validation probe per held-out episode. The wrapper preserves original
full-cache indices rather than renumbering cached samples.

CPU checks passed for resolved configuration, selected window contents,
full-dataset index mapping, validation manifest, model/data contract, and
dataset serialization. The new cache is still computing; GPU training and
real robot closed-loop validation have not run. This audit did not launch
training. The older generic workflow still represents the all-task v2
experiment; use the explicit single-task entry for the planned new run.
