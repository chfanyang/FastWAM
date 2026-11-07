# Pick_place VAE decoder, q99.95, 2 epochs

Dedicated adapter: `scripts/realworld/finetune_pick_place_vae.py`.
Configuration: `configs/vae/realworld_pick_place_q9995_decoder_ep2.json`.
The adapter reuses the LIBERO trainer and its masked reconstruction loss via
process-local hooks. Shared trainer files and previous experiments are unchanged.

- Original Wan2.1 VAE initialization. Encoder and conv1 frozen; conv2 and decoder
  trained with FP32 master parameters, BF16 autocast, LR 1e-5, cosine decay,
  5% warmup, gradient clipping 1, and sharded AdamW as in the reference run.
- 8 GPUs, batch 2/GPU, accumulation 4, effective batch 64, 2 epochs / 646 updates.
- Same four-task train-only q99.95 stats and absolute RAY0 bounds as DiT training.
- 99 pick_place training episodes: 19042 complete windows plus 1584 episode-tail
  windows. All windows contain current frame plus t+1..t+16 measured poses.
  Missing future values repeat the last recorded pose and gripper. Timing-gap
  rejection remains active; no window crosses a bad timestamp interval.
- Reconstruction loss uses ONLY valid decoded frames, including RAY0. Padding
  has zero direct reconstruction supervision, with a global valid-frame
  denominator across DDP ranks. This is the existing LIBERO loss, not a latent
  regression loss; padded latent entries are not zeroed or detached.
- Original BF16 encoder latents for complete windows reuse the verified q99.95
  cache. Eight online probes must match bit-exactly before training; the adapter
  computes a separate supplemental tail/validation cache and never edits DiT caches.
- Validation uses the same 2 held-out episodes, 5 valid complete windows each.
  Evaluation runs with BF16 deployment decoder weights and the existing legacy
  ray decoder, reporting future position, rotation and gripper errors.
- At step 500 and final step 646, save independent full optimizer checkpoints and
  complete BF16 VAE exports. Decoder evaluation does not change the frozen encoder.

CPU checks passed: padding-target perturbation leaves loss unchanged, padded
output gradients are zero, valid/RAY0 gradients nonzero, full-window loss equals
unmasked loss; realworld t+1 tail masks, target equivalence, and split isolation.

Output: `runs/realworld_pick_place_q9995_decoder_ep2`.
Launch log: `evaluate_results/realworld/pick_place_vae_20260924_042543/train.log`.
The legacy export filename includes `libero_rothko`; metadata identifies this run
as realworld pick_place with the new norm fingerprint.
