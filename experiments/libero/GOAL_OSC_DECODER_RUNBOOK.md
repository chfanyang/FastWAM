# Goal OSC decoder fine-tuning

2026-09-21 local launch: tmux `libero_goal_osc_vae`; logs and command in `evaluate_results/libero/goal_osc_vae_launch_20260921/`. Dedicated entry `scripts/prepare_goal_osc_decoder.py` and config `configs/vae/libero_goal_osc_delta_decoder_wan21_bs2_ga4_lr1e-5_ep2.json`. Existing absolute workflow and common trainer unchanged.

Original Wan2.1 VAE, encoder/conv1 frozen; conv2+decoder trainable. Eight GPUs, batch2/rank, GA4, effective64; two epochs, 789 steps/epoch, 1578 total; LR1e-5, warmup78, cosine to1e-7; FP32 parameters/optimizer with BF16 autocast and sharded AdamW.

Uses completed `data/libero_goal_osc_delta_2cam224_wan21_bf16_h16_latents` cache. RAY0 is original observed absolute world EE; future16 encode original OSC commands, translation scaled0.05m, rotation0.5rad. Tail motion is zero, gripper held. All starts and padded image targets enter reconstruction loss; validation metrics exclude padded future steps. This is not the all-absolute action representation.

Fixed seed20260801 split: 413 train episodes/50445 windows, 20 validation episodes/200 uniformly spaced windows. Manifest and cache audit under `evaluate_results/libero/goal_osc_decoder_preparation_20260921/`. 60 probes covering each task, train/validation, first/full-tail/final padded start passed bit-exact original BF16 encoding versus cache before launch. No online encoder calls during training.

Saves complete rolling `checkpoint_latest.pt` every400 steps, each epoch end, and final. Independent weights checkpoint every800 and final; VAE exports every800 and final. Validation every800, each epoch end, and final. Latest includes optimizer; independent step checkpoints follow historical weights-only format. Run: `runs/libero_goal_osc_delta_decoder_wan21_bs2_ga4_lr1e-5_ep2/`.

At launch GPU3 was still running Goal closed-loop evaluation. Existing evaluation was preserved; compute sharing may slow both jobs until evaluation completes.
