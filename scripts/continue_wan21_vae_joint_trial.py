"""Opt-in full-state 7498 continuation; same loss, exported-BF16 joint validation."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import finetune_rothko_vae_decoder as core
from compare_libero_rothko_decoders import errors, summarize
from fastwam.representations.libero_rothko import LiberoRothkoCodec

@torch.no_grad()
def joint_evaluate(vae, store, episodes, stats, args, device, dtype):
    windows = core.sample_uniform_windows_per_episode(store, episodes, args.eval_windows_per_episode, 16)
    expected = len(episodes) * args.eval_windows_per_episode
    assert len(windows) == expected
    manifest = [{'dataset': w.episode.dataset_root, 'episode': w.episode.episode_index,
                 'task': w.episode.task, 'start': w.start} for w in windows]
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if not args.smoke:
        baseline = json.loads((ROOT/'evaluate_results/libero_decoder_offline/wan21_wan22_step7498_joint_anchor0_heldout800_20260913/summary.json').read_text())
        assert digest == baseline['manifest_sha256'], 'Validation windows changed'
    # Match deployment exports: BF16 weights, not FP32-master training forward.
    evaluation_vae = core.load_base_vae(args.base_vae, args.vae_variant, device, torch.bfloat16)
    evaluation_vae.model.decoder.load_state_dict(vae.model.decoder.state_dict(), strict=True)
    evaluation_vae.model.conv2.load_state_dict(vae.model.conv2.state_dict(), strict=True)
    evaluation_vae.model.eval()
    codec = LiberoRothkoCodec(norm_stats=stats, expected_action_horizon=16,
                              decode_mode='robust_joint', decode_anchor_alpha=0, decode_block_grid=4)
    rows = []
    gripper_sum = 0.0
    gripper_count = 0
    for i, w in enumerate(windows):
        target, encoded = core.build_target_batch([w], stats, device, torch.bfloat16, args)
        truth = encoded[0].pose.unsqueeze(0)
        latent = core.vae_encode(evaluation_vae.model, target, evaluation_vae.scale)
        reconstruction = evaluation_vae.model.decode(latent, evaluation_vae.scale).float().clamp(-1, 1)
        prediction, _ = codec.decode(reconstruction, truth[:, 0])
        pe, re = errors(prediction[:, 1:], truth[:, 1:])
        if not torch.isfinite(pe).all() or not torch.isfinite(re).all():
            raise FloatingPointError('Non-finite validation error')
        rows.append({'errors': {'joint': {'translation_mm': pe[0].cpu().tolist(), 'rotation_deg': re[0].cpu().tolist()}}})
        gripper = core.read_gripper(reconstruction, args)
        gripper_error = (gripper[:, 1:] - encoded[0].gripper.unsqueeze(0)[:, 1:]).abs()
        gripper_sum += gripper_error.sum().item()
        gripper_count += gripper_error.numel()
        if (i + 1) % 200 == 0:
            print(f'Joint validation {i+1}/{len(windows)}', flush=True)
    metrics = summarize(rows)['joint']
    result = {'future_position_mae_m': metrics['translation_mm']['mean']/1000,
              'future_rotation_mean_deg': metrics['rotation_deg']['mean'],
              'future_gripper_mae': gripper_sum/gripper_count,
              'num_windows': len(windows), 'manifest_sha256': digest,
              'decode_mode': 'robust_joint', 'anchor_alpha': 0,
              'precision': 'bf16 deployment weights', 'metrics': metrics}
    del evaluation_vae, latent, reconstruction, prediction, target, gripper
    gc.collect()
    torch.cuda.empty_cache()
    return result

def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--trial-steps', type=int, default=1000)
    p.add_argument('--smoke', action='store_true')
    custom, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = core.parse_args()
    if Path(args.output_dir).exists():
        raise FileExistsError('Trial must use a fresh output directory')
    if args.lr not in (1e-6, 3e-6):
        raise ValueError('Unexpected trial LR')
    args.continuation_steps = custom.trial_steps
    args.continuation_start_lr = 1e-7
    args.continuation_warmup_steps = 50
    args.smoke = custom.smoke
    args.epochs = 3
    args.eval_every = 200
    args.save_every = 200
    args.export_every = 200
    args.step_checkpoint_every = 200
    args.log_every = 10 if not custom.smoke else 1
    args.wandb = False
    args.eval_windows_per_episode = 1 if custom.smoke else 10
    core.evaluate = joint_evaluate
    core.train(args)

if __name__ == '__main__':
    main()
