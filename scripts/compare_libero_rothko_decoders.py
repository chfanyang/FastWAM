#!/usr/bin/env python
"""Paired decoder errors on stratified training windows (no DiT or rollout)."""
import argparse
import json
import time
from pathlib import Path

import torch
import evaluate_libero_rothko_vae_checkpoints as sampling
from fastwam.representations.libero_rothko import LiberoRothkoCodec

core = sampling.core
ROOT = Path(__file__).resolve().parents[1]


def errors(pred, truth):
    position = (pred[..., :3].double() - truth[..., :3].double()).norm(dim=-1) * 1000
    q = torch.nn.functional.normalize(pred[..., 3:].double(), dim=-1)
    t = torch.nn.functional.normalize(truth[..., 3:].double(), dim=-1)
    t = torch.where((q * t).sum(-1, keepdim=True) < 0, -t, t)
    rotation = 4 * torch.atan2((q - t).norm(dim=-1), (q + t).norm(dim=-1)) * (180 / torch.pi)
    return position, rotation


def summarize(rows):
    result = {}
    for mode in rows[0]['errors']:
        result[mode] = {}
        for metric in ('translation_mm', 'rotation_deg'):
            a = torch.tensor([r['errors'][mode][metric] for r in rows], dtype=torch.float64)
            result[mode][metric] = {
                'mean': a.mean().item(), 'median': a.median().item(),
                'p95': a.quantile(.95).item(), 'max': a.max().item(),
                'first8_mean': a[:, :8].mean().item(),
                'by_horizon': a.mean(0).tolist(),
            }
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes-per-task', type=int, default=5)
    parser.add_argument('--sample-seed', type=int, default=20260909)
    parser.add_argument('--suite', choices=['libero_10', 'libero_goal', 'libero_object', 'libero_spatial'])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    config_path = ROOT / 'runs/libero_rothko_vae_decoder_all4_h16_bs2_ga8_lr1e-5_ep2_rerun_fixed/training_config.json'
    config = argparse.Namespace(**json.loads(config_path.read_text())['arguments'])
    train, heldout = core.discover_episodes(config)
    refs, records = sampling.select_windows(train, episodes_per_task=args.episodes_per_task,
                                            horizon=16, seed=args.sample_seed)
    if args.suite:
        selected = [(ref, rec) for ref, rec in zip(refs, records) if rec['suite'] == args.suite]
        refs, records = map(list, zip(*selected))
    windows = core.materialize_windows(core.EpisodeStore(64), refs, 16)
    checkpoint = ROOT / 'runs/libero_rothko_vae_decoder_all4_h16_bs2_ga8_lr1e-5_ep2_rerun_fixed/Wan2.2_VAE_libero_rothko_step007498.safetensors'
    metadata = {'samples': records, 'sample_seed': args.sample_seed, 'checkpoint': str(checkpoint),
                'norm_stats': config.norm_stats_path, 'precision': 'bf16 weights and VAE computation',
                'description': 'GT VAE round-trip; future frames 1..16 only; alpha=0; shared reconstruction',
                'split_seed': config.split_seed, 'heldout_episodes': len(heldout)}
    (out / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    stats = core.load_norm_stats(config.norm_stats_path, config)
    codecs = {name: LiberoRothkoCodec(norm_stats=stats, expected_action_horizon=16,
               decode_mode=mode, decode_anchor_alpha=0, decode_block_grid=grid)
              for name, mode, grid in [('joint', 'robust_joint', 4),
                                       ('grid4', 'robust_block_consensus', 4),
                                       ('grid2', 'robust_block_consensus', 2)]}
    device = torch.device('cuda:0')
    vae = core.load_base_vae(str(checkpoint), config.vae_variant, device, torch.bfloat16)
    rows, cache = [], []
    start = time.monotonic()
    for i, window in enumerate(windows):
        target, encoded = core.build_target_batch([window], stats, device, torch.float32, config)
        truth = encoded[0].pose.unsqueeze(0)
        latent = core.vae_encode(vae.model, target.to(torch.bfloat16), vae.scale)
        reconstruction = vae.model.decode(latent, vae.scale).float().clamp(-1, 1)
        cache.append({'latent': latent[0].cpu(), 'pose': truth[0].cpu(),
                      'gripper': encoded[0].gripper.cpu(), 'sample': records[i]})
        row = dict(records[i], errors={})
        for name, codec in codecs.items():
            prediction, _ = codec.decode(reconstruction, truth[:, 0])
            position, rotation = errors(prediction[:, 1:], truth[:, 1:])
            assert torch.isfinite(position).all() and torch.isfinite(rotation).all()
            row['errors'][name] = {'translation_mm': position[0].cpu().tolist(),
                                   'rotation_deg': rotation[0].cpu().tolist()}
            if i == 0:
                ideal, _ = codec.decode(target, truth[:, 0])
                pe, re = errors(ideal[:, 1:], truth[:, 1:])
                print('IDEAL_ROUNDTRIP', name, pe.max().item(), re.max().item(), flush=True)
                assert pe.max() < .1 and re.max() < .01, 'GT codec sanity check failed'
        rows.append(row)
        if (i + 1) % 10 == 0 or i == 0:
            print(f'{i+1}/{len(windows)} windows, elapsed={time.monotonic()-start:.1f}s', flush=True)
    torch.save({'metadata': metadata, 'samples': cache}, out / 'encoded_training_windows.pt')
    (out / 'per_window_errors.json').write_text(json.dumps(rows))
    report = {'windows': len(rows), 'future_poses': len(rows)*16, 'overall': summarize(rows),
              'by_suite': {s: summarize([r for r in rows if r['suite']==s]) for s in sorted({r['suite'] for r in rows})},
              'elapsed_seconds': time.monotonic()-start}
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
