#!/usr/bin/env python
"""Paired decode-only comparison on existing GT training latents, no resampling."""
import argparse
import json
import time
from pathlib import Path
import torch
from compare_libero_rothko_decoders import ROOT, core, errors, summarize, LiberoRothkoCodec


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    payload = torch.load(args.cache, map_location='cpu', weights_only=False)
    metadata = payload['metadata']
    config = argparse.Namespace(**json.loads((ROOT / 'runs/libero_rothko_vae_decoder_all4_h16_bs2_ga8_lr1e-5_ep2_rerun_fixed/training_config.json').read_text())['arguments'])
    checkpoint = ROOT / 'runs/libero_rothko_vae_decoder_all4_h16_bs2_ga8_lr1e-5_ep2_rerun_fixed/Wan2.2_VAE_libero_rothko_step007498.safetensors'
    assert Path(metadata['checkpoint']).resolve() == checkpoint.resolve()
    assert Path(metadata['norm_stats']).resolve() == Path(config.norm_stats_path).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stats = core.load_norm_stats(config.norm_stats_path, config)
    specs = [('joint', 'robust_joint', 4), ('grid4', 'robust_block_consensus', 4),
             ('grid2', 'robust_block_consensus', 2),
             ('weighted_grid4', 'robust_block_weighted_joint', 4),
             ('weighted_grid2', 'robust_block_weighted_joint', 2)]
    codecs = {name: LiberoRothkoCodec(norm_stats=stats, expected_action_horizon=16,
              decode_mode=mode, decode_anchor_alpha=0, decode_block_grid=grid)
              for name, mode, grid in specs}
    manifest = dict(metadata, source_cache=str(args.cache.resolve()), decoders=specs,
                    method_version='block_weighted_joint_v1', anchor_alpha=0)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    vae = core.load_base_vae(str(checkpoint), config.vae_variant, torch.device('cuda:0'), torch.bfloat16)
    rows, timings = [], {name: 0.0 for name in codecs}
    start = time.monotonic()
    for i, sample in enumerate(payload['samples']):
        latent = sample['latent'].unsqueeze(0).to('cuda:0')
        truth = sample['pose'].unsqueeze(0).to('cuda:0')
        video = vae.model.decode(latent, vae.scale).float().clamp(-1, 1)
        row = dict(sample['sample'], errors={})
        for name, codec in codecs.items():
            torch.cuda.synchronize()
            tick = time.monotonic()
            pred, _ = codec.decode(video, truth[:, 0])
            torch.cuda.synchronize()
            timings[name] += time.monotonic() - tick
            pe, re = errors(pred[:, 1:], truth[:, 1:])
            assert torch.isfinite(pe).all() and torch.isfinite(re).all()
            row['errors'][name] = {'translation_mm': pe[0].cpu().tolist(), 'rotation_deg': re[0].cpu().tolist()}
        rows.append(row)
        if i == 0 or (i+1) % 10 == 0:
            print(f'{i+1}/{len(payload["samples"])} elapsed={time.monotonic()-start:.1f}s', flush=True)
    (args.output_dir / 'per_window_errors.json').write_text(json.dumps(rows))
    report = {'windows': len(rows), 'overall': summarize(rows),
              'decoder_mean_seconds': {k:v/len(rows) for k,v in timings.items()},
              'elapsed_seconds': time.monotonic()-start}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
