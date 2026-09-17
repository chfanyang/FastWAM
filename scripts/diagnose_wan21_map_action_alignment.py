"""Read-only checkpoint comparison: exact cached validation plus training map L1."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from train_wan21_decoder_cached import ROOT, CachedWindows, cached_evaluate, core, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--new-steps', type=int, nargs='*', default=[1500, 1800, 2100, 2400])
    parser.add_argument('--old-steps', type=int, nargs='*', default=[4800, 7498])
    opts = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.15, device)
    dist.init_process_group('nccl', device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    run = ROOT/'runs/libero_wan21_vae_decoder_cached_bs128_lr1e5_const_w100_ep2_fast_20260913'
    old = ROOT/'runs/libero_rothko_vae_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2'
    args = argparse.Namespace(**json.loads((run/'training_config.json').read_text())['arguments'])
    assert args.parallel_cached_eval and not args.cached_smoke
    cache = CachedWindows(args)
    assert cache.manifest == args.cache_contract
    _, episodes = core.discover_episodes(args, rank=rank)
    stats = core.load_norm_stats(args.norm_stats_path, args)
    store = core.EpisodeStore(100)
    windows = core.sample_uniform_windows_per_episode(store, episodes, 10, 16)
    selected = list(enumerate(windows))[rank::world]
    masks = core.build_loss_masks(args, device)
    original_loader = core.load_base_vae
    checkpoints = [(f'new_{s}', run/f'Wan2.1_VAE_libero_rothko_step{s:06d}.safetensors')
                   for s in opts.new_steps]
    checkpoints += [(f'old_{s}', old/f'Wan2.1_VAE_libero_rothko_step{s:06d}.safetensors')
                    for s in opts.old_steps]
    assert checkpoints and len({name for name, _ in checkpoints}) == len(checkpoints)
    assert all(p.is_file() for _, p in checkpoints)
    if rank == 0:
        opts.output.mkdir(parents=True, exist_ok=False)
        (opts.output/'manifest.json').write_text(json.dumps({
            'checkpoints': {name: str(p) for name, p in checkpoints},
            'cache_contract': cache.manifest, 'windows': 800,
            'loss': 'Exact training compute_reconstruction_loss on raw unclamped reconstruction and CPU codec FP32 target, all 17 frames',
            'actions': 'Exact cached_evaluate; clamped map, future 16 frames, BF16 deployment decoder, robust_joint anchor0',
            'validation_script_sha256': hashlib.sha256((ROOT/'scripts/train_wan21_decoder_cached.py').read_bytes()).hexdigest(),
        }, indent=2))
    dist.barrier()
    summary = {}
    for name, path in checkpoints:
        rows = []
        original_codec_decode = cache.codec.decode

        def observed_action_decode(*decode_args, **decode_kwargs):
            pred, grip = original_codec_decode(*decode_args, **decode_kwargs)
            _, w = selected[len(rows)-1]
            truth = torch.from_numpy(w.pose).unsqueeze(0).to(device)
            pe, re = errors(pred[:, 1:], truth[:, 1:])
            rows[-1]['translation_mm'] = pe[0].cpu().tolist()
            rows[-1]['rotation_deg'] = re[0].cpu().tolist()
            return pred, grip

        def instrumented_loader(*unused_args, **unused_kwargs):
            ev = original_loader(str(path), args.vae_variant, device, torch.bfloat16)
            original_decode = ev.model.decode

            def observed_decode(*decode_args, **decode_kwargs):
                reconstruction = original_decode(*decode_args, **decode_kwargs)
                i, w = selected[len(rows)]
                target = cache.codec.encode(torch.from_numpy(w.pose), torch.from_numpy(w.gripper)).unsqueeze(0).to(device)
                total, components = core.compute_reconstruction_loss(reconstruction.float(), target.float(), masks, args)
                rows.append({'index': i, 'dataset': w.episode.dataset_root, 'episode': w.episode.episode_index,
                             'task': w.episode.task, 'start': w.start, 'total_l1': total.item(),
                             **{f'{k}_l1': v.item() for k, v in components.items()}})
                return reconstruction

            ev.model.decode = observed_decode
            # cached_evaluate copies the source state; return this exact instance as source too.
            return ev

        source = original_loader(str(path), args.vae_variant, torch.device('cpu'), torch.bfloat16)
        core.load_base_vae = instrumented_loader
        cache.codec.decode = observed_action_decode
        if rank == 0:
            print('EVALUATING', name, flush=True)
        try:
            result = cached_evaluate(cache, source, store, episodes, stats, args, device, torch.float32)
        finally:
            core.load_base_vae = original_loader
            cache.codec.decode = original_codec_decode
        assert len(rows) == len(selected)
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        if rank == 0:
            all_rows = sorted([r for shard in gathered for r in shard], key=lambda r: r['index'])
            assert [r['index'] for r in all_rows] == list(range(800))
            result['map_l1'] = {k: sum(r[k] for r in all_rows)/800 for k in ('total_l1','center_l1','direction_l1','gripper_l1')}
            result['checkpoint'] = str(path)
            result['windows'] = all_rows
            (opts.output/f'{name}.json').write_text(json.dumps(result, indent=2))
            summary[name] = {k: v for k, v in result.items() if k != 'windows'}
            (opts.output/'summary.json').write_text(json.dumps(summary, indent=2))
            print('RESULT', name, json.dumps(summary[name]), flush=True)
        del source
        dist.barrier()
    if rank == 0:
        print('ALL_COMPLETE', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
