"""Paired precision ablation; no training updates or simulator rollouts."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--prepare', action='store_true')
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path)
p.add_argument('--small', action='store_true')
p.add_argument('--full-chain', action='store_true', help='Compare complete BF16/FP32 chains and save both native latents')
p.add_argument('--shard', type=int, default=0)
p.add_argument('--num-shards', type=int, default=1)
p.add_argument('--previous', type=Path)
p.add_argument('--memory-fraction', type=float, default=0.20)
a = p.parse_args()
sys.path[:0] = [str(a.root), str(a.root/'src')]
try:
    import precision_core_snapshot as core
except ModuleNotFoundError:
    import finetune_rothko_vae_decoder as core
from fastwam.representations.libero_rothko import LiberoRothkoCodec

torch.set_num_threads(2)
torch.manual_seed(42)
run = a.root/'runs/libero_rothko_vae_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2'
if a.prepare:
    c = argparse.Namespace(**json.loads((run/'training_config.json').read_text())['arguments'])
    _, heldout = core.discover_episodes(c)
    windows = core.sample_uniform_windows_per_episode(core.EpisodeStore(80), heldout, 10, 16)
    manifest = [{'dataset': w.episode.dataset_root, 'episode': w.episode.episode_index,
                 'task': w.episode.task, 'start': w.start} for w in windows]
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    samples = [{'pose': torch.from_numpy(w.pose.copy()), 'gripper': torch.from_numpy(w.gripper.copy()),
                'suite': Path(w.episode.dataset_root).name} for w in windows]
    assert len(samples) == 800
    torch.save({'samples': samples, 'manifest': manifest, 'sha256': digest,
                'stats': core.load_norm_stats(c.norm_stats_path, c), 'variant': c.vae_variant}, a.input)
    print('PREPARED', len(samples), digest, flush=True)
    sys.exit(0)


def metrics(pred, truth):
    pe = (pred[..., :3].double()-truth[..., :3].double()).norm(dim=-1)*1000
    q = torch.nn.functional.normalize(pred[..., 3:].double(), dim=-1)
    t = torch.nn.functional.normalize(truth[..., 3:].double(), dim=-1)
    t = torch.where((q*t).sum(-1, keepdim=True)<0, -t, t)
    re = 4*torch.atan2((q-t).norm(dim=-1), (q+t).norm(dim=-1))*(180/torch.pi)
    assert torch.isfinite(pe).all() and torch.isfinite(re).all()
    return {'translation_mm': pe[0].tolist(), 'rotation_deg': re[0].tolist()}


def summarize(rows):
    result = {}
    for mode in rows[0]['errors']:
        result[mode] = {}
        for metric in ('translation_mm', 'rotation_deg'):
            x = torch.tensor([r['errors'][mode][metric] for r in rows], dtype=torch.float64)
            result[mode][metric] = {'mean': x.mean().item(), 'p95': x.quantile(.95).item(),
                                  'max': x.max().item(), 'first8_mean': x[:, :8].mean().item()}
    return result


@torch.inference_mode()
def evaluate():
    a.output.mkdir(parents=True, exist_ok=False)
    data = torch.load(a.input, map_location='cpu', weights_only=False)
    device = torch.device('cuda:0')
    # Bound this diagnostic's allocator while sharing a GPU with training.
    assert 0 < a.memory_fraction <= 1
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    checkpoint = torch.load(run/'checkpoint_step007498.pt', map_location='cpu', weights_only=False)
    assert checkpoint['step'] == 7498
    assert all(v.dtype == torch.float32 for k in ('decoder', 'conv2') for v in checkpoint[k].values() if v.is_floating_point())
    base = a.root/'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
    fp = core.load_base_vae(str(base), data['variant'], device, torch.float32)
    fp.model.decoder.load_state_dict(checkpoint['decoder'], strict=True)
    fp.model.conv2.load_state_dict(checkpoint['conv2'], strict=True)
    bf = core.load_base_vae(str(base), data['variant'], device, torch.bfloat16)
    bf.model.decoder.load_state_dict(checkpoint['decoder'], strict=True)
    bf.model.conv2.load_state_dict(checkpoint['conv2'], strict=True)
    codec = LiberoRothkoCodec(norm_stats=data['stats'], expected_action_horizon=16,
                             decode_mode='robust_joint', decode_anchor_alpha=0, decode_block_grid=4)
    indices = list(range(0, 800, 10)) if a.small else list(range(800))
    if a.full_chain:
        # Round-robin suites so partial reports do not consist of only libero10.
        groups = {}
        for i in indices:
            groups.setdefault(data['samples'][i]['suite'], []).append(i)
        indices = [g[j] for j in range(max(map(len, groups.values())))
                   for g in groups.values() if j < len(g)]
        (a.output/'latent_cache').mkdir()
    assert 0 <= a.shard < a.num_shards
    if a.previous:
        previous_meta = json.loads((a.previous/'manifest.json').read_text())
        assert previous_meta['manifest_sha256'] == data['sha256']
        assert previous_meta['note'].startswith('A: BF16 encoder/latent/decoder; E: FP32')
        done = {r['index'] for r in json.loads((a.previous/'per_window_errors.json').read_text())}
        indices = [i for i in indices if i not in done]
    indices = indices[a.shard::a.num_shards]
    metadata = {'manifest_sha256': data['sha256'], 'indices': indices, 'manifest': data['manifest'],
                'checkpoint_step': 7498, 'target': 'shared CPU policy codec FP32',
                'tf32': False, 'decode_mode': 'robust_joint', 'anchor_alpha': 0,
                'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
                'note': 'C latent cast BF16 before BF16 decode; D uses identical BF16 latent promoted to FP32'}
    metadata.update(shard=a.shard, num_shards=a.num_shards, previous=str(a.previous))
    if a.full_chain:
        metadata['note'] = 'A: BF16 encoder/latent/decoder; E: FP32 encoder/latent/decoder, no autocast or TF32. Shared robust_joint anchor0.'
        metadata['cache'] = 'latent_cache/windowNNNN.pt; native FP32 and BF16 independently encoded; includes pose and gripper'
        metadata['base_vae'] = str(base)
        metadata['decoder_checkpoint'] = str(run/'checkpoint_step007498.pt')
    (a.output/'manifest.json').write_text(json.dumps(metadata, indent=2))
    rows = []
    started = time.monotonic()
    for i in indices:
        s = data['samples'][i]
        target = codec.encode(s['pose'].float(), s['gripper'].float()).unsqueeze(0).to(device)
        truth = s['pose'].float().unsqueeze(0).to(device)
        za = core.vae_encode(bf.model, target.bfloat16(), bf.scale)
        if not a.full_chain:
            with core.autocast_context(device, True):
                zb = core.vae_encode(fp.model, target, fp.scale)
        zc = core.vae_encode(fp.model, target, fp.scale)
        assert zc.dtype == torch.float32 and za.dtype == torch.bfloat16
        if a.full_chain:
            torch.save({'index': i, 'manifest_sha256': data['sha256'],
                        'latent_fp32': zc.cpu(), 'latent_bf16': za.cpu(),
                        'pose': s['pose'], 'gripper': s['gripper']},
                       a.output/'latent_cache'/f'window{i:04d}.pt')
        row = {'index': i, 'suite': s['suite'], 'errors': {},
               'latent_mae_vs_A': {'C': (zc.float()-za.float()).abs().mean().item()}}
        variants = [('A_bf16_bf16', bf, za), ('E_fp32_fp32', fp, zc)] if a.full_chain else [
            ('A_bf16_bf16', bf, za), ('B_autocast_bf16', bf, zb.bfloat16()),
            ('C_fp32_bf16', bf, zc.bfloat16()), ('D_bf16_fp32', fp, za.float())]
        for name, model, z in variants:
            raw = model.model.decode(z, model.scale)
            assert raw.dtype == z.dtype, (name, raw.dtype, z.dtype)
            recon = raw.float().clamp(-1, 1)
            pred, _ = codec.decode(recon, truth[:, 0])
            row['errors'][name] = metrics(pred[:, 1:], truth[:, 1:])
        rows.append(row)
        if len(rows) == 1 or len(rows)%10 == 0:
            (a.output/'per_window_errors.json').write_text(json.dumps(rows))
            print('PROGRESS', len(rows), len(indices), 'seconds', round(time.monotonic()-started, 1),
                  'peak_allocated_GiB', round(torch.cuda.max_memory_allocated()/2**30, 2), flush=True)
    report = {'windows': len(rows), 'overall': summarize(rows),
              'by_suite': {s: summarize([r for r in rows if r['suite']==s]) for s in sorted({r['suite'] for r in rows})},
              'seconds': time.monotonic()-started}
    report['paired_vs_A'] = {}
    for mode in rows[0]['errors']:
        if mode == 'A_bf16_bf16':
            continue
        report['paired_vs_A'][mode] = {}
        for metric in ('translation_mm', 'rotation_deg'):
            delta = torch.tensor([r['errors'][mode][metric] for r in rows], dtype=torch.float64)-torch.tensor([r['errors']['A_bf16_bf16'][metric] for r in rows], dtype=torch.float64)
            report['paired_vs_A'][mode][metric] = {'mean_delta': delta.mean().item(), 'fraction_improved': (delta<0).double().mean().item()}
    (a.output/'summary.json').write_text(json.dumps(report, indent=2))
    (a.output/'per_window_errors.json').write_text(json.dumps(rows))
    print('COMPLETE', json.dumps(report['overall']), flush=True)


if __name__ == '__main__':
    evaluate()
