#!/usr/bin/env python3
"""Isolated realworld adapter for the LIBERO masked decoder training loop."""
import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts')]
import prepare_all4_absolute_ray0_decoder as reference
core = reference.core
from fastwam.realworld.dataset import RealWorldVideoDataset
from fastwam.realworld.common import load_episode, gripper_open, sha256_file

CONFIG = ROOT / 'configs/vae/realworld_pick_place_q9995_decoder_ep2.json'


def valid_steps(window, horizon=16):
    return torch.arange(1, horizon + 1) + window.start < window.episode.length


class Inputs:
    def __init__(self):
        self.c = json.loads(CONFIG.read_text())
        self.args = SimpleNamespace(**self.c, config=str(CONFIG))
        artifacts = Path(self.c['artifacts'])
        self.ds = RealWorldVideoDataset(dataset_dir=self.c['data_root'],
            split_manifest=artifacts / 'split.json', rothko_norm_stats=artifacts / 'rothko_stats.pt',
            include_text_context=False, latent_cache_dir=artifacts / 'latents_wan21_bf16',
            latent_cache_only=True)
        self.codec = self.ds.codec
        self.cache_sha = sha256_file(artifacts / 'latents_wan21_bf16/metadata.json')
        self.train, self.val, self.refs = [], [], []
        self.arrays, self.full_indices = {}, {}
        offset = 0
        for i, r in enumerate(self.ds.manifest['episodes']):
            if r['task'] == 'pick_place':
                ep = core.EpisodeRef(str(Path(self.c['data_root']).resolve()), i, r['task'], r['length'])
                path = Path(self.c['data_root']) / r['path']
                assert sha256_file(path / 'states.npz') == r['states_sha256']
                poses, widths, stamps, _ = load_episode(path)
                self.arrays[ep] = (poses, gripper_open(widths, self.ds.threshold)[:, None])
                pool = self.train if r['split'] == 'train' else self.val
                pool.append(ep)
                if r['split'] == 'train':
                    for j, start in enumerate(r['valid_starts']):
                        self.full_indices[core.WindowRef(ep, start)] = offset + j
                    # Preserve timing-gap rejection; add only genuine episode tails.
                    tails = [s for s in range(max(0, ep.length - 16), ep.length)
                             if np.all((np.diff(stamps[s:]) >= .05) & (np.diff(stamps[s:]) <= .15))]
                    self.refs.extend(core.WindowRef(ep, s) for s in r['valid_starts'] + tails)
            if r['split'] == 'train':
                offset += len(r['valid_starts'])
        self.val_windows = []
        for ep in self.val:
            r = self.ds.manifest['episodes'][ep.episode_index]
            starts = np.asarray(r['valid_starts'])
            for j in np.unique(np.rint(np.linspace(0, len(starts)-1, 5)).astype(int)):
                self.val_windows.append(self.window(core.WindowRef(ep, int(starts[j]))))
        self.missing = [r for r in self.refs if r not in self.full_indices]
        self.missing += [core.WindowRef(w.episode, w.start) for w in self.val_windows]
        self.extra = {}
        self.reader = self
        self.roots = [str(Path(self.c['data_root']).resolve())]
        self.manifest = dict(environment='realworld', task='pick_place',
            train_episodes=[dataclasses.asdict(e) for e in self.train],
            val_episodes=[dataclasses.asdict(e) for e in self.val],
            train_windows=len(self.refs), full_windows=len(self.full_indices),
            padded_tail_windows=len(self.refs)-len(self.full_indices),
            stats_fingerprint=self.codec.norm_stats.fingerprint(), cache_metadata_sha256=self.cache_sha,
            base_vae_sha256=sha256_file(self.args.base_vae), config_sha256=sha256_file(CONFIG),
            padding_policy='episode_tail_edge_repeat_realworld_t_plus_1',
            reconstruction_loss_includes_padding=False, validation_metrics_exclude_padding=True,
            supervision='valid_decoded_frames_including_absolute_RAY0',
            train_refs=[(r.episode.episode_index, r.start) for r in self.refs],
            val_refs=[(w.episode.episode_index, w.start) for w in self.val_windows])
        self.identity = reference.digest_obj(self.manifest)
        assert len(self.train) == 99 and len(self.val) == 2
        assert len(self.full_indices) == 19042

    def window(self, ref):
        p, g = self.arrays[ref.episode]
        indices = np.minimum(np.arange(ref.start, ref.start + 17), ref.episode.length - 1)
        return core.ActionWindow(ref.episode, ref.start, p[indices].copy(), g[indices].copy())

    def target(self, window):
        return self.codec.encode(torch.from_numpy(window.pose), torch.from_numpy(window.gripper))

    def index(self, ref):
        return ref

    def __getitem__(self, ref):
        if ref in self.full_indices:
            return self.ds.latent_cache[self.full_indices[ref]]
        return None, self.extra[(ref.episode.episode_index, ref.start)]

    @torch.no_grad()
    def prepare_extra(self, device, rank, world):
        output = Path(self.args.supplemental_cache)
        output.mkdir(parents=True, exist_ok=True)
        vae = core.load_base_vae(self.args.base_vae, self.args.vae_variant, device, torch.bfloat16)
        vae.eval().requires_grad_(False)
        # Online equivalence for reused full-window ray latents, one probe/rank.
        ref = self.refs[rank * 17]
        z = vae.encode(self.target(self.window(ref))[None].to(device, torch.bfloat16), device=device, tiled=False)[0].cpu()
        torch.testing.assert_close(z, self[ref][1], rtol=0, atol=0)
        local = {}
        selected = self.missing[rank::world]
        for j in range(0, len(selected), 2):
            group = selected[j:j+2]
            target = torch.stack([self.target(self.window(r)) for r in group]).to(device, torch.bfloat16)
            latents = vae.encode(target, device=device, tiled=False).cpu()
            for r, latent in zip(group, latents):
                local[(r.episode.episode_index, r.start)] = latent.contiguous()
        path = output / f'rank{rank}.pt'
        tmp = path.with_suffix('.tmp')
        torch.save(dict(identity=self.identity, latents=local), tmp)
        tmp.replace(path)
        del vae
        torch.cuda.empty_cache()
        torch.distributed.barrier()
        for r in range(world):
            payload = torch.load(output / f'rank{r}.pt', map_location='cpu', weights_only=False)
            assert payload['identity'] == self.identity
            self.extra.update(payload['latents'])
        assert len(self.extra) == len(self.missing)
        if rank == 0:
            reference.atomic(output / 'metadata.json', dict(identity=self.identity,
                num_samples=len(self.extra), full_cache_bit_exact_probes=world, complete=True))
            print(f'Supplemental cache ready: {len(self.extra)} tail/validation windows; {world} full-cache probes bit-exact', flush=True)

    def epoch_batches(self, refs, start_step, steps, effective, world, rank, device):
        for step in range(start_step, steps):
            for acc in range(self.args.grad_accum_steps):
                offset = step * effective + acc * self.args.batch_size * world + rank * self.args.batch_size
                batch = refs[offset:offset + self.args.batch_size]
                windows = [self.window(r) for r in batch]
                self.current_valid_frames = torch.stack([torch.cat((torch.ones(1, dtype=torch.bool), valid_steps(w))) for w in windows]).to(device)
                yield (torch.stack([self.target(w) for w in windows]).to(device),
                       torch.stack([self[r][1] for r in batch]).to(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    opts = parser.parse_args()
    inputs = Inputs()
    if not opts.execute:
        print(json.dumps(dict(inputs.manifest, train_refs='omitted',
            steps=2 * math.ceil(len(inputs.refs)/64)), indent=2))
        return
    assert int(os.environ.get('WORLD_SIZE', '0')) == 8
    _, rank, world, local_rank = core._init_distributed()
    output = Path(inputs.args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output}')
    torch.distributed.barrier()
    inputs.prepare_extra(torch.device(f'cuda:{local_rank}'), rank, world)
    a = inputs.args
    a.cached_prefetch = True
    a.parallel_cached_eval = True
    a.independent_full_checkpoints = True
    core.discover_episodes = lambda *xs, **kw: (inputs.train, inputs.val)
    core.enumerate_window_refs = lambda *xs: inputs.refs
    core.cached_epoch_batches = inputs.epoch_batches
    core.load_norm_stats = lambda *xs: inputs.codec.norm_stats
    core.compute_reconstruction_loss = lambda pred, target, masks, args: reference.reconstruction_loss_valid_frames(pred, target, masks, args, inputs.current_valid_frames)
    reference.valid_steps = valid_steps
    core.evaluate = lambda *xs: reference.evaluate(inputs, *xs)
    core.vae_encode = lambda *xs: (_ for _ in ()).throw(RuntimeError('Unexpected encoder use in decoder training'))
    old_init = core.init_wandb
    def init(args, metadata, rank):
        metadata.update(environment='realworld', encoding='absolute_RAY0_future_measured_poses_relative_to_current',
            realworld_identity=inputs.identity, norm_stats_fingerprint=inputs.codec.norm_stats.fingerprint(),
            codec_metadata=inputs.codec.metadata(), reconstruction_loss_includes_padding=False,
            validation_metrics_exclude_padding=True, cache_metadata_sha256=inputs.cache_sha,
            padding_policy=inputs.manifest['padding_policy'])
        if rank == 0:
            reference.atomic(Path(args.output_dir) / 'training_config.json', metadata)
            reference.atomic(Path(args.output_dir) / 'data_manifest.json', inputs.manifest)
        return old_init(args, metadata, rank)
    core.init_wandb = init
    core.train(a)


if __name__ == '__main__':
    main()
