"""Independent shuffled-action cache; copy RGB bits from the verified Goal cache."""
import argparse
import bisect
import copy
import json
import math
import os
from pathlib import Path
import sys
from collections import OrderedDict
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.datasets.latent_cache import LatentCacheReader, sha256_file
from fastwam.representations.libero_rothko import LiberoRothkoCodec, LiberoRothkoCodecConfig
from fastwam.representations.shuffled_raymap import ShuffledRaymapCodec
from fastwam.models.wan22.helpers.loader import _load_registered_model


def config():
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=['task=libero_goal_shuffled_raymap_wan21'])
    return cfg


class NumericWindows:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg.data.train.dataset_dirs[0])
        source = Path(cfg.data.train.source_latent_cache_dir)
        self.metadata = json.loads((source / 'metadata.json').read_text())
        self.source = LatentCacheReader(source, expected_dataset_contract=self.metadata['dataset_contract'])
        codec = LiberoRothkoCodec(config=LiberoRothkoCodecConfig(**OmegaConf.to_container(cfg.data.train.rothko_config, resolve=True)),
                                 norm_stats=cfg.data.train.rothko_norm_stats)
        if codec.metadata() != self.metadata['dataset_contract']['raymap_codec_metadata']:
            raise ValueError('Source cache codec does not match baseline')
        self.codec = ShuffledRaymapCodec(codec)
        self.episodes = sorted((json.loads(x) for x in (self.root / 'meta/episodes.jsonl').read_text().splitlines()), key=lambda x:x['episode_index'])
        self.offsets = [0]
        for ep in self.episodes:
            self.offsets.append(self.offsets[-1] + ep['length'])
        assert self.offsets[-1] == len(self.source) == 52895
        self.arrays = OrderedDict()

    def pose_grip(self, index):
        ep_index = bisect.bisect_right(self.offsets, index) - 1
        ep = self.episodes[ep_index]
        if ep_index not in self.arrays:
            f = next(self.root.glob(f"data/chunk-*/episode_{ep['episode_index']:06d}.parquet"))
            keys = ['observation.state.ee_pose_wxyz','action.osc_target_pose_wxyz','observation.state.gripper_open','action']
            table = pq.read_table(f, columns=keys)
            self.arrays[ep_index] = [torch.tensor(table[k].to_pylist(), dtype=torch.float32) for k in keys]
            if len(self.arrays) > 3:
                self.arrays.popitem(last=False)
        self.arrays.move_to_end(ep_index)
        state, action, grip, command = self.arrays[ep_index]
        start = index - self.offsets[ep_index]
        idx = torch.arange(start, start + 16).clamp_max(ep['length'] - 1)
        return torch.cat((state[start:start+1], action[idx])), torch.cat((grip[start:start+1].reshape(1,1), command[idx,-1:])).clamp(0,1)

    def raymap(self, index):
        return self.codec.encode(*self.pose_grip(index))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--samples-per-shard', type=int, default=256)
    a = p.parse_args()
    torch.set_num_threads(1); pa.set_cpu_count(1); pa.set_io_thread_count(1)
    rank = int(os.environ.get('RANK',0)); world = int(os.environ.get('WORLD_SIZE',1))
    device = torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK",0)) % torch.cuda.device_count()}')
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('gloo', timeout=timedelta(hours=2))
    cfg = config(); ds = NumericWindows(cfg)
    output = a.output or Path(cfg.data.train.latent_cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    n = min(a.limit, len(ds.source)) if a.limit else len(ds.source)
    count = math.ceil(n/a.samples_per_shard)
    shards = [dict(index=i,start=i*a.samples_per_shard,end=min(n,(i+1)*a.samples_per_shard),file=f'latents-{i:05d}-of-{count:05d}.bin') for i in range(count)]
    meta = copy.deepcopy(ds.metadata)
    meta.update(complete=False,num_samples=n,samples_per_shard=a.samples_per_shard,shards=shards,
                source_task='libero_goal_shuffled_raymap_wan21',created_at_utc=datetime.now(timezone.utc).isoformat(),
                reused_rgb=dict(source_cache=str(ds.source.cache_dir),source_metadata_sha256=sha256_file(ds.source.cache_dir/'metadata.json'),mode='bit_exact_copy'),
                benchmark_only=bool(a.limit))
    meta['dataset_contract']['raymap_codec_metadata'] = ds.codec.metadata()
    if rank == 0:
        if (output/'metadata.json').exists() or (output/'_SUCCESS').exists():
            raise FileExistsError(f'Refusing to overwrite cache {output}')
        (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        torch.save(dict(permutation=ds.codec.permutation,inverse=ds.codec.inverse),output/'permutation.pt')
    if world > 1:dist.barrier()
    vaep = Path('checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    vae = _load_registered_model(str(vaep),'wan_video_vae',torch_dtype=torch.bfloat16,device=str(device)).eval().requires_grad_(False)
    shape = tuple(meta['latent_shape'])
    with torch.inference_mode():
        # Per-worker online sanity: unchanged original action cache is bit-exact.
        probe = min(rank * 31, n-1)
        pose, grip = ds.pose_grip(probe)
        original = ds.codec.base.encode(pose,grip)
        z = vae.encode(original[None].to(device,torch.bfloat16),device=str(device),tiled=False)[0].cpu()
        torch.testing.assert_close(z,ds.source[probe][1],rtol=0,atol=0)
        for shard in shards[rank::world]:
            final = output/shard['file']; partial = final.with_suffix('.partial')
            if final.exists() or partial.exists():raise FileExistsError(final)
            buf = np.memmap(partial,mode='w+',dtype=np.uint16,shape=(shard['end']-shard['start'],2,*shape))
            for start in range(shard['start'],shard['end'],a.batch_size):
                indices = list(range(start,min(start+a.batch_size,shard['end'])))
                pixels = torch.stack([ds.raymap(i) for i in indices]).to(device,torch.bfloat16)
                latents = vae.encode(pixels,device=str(device),tiled=False).cpu().contiguous()
                assert torch.isfinite(latents).all()
                for j,index in enumerate(indices):
                    buf[index-shard['start'],0] = ds.source[index][0].contiguous().view(torch.uint16).numpy()
                    buf[index-shard['start'],1] = latents[j].view(torch.uint16).numpy()
            buf.flush();del buf;partial.replace(final)
            print(f'rank={rank} shard={shard["index"]+1}/{count} end={shard["end"]} peak_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}',flush=True)
    if world > 1:dist.barrier()
    if rank == 0:
        for shard in shards:
            assert (output/shard['file']).stat().st_size == (shard['end']-shard['start'])*2*math.prod(shape)*2
        meta.update(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),baseline_online_bit_exact_probes=world)
        (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        (output/'_SUCCESS').write_text('complete\n')
        print(f'COMPLETE {n} windows',flush=True)
    if world > 1:dist.destroy_process_group()

if __name__ == '__main__':main()
