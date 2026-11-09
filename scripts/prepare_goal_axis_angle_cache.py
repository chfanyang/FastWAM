"""Independent, resumable axis-angle action cache with bit-exact RGB reuse.

Modes do not start training. Workers own disjoint shards; completed shards and
their hashes can be reused when changing worker/GPU counts after stopping the
old workers. Diagnostic caches cannot be consumed by the training dataset.
"""
import argparse
import bisect
from collections import OrderedDict
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
TASK='libero_goal_axis_angle_constant_wan21'


def write_json(path,data):
    partial=path.with_suffix('.json.partial')
    partial.write_text(json.dumps(data,indent=2)+'\n');partial.replace(path)


def config():
    from hydra import compose,initialize_config_dir
    from fastwam.utils.config_resolvers import register_default_resolvers
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
        return compose(config_name='train',overrides=['task='+TASK])


class Windows:
    def __init__(self):
        import torch
        from omegaconf import OmegaConf
        from fastwam.datasets.latent_cache import LatentCacheReader,build_dataset_contract
        from fastwam.representations.libero_rothko import LiberoRothkoCodec,LiberoRothkoCodecConfig
        from fastwam.representations.libero_axis_angle import LiberoAxisAngleCodec
        self.cfg=config();self.root=ROOT/self.cfg.data.train.dataset_dirs[0]
        self.source_path=ROOT/self.cfg.data.train.source_latent_cache_dir
        self.metadata=json.loads((self.source_path/'metadata.json').read_text())
        self.base=LiberoRothkoCodec(LiberoRothkoCodecConfig(**OmegaConf.to_container(self.cfg.data.train.rothko_config,resolve=True)),
            ROOT/self.cfg.data.train.rothko_norm_stats,expected_action_horizon=16)
        expected=build_dataset_contract(dataset_dirs=[str(self.root)],dataset_length=52895,
            num_frames=17,video_size=[224,448],raymap_representation='libero_rothko',
            raymap_codec_metadata=self.base.metadata(),norm_stats_sha256=self.base.norm_stats.fingerprint())
        self.source=LatentCacheReader(self.source_path,expected_dataset_contract=expected)
        if self.metadata['model_variant']!='wan2.1-t2v-1.3b' or self.metadata['encoding_torch_dtype']!='torch.bfloat16':
            raise ValueError('Source RGB cache must be the original BF16 Wan2.1 h16 cache.')
        self.codec=LiberoAxisAngleCodec(self.base)
        self.episodes=sorted([json.loads(x) for x in (self.root/'meta/episodes.jsonl').read_text().splitlines()],key=lambda e:e['episode_index'])
        self.offsets=[0]
        for ep in self.episodes:self.offsets.append(self.offsets[-1]+ep['length'])
        assert self.offsets[-1]==len(self.source)==52895
        self.arrays=OrderedDict()

    def pose_grip(self,index):
        import torch
        import pyarrow.parquet as pq
        i=bisect.bisect_right(self.offsets,index)-1;ep=self.episodes[i]
        if i not in self.arrays:
            table=pq.read_table(next(self.root.glob(f'data/chunk-*/episode_{ep["episode_index"]:06d}.parquet')),
                columns=['observation.state.ee_pose_wxyz','action.osc_target_pose_wxyz','observation.state.gripper_open','action'])
            self.arrays[i]=[torch.tensor(table[k].to_pylist(),dtype=torch.float32) for k in table.column_names]
            if len(self.arrays)>3:self.arrays.popitem(last=False)
        self.arrays.move_to_end(i)
        state,action,grip,command=self.arrays[i]
        start=index-self.offsets[i];future=torch.arange(start,start+16).clamp_max(ep['length']-1)
        return torch.cat((state[start:start+1],action[future])),torch.cat((grip[start:start+1].reshape(1,1),command[future,-1:])).clamp(0,1)

    def pixels(self,index):
        return self.codec.encode(*self.pose_grip(index))


def initialize(a):
    from fastwam.datasets.latent_cache import sha256_file
    ds=Windows();out=a.output or ROOT/ds.cfg.data.train.latent_cache_dir
    if out.exists():raise FileExistsError(f'Cache already exists; use worker/finalize/check to resume: {out}')
    n=min(a.limit,len(ds.source)) if a.limit else len(ds.source)
    count=math.ceil(n/a.samples_per_shard)
    meta=copy.deepcopy(ds.metadata)
    vae=ROOT/'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
    assert sha256_file(vae)==meta['vae_identity']['sha256']
    meta.update(complete=False,num_samples=n,samples_per_shard=a.samples_per_shard,
        shards=[dict(index=i,start=i*a.samples_per_shard,end=min(n,(i+1)*a.samples_per_shard),file=f'latents-{i:05d}-of-{count:05d}.bin') for i in range(count)],
        source_task=TASK,created_at_utc=datetime.now(timezone.utc).isoformat(),
        benchmark_only=bool(a.limit),encoding_batch_size=a.batch_size,
        codec_source_sha256=sha256_file(ROOT/'src/fastwam/representations/libero_axis_angle.py'),
        reused_rgb=dict(source_cache=str(ds.source_path),source_metadata_sha256=sha256_file(ds.source_path/'metadata.json'),mode='bit_exact_copy'))
    meta.pop('verification',None)
    meta['source_overrides']=['task='+TASK,'action_codec=libero_axis_angle_constant',
        'rgb_latents=bit_exact_source_copy','action_encoding_torch_dtype=torch.bfloat16',
        f'action_encoding_batch_size={a.batch_size}']
    meta['dataset_contract'].update(raymap_representation=ds.codec.representation,raymap_codec_metadata=ds.codec.metadata())
    out.mkdir(parents=True)
    (out/'receipts').mkdir();(out/'probes').mkdir()
    write_json(out/'metadata.json',meta)
    print(json.dumps(dict(output=str(out),windows=n,shards=count,bytes=n*2*math.prod(meta['latent_shape'])*2,diagnostic=bool(a.limit))),flush=True)


def authenticated(out):
    from fastwam.datasets.latent_cache import sha256_file
    ds=Windows();meta=json.loads((out/'metadata.json').read_text())
    if meta['codec_source_sha256']!=sha256_file(ROOT/'src/fastwam/representations/libero_axis_angle.py'):
        raise ValueError('Codec source changed after cache initialization; choose a new cache.')
    if meta['dataset_contract']['raymap_codec_metadata']!=ds.codec.metadata():raise ValueError('Codec identity mismatch.')
    if meta['reused_rgb']['source_metadata_sha256']!=sha256_file(ds.source_path/'metadata.json'):
        raise ValueError('Source cache identity changed.')
    if meta['vae_identity']!=ds.metadata['vae_identity']:raise ValueError('VAE identity mismatch.')
    return ds,meta


def verify_shard(out,meta,shard,hash_check=True):
    from fastwam.datasets.latent_cache import sha256_file
    final=out/shard['file'];receipt=out/'receipts'/f'{shard["index"]:05d}.json'
    expected=(shard['end']-shard['start'])*2*math.prod(meta['latent_shape'])*2
    if not final.is_file() or final.stat().st_size!=expected or not receipt.is_file():raise ValueError(f'Incomplete shard: {final}')
    r=json.loads(receipt.read_text())
    if r['codec_source_sha256']!=meta['codec_source_sha256'] or not r['rgb_bit_exact'] or r['windows']!=shard['end']-shard['start']:
        raise ValueError(f'Invalid receipt: {receipt}')
    if hash_check and sha256_file(final)!=r['sha256']:raise ValueError(f'Corrupt shard: {final}')


def worker(a):
    import numpy as np
    import torch
    import pyarrow as pa
    from filelock import FileLock
    from fastwam.datasets.latent_cache import sha256_file
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    torch.set_num_threads(1);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    ds,meta=authenticated(a.output)
    if a.batch_size!=meta['encoding_batch_size']:raise ValueError('Use the initialized encoding batch size.')
    if (a.output/'_SUCCESS').exists():print('Cache already complete.',flush=True);return
    torch.cuda.set_device(0)
    vae=_load_registered_model(str(ROOT/'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'),
        'wan_video_vae',torch_dtype=torch.bfloat16,device='cuda:0').eval().requires_grad_(False)
    shape=tuple(meta['latent_shape']);torch.cuda.reset_peak_memory_stats()
    n=meta['num_samples'];started=time.monotonic()
    with torch.inference_mode():
        probe=[0,min(137,n-1),n//2,n-1][a.rank%4]
        pose,grip=ds.pose_grip(probe)
        baseline=vae.encode(ds.base.encode(pose,grip)[None].to('cuda:0',torch.bfloat16),device='cuda:0',tiled=False)[0].cpu()
        torch.testing.assert_close(baseline,ds.source[probe][1],rtol=0,atol=0)
        probes=[probe,min(probe+1,n-1)]
        pixels=torch.stack([ds.pixels(i) for i in probes]).to('cuda:0',torch.bfloat16)
        encoded=vae.encode(pixels,device='cuda:0',tiled=False).cpu()
        for j in range(2):
            single=vae.encode(pixels[j:j+1],device='cuda:0',tiled=False)[0].cpu()
            torch.testing.assert_close(encoded[j],single,rtol=0,atol=0)
        write_json(a.output/'probes'/f'worker{a.rank}_pid{os.getpid()}.json',dict(rank=a.rank,world=a.world,
            baseline_bit_exact=True,axis_angle_batch1_batch2_bit_exact=True,indices=probes))
        for shard in meta['shards'][a.rank::a.world]:
            final=a.output/shard['file']
            with FileLock(str(final)+'.lock'):
                if final.exists():verify_shard(a.output,meta,shard);print(f'SKIP shard={shard["index"]}',flush=True);continue
                partial=final.with_suffix(f'.pid{os.getpid()}.partial')
                if partial.exists():raise FileExistsError(partial)
                buf=np.memmap(partial,mode='w+',dtype=np.uint16,shape=(shard['end']-shard['start'],2,*shape))
                for start in range(shard['start'],shard['end'],a.batch_size):
                    indices=list(range(start,min(start+a.batch_size,shard['end'])))
                    x=torch.stack([ds.pixels(i) for i in indices]).to('cuda:0',torch.bfloat16)
                    z=vae.encode(x,device='cuda:0',tiled=False).cpu().contiguous()
                    if tuple(z.shape)!=(len(indices),*shape) or not torch.isfinite(z).all():raise ValueError('Invalid encoded action latent.')
                    for j,index in enumerate(indices):
                        rgb=ds.source[index][0].contiguous().view(torch.uint16).numpy()
                        buf[index-shard['start'],0]=rgb
                        assert np.array_equal(buf[index-shard['start'],0],rgb)
                        buf[index-shard['start'],1]=z[j].view(torch.uint16).numpy()
                buf.flush();del buf
                checksum=sha256_file(partial)
                partial.replace(final)
                write_json(a.output/'receipts'/f'{shard["index"]:05d}.json',dict(windows=shard['end']-shard['start'],
                    sha256=checksum,rgb_bit_exact=True,codec_source_sha256=meta['codec_source_sha256']))
            print(f'rank={a.rank}/{a.world} shard={shard["index"]+1}/{len(meta["shards"])} end={shard["end"]} elapsed={time.monotonic()-started:.1f}s peakGiB={torch.cuda.max_memory_allocated()/2**30:.2f}',flush=True)


def finalize(a):
    from fastwam.datasets.latent_cache import LatentCacheReader
    _,meta=authenticated(a.output)
    for shard in meta['shards']:verify_shard(a.output,meta,shard)
    probes=list((a.output/'probes').glob('*.json'))
    if not probes:raise ValueError('No online/batch equivalence probes.')
    for p in probes:
        r=json.loads(p.read_text());assert r['baseline_bit_exact'] and r['axis_angle_batch1_batch2_bit_exact']
    meta.update(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),
        verification=dict(all_shard_hashes_checked=True,rgb_all_windows_bit_exact=True,baseline_online_bit_exact=True,
            axis_angle_batch1_batch2_bit_exact=True,probe_records=len(probes)))
    write_json(a.output/'metadata.json',meta)
    if not (a.output/'_SUCCESS').exists():(a.output/'_SUCCESS').write_text('complete\n')
    reader=LatentCacheReader(a.output,expected_dataset_contract=meta['dataset_contract'])
    assert len(reader)==meta['num_samples']
    print(json.dumps(dict(complete=True,windows=len(reader),diagnostic=meta['benchmark_only'],verification=meta['verification'])),flush=True)


def check(a):
    import torch
    from fastwam.datasets.latent_cache import LatentCacheReader
    ds,meta=authenticated(a.output)
    if not meta['complete'] or not (a.output/'_SUCCESS').exists():raise ValueError('Cache incomplete.')
    for shard in meta['shards']:verify_shard(a.output,meta,shard,hash_check=not a.skip_hash)
    reader=LatentCacheReader(a.output,expected_dataset_contract=meta['dataset_contract'])
    indices=sorted(set([0,min(122,len(reader)-1),min(137,len(reader)-1),len(reader)//2,len(reader)-1]))
    online=None
    if a.online:
        from fastwam.models.wan22.helpers.loader import _load_registered_model
        torch.cuda.set_device(0)
        online=_load_registered_model(str(ROOT/'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'),'wan_video_vae',torch_dtype=torch.bfloat16,device='cuda:0').eval().requires_grad_(False)
    with torch.inference_mode():
        for index in indices:
            rgb,action=reader[index]
            torch.testing.assert_close(rgb,ds.source[index][0],rtol=0,atol=0)
            if online is not None:
                encoded=online.encode(ds.pixels(index)[None].to('cuda:0',torch.bfloat16),device='cuda:0',tiled=False)[0].cpu()
                torch.testing.assert_close(encoded,action,rtol=0,atol=0)
    report=dict(complete=True,windows=len(reader),diagnostic=meta['benchmark_only'],indices=indices,
                rgb_bit_exact=True,axis_angle_online_bit_exact=a.online,hashes_checked=not a.skip_hash)
    write_json(a.output/f'check_{"online" if a.online else "cpu"}.json',report)
    print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initialize','worker','finalize','check'])
    p.add_argument('--output',type=Path);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--samples-per-shard',type=int,default=256)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--online',action='store_true');p.add_argument('--skip-hash',action='store_true')
    a=p.parse_args()
    if a.limit<0 or a.batch_size<1 or a.samples_per_shard<1 or not 0<=a.rank<a.world:raise ValueError('Invalid cache arguments.')
    if a.mode=='initialize':initialize(a)
    else:
        if a.output is None:p.error('--output is required')
        {'worker':worker,'finalize':finalize,'check':check}[a.mode](a)


if __name__=='__main__':main()
