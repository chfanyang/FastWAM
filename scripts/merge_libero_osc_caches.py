"""Merge completed Goal + other3 OSC caches by exact-byte repacking.

Writes a NEW incomplete all4 cache; the ordinary cache precomputer must then
perform its online latent/training_loss verification before this is usable.
"""
import argparse,hashlib,json,math,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASK='libero_all4_rothko_osc_delta_2cam224_full_wan21_1_3b_1e-4'


def copy_segments(segments,start,end,item_bytes,destination):
    """segments: (global start, global end, file, source byte offset)."""
    cursor=start;digest=hashlib.sha256()
    with Path(destination).open('xb') as out:
        for lo,hi,path,byte_offset in segments:
            a,b=max(start,lo),min(end,hi)
            if a>=b:continue
            if a!=cursor:raise ValueError(f'Gap/overlap at {cursor}, got {a}')
            remaining=(b-a)*item_bytes
            with Path(path).open('rb') as source:
                source.seek(byte_offset+(a-lo)*item_bytes)
                while remaining:
                    chunk=source.read(min(8*1024**2,remaining))
                    if not chunk:raise EOFError(path)
                    out.write(chunk);digest.update(chunk);remaining-=len(chunk)
            cursor=b
        if cursor!=end:raise ValueError(f'Incomplete coverage {cursor}/{end}')
    check=hashlib.sha256()
    with Path(destination).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):check.update(chunk)
    if check.digest()!=digest.digest():raise IOError('Copy checksum mismatch')
    return digest.hexdigest()


def main():
    from hydra import compose,initialize_config_dir
    from omegaconf import OmegaConf
    from fastwam.utils.config_resolvers import register_default_resolvers
    from fastwam.datasets.latent_cache import build_dataset_contract,LatentCacheReader
    from fastwam.representations.libero_rothko import LiberoRothkoCodecConfig
    from fastwam.representations.libero_rothko_osc_delta import LiberoOSCDeltaRothkoCodec
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--goal',type=Path,required=True);p.add_argument('--other3',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    os.chdir(ROOT);register_default_resolvers()
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
        cfg=compose(config_name='train',overrides=['task='+TASK])
    codec=LiberoOSCDeltaRothkoCodec(LiberoRothkoCodecConfig(**OmegaConf.to_container(cfg.data.train.rothko_config)),cfg.data.train.rothko_norm_stats,expected_action_horizon=16)
    roots=[str(Path(x).resolve()) for x in cfg.data.train.dataset_dirs]
    counts=[json.loads((Path(x)/'meta/info.json').read_text())['total_frames'] for x in roots]
    expected=build_dataset_contract(dataset_dirs=roots,dataset_length=sum(counts),num_frames=17,video_size=[224,448],raymap_representation=codec.representation,raymap_codec_metadata=codec.metadata(),norm_stats_sha256=codec.norm_stats.fingerprint())
    groups=[];locations={};identity_keys=['cache_version','latent_shape','modalities','dtype','encoding_torch_dtype','model_variant','vae_identity']
    for path in [a.goal.resolve(),a.other3.resolve()]:
        m=json.loads((path/'metadata.json').read_text());contract=m['dataset_contract'];source_roots=[x['path'] for x in contract['dataset_roots']]
        contract_want=dict(expected,dataset_roots=[expected['dataset_roots'][roots.index(x)] for x in source_roots],dataset_length=sum(counts[roots.index(x)] for x in source_roots))
        if contract!=contract_want or m.get('benchmark_only') or not m.get('verification',{}).get('training_loss_bit_exact'):raise ValueError(f'Incompatible/unverified source: {path}')
        LatentCacheReader(path,expected_dataset_contract=contract_want)
        if groups and any(m[k]!=groups[0][1][k] for k in identity_keys):raise ValueError('Sources have different VAE/shape/precision')
        offset=0
        for root in source_roots:
            if root in locations:raise ValueError('Duplicate suite')
            count=counts[roots.index(root)];locations[root]=(path,m,offset,count);offset+=count
        groups.append((path,m))
    if set(locations)!=set(roots):raise ValueError('Must cover all4 exactly once')
    item_bytes=2*math.prod(groups[0][1]['latent_shape'])*2
    segments=[];global_offset=0
    for root in roots:
        path,m,source_start,count=locations[root]
        for shard in m['shards']:
            lo=max(source_start,shard['start']);hi=min(source_start+count,shard['end'])
            if lo<hi:segments.append((global_offset+lo-source_start,global_offset+hi-source_start,path/shard['file'],(lo-shard['start'])*item_bytes))
        global_offset+=count
    segments.sort(key=lambda x:x[0]);size=1024;n=sum(counts);nshards=math.ceil(n/size)
    shards=[dict(index=i,start=i*size,end=min(n,(i+1)*size),file=f'latents-{i:05d}-of-{nshards:05d}.bin') for i in range(nshards)]
    meta=dict(groups[0][1],complete=False,source_task=TASK,num_samples=n,dataset_contract=expected,samples_per_shard=size,shards=shards)
    for k in ['verification','completed_at_utc','source_overrides']:meta.pop(k,None)
    a.output.mkdir(parents=True);(a.output/'metadata.json').write_text(json.dumps(meta,indent=2));hashes={}
    for shard in shards:
        partial=a.output/(shard['file']+'.merge-partial');hashes[shard['file']]=copy_segments(segments,shard['start'],shard['end'],item_bytes,partial);partial.rename(a.output/shard['file']);print('COPIED',shard['end'],'/',n,flush=True)
    (a.output/'merge_provenance.json').write_text(json.dumps({'sources':[str(x[0]) for x in groups],'suite_order':roots,'counts':counts,'sha256':hashes,'byte_exact_copy':True,'requires_online_verification':True},indent=2))
    print('Run precompute_visual_action_latents.py with all4 OSC task and samples-per-shard=1024 to verify; no _SUCCESS created.')

if __name__=='__main__':main()
