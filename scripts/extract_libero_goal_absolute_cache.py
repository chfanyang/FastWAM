#!/usr/bin/env python3
"""Copy Goal's contiguous range from the verified all-suite absolute cache.

Writes a NEW, incomplete cache. Run precompute_visual_action_latents.py with
the Goal task and shard size 1024 afterwards: it reuses these bytes and performs
the usual online latent and training-loss checks before writing _SUCCESS.
"""
import bisect
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = 'libero_goal_rothko_all_absolute_2cam224_full_wan21_1_3b_1e-4'


def copy_range(source, shards, start, end, item_bytes, destination):
    """Copy a sample range across shard boundaries without interpreting floats."""
    starts = [s['start'] for s in shards]
    digest = hashlib.sha256()
    with destination.open('xb') as writer:
        while start < end:
            shard = shards[bisect.bisect_right(starts, start)-1]
            stop = min(end, shard['end'])
            if stop <= start:
                raise ValueError('Non-contiguous source shards')
            remaining = (stop-start)*item_bytes
            with (source/shard['file']).open('rb') as reader:
                reader.seek((start-shard['start'])*item_bytes)
                while remaining:
                    chunk = reader.read(min(8*1024**2, remaining))
                    if not chunk:
                        raise EOFError('Truncated source shard')
                    writer.write(chunk); digest.update(chunk); remaining -= len(chunk)
            start = stop
    actual = hashlib.sha256()
    with destination.open('rb') as reader:
        for chunk in iter(lambda: reader.read(8*1024**2), b''):
            actual.update(chunk)
    if actual.digest() != digest.digest():
        raise ValueError('Copied shard bytes changed')
    return actual.hexdigest()


def main():
    import os
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from fastwam.datasets.latent_cache import build_dataset_contract
    from fastwam.representations.libero_rothko import LiberoRothkoCodecConfig
    from fastwam.representations.libero_rothko_all_absolute import LiberoAllAbsoluteRothkoCodec
    from fastwam.representations.rothko import RothkoNormStats
    from libero_all_absolute_workflow import check_inputs, CACHE
    os.chdir(ROOT)
    checked = check_inputs(require_cache=True)
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=['task='+TASK])
    OmegaConf.resolve(cfg)
    target = Path(cfg.data.train.latent_cache_dir).resolve()
    if target.exists():
        raise FileExistsError(target)
    source = json.loads((CACHE/'metadata.json').read_text())
    roots = source['dataset_contract']['dataset_roots']
    counts = [json.loads((Path(r['path'])/'meta/info.json').read_text())['total_frames'] for r in roots]
    goal = str(Path(cfg.data.train.dataset_dirs[0]).resolve())
    assert len(cfg.data.train.dataset_dirs)==1
    ordinal = [r['path'] for r in roots].index(goal)
    offset, count = sum(counts[:ordinal]), counts[ordinal]
    assert offset==120538 and count==52895
    stats = RothkoNormStats.load(cfg.data.train.rothko_norm_stats)
    codec = LiberoAllAbsoluteRothkoCodec(LiberoRothkoCodecConfig(
        **OmegaConf.to_container(cfg.data.train.rothko_config)), stats)
    contract = build_dataset_contract(dataset_dirs=[goal], dataset_length=count,
        num_frames=cfg.data.train.num_frames, video_size=list(cfg.data.train.video_size),
        raymap_representation=cfg.data.train.raymap_representation,
        raymap_codec_metadata=codec.metadata(), norm_stats_sha256=stats.fingerprint())
    expected = dict(source['dataset_contract'], dataset_roots=[roots[ordinal]], dataset_length=count)
    if contract != expected:
        raise ValueError('Goal encoding contract differs from all-suite cache')
    size = 1024
    shards = [dict(index=i, start=i*size, end=min((i+1)*size,count),
                   file=f'latents-{i:05d}-of-{math.ceil(count/size):05d}.bin')
              for i in range(math.ceil(count/size))]
    target.mkdir()
    metadata = dict(source, complete=False, source_task=TASK, num_samples=count,
                    dataset_contract=contract, samples_per_shard=size, shards=shards)
    for key in ('verification', 'completed_at_utc', 'source_overrides'):
        metadata.pop(key, None)
    (target/'metadata.json').write_text(json.dumps(metadata, indent=2))
    item_bytes = 2*math.prod(source['latent_shape'])*2
    hashes = {}
    for shard in shards:
        partial = target/(shard['file']+'.extract-partial')
        hashes[shard['file']] = copy_range(CACHE, source['shards'],
            offset+shard['start'], offset+shard['end'], item_bytes, partial)
        partial.rename(target/shard['file'])
        print(f"Copied {shard['end']}/{count} samples", flush=True)
    (target/'extraction_provenance.json').write_text(json.dumps(dict(
        source=str(CACHE), source_metadata_sha256=checked['cache_metadata_sha256'],
        source_start=offset, source_end=offset+count, target=str(target),
        sha256=hashes, verified_copy_bytes=True, requires_online_verification=True), indent=2))
    print('Copied exact bytes; online verification still required.', flush=True)


if __name__=='__main__':
    main()
