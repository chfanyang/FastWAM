#!/usr/bin/env python3
"""Fit q99.95 over ALL training windows; preserve split and absolute RAY0."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from fastwam.realworld.common import load_episode, sha256_file, write_json_checked
from fastwam.representations.rothko import RothkoNormStats, quaternion_wxyz_to_matrix, _center_and_read_masks


def prepare(source, output, data_root):
    source, output, data_root = Path(source), Path(output), Path(data_root)
    manifest = json.loads((source / 'split.json').read_text())
    old = RothkoNormStats.load(source / 'rothko_stats.pt')
    assert old.metadata['split_manifest_sha256'] == sha256_file(source / 'split.json')
    hist = np.zeros((3, 100000), dtype=np.int64)
    n = 0
    maximum = np.zeros(3)
    for record in manifest['episodes']:
        if record['split'] != 'train':
            continue
        ep = data_root / record['path']
        if sha256_file(ep / 'states.npz') != record['states_sha256']:
            raise ValueError(f'Source states changed: {ep}')
        poses, _, _, _ = load_episode(ep)
        starts = np.asarray(record['valid_starts'])
        rotations = quaternion_wxyz_to_matrix(torch.from_numpy(poses[starts, 3:])).numpy()
        future = poses[starts[:, None] + np.arange(1, 17)[None, :], :3]
        relative = np.abs(np.einsum('nji,ntj->nti', rotations, future - poses[starts, None, :3])).reshape(-1, 3)
        if not np.isfinite(relative).all() or np.any(relative >= 1):
            raise ValueError('Invalid relative position or histogram overflow')
        maximum = np.maximum(maximum, relative.max(0))
        for axis in range(3):
            hist[axis] += np.histogram(relative[:, axis], bins=100000, range=(0, 1))[0]
        n += len(relative)
    bounds = np.array([(np.searchsorted(np.cumsum(h), np.ceil(n * .9995), side='left') + 1) / 100000 for h in hist])
    lo, hi = old.lo.clone(), old.hi.clone()
    center, _, _ = _center_and_read_masks(224, 224, center_frac=.5, boundary_margin=0, outer_margin=0, device=lo.device)
    center = torch.cat((center, center), dim=1)
    hi[..., center] = torch.tensor(bounds, dtype=hi.dtype)[None, :, None]
    lo[..., center] = -hi[..., center]
    meta = dict(old.metadata, normalization_method='all_valid_train_windows_symmetric_q99p95_hist100000',
        translation_abs_bounds_xyz_m=bounds.tolist(), translation_quantile=.9995,
        histogram_bins=100000, histogram_max_abs_m=1., num_relative_vectors=n,
        observed_max_abs_xyz_m=maximum.tolist(), source_stats_fingerprint=old.fingerprint())
    new = RothkoNormStats(lo, hi, meta)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('split.json', 'validation_windows.json'):
        value = (source / name).read_bytes()
        dest = output / name
        if dest.exists() and dest.read_bytes() != value:
            raise ValueError(f'Refusing different source manifest: {dest}')
        if not dest.exists():
            dest.write_bytes(value)
    path = output / 'rothko_stats.pt'
    if path.exists():
        if RothkoNormStats.load(path).fingerprint() != new.fingerprint():
            raise ValueError('Existing quantile statistics differ')
    else:
        tmp = path.with_suffix('.partial')
        torch.save(dict(lo=lo, hi=hi, metadata=meta), tmp)
        tmp.replace(path)
    write_json_checked(output / 'rothko_stats.json', meta)
    audit = json.loads((source / 'audit.json').read_text())
    audit.update(relative_position_bounds=bounds.tolist(), normalization_method=meta['normalization_method'],
        source_artifacts=str(source.resolve()), num_relative_vectors=n)
    write_json_checked(output / 'audit.json', audit)
    text = output / 'text'
    if text.exists() or text.is_symlink():
        if text.resolve() != (source / 'text').resolve():
            raise ValueError('Text cache link differs')
    else:
        text.symlink_to((source / 'text').resolve(), target_is_directory=True)
    print(json.dumps(dict(bounds_m=bounds.tolist(), target_vectors=n, fingerprint=new.fingerprint(),
                         absolute_ray0_unchanged=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_v2')
    parser.add_argument('--output', default='data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_q9995_v1')
    parser.add_argument('--data-root', default='data/realworld_data')
    args = parser.parse_args()
    torch.set_num_threads(1)
    prepare(args.source, args.output, args.data_root)
