"""Fixed, auditable window sampling for VLABench VAE validation (stdlib only)."""
import argparse
import bisect
from collections import Counter
import hashlib
import json
from pathlib import Path
import random


def build_manifest(split_path, episode_path, seed=42):
    split = json.loads(split_path.read_text())
    lengths = {r['episode_index']: r['length'] for r in
               map(json.loads, episode_path.read_text().splitlines())}
    tasks = {r['episode_index']: r['base_task'] for r in split['records']}
    layout = [[ep, lengths[ep]] for ep in split['val_episode_indices']]
    total = sum(length for _, length in layout)
    selected = sorted(random.Random(seed).sample(range(total), total // 2))
    ends = []; offset = 0
    for ep, length in layout:
        offset += length; ends.append(offset)
    rows = []
    for index in selected:
        pos = bisect.bisect_right(ends, index)
        ep = layout[pos][0]
        rows.append(dict(val_index=index, episode_index=ep,
                         frame_start=index-(ends[pos-1] if pos else 0), task=tasks[ep]))
    return dict(schema_version=1, sampling='uniform_without_replacement',
                fraction=0.5, rounding='floor', seed=seed, total_windows=total,
                selected_windows=len(selected),
                split_manifest_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                episode_metadata_sha256=hashlib.sha256(episode_path.read_bytes()).hexdigest(),
                episode_layout=layout, indices=selected, windows=rows,
                selected_per_task=dict(Counter(r['task'] for r in rows)))


def load_indices(path, *, total_windows, episode_layout, split_sha256):
    manifest = json.loads(Path(path).read_text())
    if manifest['schema_version'] != 1 or manifest['total_windows'] != total_windows:
        raise ValueError('Validation subset size/schema mismatch')
    if manifest['episode_layout'] != [list(x) for x in episode_layout]:
        raise ValueError('Validation episode order/length mismatch')
    if manifest['split_manifest_sha256'] != split_sha256:
        raise ValueError('Validation split fingerprint mismatch')
    indices = manifest['indices']
    if (not indices or any(type(i) is not int or not 0 <= i < total_windows for i in indices)
            or indices != sorted(set(indices)) or len(indices) != manifest['selected_windows']):
        raise ValueError('Invalid validation window indices')
    return indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-manifest', type=Path, required=True)
    p.add_argument('--episode-metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    report = build_manifest(args.split_manifest, args.episode_metadata, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f: json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps({k:report[k] for k in ['total_windows','selected_windows','seed','selected_per_task']},indent=2))


if __name__ == '__main__':
    main()
