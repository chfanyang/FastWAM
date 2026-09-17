"""Freeze task-balanced monitoring windows from the existing held-out split."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as pq


def build(root, split_path, seed=42):
    split = json.loads(split_path.read_text())
    records = {r['episode_index']: r for r in split['records']}
    groups = defaultdict(list)
    offsets = {}
    offset = 0
    for episode in split['val_episode_indices']:
        record = records[episode]
        offsets[episode] = offset
        offset += record['length']
        groups[record['base_task']].append(episode)
    samples = []
    for task, episodes in sorted(groups.items()):
        rng = random.Random(hashlib.sha256(f'{seed}:{task}'.encode()).hexdigest())
        if len(episodes) < 2:
            raise ValueError(f'Need two held-out episodes for {task}')
        for n, episode in enumerate(rng.sample(sorted(episodes), 2)):
            record = records[episode]
            # Keep all frame starts eligible, including padded tails.
            frame = rng.randrange(record['length'])
            task_index = int(pq.read_table(root / record['data_path'], columns=['task_index'])['task_index'][frame].as_py())
            samples.append(dict(sample_id=f'{task}_ep{episode}_frame{frame}',
                                base_task=task, episode_index=episode, frame_index=frame,
                                task_index=task_index, val_dataset_index=offsets[episode]+frame,
                                diffusion_seed=seed+len(samples), run_visual=(n == 0),
                                image_padding_frames=max(0, frame+17-record['length']),
                                action_padding_steps=max(0, frame+16-record['length'])))
    return dict(version=1, environment='vlabench', seed=seed,
                split_manifest_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                sampling='Two distinct held-out episodes/task; uniform all frame starts, padding allowed',
                val_dataset_length=offset, samples=samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, default=Path('data/vlabench_primitive_ft_lerobot'))
    args = parser.parse_args()
    folder = args.dataset_root / 'manifests'
    payload = build(args.dataset_root, folder / 'episode_split_val01_seed42.json')
    output = folder / 'validation_windows_10task_20samples_seed42.json'
    if output.exists():
        if json.loads(output.read_text()) != payload:
            raise ValueError(f'Refusing to replace a different frozen manifest: {output}')
    else:
        with output.open('x') as f:
            json.dump(payload, f, indent=2)
    print(output)
    print('loss samples:', len(payload['samples']), 'visual samples:', sum(r['run_visual'] for r in payload['samples']))
