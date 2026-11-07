#!/usr/bin/env python3
"""Prepare only pick_place indices, using unchanged full-training-set q99.95 stats."""
import argparse
import json
from pathlib import Path

from fastwam.realworld.common import sha256_file, write_json_checked, TASKS
from fastwam.representations.rothko import RothkoNormStats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=['insert_pen', 'stack_bowls_three', 'white_board'])
    task = parser.parse_args().task
    source = Path('data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_q9995_v1')
    output = Path(f'data/realworld_artifacts/{task}_q9995_v1')
    manifest = json.loads((source / 'split.json').read_text())
    indices, offsets, probes = {'train': [], 'val': []}, {'train': 0, 'val': 0}, []
    for ep, record in enumerate(manifest['episodes']):
        split, count = record['split'], len(record['valid_starts'])
        if record['task'] == task:
            if split == 'val':
                local = count // 2
                probes.append(dict(sample_id=record['path'].replace('/', '_') + f'_{local}',
                    val_dataset_index=len(indices['val']) + local, episode_index=ep,
                    frame_index=record['valid_starts'][local], task_index=TASKS.index(task),
                    diffusion_seed=42 + len(probes), run_visual=True))
            indices[split].extend(range(offsets[split], offsets[split] + count))
        offsets[split] += count
    stats = RothkoNormStats.load(source / 'rothko_stats.pt')
    digest = sha256_file(source / 'split.json')
    assert len(probes) == 2 and len(indices['train']) > 0
    write_json_checked(output / 'subset.json', dict(version=1, task=task,
        source_split_sha256=digest, norm_stats_fingerprint=stats.fingerprint(), indices=indices))
    write_json_checked(output / 'validation_windows.json', dict(version=1,
        split_manifest_sha256=digest, samples=probes, val_dataset_length=len(indices['val']), environment='realworld'))
    print(f'Prepared {task}: {len(indices["train"])} train windows, {len(indices["val"])} val windows')


if __name__ == '__main__':
    main()
