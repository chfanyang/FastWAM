"""Deduplicate paired windows and bootstrap episodes within each fixed task."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from compare_libero_rothko_decoders import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    all_rows, batches = [], {}
    for root in args.roots:
        files = sorted(root.glob('*/per_window_errors.json'))
        if len(files) != 4:
            raise ValueError(f'Expected four completed suites: {root}')
        rows = [r for f in files for r in json.loads(f.read_text())]
        assert len(rows) == 800
        batches[root.name] = summarize(rows)
        all_rows.extend(rows)
    unique = {}
    for row in all_rows:
        key = (row['suite'], row['task'], row['episode_index'], row['window_start'])
        unique.setdefault(key, row)
    rows = list(unique.values())
    tasks = defaultdict(lambda: defaultdict(list))
    for row in rows:
        tasks[(row['suite'], row['task'])][row['episode_index']].append(row)
    assert len(tasks) == 40
    rng = np.random.default_rng(20260909)
    bootstrap = {}
    for other in (name for name in rows[0]['errors'] if name != 'joint'):
        bootstrap[other + '_minus_joint'] = {}
        for metric in ('translation_mm', 'rotation_deg'):
            for horizon in (8, 16):
                sums = np.zeros(3000)
                counts = np.zeros(3000)
                point_sum, point_count = 0., 0
                for episodes in tasks.values():
                    cluster = []
                    for group in episodes.values():
                        diffs = [np.mean(r['errors'][other][metric][:horizon]) -
                                 np.mean(r['errors']['joint'][metric][:horizon]) for r in group]
                        cluster.append((sum(diffs), len(diffs)))
                    a = np.asarray(cluster)
                    idx = rng.integers(0, len(a), size=(3000, len(a)))
                    sums += a[idx, 0].sum(1)
                    counts += a[idx, 1].sum(1)
                    point_sum += a[:, 0].sum()
                    point_count += a[:, 1].sum()
                bootstrap[other + '_minus_joint'][f'{metric}_first{horizon}'] = {
                    'mean_difference': point_sum / point_count,
                    'ci95': np.quantile(sums / counts, [.025, .975]).tolist(),
                }
    report = {'sample_draws': len(all_rows), 'unique_windows': len(rows),
              'distinct_episodes': sum(len(g) for g in tasks.values()),
              'future_poses': len(rows) * 16, 'overall': summarize(rows),
              'by_seed': batches,
              'by_suite': {s: summarize([r for r in rows if r['suite'] == s])
                           for s in sorted({r['suite'] for r in rows})},
              'paired_episode_bootstrap': bootstrap,
              'bootstrap_method': '3000 paired resamples of episodes within each of 40 fixed tasks; all windows of a sampled episode stay together'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in ('overall','by_seed','by_suite')}, indent=2))
    for name, stats in {'combined': report['overall'], **batches}.items():
        for mode, metrics in stats.items():
            print(name, mode, {k: {key: v[key] for key in ('mean','p95','first8_mean')} for k,v in metrics.items()})


if __name__ == '__main__':
    main()
