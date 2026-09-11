"""Read successful trial IDs + final tqdm counts; exclude initialization waits."""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def stats(values):
    values = sorted(values)
    if not values:
        return None
    def q(p):
        return values[max(0, math.ceil(p * len(values)) - 1)]
    return dict(n=len(values), min=values[0], p50=q(.5), p90=q(.9),
                p95=q(.95), p99=q(.99), max=values[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    records, missing, duplicates = [], [], []
    seen = set()
    for run in sorted(args.root.iterdir()):
        config = run / 'manager_config.yaml'
        if not config.exists():
            continue
        wait_match = re.search(r'^\s+num_steps_wait:\s*(\d+)', config.read_text(), re.M)
        if not wait_match:
            raise ValueError(f'Missing num_steps_wait in {config}')
        wait = int(wait_match[1])
        for path in sorted(run.glob('libero_*/*results.json')):
            result = json.loads(path.read_text())
            suite, task = result['task_suite'], result['task_id']
            log = run / 'task_logs' / f'{suite}_task{task}_gpu{result["gpu_id"]}.log'
            counts = {}
            if log.exists():
                for m in re.finditer(r'Episode (\d+):[^\r\n]*?\|\s*(\d+)/(\d+)', log.read_text()):
                    counts[int(m[1]) - 1] = (int(m[2]), int(m[3]))
            for trial in result['success_episodes']:
                key = (run.name, suite, task, trial)
                if key in seen:
                    duplicates.append(key)
                    continue
                seen.add(key)
                if trial not in counts:
                    missing.append(dict(run=run.name, suite=suite, task=task, trial=trial,
                                        expected_log=str(log)))
                    continue
                n, total = counts[trial]
                if not wait < n <= total:
                    raise ValueError(f'Invalid step count {key}: {n}/{total}, wait={wait}')
                records.append(dict(run=run.name, suite=suite, task=task, trial=trial,
                                    steps=n-wait, wait_steps=wait, original_limit=total-wait,
                                    task_description=result.get('task_description'), log=str(log)))
    grouped, run_grouped, task_grouped = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        grouped[r['suite']].append(r['steps'])
        run_grouped[(r['run'], r['suite'])].append(r['steps'])
        task_grouped[(r['suite'], r['task'])].append(r['steps'])
    thresholds = [150, 200, 220, 250, 300, 350, 400, 450, 500, 550, 600, 650]
    report = dict(
        definition='successful policy env.step calls; final tqdm n minus initialization wait',
        suites={k: stats(v) for k,v in grouped.items()},
        runs={run: {suite:stats(v) for (rr,suite),v in run_grouped.items() if rr==run}
              for run in sorted({r['run'] for r in records})},
        tasks={f'{s}/task{t}':stats(v) for (s,t),v in sorted(task_grouped.items())},
        successful_trials_lost_by_limit={s:{str(t):sum(x>t for x in v) for t in thresholds}
                                        for s,v in grouped.items()},
        missing=missing, duplicate_result_keys=duplicates,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    with (args.output_dir/'successful_trials.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
