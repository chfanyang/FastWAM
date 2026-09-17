"""Opt-in task-list wrapper; leave the existing Plus manager/worker unchanged.

Freeze up to ten variants per base-task/category (seed 42). Results retain
official suite/task IDs, so a later ordinary full manager run can fill gaps.
Set PLUS_SUBSET_ONLY=1 to generate/check the manifest without starting workers.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
import hydra
from experiments.libero_plus import run_libero_plus_manager as manager

original_create_rows = manager._create_task_rows
SEED = 42
LIMIT = 10


def subset_rows(cfg, output_dir):
    full_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    full_cfg.MULTIRUN.resume = False
    full_cfg.MULTIRUN.task_start = 0
    full_cfg.MULTIRUN.task_end = None
    full_cfg.MULTIRUN.max_tasks_per_suite = None
    full_rows = original_create_rows(full_cfg, output_dir)
    class_path = ROOT / 'third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json'
    classification = json.loads(class_path.read_text())
    tree = ast.parse((ROOT / 'third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py').read_text())
    base_map = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in n.targets)))
    by_name = {(suite, row['name']): row['category']
               for suite, rows in classification.items() for row in rows}
    groups = defaultdict(list)
    for row in full_rows:
        suite = row['suite']
        name = Path(row['bddl_file']).stem
        matches = [base for base in base_map[suite] if name == base or name.startswith(base + '_')]
        if len(matches) != 1:
            raise ValueError(f'Unexpected base task mapping: {suite}/{name}: {matches}')
        category = by_name[(suite, name)]
        groups[(suite, matches[0], category)].append(dict(row, base_task=matches[0], category=category))
    rng = random.Random(SEED)
    selected_groups = {}
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: r['task_id'])
        selected_groups[key] = rng.sample(rows, min(LIMIT, len(rows)))
    selected = []
    # Preserve suite order; interleave base/category groups within each suite.
    for suite in cfg.MULTIRUN.task_suite_names:
        keys = sorted(k for k in selected_groups if k[0] == suite)
        for offset in range(LIMIT):
            for key in keys:
                if offset < len(selected_groups[key]):
                    selected.append(selected_groups[key][offset])
    if len(selected) != 2757 or len({(r['suite'], r['task_id']) for r in selected}) != 2757:
        raise ValueError(f'Unexpected subset count: {len(selected)}')
    manifest = dict(seed=SEED, max_per_base_category=LIMIT, num_tasks=len(selected),
                    classification_sha256=hashlib.sha256(class_path.read_bytes()).hexdigest(),
                    suite_order=list(cfg.MULTIRUN.task_suite_names), tasks=selected)
    path = output_dir / 'stratified10_seed42_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Frozen selection differs; do not overwrite the existing manifest')
    else:
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    counts = Counter(row['suite'] for row in selected)
    pending = [r for r in selected if not (output_dir / r['suite'] / 'results' / f"task{r['task_id']:04d}.json").exists()]
    print(f'STRATIFIED total={len(selected)} pending={len(pending)} counts={dict(counts)} manifest={path}', flush=True)
    if os.environ.get('PLUS_SUBSET_ONLY') == '1':
        raise SystemExit(0)
    return pending


@hydra.main(version_base='1.3', config_path=str(ROOT / 'configs'), config_name='sim_libero_plus')
def main(cfg):
    manager._create_task_rows = subset_rows
    manager.main.__wrapped__(cfg)


if __name__ == '__main__':
    main()
