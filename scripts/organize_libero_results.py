"""One-time, reversible LIBERO result organization; never deletes result data."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
from collections import Counter

import yaml

ROOT = Path(__file__).resolve().parents[1] / 'evaluate_results/libero'
META = ROOT / '_organization_20261107'
CATEGORIES = {'closed_loop', 'offline_audits', 'preparation', 'launch_records', 'archive'}
TEMP = {'cachetmp', 'gtabletmp', 'oscevaltmp', 'osctmp', 'osctrain_tmp', 'ray0tmp', 'w21tmp053905'}


def category(p):
    n = p.name
    if n in TEMP:
        return 'archive/temporary'
    if 'startup_' in n or n.startswith('withdraw_'):
        return 'archive/failed_or_withdrawn'
    if p.is_file() or 'launch' in n or 'resume_' in n or 'restart_' in n or '3090_prepare' in n:
        return 'launch_records'
    if 'smoke' in n or 'e2e' in n or 'validation_fix' in n:
        return 'preparation'
    if n.startswith('libero_') or list(p.glob('*/gpu*_task*_results.json')) or n.startswith('goal_osc_step4140_vae400_replan8'):
        return 'closed_loop'
    if any(x in n for x in ('repr100', 'roundtrip', 'axis_angle_constant', 'repr_50episodes', 'vae_audit', 'robust30', 'learning_check', 'effective30', 'alphabet_soup', 'object_goal_10windows', 'flatten_layout')):
        return 'offline_audits'
    return 'preparation'


def evaluation(p):
    """Only count per-task result files; no log/summary double counting."""
    cfgpath = p / 'manager_config.yaml'
    cfg = yaml.safe_load(cfgpath.read_text()) if cfgpath.exists() else {}
    cfg = cfg or {}
    ev, model = cfg.get('EVALUATION', {}), cfg.get('model', {})
    results, issues = {}, []
    for f in sorted(p.glob('*/gpu*_task*_results.json')):
        d = json.loads(f.read_text())
        key = (d['task_suite'], int(d['task_id']))
        value = (int(d['successes']), int(d['total_episodes']))
        if key in results and results[key] != value:
            issues.append('conflicting_duplicate:' + str(key))
        results.setdefault(key, value)
    expected = set()
    taskfile = p / 'tasks.txt'
    if taskfile.exists():
        for line in taskfile.read_text().splitlines():
            m = re.fullmatch(r'\s*(libero_\w+)\s*,\s*(\d+)\s*', line)
            if m:
                expected.add((m[1], int(m[2])))
    trials = ev.get('num_trials')
    if not isinstance(trials, int):
        trials = None
    count = sum(x[1] for x in results.values())
    successes = sum(x[0] for x in results.values())
    missing = expected - results.keys()
    extra = results.keys() - expected if expected else set()
    if extra:
        issues.append('results_outside_tasks_manifest')
    if expected and trials:
        complete = not missing and not extra and all(results[k][1] == trials for k in expected)
        status = 'complete' if complete else 'partial'
    else:
        status = 'results_present_completion_unverified' if results else 'no_task_results'
    if issues:
        status = 'needs_review'
    ckpt = str(cfg.get('ckpt') or '')
    step = re.search(r'step_(\d+)', ckpt)
    vae = model.get('vae_safetensors_path', 'unknown')
    if vae is None:
        vae = 'original'
    per_suite = {}
    for suite in sorted({k[0] for k in results}):
        v = [x for k, x in results.items() if k[0] == suite]
        s, n = sum(x[0] for x in v), sum(x[1] for x in v)
        per_suite[suite] = f'{s}/{n} ({100*s/n:.2f}%)' if n else '0/0'
    return dict(status=status, checkpoint=ckpt, dit_step=int(step[1]) if step else '',
                vae=str(vae), decoder=model.get('rothko_decode_mode', 'unknown'),
                anchor=model.get('rothko_decode_anchor_alpha', ''), replan=ev.get('replan_steps', ''),
                denoising_steps=ev.get('num_inference_steps', ''), suites=';'.join(per_suite),
                result_tasks=len(results), expected_tasks=len(expected) if expected else '',
                completed_episodes=count, expected_episodes=len(expected)*trials if expected and trials else '',
                successes=successes, sr_percent=round(100*successes/count, 4) if count and not issues else '',
                per_suite=json.dumps(per_suite, ensure_ascii=False),
                notes=';'.join(issues), config_path=str(cfgpath.relative_to(ROOT)) if cfgpath.exists() else '')


def index_rows(entries, relocated):
    rows = []
    for e in entries:
        old = ROOT / e['old']
        p = ROOT / e['new'] if relocated else old
        cat = e['new'].split('/')[0]
        row = dict(name=e['old'], category=cat, kind='directory' if p.is_dir() else 'file',
                   path=e['new'], legacy_path=e['old'], status='not_an_evaluation')
        if cat == 'closed_loop':
            if e['old'].startswith('libero_'):
                row['status'] = 'experiment_family'
                rows.append(row)
                for child in sorted(p.iterdir()):
                    if not child.is_dir():
                        continue
                    r = dict(name=child.name, category=cat, kind='evaluation_or_analysis',
                             path=e['new']+'/'+child.name, legacy_path=e['old']+'/'+child.name)
                    r.update(evaluation(child))
                    rows.append(r)
                continue
            row.update(evaluation(p))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if META.exists():
        raise SystemExit('Organization record exists; inspect it rather than rerunning moves.')
    entries = []
    for p in sorted(ROOT.iterdir()):
        if p.name in CATEGORIES or p.name.startswith('_organization') or p.name in ('INDEX.csv', 'README.md'):
            raise RuntimeError(f'Unexpected pre-existing destination: {p}')
        if p.is_symlink():
            raise RuntimeError(f'Inspect existing top-level symlink first: {p}')
        target = ROOT / category(p) / p.name
        assert not target.exists()
        st = p.stat()
        entries.append(dict(old=p.name, new=str(target.relative_to(ROOT)), inode=st.st_ino, device=st.st_dev))
    print('Plan:', dict(Counter(e['new'].split('/')[0] for e in entries)), flush=True)
    # Parse before making any structural change; malformed records stop the run.
    rows_before = index_rows(entries, False)
    if not args.apply:
        print(json.dumps(entries, indent=2))
        return
    META.mkdir()
    (META / 'moves.json').write_text(json.dumps(entries, indent=2)+'\n')
    (META / 'index_before.json').write_text(json.dumps(rows_before, indent=2)+'\n')
    for e in entries:
        old, new = ROOT / e['old'], ROOT / e['new']
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        try:
            old.symlink_to(e['new'], target_is_directory=new.is_dir())
        except Exception:
            new.rename(old)
            raise
        st = old.stat()
        assert (st.st_dev, st.st_ino) == (e['device'], e['inode'])
        with (META / 'completed.jsonl').open('a') as f:
            f.write(json.dumps(e)+'\n')
    rows = index_rows(entries, True)
    for a, b in zip(rows_before, rows):
        # Only canonical config paths change after relocation.
        assert {k:v for k,v in a.items() if k!='config_path'} == {k:v for k,v in b.items() if k!='config_path'}
    assert len(rows_before) == len(rows)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT / 'INDEX.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (META / 'verification.json').write_text(json.dumps(dict(entries=len(entries), index_rows=len(rows),
        old_paths_resolve_to_original_inodes=True, evaluation_counts_unchanged=True,
        deleted_result_files=0), indent=2)+'\n')
    print('Verified:', len(entries), 'moves;', len(rows), 'index rows', flush=True)


if __name__ == '__main__':
    main()
