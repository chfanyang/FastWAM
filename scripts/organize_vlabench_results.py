"""Organize VLABench artifacts without deleting data or retaining old links."""
from pathlib import Path
import argparse
from collections import Counter
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1] / 'evaluate_results/vlabench'
META = ROOT / '_organization_20261107'


def read(p):
    return json.loads(p.read_text()) if p.exists() else {}


def category(p):
    n = p.name
    if 'tmp' in n or n == 'vmt1021':
        return 'archive/temporary'
    if n == 'git_batches_20260917':
        return 'archive/repository_records'
    if p.is_file() or n.startswith(('launch_', 'resume_', 'vae_resume')) or 'launch' in n:
        return 'launch_records'
    if n.startswith('vlabench_') or (p/'plan.json').exists() or n.startswith('all10_step'):
        return 'closed_loop'
    if any(x in n for x in ('vae_audit', 'replay', 'rotation_diagnosis', 'compare', 'comparison', 'previews', 'clip_distance', 'quantiles')):
        return 'offline_audits'
    return 'preparation'


def evaluation(p):
    plan = read(p/'plan.json')
    cfg = plan.get('options') or read(p/'manager_config.json')
    summary = read(p/'summary.json')
    source = 'summary.json' if summary else ''
    if not summary:
        summary = read(p/'status.json')
        source = 'status.json' if summary else ''
    rows = {}
    issues = []
    def add(d):
        if not isinstance(d,dict) or not all(k in d for k in ('task','episode','success')):
            return
        key = (d['task'],int(d['episode']))
        v = (bool(d['success']),d.get('intention_score'),d.get('progress_score'))
        if key in rows and rows[key] != v:
            issues.append('conflicting_episode:'+str(key))
        else:
            rows[key] = v
    for d in summary.get('episodes', []):
        add(d)
    records = summary.get('records', [])
    for d in records:
        add(d.get('result', {}))
    # Limited known layouts avoid traversing videos, latent caches and diagnostics.
    files = list(p.glob('worker*/episode_*.json')) + list(p.glob('*/episode_*/episode_*.json')) + list(p.glob('episode_*.json'))
    for f in files:
        add(read(f))
    n, s = len(rows), sum(x[0] for x in rows.values())
    if summary.get('completed') is not None and (n != summary['completed'] or s != summary.get('successes',s)):
        issues.append('summary_differs_from_available_records')
    expected = summary.get('expected')
    if expected is None and isinstance(cfg.get('episodes'),int):
        expected = cfg['episodes'] * len(plan.get('tasks', ['select_book']))
    status = 'complete' if n and n == expected else ('partial' if expected else 'completion_unverified')
    if not n:
        status = 'no_episode_records'
    if issues:
        status = 'needs_review'
    commands = cfg.get('commands') or []
    cmd = commands[0] if commands else []
    def option(key,flag):
        if key in cfg:
            return cfg[key]
        return cmd[cmd.index(flag)+1] if flag in cmd else 'unknown'
    ckpt = cfg.get('checkpoint','')
    step = re.search(r'step_(\d+)',ckpt)
    per_task = {}
    for task in sorted({k[0] for k in rows}):
        values = [v for k,v in rows.items() if k[0]==task]
        per_task[task] = dict(successes=sum(v[0] for v in values),completed=len(values))
    def mean_component(i):
        values = [v[i] for v in rows.values() if isinstance(v[i],(int,float))]
        return round(sum(values)/len(values),6) if values else ''
    return dict(status=status,checkpoint=ckpt,dit_step=int(step[1]) if step else '',
        vae=option('vae_safetensors_path','--vae-safetensors-path'),decoder=option('decode_mode','--decode-mode'),
        replan=cfg.get('replan_steps',''),seed=cfg.get('seed',''),completed=n,expected=expected or '',
        successes=s,sr_percent=round(100*s/n,4) if n and not issues else '',
        is_raw=mean_component(1),ps_raw=mean_component(2),
        metric_records=sum(isinstance(v[1],(int,float)) and isinstance(v[2],(int,float)) for v in rows.values()),
        per_task=json.dumps(per_task),inherited_records=sum(bool(x.get('inherited')) for x in records),
        completed_from=cfg.get('completed_from',''),summary_source=source,
        notes=';'.join(sorted(set(issues))))


def make_rows(entries, relocated):
    result = []
    for e in entries:
        p = ROOT/(e['new'] if relocated else e['old'])
        cat = e['new'].split('/')[0]
        row = dict(name=Path(e['old']).name,category=cat,path=e['new'],legacy_path=e['old'],status='not_an_evaluation')
        if cat == 'closed_loop':
            if p.name.startswith('vlabench_'):
                row['status']='experiment_family'
                result.append(row)
                for sub in sorted(p.iterdir()):
                    if not sub.is_dir() or sub.name.startswith('heldout'):
                        continue
                    item=dict(name=sub.name,category=cat,path=e['new']+'/'+sub.name,legacy_path=e['old']+'/'+sub.name)
                    item.update(evaluation(sub));result.append(item)
                continue
            row.update(evaluation(p))
        result.append(row)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    if META.exists():raise RuntimeError('Organization already started; inspect journal before any further changes.')
    entries=[]
    for p in sorted(ROOT.iterdir()):
        if p.is_symlink() or p.name in ('closed_loop','offline_audits','preparation','launch_records','archive','INDEX.csv','README.md'):
            raise RuntimeError(f'Unexpected existing destination/link: {p}')
        st=p.stat();entries.append(dict(old=p.name,new=category(p)+'/'+p.name,inode=st.st_ino,device=st.st_dev))
    # Extract the offline heldout audit from the historical model family first.
    heldout='vlabench_select_book_rothko_3cam192_wan21/heldout382_4ckpt_vae562'
    if (ROOT/heldout).exists():
        st=(ROOT/heldout).stat();entries.insert(0,dict(old=heldout,new='offline_audits/'+heldout,inode=st.st_ino,device=st.st_dev))
    before=make_rows(entries,False)
    print('Plan',dict(Counter(x['new'].split('/')[0] for x in entries)),flush=True)
    if not args.apply:return
    META.mkdir();(META/'moves.json').write_text(json.dumps(entries,indent=2)+'\n')
    (META/'index_before.json').write_text(json.dumps(before,indent=2)+'\n')
    for e in entries:
        old,new=ROOT/e['old'],ROOT/e['new']
        assert not new.exists(),new
        new.parent.mkdir(parents=True,exist_ok=True)
        old.rename(new)
        st=new.stat();assert (st.st_dev,st.st_ino)==(e['device'],e['inode'])
        with (META/'completed.jsonl').open('a') as f:f.write(json.dumps(e)+'\n')
    after=make_rows(entries,True);assert before==after
    fields=list(dict.fromkeys(k for row in after for k in row))
    with (ROOT/'INDEX.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(after)
    for e in entries:assert (ROOT/e['new']).exists() and not (ROOT/e['old']).exists()
    (META/'verification.json').write_text(json.dumps(dict(moved_entries=len(entries),index_rows=len(after),
        evaluation_statistics_unchanged=True,inodes_preserved=True,created_symlinks=0,deleted_data_files=0),indent=2)+'\n')
    print('Verified',len(entries),'moves;',len(after),'index rows;',dict(Counter(x['status'] for x in after)),flush=True)


if __name__=='__main__':main()
