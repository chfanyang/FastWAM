"""Freeze the missing three-suite tasks; balance every suite/category across hosts."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.libero_plus import run_libero_plus_manager as manager
from omegaconf import OmegaConf

def main():
    template = json.loads((ROOT/'experiments/libero_plus/a800_paired_3090_20260913/nav_manifest.json').read_text())
    baseline = Path(template['baseline_dir'])
    legacy = baseline.with_name('ckpt021700_vaeDecoderStep7498_replan8_ensembleOff')
    suites = {'libero_spatial':2402, 'libero_goal':2591, 'libero_10':2519}
    cfg = OmegaConf.create({'MULTIRUN':dict(task_suite_names=list(suites),resume=False)})
    full = manager._create_task_rows(cfg, baseline)
    assert Counter(r['suite'] for r in full) == Counter(suites)
    classification=json.loads((ROOT/'third_party/LIBERO-plus/libero/libero/benchmark/task_classification.json').read_text())
    category={(s,r['name']):r['category'] for s,rs in classification.items() for r in rs}
    groups=defaultdict(list)
    seen=set()
    completed=[]
    durations={}
    medians=defaultdict(list)
    for row in full:
        key=(row['suite'],row['task_id'])
        assert key not in seen
        seen.add(key)
        c=category[(row['suite'],Path(row['bddl_file']).stem)]
        row['category']=c
        path=manager._result_path(baseline,*key)
        if path.exists():
            d=json.loads(path.read_text())
            assert d['task_suite']==key[0] and d['task_id']==key[1]
            assert Path(d['bddl_file']).name==Path(row['bddl_file']).name
            assert d['total_episodes']==1 and d['successes'] in (0,1)
            completed.append(key)
            continue
        groups[(row['suite'],c)].append(row)
        old=manager._result_path(legacy,*key)
        if old.exists():
            d=json.loads(old.read_text())
            assert Path(d['bddl_file']).name==Path(row['bddl_file']).name
            durations[key]=float(d['duration'])
            medians[(row['suite'],c)].append(float(d['duration']))
    assert len(completed)==2068 and sum(map(len,groups.values()))==5444
    buckets=[[],[]]; loads=[0.,0.]
    counts=[]
    for key,rows in sorted(groups.items()):
        for r in rows:
            # Scheduling proxy only: measured legacy runtime, not a joint result.
            r['a800_duration']=durations.get((r['suite'],r['task_id']),statistics.median(medians[key]))
            r['duration_source']='legacy_runtime_proxy'
        rows.sort(key=lambda r:(-r['a800_duration'],r['task_id']))
        before=[len(b) for b in buckets]
        for i in range(0,len(rows),2):
            pair=rows[i:i+2]
            h=min(range(2),key=lambda j:(loads[j],len(buckets[j]),j)) if len(pair)==2 else min(range(2),key=lambda j:(len(buckets[j]),loads[j],j))
            for j,r in enumerate(pair):
                target=h if j==0 else 1-h
                buckets[target].append(r);loads[target]+=r['a800_duration']
        sizes=[len(buckets[j])-before[j] for j in range(2)]
        assert abs(sizes[0]-sizes[1])<=1
        counts.append(dict(suite=key[0],category=key[1],manipulation=sizes[0],nav=sizes[1]))
    keys=[{(r['suite'],r['task_id']) for r in b} for b in buckets]
    assert not keys[0]&keys[1] and not (keys[0]|keys[1])&set(completed)
    assert keys[0]|keys[1]|set(completed)==seen
    assert abs(len(buckets[0])-len(buckets[1]))<=1
    out=ROOT/'experiments/libero_plus/remaining_joint_20260914'
    out.mkdir(exist_ok=True)
    for h,rows in zip(('manipulation','nav'),buckets):
        m={k:template[k] for k in ('baseline_dir','baseline_config','file_sha256')}
        m.update(host=h,num_tasks=len(rows),tasks=rows,selection='missing_from_A800_joint_subset',suite_counts=dict(Counter(r['suite'] for r in rows)),suite_category_counts=counts)
        path=out/f'{h}_manifest.json'
        if path.exists():assert json.loads(path.read_text())==m
        else:path.write_text(json.dumps(m,indent=2)+'\n')
    print(json.dumps(dict(completed=len(completed),pending=sum(map(len,buckets)),counts=counts,host_tasks=list(map(len,buckets)),proxy_gpu_hours=[x/3600 for x in loads]),indent=2))

if __name__=='__main__':main()
