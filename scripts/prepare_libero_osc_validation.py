"""Generate deterministic training probes for every LIBERO task (not held-out)."""
import json
from pathlib import Path
from collections import defaultdict
import hashlib
ROOT=Path(__file__).resolve().parents[1]

def build(suites):
    records=[];base=0;identity={}
    for suite in suites:
        root=ROOT/'data/libero_mujoco3.3.2'/f'libero_{suite}_no_noops_lerobot'
        groups=defaultdict(list);offset=0
        for line in (root/'meta/episodes.jsonl').read_text().splitlines():
            e=json.loads(line);assert len(e['tasks'])==1
            groups[e['tasks'][0]].append((e,offset));offset+=e['length']
        tasks=[json.loads(x) for x in (root/'meta/tasks.jsonl').read_text().splitlines()]
        assert offset==json.loads((root/'meta/info.json').read_text())['total_frames']
        identity[suite]={f:hashlib.sha256((root/'meta'/f).read_bytes()).hexdigest() for f in ['episodes.jsonl','tasks.jsonl','info.json']}
        for t in tasks:
            eps=groups[t['task']];assert len(eps)>=3
            for phase,ep_slot in [('start',0),('middle',len(eps)//2),('tail',len(eps)-1)]:
                e,off=eps[ep_slot];n=e['length'];assert n>=17
                frame={'start':0,'middle':min(n//2,n-16),'tail':n-8}[phase]
                records.append(dict(suite=suite,task_index=t['task_index'],task=t['task'],episode_index=e['episode_index'],frame_index=frame,dataset_index=base+off+frame,phase=phase,valid_action_steps=min(16,n-frame),diffusion_seed=42+len(records)))
        base+=offset
    assert len({r['dataset_index'] for r in records})==len(records)
    return dict(version=1,scope='training_probes_not_heldout',suite_order=suites,dataset_length=base,metadata_sha256=identity,samples=records)

if __name__=='__main__':
    for label,suites in [('goal',['goal']),('all4',['spatial','object','goal','10'])]:
        m=build(suites);p=ROOT/'configs/validation'/f'libero_{label}_osc_probes.json';p.write_text(json.dumps(m,indent=2)+'\n')
        cfg=ROOT/'configs/task'/f'libero_{label}_rothko_osc_delta_2cam224_full_wan21_1_3b_1e-4.yaml'
        marker='# Fixed task-balanced training probes (not held-out).'
        s=cfg.read_text().split(marker)[0].rstrip()+'\n'
        s+=marker+'\neval_num_samples: '+str(len(m['samples']))+'\neval_sample_indices: '+json.dumps([r['dataset_index'] for r in m['samples']])+'\n'
        cfg.write_text(s);print(label,len(m['samples']))
