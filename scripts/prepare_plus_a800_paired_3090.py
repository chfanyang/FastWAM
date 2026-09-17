"""Freeze all completed A800 subset tasks into balanced two-host rerun manifests."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'evaluate_results/libero_plus/libero_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-4/ckpt021700_vaeDecoderStep7498_replan8_ensembleOff_robustJointAnchor0_stratified10_seed42'
OUT=ROOT/'experiments/libero_plus/a800_paired_3090_20260913'
OUT.mkdir(exist_ok=True)
assert not any(OUT.iterdir()), 'Refusing to overwrite an existing split'
source=json.loads((BASE/'stratified10_seed42_manifest.json').read_text())
groups=defaultdict(list)
for row in source['tasks']:
    result=json.loads((BASE/row['suite']/'results'/f"task{row['task_id']:04d}.json").read_text())
    assert Path(result['bddl_file']).name==Path(row['bddl_file']).name
    assert result['total_episodes']==1
    groups[(row['suite'],row['category'],row['base_task'])].append(dict(row,
        a800_success=int(result['successes']),a800_duration=float(result['duration'])))
parts=[[],[]];loads=[0.,0.];cat_counts=[Counter(),Counter()];suite_counts=[Counter(),Counter()]
def put(row,part):
    parts[part].append(row);loads[part]+=row['a800_duration']
    cat_counts[part][(row['suite'],row['category'])]+=1;suite_counts[part][row['suite']]+=1
for key,rows in sorted(groups.items()):
    rows=sorted(rows,key=lambda r:(-r['a800_duration'],r['task_id']))
    for j in range(0,len(rows)-1,2):
        part=0 if loads[0]<=loads[1] else 1
        put(rows[j],part);put(rows[j+1],1-part)
    if len(rows)%2:
        row=rows[-1];ck=key[:2]
        part=min(range(2),key=lambda p:(cat_counts[p][ck],suite_counts[p][row['suite']],len(parts[p]),loads[p]))
        put(row,part)
assert sorted(map(len,parts))==[1378,1379]
assert all(abs(cat_counts[0][k]-cat_counts[1][k])<=1 for k in cat_counts[0]|cat_counts[1])
assert all(abs(suite_counts[0][k]-suite_counts[1][k])<=1 for k in suite_counts[0]|suite_counts[1])
keys=[{(r['suite'],r['task_id']) for r in part} for part in parts]
assert not keys[0]&keys[1] and len(keys[0]|keys[1])==2757
cfg=OmegaConf.load(BASE/'manager_config.yaml')
baseline={k:cfg[k] for k in ('ckpt','seed','eval_num_inference_steps','eval_random_seed')}
baseline.update(model=OmegaConf.to_container(cfg.model,resolve=True),EVALUATION=OmegaConf.to_container(cfg.EVALUATION,resolve=True))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
files=[baseline['ckpt'],baseline['model']['vae_safetensors_path'],baseline['EVALUATION']['dataset_stats_path'],
       str(ROOT/'data/libero_mujoco3.3.2/libero_rothko_region_symmetric_q99p95_h16_224x448.pt')]
identity={str(Path(p).relative_to(ROOT)):sha(p) for p in files}
for p,host in enumerate(('manipulation','nav')):
    # Longest-first worker assignment is applied remotely; each host has a frozen task set.
    manifest={'host':host,'num_tasks':len(parts[p]),'baseline_dir':str(BASE),'baseline_config':baseline,
              'source_manifest_sha256':sha(BASE/'stratified10_seed42_manifest.json'),'file_sha256':identity,
              'suite_counts':dict(suite_counts[p]),'estimated_a800_duration_sum':loads[p],
              'suite_category_counts':{'|'.join(k):v for k,v in sorted(cat_counts[p].items())},'tasks':parts[p]}
    (OUT/f'{host}_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'counts':[len(p) for p in parts],'suites':[dict(c) for c in suite_counts],
                  'estimated_load_hours':[v/3600 for v in loads],'directory':str(OUT)},indent=2))
