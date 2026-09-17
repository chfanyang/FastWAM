"""Rerun frozen A800 task IDs, checking BDDL identity and baseline settings."""
import hashlib
import json
import os
from pathlib import Path
import sys
import hydra
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.libero_plus import run_libero_plus_manager as manager
original=manager._create_task_rows

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def normalized(x):
    if isinstance(x,str):return x.replace('/mnt/hwdata/cfy/FastWAM',str(ROOT))
    if isinstance(x,dict):return {k:normalized(v) for k,v in x.items()}
    if isinstance(x,list):return [normalized(v) for v in x]
    return x

def select(cfg,out):
    manifest_dir=Path(os.environ.get('PAIRED_MANIFEST_DIR',str(ROOT/'experiments/libero_plus/a800_paired_3090_20260913')))
    manifest=json.loads((manifest_dir/f"{os.environ['PAIRED_HOST']}_manifest.json").read_text())
    actual={k:cfg[k] for k in ('ckpt','seed','eval_num_inference_steps','eval_random_seed')}
    actual.update(model=OmegaConf.to_container(cfg.model,resolve=True),EVALUATION=OmegaConf.to_container(cfg.EVALUATION,resolve=True))
    base=normalized(manifest['baseline_config'])
    for k in ('seed','eval_num_inference_steps','eval_random_seed','ckpt'):
        assert actual[k]==base[k], (k,actual[k],base[k])
    for k,v in base['EVALUATION'].items():
        if k not in ('task_suite_name','task_id','output_dir','device'):
            assert actual['EVALUATION'].get(k)==v, ('EVALUATION.'+k,actual['EVALUATION'].get(k),v)
    for k in ('vae_safetensors_path','allow_vae_mismatch','rothko_decode_mode','rothko_decode_anchor_alpha',
              'rothko_decode_block_grid','model_variant','scheduler'):
        assert actual['model'][k]==base['model'][k], ('model.'+k,actual['model'][k],base['model'][k])
    for rel,expected in manifest['file_sha256'].items():
        assert sha(ROOT/rel)==expected, f'Weight/stats mismatch: {rel}'
    lookup={(r['suite'],r['task_id']):r for r in original(cfg,out)}
    frozen=out/'paired_a800_manifest.json'
    if frozen.exists():assert json.loads(frozen.read_text())==manifest
    else:frozen.write_text(json.dumps(manifest,indent=2))
    selected=[]
    for row in manifest['tasks']:
        if manager._result_path(out,row['suite'],row['task_id']).exists():continue
        found=lookup[(row['suite'],row['task_id'])]
        assert Path(found['bddl_file']).name==Path(row['bddl_file']).name
        selected.append(dict(row))
    print('PAIRED_BASELINE_CHECK_OK',manifest['host'],len(selected),'of',manifest['num_tasks'],flush=True)
    return selected

def balanced_workers(rows,gpu_ids,suite_names,*,workers_per_gpu,assignment_mode):
    assert workers_per_gpu==1 and len(gpu_ids)==8
    buckets=[[] for _ in gpu_ids];loads=[0.]*len(gpu_ids)
    for row in sorted(rows,key=lambda r:(-r['a800_duration'],r['suite'],r['task_id'])):
        i=min(range(len(buckets)),key=lambda i:(loads[i],len(buckets[i]),i))
        buckets[i].append(row);loads[i]+=row['a800_duration']
    print('BALANCED_WORKERS',json.dumps({'counts':[len(b) for b in buckets],'a800_hours':[v/3600 for v in loads]}),flush=True)
    return list(zip(gpu_ids,buckets))

@hydra.main(version_base='1.3',config_path=str(ROOT/'configs'),config_name='sim_libero_plus')
def main(cfg):
    if bool(cfg.MULTIRUN.get('create_only',False)):
        out=Path(str(cfg.EVALUATION.output_dir))
        out.mkdir(parents=True,exist_ok=True)
        rows=select(cfg,out)
        balanced_workers(rows,manager._gpu_ids(cfg),list(cfg.MULTIRUN.task_suite_names),workers_per_gpu=1,assignment_mode='round_robin')
        return
    manager._create_task_rows=select
    manager._assign_tasks=balanced_workers
    manager.main.__wrapped__(cfg)

if __name__=='__main__':main()
