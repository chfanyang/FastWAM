"""Opt-in three-host LIBERO-Plus plan. No changes to existing managers/workers."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import hydra
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.libero_plus import run_libero_plus_manager as manager
PLAN_DIR=ROOT/'experiments/libero_plus/absolute_ray0_three_host'
PLAN=json.loads((PLAN_DIR/'plan.json').read_text())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def all_rows():
    manager.assert_libero_plus_install()
    result=[]
    for name,count in PLAN['expected_counts'].items():
        with contextlib.redirect_stdout(io.StringIO()):
            suite=manager.benchmark.get_benchmark_dict()[name]()
        assert suite.n_tasks==count,(name,suite.n_tasks,count)
        for i in range(count):
            t=suite.get_task(i)
            result.append(dict(suite=name,task_id=i,language=str(t.language),bddl_file=str(t.bddl_file),init_states_file=str(t.init_states_file)))
    return result

def select_role(rows,role):
    spec=PLAN['roles'][role]
    return sorted([r for r in rows if r['suite']==spec['full_suite'] or (r['suite']=='libero_spatial' and spec['spatial_parity'] is not None and r['task_id']%2==spec['spatial_parity'])],key=lambda r:(r['task_id'],r['suite']))

def build():
    rows=all_rows();seen=set();counts={}
    for role in PLAN['roles']:
        selected=select_role(rows,role);keys={(r['suite'],r['task_id']) for r in selected}
        assert len(keys)==len(selected) and not keys&seen;seen|=keys
        path=PLAN_DIR/f'{role}.json';payload=dict(role=role,tasks=selected)
        if path.exists():assert json.loads(path.read_text())==payload
        else:path.write_text(json.dumps(payload,indent=2)+'\n')
        counts[role]=len(selected)
    assert seen=={(r['suite'],r['task_id']) for r in rows} and len(seen)==10030
    print(json.dumps({'partition':counts,'total':len(seen),'spatial_split':'even/odd task IDs; 1201 each'},indent=2))

def pending(cfg,out):
    role=os.environ['PLUS_ROLE'];frozen=json.loads((PLAN_DIR/f'{role}.json').read_text())['tasks']
    assert frozen==select_role(all_rows(),role),'Benchmark task identity/order differs from frozen manifest'
    selected=[]
    for row in frozen:
        path=manager._result_path(out,row['suite'],row['task_id'])
        if path.exists():
            result=json.loads(path.read_text())
            assert result['task_suite']==row['suite'] and result['task_id']==row['task_id']
            assert result['bddl_file']==row['bddl_file'] and result['total_episodes']==1
            assert result['checkpoint']==str(cfg.ckpt)
            assert isinstance(result['successes'],int) and result['successes'] in (0,1)
            continue
        selected.append(row)
    print(f'{role}: {len(selected)}/{len(frozen)} pending',flush=True)
    return selected

def verify_artifacts(cfg):
    evidence=json.loads((ROOT/'artifacts/READY.json').read_text())
    assert evidence['dit_step']==PLAN['dit_step'] and evidence['vae_step']==PLAN['vae_step']
    required={'artifacts/dit_step021700.pt','artifacts/vae_step008264.safetensors','artifacts/dataset_stats.json'}
    assert required <= set(evidence['sha256']), 'Incomplete artifact identity'
    for rel,expected in evidence['sha256'].items():assert sha(ROOT/rel)==expected,rel
    assert Path(cfg.ckpt).resolve()==ROOT/'artifacts/dit_step021700.pt'
    assert Path(cfg.model.vae_safetensors_path).resolve()==ROOT/'artifacts/vae_step008264.safetensors'
    return evidence

@hydra.main(version_base='1.3',config_path=str(ROOT/'configs'),config_name='sim_libero_plus')
def main(cfg):
    role=os.environ['PLUS_ROLE'];spec=PLAN['roles'][role]
    assert int(cfg.seed)==42 and cfg.model.model_variant=='wan2.1-t2v-1.3b'
    assert cfg.data.train.raymap_representation=='libero_rothko'
    assert cfg.data.train.rothko_config.frame0_pose_mode=='absolute'
    assert cfg.EVALUATION.num_trials==1 and cfg.EVALUATION.replan_steps==8
    assert cfg.EVALUATION.num_inference_steps==20 and not cfg.EVALUATION.use_action_ensembler
    assert cfg.model.rothko_decode_mode=='legacy' and cfg.model.rothko_decode_anchor_alpha==0
    assert cfg.MULTIRUN.workers_per_gpu==spec['workers_per_gpu']
    assert cfg.MULTIRUN.num_gpus==8 and cfg.MULTIRUN.render_on_worker_gpu
    expected=[spec['full_suite']]+(['libero_spatial'] if spec['spatial_parity'] is not None else [])
    assert list(cfg.MULTIRUN.task_suite_names)==expected
    out=Path(cfg.EVALUATION.output_dir)
    if cfg.MULTIRUN.get('create_only',False):
        rows=select_role(all_rows(),role)
        assert rows==json.loads((PLAN_DIR/f'{role}.json').read_text())['tasks']
        print(json.dumps({'role':role,'tasks':len(rows),'workers':8*spec['workers_per_gpu'],'create_only':True}))
        return
    evidence=verify_artifacts(cfg)
    identity=dict(artifacts=evidence,role=role,manifest=sha(PLAN_DIR/f'{role}.json'),model=OmegaConf.to_container(cfg.model,resolve=True),evaluation=OmegaConf.to_container(cfg.EVALUATION,resolve=True),seed=int(cfg.seed))
    marker=out/'evaluation_identity.json'
    if marker.exists():assert json.loads(marker.read_text())==identity,'Refusing mixed-protocol results'
    else:
        assert not out.exists() or not any(out.iterdir()),'Unidentified existing result directory'
        out.mkdir(parents=True,exist_ok=True);marker.write_text(json.dumps(identity,indent=2))
    # Preserve prior manager metadata/logs on an explicit resume.
    import shutil,time
    items=[out/n for n in ['manager_config.yaml','tasks.jsonl','worker_tasks','worker_logs','summary.json'] if (out/n).exists()]
    if items:
        backup=out/f'resume_backup_{time.time_ns()}';backup.mkdir()
        for path in items:shutil.move(str(path),str(backup/path.name))
    manager._create_task_rows=pending
    manager.main.__wrapped__(cfg)

if __name__=='__main__':
    if sys.argv[1:]==['--build-manifests']:build()
    else:main()
