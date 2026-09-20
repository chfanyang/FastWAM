"""Full-episode, sharded GT roundtrip audit, independent of training."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import torch
import pyarrow.parquet as pq
import pyarrow as pa
from audit_goal_representation_roundtrip import (
    flat_chw_encode,flat_chw_decode,permute_tiles,stats,
    LiberoRothkoCodecConfig,LiberoAllAbsoluteRothkoCodec,_load_registered_model)

NAMES=('rothko','flatten_chw','shuffled')
STAGES=('direct','bf16_pixels','vae')
KEYS=('position_mm','rotation_deg','gripper_abs','gripper_binary')
DATA=Path('data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot')

def atomic(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    groups={}
    for line in (DATA/'meta/episodes.jsonl').read_text().splitlines():
        ep=json.loads(line);groups.setdefault(ep['tasks'][0],[]).append(ep)
    assert len(groups)==10
    rng=random.Random(42);episodes=[]
    for task,eps in sorted(groups.items()):
        for ep in rng.sample(eps,5):
            episodes.append(dict(task=task,episode=ep['episode_index'],length=ep['length']))
    # Balance intact episodes among 8 GPUs; sample selection is independent of sharding.
    counts=[0]*8
    for ep in sorted(episodes,key=lambda e:-e['length']):
        rank=min(range(8),key=lambda r:counts[r]);ep['shard']=rank;counts[rank]+=ep['length']
    manifest=dict(seed=42,episodes=episodes,num_shards=8,total_windows=sum(counts),shard_windows=counts,
                  representations=NAMES,stages=STAGES,padding='edge repeat; exclude padded future targets from all metrics')
    atomic(out/'manifest.json',manifest)
    print(json.dumps(manifest),flush=True)

def run(out,rank):
    manifest=json.loads((out/'manifest.json').read_text())
    shard=out/f'shard{rank}';shard.mkdir(exist_ok=False)
    torch.set_num_threads(2);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    torch.cuda.set_per_process_memory_fraction(.15)
    sp=Path('data/libero_mujoco3.3.2/libero_all4_rothko_all_absolute_minmax_margin01_h16_224x448_centerfrac05.pt')
    meta=json.loads(sp.with_suffix('.json').read_text())
    cfg=LiberoRothkoCodecConfig(frame0_pose_mode='absolute',absolute_position_min=meta['absolute_position_min'],absolute_position_max=meta['absolute_position_max'])
    codec=LiberoAllAbsoluteRothkoCodec(config=cfg,norm_stats=sp)
    lo=torch.tensor(cfg.absolute_position_min);hi=torch.tensor(cfg.absolute_position_max)
    perm=torch.randperm(224*224,generator=torch.Generator().manual_seed(42));inv=perm.argsort()
    torch.save(dict(permutation=perm,inverse=inv),shard/'permutation.pt')
    vaep=Path('checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    atomic(shard/'identity.json',dict(vae_sha256=hashlib.sha256(vaep.read_bytes()).hexdigest(),
        stats_fingerprint=codec.norm_stats.fingerprint(),permutation_sha256=hashlib.sha256(perm.numpy().tobytes()).hexdigest()))
    vae=_load_registered_model(str(vaep),'wan_video_vae',torch_dtype=torch.bfloat16,device='cuda')
    vae.eval().requires_grad_(False)
    completed=0;began=time.monotonic()
    with torch.inference_mode():
        for ep in manifest['episodes']:
            if ep['shard']!=rank:continue
            f=next(DATA.glob(f"data/chunk-*/episode_{ep['episode']:06d}.parquet"))
            keys=['observation.state.ee_pose_wxyz','action.osc_target_pose_wxyz','observation.state.gripper_open','action']
            table=pq.read_table(f,columns=keys);assert len(table)==ep['length']
            t={k:torch.tensor(table[k].to_pylist()) for k in keys}
            dest=shard/f"episode_{ep['episode']:06d}.jsonl"
            with dest.open('x') as stream:
                for start in range(ep['length']):
                    idx=torch.arange(start,start+16);valid=idx<len(table);idx=idx.clamp_max(len(table)-1)
                    pose=torch.cat((t[keys[0]][start:start+1],t[keys[1]][idx]))
                    grip=torch.cat((t[keys[2]][start:start+1],t['action'][idx,-1:].clamp(0,1)))
                    base=codec.encode(pose,grip);shuffled=permute_tiles(base,perm)
                    assert torch.equal(base[...,:224],base[...,224:])
                    assert torch.equal(permute_tiles(shuffled,inv),base)
                    variants=[('rothko',base),('flatten_chw',flat_chw_encode(pose,grip,lo,hi)),('shuffled',shuffled)]
                    rows=[]
                    for name,x in variants:
                        z=vae.encode(x[None].cuda().bfloat16(),device='cuda',tiled=False)
                        y=vae.decode(z,device='cuda',tiled=False)[0].float().clamp(-1,1).cpu()
                        for stage,pixels in [('direct',x),('bf16_pixels',x.bfloat16().float()),('vae',y)]:
                            bad=0
                            if name=='flatten_chw':pred,g,bad=flat_chw_decode(pixels,lo,hi)
                            else:pred,g=codec.decode(permute_tiles(pixels,inv) if name=='shuffled' else pixels)
                            assert torch.isfinite(pred).all() and torch.isfinite(g).all()
                            q=torch.nn.functional.normalize(pred[1:,3:].double(),dim=-1)
                            gt=torch.nn.functional.normalize(pose[1:,3:].double(),dim=-1)
                            pos=(pred[1:,:3].double()-pose[1:,:3].double()).norm(dim=-1)*1000
                            rot=torch.rad2deg(2*torch.acos((q*gt).sum(-1).abs().clamp(0,1)))
                            ge=(g[1:].reshape(-1)-grip[1:].reshape(-1)).abs()
                            gb=((g[1:].reshape(-1)>.5)!=(grip[1:].reshape(-1)>.5)).float()
                            if stage=='direct':
                                assert pos.max()<.01 and rot.max()<.01 and ge.max()<1e-5
                            rows.append(dict(task=ep['task'],episode=ep['episode'],start=start,representation=name,stage=stage,
                                valid=valid.tolist(),position_mm=pos.tolist(),rotation_deg=rot.tolist(),
                                gripper_abs=ge.tolist(),gripper_binary=gb.tolist(),degenerate_quaternions=bad))
                        del z,y
                    stream.write(''.join(json.dumps(row)+'\n' for row in rows));stream.flush()
                    completed+=1
                    if completed%10==0 or completed==1:
                        elapsed=time.monotonic()-began
                        status=dict(completed_windows=completed,total_windows=manifest['shard_windows'][rank],
                            elapsed_seconds=elapsed,seconds_per_window=elapsed/completed,peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)
                        atomic(shard/'progress.json',status);print(json.dumps(status),flush=True)
            atomic(shard/f"episode_{ep['episode']:06d}.done.json",dict(windows=ep['length'],rows=ep['length']*9))
    assert completed==manifest['shard_windows'][rank]
    atomic(shard/'complete.json',dict(windows=completed,seconds=time.monotonic()-began))
    print('COMPLETE',rank,completed,flush=True)

def summarize(rows):
    result={}
    for name in NAMES:
        result[name]={}
        for stage in STAGES:
            subset=[r for r in rows if r['representation']==name and r['stage']==stage]
            result[name][stage]={str(h):{k:stats([v for r in subset for v,ok in zip(r[k][:h],r['valid'][:h]) if ok])
                for k in KEYS} for h in (8,16)}
            result[name][stage]['degenerate_quaternions']=sum(r['degenerate_quaternions'] for r in subset)
    return result

def aggregate(out):
    manifest=json.loads((out/'manifest.json').read_text());rows=[];identities=[]
    for rank in range(8):
        shard=out/f'shard{rank}'
        assert json.loads((shard/'complete.json').read_text())['windows']==manifest['shard_windows'][rank]
        identities.append(json.loads((shard/'identity.json').read_text()))
    assert all(x==identities[0] for x in identities)
    for ep in manifest['episodes']:
        shard=out/f"shard{ep['shard']}"
        assert (shard/f"episode_{ep['episode']:06d}.done.json").exists()
        rr=[json.loads(line) for line in (shard/f"episode_{ep['episode']:06d}.jsonl").read_text().splitlines()]
        expected={(s,n,k) for s in range(ep['length']) for n in NAMES for k in STAGES}
        assert len(rr)==len(expected) and {(r['start'],r['representation'],r['stage']) for r in rr}==expected
        for r in rr:
            assert r['task']==ep['task'] and r['episode']==ep['episode']
            assert r['valid']==[r['start']+j<ep['length'] for j in range(16)]
        rows.extend(rr)
    report=dict(episodes=50,windows=manifest['total_windows'],identity=identities[0],summary=summarize(rows),
        per_task={task:summarize([r for r in rows if r['task']==task]) for task in sorted({e['task'] for e in manifest['episodes']})},
        note='GT roundtrip, original BF16 VAE, all starts of 5 sampled episodes/task, overlapping-window prediction-step weighting, padding excluded; not closed-loop SR.')
    atomic(out/'report.json',report);print('ALL_COMPLETE',manifest['total_windows'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run','aggregate']);p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int)
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.output)
    elif a.stage=='run':run(a.output,a.rank)
    else:aggregate(a.output)
