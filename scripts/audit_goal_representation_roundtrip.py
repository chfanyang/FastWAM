"""Isolated Goal representation audit; no training/evaluation behavior changes."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
import pyarrow.parquet as pq
import pyarrow as pa
from fastwam.representations.libero_rothko import LiberoRothkoCodecConfig
from fastwam.representations.libero_rothko_all_absolute import LiberoAllAbsoluteRothkoCodec
from fastwam.models.wan22.helpers.loader import _load_registered_model

def canonical(q):
    q=torch.nn.functional.normalize(q,dim=-1)
    first=(q!=0).to(torch.int64).argmax(-1,keepdim=True)
    return q*torch.where(q.gather(-1,first)<0,-1.,1.)

def flat_encode(p,g,lo,hi):
    v=torch.cat((2*(p[:,:3]-lo)/(hi-lo)-1,canonical(p[:,3:]),2*g-1),-1)
    tile=v.repeat(1,18816).reshape(-1,224,224,3).permute(3,0,1,2)
    return torch.cat((tile,tile),-1)

def flat_chw_encode(p,g,lo,hi):
    v=torch.cat((2*(p[:,:3]-lo)/(hi-lo)-1,canonical(p[:,3:]),2*g-1),-1)
    tile=v.repeat(1,18816).reshape(-1,3,224,224).permute(1,0,2,3)
    return torch.cat((tile,tile),-1)

def flat_chw_decode(x,lo,hi):
    # Reorder to HWC representation only to reuse the identical numeric reader.
    converted=torch.cat([t.permute(1,0,2,3).reshape(17,224,224,3).permute(3,0,1,2)
                         for t in x.split(224,-1)],-1)
    return flat_decode(converted,lo,hi)

def flat_decode(x,lo,hi):
    v=torch.stack([t.permute(1,2,3,0).reshape(17,-1,8) for t in x.split(224,-1)]).mean((0,2))
    norm=v[:,3:7].norm(dim=-1)
    q=v[:,3:7].clone()
    bad=norm<1e-8
    q[bad]=torch.tensor([1.,0,0,0])
    return torch.cat(((v[:,:3]+1)*.5*(hi-lo)+lo,canonical(q)),-1),(v[:,7:]+1)*.5,int(bad.sum())

def permute_tiles(x,perm):
    return torch.cat([tile.flatten(-2)[...,perm].reshape_as(tile) for tile in x.split(224,-1)],-1)

def stats(v):
    x=torch.tensor(v,dtype=torch.float64)
    return dict(n=len(v),mean=x.mean().item(),p95=x.quantile(.95).item(),max=x.max().item())

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--windows-per-task',type=int,default=2)
    a.add_argument('--chw-only',action='store_true')
    args=a.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    torch.cuda.set_per_process_memory_fraction(.15)
    root=Path('data/libero_mujoco3.3.2')
    sp=root/'libero_all4_rothko_all_absolute_minmax_margin01_h16_224x448_centerfrac05.pt'
    meta=json.loads(sp.with_suffix('.json').read_text())
    cfg=LiberoRothkoCodecConfig(frame0_pose_mode='absolute',absolute_position_min=meta['absolute_position_min'],absolute_position_max=meta['absolute_position_max'])
    codec=LiberoAllAbsoluteRothkoCodec(config=cfg,norm_stats=sp)
    lo=torch.tensor(cfg.absolute_position_min);hi=torch.tensor(cfg.absolute_position_max)
    perm=torch.randperm(224*224,generator=torch.Generator().manual_seed(42));inv=perm.argsort()
    torch.save(dict(permutation=perm,inverse=inv),args.output/'permutation.pt')
    data=root/'libero_goal_no_noops_lerobot'
    groups={}
    for line in (data/'meta/episodes.jsonl').read_text().splitlines():
        ep=json.loads(line);groups.setdefault(ep['tasks'][0],[]).append(ep)
    rng=random.Random(42);selection=[]
    for task,eps in sorted(groups.items()):
        for ep in rng.sample(eps,args.windows_per_task):
            selection.append(dict(task=task,episode=ep['episode_index'],start=rng.randrange(ep['length']),length=ep['length']))
    (args.output/'selection.json').write_text(json.dumps(selection,indent=2))
    vaep=Path('checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    vae=_load_registered_model(str(vaep),'wan_video_vae',torch_dtype=torch.bfloat16,device='cuda')
    vae.eval().requires_grad_(False)
    rows=[]
    with torch.inference_mode():
        for sel in selection:
            f=next(data.glob(f"data/chunk-*/episode_{sel['episode']:06d}.parquet"))
            keys=['observation.state.ee_pose_wxyz','action.osc_target_pose_wxyz','observation.state.gripper_open','action']
            table=pq.read_table(f,columns=keys)
            t={k:torch.tensor(table[k].to_pylist()) for k in keys}
            start=sel['start'];idx=torch.arange(start,start+16);valid=idx<len(table);idx=idx.clamp_max(len(table)-1)
            pose=torch.cat((t[keys[0]][start:start+1],t[keys[1]][idx]))
            grip=torch.cat((t[keys[2]][start:start+1],t['action'][idx,-1:].clamp(0,1)))
            base=codec.encode(pose,grip)
            shuffled=permute_tiles(base,perm)
            assert torch.equal(permute_tiles(shuffled,inv),base)
            variants=[('flatten_chw',flat_chw_encode(pose,grip,lo,hi))] if args.chw_only else [('rothko',base),('flatten_hwc',flat_encode(pose,grip,lo,hi)),('shuffled',shuffled)]
            for name,x in variants:
                z=vae.encode(x[None].cuda().bfloat16(),device='cuda',tiled=False)
                y=vae.decode(z,device='cuda',tiled=False)[0].float().clamp(-1,1).cpu()
                for stage,pixels in [('direct',x),('bf16_pixels',x.bfloat16().float()),('vae',y)]:
                    bad=0
                    if name=='flatten_chw':pred,g,bad=flat_chw_decode(pixels,lo,hi)
                    elif name=='flatten_hwc':pred,g,bad=flat_decode(pixels,lo,hi)
                    else:pred,g=codec.decode(permute_tiles(pixels,inv) if name=='shuffled' else pixels)
                    q=torch.nn.functional.normalize(pred[1:,3:].double(),dim=-1)
                    gt=torch.nn.functional.normalize(pose[1:,3:].double(),dim=-1)
                    pos=(pred[1:,:3].double()-pose[1:,:3].double()).norm(dim=-1)*1000
                    rot=torch.rad2deg(2*torch.acos((q*gt).sum(-1).abs().clamp(0,1)))
                    ge=(g[1:].reshape(-1)-grip[1:].reshape(-1)).abs()
                    gb=((g[1:].reshape(-1)>.5)!=(grip[1:].reshape(-1)>.5)).float()
                    rows.append(dict(**sel,representation=name,stage=stage,valid=valid.tolist(),position_mm=pos.tolist(),rotation_deg=rot.tolist(),gripper_abs=ge.tolist(),gripper_binary=gb.tolist(),degenerate_quaternions=bad))
                del z,y
            print('DONE',len(rows)//(3 if args.chw_only else 9),sel['episode'], 'peak_GiB',torch.cuda.max_memory_allocated()/2**30,flush=True)
            (args.output/'rows.json').write_text(json.dumps(rows))
    summary={}
    for name in (('flatten_chw',) if args.chw_only else ('rothko','flatten_hwc','shuffled')):
        summary[name]={}
        for stage in ('direct','bf16_pixels','vae'):
            subset=[r for r in rows if r['representation']==name and r['stage']==stage]
            summary[name][stage]={str(h):{k:stats([v for r in subset for v,ok in zip(r[k][:h],r['valid'][:h]) if ok]) for k in ('position_mm','rotation_deg','gripper_abs','gripper_binary')} for h in (8,16)}
    report=dict(windows=len(selection),seed=42,stats_fingerprint=codec.norm_stats.fingerprint(),vae_sha256=hashlib.sha256(vaep.read_bytes()).hexdigest(),peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,summary=summary,notes='GT roundtrip only; 2 random episodes/task, 1 random start/episode; padding excluded; original BF16 VAE; clamp output [-1,1]; fixed same tile permutation across frames and duplicate tiles.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
