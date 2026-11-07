"""Offline OSC checkpoint audit, preserving predictions and excluding padding in metrics."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils import misc

def errors(p,g,pg,gg):
    d=p[:,:3].double()-g[:,:3].double()
    q=torch.nn.functional.normalize(p[:,3:].double(),dim=-1);r=torch.nn.functional.normalize(g[:,3:].double(),dim=-1)
    return dict(position_axis_mae_mm=(d.abs().mean(-1)*1000).tolist(),position_l2_mm=(d.norm(dim=-1)*1000).tolist(),rotation_deg=torch.rad2deg(2*torch.acos((q*r).sum(-1).abs().clamp(0,1))).tolist(),gripper_abs_error=(pg-gg).abs().flatten().tolist())

def summarize(rows,kind,horizon,valid_only=True):
    result={}
    for key in rows[0][kind]:
        v=[value for row in rows for j,value in enumerate(row[kind][key][:horizon]) if not valid_only or not row['action_is_pad'][j]]
        result[key]=dict(count=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(np.max(v)))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--step',type=int,default=4140);p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=8);p.add_argument('--aggregate',action='store_true');a=p.parse_args()
    cfg=OmegaConf.load(a.run/'config.yaml');indices=list(cfg.eval_sample_indices)
    if a.aggregate:
        rows=[]
        for rank in range(a.shards):
            part=json.loads((a.output/f'shard{rank}.json').read_text());assert part['complete'] and part['step']==a.step;rows+=part['rows']
        rows.sort(key=lambda r:r['probe_offset']);assert [r['index'] for r in rows]==indices
        result=dict(complete=True,step=a.step,checkpoint=str(a.run/f'checkpoints/weights/step_{a.step:06d}.pt'),num_windows=len(rows),scope='fixed training probes, not heldout',padding='OSC motion zero, gripper hold; padding excluded from primary metrics',summary={str(h):{k:summarize(rows,k,h) for k in ['prediction','zero_increment']} for h in [8,16]},all16_including_padding=summarize(rows,'prediction',16,False),by_task={task:{str(h):{k:summarize([r for r in rows if r['task']==task],k,h) for k in ['prediction','zero_increment']} for h in [8,16]} for task in sorted({r['task'] for r in rows})},rows=rows)
        (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['summary'],indent=2));return
    torch.set_num_threads(2);register_default_resolvers();a.output.mkdir(parents=True,exist_ok=True);misc.register_work_dir(a.output)
    cfg.data.train.latent_cache_only=False;cfg.data.train.latent_cache_dir=None
    cfg.model.load_text_encoder=False;cfg.model.skip_dit_load_from_pretrain=True
    assert cfg.model.vae_safetensors_path is None and cfg.model.raymap_representation=='libero_rothko_osc_delta'
    ds=instantiate(cfg.data.train);model=instantiate(cfg.model,device='cuda:0',model_dtype=torch.bfloat16)
    model.load_checkpoint(str(a.run/f'checkpoints/weights/step_{a.step:06d}.pt'));model.eval().requires_grad_(False)
    path=a.output/f'shard{a.rank}.json';assert not path.exists();rows=[]
    with torch.inference_mode():
        for offset in range(a.rank,len(indices),a.shards):
            idx=indices[offset];seed=int(cfg.eval_random_seed)+offset;sample=ds[idx]
            pred=model.infer(prompt=None,input_image=sample['video'][:,0][None],input_raymap=sample['raymap'][:,0][None],proprio=sample['proprio'][0],current_endpose=sample['current_endpose'],context=sample['context'],context_mask=sample['context_mask'],num_frames=17,num_inference_steps=20,seed=seed,tiled=False,decode_future_rgb=False)
            pose=pred['pose'][0,1:].float().cpu();grip=pred['gripper'][0,1:].float().cpu();gt=sample['future_endpose'].float().cpu();gg=sample['future_gripper'].float().cpu();zero=torch.zeros_like(gt);zero[:,3]=1
            current_grip=ds.raymap_codec.read_gripper(sample['raymap'].permute(1,0,2,3))[0].reshape(1,1).expand_as(gg)
            rows.append(dict(index=idx,probe_offset=offset,seed=seed,task=sample['prompt'],action_is_pad=sample['action_is_pad'].tolist(),prediction=errors(pose,gt,grip,gg),zero_increment=errors(zero,gt,current_grip,gg),predicted_pose=pose.tolist(),target_pose=gt.tolist(),predicted_gripper=grip.tolist(),target_gripper=gg.tolist()))
            payload=dict(complete=False,step=a.step,rank=a.rank,rows=rows);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(payload));tmp.replace(path);print('DONE',idx,flush=True)
    payload['complete']=True;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(payload));tmp.replace(path)
if __name__=='__main__':main()
