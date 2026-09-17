"""Read-only held-out checkpoint comparison; per-window deterministic noise."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import default_collate
from fastwam.utils.config_resolvers import register_default_resolvers
from experiments.vlabench.audit_three_camera_vae_reconstruction import metrics, summarize


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--vae',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--rank',type=int,required=True)
    p.add_argument('--shards',type=int,default=2)
    a=p.parse_args()
    torch.set_num_threads(2)
    register_default_resolvers()
    cfg=OmegaConf.load(a.run/'config.yaml')
    cfg.model.load_text_encoder=False
    cfg.model.vae_safetensors_path=str(a.vae.resolve())
    cfg.model.allow_vae_mismatch=True
    cfg.model.rothko_decode_mode='legacy'
    cfg.model.rothko_decode_anchor_alpha=0.
    ds=instantiate(cfg.data.val)
    model=instantiate(cfg.model,device='cuda',model_dtype=torch.bfloat16)
    model.load_checkpoint(str(a.run/f'checkpoints/weights/step_{a.step:06d}.pt'))
    model.validate_dataset_stats(cfg.data.val.pretrained_norm_stats)
    model.eval().requires_grad_(False)
    a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/f'step{a.step:06d}_shard{a.rank}.json'
    if path.exists(): raise FileExistsError(path)
    rows=[]
    with torch.no_grad():
        for i in range(a.rank,len(ds),a.shards):
            sample=ds[i]
            seed=420000+i
            torch.manual_seed(seed);torch.cuda.manual_seed(seed)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss,parts=model.training_loss(default_collate([sample]))
            pred=model.infer(prompt=None,input_image=sample['video'][:,0][None],
                input_raymap=sample['raymap'][:,0][None],
                proprio=sample['proprio'][0],current_endpose=sample['current_endpose'],
                context=sample['context'],context_mask=sample['context_mask'],
                num_frames=17,num_inference_steps=20,seed=seed,tiled=False,
                decode_future_rgb=False)
            error=metrics(pred['pose'][0,1:].cpu(),pred['gripper'][0,1:].cpu(),
                          sample['future_endpose'],sample['future_gripper'])
            row=dict(val_index=i,seed=seed,val_loss=float(loss),
                     action_is_pad=sample['action_is_pad'].tolist(),prediction_vs_target=error)
            rows.append(row)
            payload=dict(step=a.step,vae=str(a.vae),rank=a.rank,shards=a.shards,
                expected_windows=len(ds),windows=rows,complete=False)
            tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(payload));tmp.replace(path)
            print(f'step={a.step} window={i} done={len(rows)} loss={float(loss):.6f}',flush=True)
    payload.update(complete=True,val_loss=float(np.mean([r['val_loss'] for r in rows])),
        first8=summarize(rows,'prediction_vs_target',8),all16=summarize(rows,'prediction_vs_target',16))
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(payload));tmp.replace(path)


if __name__=='__main__': main()
