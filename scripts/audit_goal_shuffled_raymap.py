"""Scoped data parity, one-step train, inference and checkpoint round-trip check."""
import argparse
import copy
import json
import pickle
import time
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from torch.utils.data import default_collate
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fastwam.utils import misc
from prepare_goal_shuffled_raymap import NumericWindows, config


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',action='store_true');parser.add_argument('--cached',action='store_true');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);report_path=args.output/('cached.json' if args.cached else 'gpu.json' if args.gpu else 'cpu.json')
    if report_path.exists():raise FileExistsError(report_path)
    torch.set_num_threads(1);misc.register_work_dir(args.output);cfg=config();numeric=NumericWindows(cfg)
    raw_cfg=copy.deepcopy(cfg.data.train);raw_cfg.latent_cache_only=False;raw_cfg.latent_cache_dir=None
    ds=instantiate(raw_cfg)
    baseline_cfg=copy.deepcopy(raw_cfg);baseline_cfg._target_='fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset'
    del baseline_cfg.source_latent_cache_dir
    baseline=instantiate(baseline_cfg)
    indices=[0,numeric.offsets[1]-16,numeric.offsets[1]-1,26447,52894]
    rows=[]
    for i in indices:
        x=ds[i];b=baseline[i]
        assert torch.equal(ds.raymap_codec.transform(x['raymap'],inverse=True),b['raymap'])
        assert torch.equal(x['raymap'],numeric.raymap(i)),f'Raw-window/cache-window mismatch at {i}'
        assert torch.equal(x['video'],b['video']) and torch.equal(x['context'],b['context'])
        assert torch.equal(x['raymap_is_pad'],b['raymap_is_pad'])
        bp,bg=baseline.raymap_codec.decode(b['raymap'],b['current_endpose'])
        sp,sg=ds.raymap_codec.decode(x['raymap'],x['current_endpose'])
        assert torch.equal(bp,sp) and torch.equal(bg,sg)
        assert torch.equal(x['raymap'][...,:224],x['raymap'][...,224:])
        restored=pickle.loads(pickle.dumps(ds.raymap_codec))
        assert torch.equal(restored.transform(x['raymap'],inverse=True),b['raymap'])
        rows.append(dict(index=i,padding_steps=int(x['action_is_pad'].sum()),direct_bit_exact=True))
    report=dict(samples=rows,codec=ds.raymap_codec.metadata(),scope='pipeline validation only')
    if args.cached:
        cached=instantiate(cfg.data.train)
        model=instantiate(cfg.model,device='cuda:0',model_dtype=torch.bfloat16)
        model.eval().requires_grad_(False);model.validate_latent_cache_metadata(cached.latent_cache_metadata)
        with torch.inference_mode():
            for i in indices:
                pixel=ds[i];z=model.vae.encode(pixel['raymap'][None].cuda().bfloat16(),device='cuda',tiled=False)[0].cpu()
                cc=cached[i]
                assert torch.equal(z,cc['raymap_latents'])
                assert torch.equal(cc['rgb_latents'],numeric.source[i][0])
            # One real cached training forward verifies that no VAE encode is invoked.
            def forbidden(*xs,**kw):raise AssertionError('Cached training unexpectedly invoked VAE encode')
            model.vae.encode=forbidden
            loss,metrics=model.training_loss(default_collate([cached[0]]))
            assert torch.isfinite(loss)
        report.update(cache_online_bit_exact=True,cached_loss=float(loss),windows=len(cached))
    elif args.gpu:
        started=time.monotonic();torch.manual_seed(42)
        model=instantiate(cfg.model,device='cuda:0',model_dtype=torch.bfloat16)
        assert model.raymap_codec.metadata()==ds.raymap_codec.metadata()
        model.requires_grad_(False);model.video_expert.requires_grad_(True)
        batch=default_collate([ds[0]]);opt=torch.optim.AdamW(model.video_expert.parameters(),lr=1e-6)
        loss,metrics=model.training_loss(batch);assert torch.isfinite(loss)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.video_expert.parameters(),1.)
        assert torch.isfinite(grad);opt.step();opt.zero_grad(set_to_none=True)
        assert abs(metrics['loss_raymap']-5*metrics['loss_raymap_raw'])<1e-5
        del opt;model.eval().requires_grad_(False);torch.cuda.empty_cache()
        sample=ds[0]
        pred=model.infer(prompt=None,input_image=sample['video'][:,0],input_raymap=sample['raymap'][:,0],current_endpose=sample['current_endpose'],context=sample['context'],context_mask=sample['context_mask'],num_inference_steps=20,seed=42,decode_future_rgb=False)
        assert pred['pose'].shape==(1,17,7) and torch.isfinite(pred['pose']).all()
        path=args.output/'smoke_step1.pt';model.save_checkpoint(path,step=1)
        payload=model.load_checkpoint(path);assert payload['step']==1
        meta=model._visual_action_checkpoint_config();bad=copy.deepcopy(meta);bad.pop('permutation_sha256')
        try:model._validate_visual_action_checkpoint_config(bad,checkpoint_path=str(path))
        except ValueError:pass
        else:raise AssertionError('Missing permutation identity was accepted')
        report.update(loss=float(loss.detach()),grad_norm=float(grad),inference_shape=list(pred['pose'].shape),
                      checkpoint_roundtrip=True,identity_rejection=True,peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-started)
    report_path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
