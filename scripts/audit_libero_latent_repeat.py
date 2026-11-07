"""Bounded integration smoke; no training run, no checkpoint overwrite."""
import argparse,json,time
from pathlib import Path
import torch
from hydra import compose,initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils import misc

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--gpu',action='store_true');a=a.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);report_path=a.output/('gpu.json' if a.gpu else 'cpu.json');assert not report_path.exists()
    torch.set_num_threads(2);register_default_resolvers();misc.register_work_dir(a.output)
    with initialize_config_dir(config_dir=str(Path('configs').resolve()),version_base='1.3'):
        cfg=compose(config_name='train',overrides=['task=libero_goal_latent_repeat_absolute_ray0_wan21'])
    ds=instantiate(cfg.data.train)
    from fastwam.representations.libero_latent_repeat import LiberoLatentRepeat
    codec=LiberoLatentRepeat(ds.raymap_codec)
    indices=[0,ds._repeat_ends[0]-16,ds._repeat_ends[0]-1,26447,52894]
    samples=[ds[i] for i in indices];rows=[]
    for i,s in zip(indices,samples):
        z=codec.encode(s['repeat_pose'][None],s['repeat_gripper'][None]);p,g,bad=codec.decode(z,s['current_endpose'][None])
        assert torch.isfinite(p).all() and not bad.any()
        assert 'raymap_latents' not in s
        video=ds.raymap_codec.encode(s['repeat_pose'],s['repeat_gripper'])
        bp,bg=ds.raymap_codec.decode(video,s['current_endpose'])
        from fastwam.representations.rothko import quaternion_wxyz_to_matrix as rot
        torch.testing.assert_close(p[0,:,:3],bp[:,:3],atol=3e-6,rtol=0)
        torch.testing.assert_close(rot(p[0,:,3:]),rot(bp[:,3:]),atol=3e-6,rtol=0)
        torch.testing.assert_close(g[0],bg,atol=2e-6,rtol=0)
        rows.append({'index':i,'padding_steps':int(s['action_is_pad'].sum()),'numerical_std':float(z.std()),'rgb_std':float(s['rgb_latents'].float().std()),'direct_matches_baseline':True})
    report={'samples':rows,'rgb_weight':float(cfg.model.loss.lambda_rgb),'action_weight':float(cfg.model.loss.lambda_raymap),'scope':'pipeline smoke only; not trained quality'}
    if a.gpu:
        started=time.monotonic();torch.manual_seed(42)
        model=instantiate(cfg.model,device='cuda:0',model_dtype=torch.bfloat16)
        model.requires_grad_(False);model.video_expert.requires_grad_(True)
        model.validate_latent_cache_metadata(ds.latent_cache.metadata)
        calls={'encode':0,'decode':0};encode=model.vae.encode;decode=model.vae.decode
        def counted_encode(*xs,**kw):calls['encode']+=1;return encode(*xs,**kw)
        def counted_decode(*xs,**kw):calls['decode']+=1;return decode(*xs,**kw)
        model.vae.encode=counted_encode;model.vae.decode=counted_decode
        # One optimizer step using the exact original cached RGB path and numerical action path.
        batch=default_collate(samples[:1]);opt=torch.optim.AdamW(model.video_expert.parameters(),lr=1e-6)
        loss,metrics=model.training_loss(batch);assert torch.isfinite(loss)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.video_expert.parameters(),1.)
        assert torch.isfinite(grad);opt.step();opt.zero_grad(set_to_none=True)
        assert calls=={'encode':0,'decode':0},calls
        assert abs(metrics['loss_raymap']-5*metrics['loss_raymap_raw'])<1e-5
        del opt;model.requires_grad_(False);torch.cuda.empty_cache()
        # Verify real online RGB encoder and the direct numerical inference reader.
        cfg.data.train.latent_cache_only=False
        raw_ds=instantiate(cfg.data.train)
        s=raw_ds[0]
        pred=model.infer(prompt=None,input_image=s['video'][:,0],input_raymap=s['raymap'][:,0],current_endpose=s['current_endpose'],context=s['context'],context_mask=s['context_mask'],num_inference_steps=20,seed=42,decode_future_rgb=False)
        assert pred['pose'].shape==(1,17,7) and torch.isfinite(pred['pose']).all()
        assert calls=={'encode':1,'decode':0},calls
        report.update(loss=float(loss.detach()),metrics=metrics,grad_norm=float(grad),vae_calls=calls,inference_shape=list(pred['pose'].shape),degenerate_quaternions=pred['degenerate_quaternions'],peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-started,checkpoint_identity=model._visual_action_checkpoint_config()['action_latent_encoding'])
    report_path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
