"""Exercise the real trainer validation adapter and full per-window evaluation."""
import json,argparse,contextlib
from pathlib import Path
from types import SimpleNamespace
import torch
from hydra import initialize_config_dir,compose
from hydra.utils import instantiate
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils import misc
from fastwam.trainer import Wan22Trainer
from fastwam.models.wan22.fastwam_latent_repeat import FastWAMLatentRepeat

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',action='store_true');a=p.parse_args()
 root=Path('evaluate_results/libero/latent_repeat_validation_fix_20260922');root.mkdir(exist_ok=True);torch.set_num_threads(2);register_default_resolvers();misc.register_work_dir(root)
 with initialize_config_dir(config_dir=str(Path('configs').resolve()),version_base='1.3'):
  cfg=compose(config_name='train',overrides=['task=libero_goal_latent_repeat_absolute_ray0_wan21','data.train.latent_cache_only=false'])
 ds=instantiate(cfg.data.train);samples=[]
 for idx in [0,137]:
  s=ds[idx];batch=Wan22Trainer._to_batched_eval_sample(s)
  assert 'repeat_pose' not in batch and 'repeat_gripper' not in batch
  pose,grip=FastWAMLatentRepeat.repeat_targets(SimpleNamespace(raymap_codec=ds.raymap_codec),batch)
  torch.testing.assert_close(pose[0],s['repeat_pose'],atol=0,rtol=0)
  torch.testing.assert_close(grip[0],s['repeat_gripper'],atol=1e-7,rtol=0)
  samples.append((idx,batch))
 report={'actual_trainer_adapter_samples':[i for i,_ in samples],'targets_exact':True,'gpu':a.gpu}
 if a.gpu:
  model=instantiate(cfg.model,device='cuda:0',model_dtype=torch.bfloat16);model.requires_grad_(False);model.eval()
  trainer=SimpleNamespace(accelerator=SimpleNamespace(device=torch.device('cuda:0'),process_index=0,autocast=lambda:torch.autocast('cuda',dtype=torch.bfloat16)),cfg=cfg,eval_num_inference_steps=20,eval_dir=str(root),global_step=0)
  report['metrics']=[]
  for idx,s in samples:
   result=Wan22Trainer._evaluate_visual_action(trainer,model,s,False,eval_sample_index=idx,eval_seed=42)
   assert torch.isfinite(torch.tensor(result['val_loss']))
   report['metrics'].append(result)
  report['peak_gpu_gib']=torch.cuda.max_memory_allocated()/2**30
 path=root/('gpu.json' if a.gpu else 'cpu.json');assert not path.exists();path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
