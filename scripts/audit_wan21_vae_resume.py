"""Read-only checkpoint/cache compatibility probe; no training updates."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import finetune_rothko_vae_decoder as core
from fastwam.representations.libero_rothko import LiberoRothkoCodec

run=ROOT/'runs/libero_rothko_vae_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2'
c=argparse.Namespace(**json.loads((run/'training_config.json').read_text())['arguments'])
payload=torch.load(run/'checkpoint_latest.pt',map_location='cpu',weights_only=False,mmap=True)
opt=payload['optimizer']
print('RESUME',json.dumps({'step':payload['step'],'optimizer_present':opt is not None,'optimizer_states':len(opt['state']) if opt else 0,'saved_lr':opt['param_groups'][0]['lr'] if opt else None}),flush=True)
assert payload['step']==7498 and opt and len(opt['state'])>0
train,heldout=core.discover_episodes(c)
root=str((Path(c.data_root)/core.SUITE_DIRS['libero_spatial']).resolve())
episode=next(e for e in train+heldout if e.dataset_root==root and e.episode_index==0)
w=core.materialize_windows(core.EpisodeStore(4),[core.WindowRef(episode,0)],16)
device=torch.device('cuda:0');stats=core.load_norm_stats(c.norm_stats_path,c)
target,_=core.build_target_batch(w,stats,device,torch.float32,c)
cache=ROOT/'data/libero_all4_rothko_centerfrac05_wan21_bf16_h16_latents'
meta=json.loads((cache/'metadata.json').read_text());shard=meta['shards'][0]
arr=np.memmap(cache/shard['file'],mode='r',dtype=np.uint16,shape=(shard['end']-shard['start'],2,*meta['latent_shape']))
cached=torch.from_numpy(np.array(arr[0,1],copy=True)).view(torch.bfloat16).unsqueeze(0).to(device)
with torch.inference_mode():
 vae=core.load_base_vae(c.base_vae,c.vae_variant,device,torch.float32)
 with core.autocast_context(device,True):online=core.vae_encode(vae.model,target,vae.scale)
 diff=(cached.float()-online.float()).abs()
 print('CACHE_PROBE',json.dumps({'mapping':'candidate global sample0 = spatial episode0 start0; not fully audited','exact':torch.equal(cached,online),'mae':diff.mean().item(),'max':diff.max().item(),'training_encoder_weights':'fp32','cache_encoder_weights':'bf16','cache_tiled':False}),flush=True)
 def compare(name,a,b):
  delta=(a.float()-b.float()).abs();print(name,json.dumps({'exact':torch.equal(a,b),'mae':delta.mean().item(),'max':delta.max().item()}),flush=True)
 vae_bf16=core.load_base_vae(c.base_vae,c.vae_variant,device,torch.bfloat16)
 same_input_bf16=core.vae_encode(vae_bf16.model,target.bfloat16(),vae_bf16.scale)
 compare('SAME_TARGET_precision_only',same_input_bf16,online)
 compare('CACHE_vs_bf16_core_target',cached,same_input_bf16)
 cpu_stats=core.load_norm_stats(c.norm_stats_path,c)
 codec=LiberoRothkoCodec(norm_stats=cpu_stats,expected_action_horizon=16)
 raymap=codec.encode(torch.from_numpy(w[0].pose),torch.from_numpy(w[0].gripper)).unsqueeze(0).to(device)
 compare('TARGET_core_vs_policy_codec',target,raymap)
 policy_latent=vae_bf16.encode(raymap.bfloat16(),device=device,tiled=False)
 compare('CACHE_vs_policy_bf16',cached,policy_latent)
