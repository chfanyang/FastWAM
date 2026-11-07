#!/usr/bin/env python3
"""CPU-only check for a cached-training transfer (training PNGs are unnecessary)."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts/realworld')]
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.realworld.common import sha256_file
import finetune_task_vae as decoder

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['insert_pen','stack_bowls_three','white_board']);a=p.parse_args()
 torch.set_num_threads(1);register_default_resolvers()
 epochs=5 if a.task=='white_board' else 8
 with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
  cfg=compose(config_name='train',overrides=[f'task=realworld_{a.task}_ray0_absolute_wan21_q9995_ep{epochs}'])
 assert cfg.num_epochs==epochs and cfg.batch_size==4 and cfg.gradient_accumulation_steps==4
 assert cfg.data.train.latent_cache_only and not cfg.model.video_dit_config.use_gradient_checkpointing
 train=instantiate(cfg.data.train); val=instantiate(cfg.data.val)
 metadata=train.base.latent_cache_metadata; cache=Path(cfg.data.train.latent_cache_dir)
 assert metadata['complete'] and (cache/'_SUCCESS').is_file()
 end=0
 for s in metadata['shards']:
  assert s['start']==end
  assert (cache/s['file']).stat().st_size==(s['end']-s['start'])*math.prod(metadata['latent_shape'])*len(metadata['modalities'])*2
  end=s['end']
 assert end==metadata['num_samples']==len(train.base)
 for r in train.records+val.records:
  assert sha256_file(Path(cfg.data.train.dataset_dir)/r['path']/'states.npz')==r['states_sha256']
 for i in (0,len(train)//2,len(train)-1):
  x=train[i]
  assert x['rgb_latents'].shape==x['raymap_latents'].shape==(16,5,28,56)
  assert torch.isfinite(x['rgb_latents']).all() and torch.isfinite(x['raymap_latents']).all()
  assert x['context'].shape==(128,4096)
 probes=json.loads(Path(cfg.eval_sample_manifest).read_text())['samples'];assert len(probes)==2
 for probe in probes:
  x=val[probe['val_dataset_index']]
  assert x['video'].shape==x['raymap'].shape==(3,17,224,448)
  assert torch.isfinite(x['video']).all() and torch.isfinite(x['raymap']).all()
 decoder.CONFIG=ROOT/f'configs/vae/realworld_{a.task}_q9995_decoder_ep1.json'
 inputs=decoder.Inputs();inputs.load_extra()
 assert inputs.args.epochs==1 and not inputs.args.reconstruction_loss_includes_padding
 print(json.dumps(dict(task=a.task,dit_epochs=epochs,dit_steps=epochs*math.ceil(len(train)/128),train_windows=len(train),val_windows=len(val),vae_steps=math.ceil(len(inputs.refs)/64),vae_train_windows=len(inputs.refs),supplemental_windows=len(inputs.missing),status='READY'),sort_keys=True),flush=True)
if __name__=='__main__':main()
