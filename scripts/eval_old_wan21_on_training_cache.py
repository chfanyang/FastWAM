"""Evaluate old FP32 decoder checkpoints using the new run's exact cached validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from train_wan21_decoder_cached import ROOT, CachedWindows, cached_evaluate, core


class SavedState:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    opts=p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    local_rank=int(os.environ['LOCAL_RANK'])
    device=torch.device('cuda',local_rank)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.15,device)
    dist.init_process_group('nccl')
    rank=dist.get_rank()
    assert dist.get_world_size()==8
    run=ROOT/'runs/libero_wan21_vae_decoder_cached_bs128_lr1e5_const_w100_ep2_fast_20260913'
    old=ROOT/'runs/libero_rothko_vae_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2'
    args=argparse.Namespace(**json.loads((run/'training_config.json').read_text())['arguments'])
    assert args.parallel_cached_eval and not args.cached_smoke
    cache=CachedWindows(args)
    assert args.cache_contract==cache.manifest
    _,episodes=core.discover_episodes(args,rank=rank)
    stats=core.load_norm_stats(args.norm_stats_path,args)
    store=core.EpisodeStore(100)
    if rank==0:
        opts.output.mkdir(parents=True,exist_ok=False)
        metadata={'new_run':str(run),'old_run':str(old),'cache_contract':cache.manifest,
                  'validation_function':'train_wan21_decoder_cached.cached_evaluate',
                  'validation_script_sha256':hashlib.sha256((ROOT/'scripts/train_wan21_decoder_cached.py').read_bytes()).hexdigest(),
                  'old_steps':[4000,4800,5600,7498], 'num_windows':800,
                  'precision':'original FP32 decoder parameters cast to BF16 by exact training validation function; existing BF16 cache',
                  'torch':torch.__version__, 'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
                  'tf32_cudnn':torch.backends.cudnn.allow_tf32}
        (opts.output/'manifest.json').write_text(json.dumps(metadata,indent=2))
    dist.barrier()
    results={}
    for step in (4000,4800,5600,7498):
        path=old/f'checkpoint_step{step:06d}.pt'
        payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert payload['step']==step
        for key in ('decoder','conv2'):
            assert all(v.dtype==torch.float32 for v in payload[key].values() if v.is_floating_point())
        view=SimpleNamespace(model=SimpleNamespace(decoder=SavedState(payload['decoder']),conv2=SavedState(payload['conv2'])))
        if rank==0:print('EVALUATING',step,'old_epoch',step/3749,flush=True)
        result=cached_evaluate(cache,view,store,episodes,stats,args,device,torch.float32)
        result.update(checkpoint=str(path),old_step=step,old_epoch=step/3749)
        if rank==0:
            (opts.output/f'eval_old_step{step:06d}.json').write_text(json.dumps(result,indent=2))
            results[str(step)]=result
            (opts.output/'summary.json').write_text(json.dumps(results,indent=2))
            print('RESULT',step,json.dumps(result['metrics']),flush=True)
        del view,payload,result
        dist.barrier()
    if rank==0:print('ALL_COMPLETE',flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
