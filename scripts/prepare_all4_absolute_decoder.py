"""Scoped four-suite absolute decoder workflow. GPU stages require --execute.

The legacy trainer is reused with process-local hooks only; its file, defaults,
other runs, and representation code are never modified.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import finetune_rothko_vae_decoder as core
from fastwam.datasets.latent_cache import LatentCacheReader,build_dataset_contract,sha256_file
from fastwam.representations.libero_rothko import LiberoRothkoCodecConfig
from fastwam.representations.libero_rothko_all_absolute import LiberoAllAbsoluteRothkoCodec

CONFIG=ROOT/'configs/vae/libero_all4_all_absolute_decoder_wan21_bs2_ga4_lr1e-5_ep2.json'
CACHE=ROOT/'data/libero_all4_rothko_all_absolute_2cam224_wan21_bf16_h16_latents'
PREP=ROOT/'evaluate_results/libero/all4_absolute_decoder_preparation_padding_20260919'

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def digest_obj(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def enumerate_padded_refs(episodes,horizon):
    return [core.WindowRef(e,start) for e in episodes for start in range(e.length)]

def materialize_padded_windows(store,refs,horizon):
    windows=[]
    for ref in refs:
        arrays=store.get(ref.episode)
        assert 0<=ref.start<ref.episode.length
        indices=np.minimum(np.arange(ref.start,ref.start+horizon),ref.episode.length-1)
        pose=np.concatenate((arrays.state_pose[ref.start:ref.start+1],arrays.action_pose[indices]),axis=0)
        grip=np.concatenate((np.clip(arrays.state_gripper[ref.start:ref.start+1],0,1),
                             np.clip(arrays.action[indices,-1:],0,1)),axis=0)
        windows.append(core.ActionWindow(ref.episode,ref.start,pose,grip))
    return windows

def padded_uniform_validation(store,episodes,count,horizon):
    refs=[]
    for ep in sorted(episodes,key=lambda e:(e.dataset_root,e.task,e.episode_index)):
        starts=np.unique(np.rint(np.linspace(0,ep.length-1,min(count,ep.length))).astype(int))
        refs.extend(core.WindowRef(ep,int(start)) for start in starts)
    return materialize_padded_windows(store,refs,horizon)

def valid_steps(window,horizon=16):
    return torch.arange(horizon)+window.start<window.episode.length

def masked_metrics(values,valid,horizon):
    x=values[:,:horizon][valid[:,:horizon]]
    return dict(n=x.numel(),mean=x.mean().item(),p95=x.quantile(.95).item(),max=x.max().item())

class Inputs:
    def __init__(self,config=CONFIG):
        self.config=Path(config).resolve();self.c=json.loads(self.config.read_text())
        self.args=SimpleNamespace(**self.c,config=str(self.config))
        assert self.args.suites==['libero_spatial','libero_object','libero_goal','libero_10'] and self.args.dataset_roots is None
        assert (self.args.batch_size,self.args.grad_accum_steps,self.args.epochs)==(2,4,2)
        assert self.args.vae_variant=='wan2.1-t2v-1.3b' and self.args.bf16 and self.args.zero_optimizer
        assert self.args.lr==1e-5 and self.args.eval_episodes_per_task==2
        self.train,self.val=core.discover_episodes(self.args,rank=1)
        self.refs=enumerate_padded_refs(self.train,16)
        store=core.EpisodeStore(2)
        self.val_windows=padded_uniform_validation(store,self.val,10,16)
        assert (len(self.train),len(self.val),len(self.refs),len(self.val_windows))==(1632,80,264409,800)
        assert not set(self.train)&set(self.val)
        meta=json.loads(Path(self.args.norm_stats_path).with_suffix('.json').read_text())
        cfg=LiberoRothkoCodecConfig(frame0_pose_mode='absolute',absolute_position_min=meta['absolute_position_min'],absolute_position_max=meta['absolute_position_max'])
        self.codec=LiberoAllAbsoluteRothkoCodec(config=cfg,norm_stats=self.args.norm_stats_path,expected_action_horizon=16)
        # Cache concatenation order must match experiment 1, not alphabetic suite order.
        self.roots=[str((Path(self.args.data_root)/core.SUITE_DIRS[suite]).resolve()) for suite in self.args.suites]
        self.offsets={};count=0
        for root in self.roots:
            records=[json.loads(x) for x in (Path(root)/'meta/episodes.jsonl').read_text().splitlines()]
            local_count=0
            for ep in sorted(records,key=lambda x:x['episode_index']):
                self.offsets[(root,ep['episode_index'])]=(count,ep['length'])
                count+=ep['length'];local_count+=ep['length']
            assert local_count==json.loads((Path(root)/'meta/info.json').read_text())['total_frames']
        assert count==277713
        self.contract=build_dataset_contract(dataset_dirs=self.roots,dataset_length=count,num_frames=17,
            video_size=[224,448],raymap_representation=self.codec.representation,
            raymap_codec_metadata=self.codec.metadata(),norm_stats_sha256=self.codec.norm_stats.fingerprint())
        self.manifest=dict(train_episodes=[dataclasses.asdict(e) for e in self.train],
            val_episodes=[dataclasses.asdict(e) for e in self.val],
            val_windows=[dict(dataset_root=w.episode.dataset_root,episode=w.episode.episode_index,start=w.start) for w in self.val_windows],
            train_cache_indices=[self.index(r) for r in self.refs],train_windows=len(self.refs),steps_per_epoch=4132,total_steps=8264,warmup_steps=413,
            stats_fingerprint=self.codec.norm_stats.fingerprint(),vae_sha256=sha256_file(self.args.base_vae),
            config_sha256=sha256_file(self.config),contract=self.contract,
            padding_policy='all_starts_edge_repeat_v1',reconstruction_loss_includes_padding=True,validation_metrics_exclude_padding=True)
        assert len(set(self.manifest['train_cache_indices']))==264409
        self.identity=digest_obj(self.manifest)
        self.reader=None

    def index(self,ref):
        assert ref.episode.dataset_root in self.roots
        offset,length=self.offsets[(ref.episode.dataset_root,ref.episode.episode_index)]
        assert length==ref.episode.length and 0<=ref.start<length
        return offset+ref.start

    def open_cache(self):
        self.reader=LatentCacheReader(CACHE,expected_dataset_contract=self.contract)
        m=self.reader.metadata
        assert m['vae_identity']['sha256']==self.manifest['vae_sha256']
        assert m['model_variant']=='wan2.1-t2v-1.3b' and m['encoding_torch_dtype']=='torch.bfloat16'
        assert m['modalities']==['rgb','raymap'] and m['latent_shape']==[16,5,28,56]
        assert not m.get('benchmark_only') and len(self.reader)==277713
        assert m.get('verification',{}).get('training_loss_bit_exact')
        self.cache_sha=sha256_file(CACHE/'metadata.json')

    def target(self,w):
        return self.codec.encode(torch.from_numpy(w.pose),torch.from_numpy(w.gripper))

    def epoch_batches(self,refs,start_step,steps,effective,world,rank,device):
        store=core.EpisodeStore(self.args.episode_cache_size)
        trainset=set(self.train)
        for step in range(start_step,steps):
            for acc in range(self.args.grad_accum_steps):
                offset=step*effective+acc*self.args.batch_size*world+rank*self.args.batch_size
                batch=refs[offset:offset+self.args.batch_size]
                assert len(batch)==self.args.batch_size and all(r.episode in trainset for r in batch)
                windows=materialize_padded_windows(store,batch,16)
                target=torch.stack([self.target(w) for w in windows])
                latent=torch.stack([self.reader[self.index(r)][1] for r in batch])
                assert target.dtype==torch.float32 and latent.dtype==torch.bfloat16
                yield target.to(device),latent.to(device)

@torch.no_grad()
def audit(inputs,output):
    assert not output.exists(),'Refusing to overwrite audit'
    vae=core.load_base_vae(inputs.args.base_vae,inputs.args.vae_variant,torch.device('cuda:0'),torch.bfloat16)
    store=core.EpisodeStore(4);probes=[]
    # Both train/held-out pools, every task, first, last complete, and final padded starts.
    for pool in [inputs.train,inputs.val]:
        for root,task in sorted({(e.dataset_root,e.task) for e in pool}):
            ep=next(e for e in pool if (e.dataset_root,e.task)==(root,task))
            for start in sorted({0,max(ep.length-16,0),ep.length-1}):
                ref=core.WindowRef(ep,start);w=materialize_padded_windows(store,[ref],16)[0]
                actual=vae.encode(inputs.target(w)[None].cuda().bfloat16(),device='cuda',tiled=False)[0].cpu()
                cached=inputs.reader[inputs.index(ref)][1]
                assert torch.equal(actual,cached),'Cache target/encoder mismatch'
                probes.append(dict(dataset_root=root,task=task,episode=ep.episode_index,start=start,exact=True))
    atomic(output,dict(identity=inputs.identity,cache_sha256=inputs.cache_sha,probes=probes))

@torch.no_grad()
def evaluate(inputs,vae,store,episodes,stats,args,device,dtype):
    assert set(episodes)==set(inputs.val)
    ev=core.load_base_vae(args.base_vae,args.vae_variant,device,torch.bfloat16)
    ev.model.decoder.load_state_dict(vae.model.decoder.state_dict());ev.model.conv2.load_state_dict(vae.model.conv2.state_dict())
    ev.eval().requires_grad_(False);positions=[];rotations=[];grippers=[]
    for w in inputs.val_windows:
        ref=core.WindowRef(w.episode,w.start)
        z=inputs.reader[inputs.index(ref)][1][None].to(device)
        video=ev.decode(z,device=device,tiled=False).float().clamp(-1,1).cpu()
        pred,g=inputs.codec.decode(video)
        target=torch.from_numpy(w.pose).double()
        pe=(pred[0,1:,:3].double()-target[1:,:3]).norm(dim=-1)
        q=torch.nn.functional.normalize(pred[0,1:,3:].double(),dim=-1);gt=torch.nn.functional.normalize(target[1:,3:],dim=-1)
        re=torch.rad2deg(2*torch.acos((q*gt).sum(-1).abs().clamp(0,1)))
        ge=(g[0,1:]-torch.from_numpy(w.gripper)[1:]).abs().flatten()
        assert all(torch.isfinite(x).all() for x in [pe,re,ge])
        positions.append(pe);rotations.append(re);grippers.append(ge)
    pe=torch.stack(positions);re=torch.stack(rotations);ge=torch.stack(grippers)
    valid=torch.stack([valid_steps(w) for w in inputs.val_windows])
    result=dict(future_position_mae_m=pe[valid].mean().item(),future_rotation_mean_deg=re[valid].mean().item(),future_gripper_mae=ge[valid].mean().item(),
        num_windows=800,num_eval_episodes=80,decode_mode='legacy',anchor_alpha=0,precision='BF16 deployment decoder',manifest_sha256=inputs.identity,
        metrics={})
    for horizon in [8,16]:
        result['metrics'][str(horizon)]={name:masked_metrics(x,valid,horizon) for name,x in [('position_m',pe),('rotation_deg',re),('gripper_abs',ge.double())]}
    result['per_suite']={}
    for root in inputs.roots:
        selected=torch.tensor([w.episode.dataset_root==root for w in inputs.val_windows])
        result['per_suite'][Path(root).name]={str(h):{name:masked_metrics(x[selected],valid[selected],h)
            for name,x in [('position_m',pe),('rotation_deg',re),('gripper_abs',ge.double())]} for h in [8,16]}
    del ev;torch.cuda.empty_cache()
    return result

def train(inputs,resume):
    assert int(os.environ.get('WORLD_SIZE','0'))==8,'Use eight torchrun ranks'
    audit_report=json.loads((PREP/'cache_audit.json').read_text())
    assert audit_report['identity']==inputs.identity and audit_report['cache_sha256']==inputs.cache_sha
    a=inputs.args;a.resume=resume
    output=Path(a.output_dir)
    if resume:
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        assert saved['metadata']['all4_absolute_identity']==inputs.identity
        assert saved['metadata']['cache_metadata_sha256']==inputs.cache_sha
        assert saved.get('optimizer') is not None
        del saved
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    # Hooks are local to this dedicated executable, never imported by a running trainer.
    a.cached_prefetch=True
    core.enumerate_window_refs=enumerate_padded_refs
    core.cached_epoch_batches=inputs.epoch_batches
    core.load_norm_stats=lambda *unused:inputs.codec.norm_stats
    core.evaluate=lambda *xs:evaluate(inputs,*xs)
    core.vae_encode=lambda *xs:(_ for _ in ()).throw(RuntimeError('Unexpected online encoder use'))
    original_wandb=core.init_wandb
    def init_wandb(args,metadata,rank):
        metadata.update(raymap_representation=inputs.codec.representation,pose_mode='all_absolute',
            all4_absolute_identity=inputs.identity,cache_metadata_sha256=inputs.cache_sha,
            validation_manifest=inputs.manifest['val_windows'],padding_policy=inputs.manifest['padding_policy'],
            reconstruction_loss_includes_padding=True,validation_metrics_exclude_padding=True)
        if rank==0:
            atomic(Path(args.output_dir)/'training_config.json',metadata)
            atomic(Path(args.output_dir)/'data_manifest.json',inputs.manifest)
        return original_wandb(args,metadata,rank)
    core.init_wandb=init_wandb
    core.train(a)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','check-cache','audit','train'])
    p.add_argument('--execute',action='store_true');p.add_argument('--resume');a=p.parse_args()
    os.chdir(ROOT);torch.set_num_threads(2)
    if a.stage in ['audit','train'] and not a.execute:
        command=('python scripts/prepare_all4_absolute_decoder.py audit --execute' if a.stage=='audit' else
            'torchrun --standalone --nproc_per_node=8 scripts/prepare_all4_absolute_decoder.py train --execute')
        print(command);return
    inputs=Inputs()
    if a.stage=='prepare':
        PREP.mkdir(parents=True,exist_ok=True)
        dest=PREP/'manifest.json'
        if dest.exists():assert json.loads(dest.read_text())==inputs.manifest
        else:atomic(dest,inputs.manifest)
        print(json.dumps(dict(identity=inputs.identity,train_windows=len(inputs.refs),steps=8264,eval_windows=800,cache_exists=CACHE.exists())));return
    inputs.open_cache()
    if a.stage=='check-cache':print('CACHE_CONTRACT_OK',inputs.cache_sha)
    elif a.stage=='audit':audit(inputs,PREP/'cache_audit.json')
    else:train(inputs,a.resume)

if __name__=='__main__':main()
