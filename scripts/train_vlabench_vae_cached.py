"""VLABench-only decoder+conv2 fine-tuning; shared training code is unchanged."""
import argparse
from bisect import bisect_right
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from hydra.utils import instantiate
from omegaconf import OmegaConf

import finetune_rothko_vae_decoder as core
from vlabench_vae_validation_subset import load_indices
from fastwam.datasets.latent_cache import sha256_file
from fastwam.datasets.vlabench_video import pose_xyz_euler_to_wxyz
from fastwam.models.wan22.helpers.loader import _load_registered_model
from experiments.vlabench.audit_three_camera_vae_reconstruction import metrics, summarize


class Windows(Dataset):
    def __init__(self, config, split):
        cfg=OmegaConf.create(OmegaConf.to_container(config.data[split], resolve=True))
        cfg.include_text_context=False
        self.ds=instantiate(cfg)
        self.parts=list(self.ds.datasets) if hasattr(self.ds,'datasets') else [self.ds]
        self.ends=np.cumsum([len(part) for part in self.parts]).tolist()
        self.episode_ids=sorted({ep for part in self.parts for ep in part.windows.reader.episodes})
        for part in self.parts:
            part.windows.camera_keys=()
            part.windows.reader.cameras=()
        if split=='train':
            metas=[part.latent_cache.metadata for part in self.parts]
            self.cache_metadata=dict(metas[0])
            for meta in metas:
                assert meta['vae_identity']==metas[0]['vae_identity']
                assert meta['model_variant']==metas[0]['model_variant']
            if len(metas)>1:
                self.cache_metadata['dataset_contract']=[meta['dataset_contract'] for meta in metas]
        self.split=split

    def __len__(self):
        return len(self.ds)

    def __getitem__(self,index):
        part_index=bisect_right(self.ends,index)
        part=self.parts[part_index]
        local_index=index-(self.ends[part_index-1] if part_index else 0)
        if part.sample_indices is not None:
            local_index=part.sample_indices[local_index]
        raw=part.windows[local_index]
        state,action=raw['raw_state']['default'],raw['raw_action']['default']
        base=pose_xyz_euler_to_wxyz(state[0])
        pose=torch.cat((base[None],pose_xyz_euler_to_wxyz(action)))
        grip=torch.cat((raw['raw_state']['gripper_open'][:1],raw['raw_action']['gripper_open']))
        result=dict(target=part.codec.encode(pose,grip),pose=pose,gripper=grip,
                    action_is_pad=raw['action_is_pad'],index=index)
        if self.split=='train':
            result['latent']=part.latent_cache[local_index][1]
        return result


def init_worker(_):
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.set_num_threads(1)


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


@torch.no_grad()
def evaluate(vae, deployment, val, val_cache, masks, loss_args, step, output, rank, world, smoke, selected_indices=None):
    deployment.model.decoder.load_state_dict(vae.model.decoder.state_dict(),strict=True)
    deployment.model.conv2.load_state_dict(vae.model.conv2.state_dict(),strict=True)
    rows=[]
    expected_indices=list(range(len(val))) if selected_indices is None else list(selected_indices)
    if smoke:
        expected_indices=expected_indices[:8]
    indices=expected_indices[rank::world]
    for index in indices:
        sample=val[index]
        if index not in val_cache:
            val_cache[index]=deployment.encode(sample['target'][None].cuda().bfloat16(),
                                              device='cuda',tiled=False).cpu()
        latent=val_cache[index].cuda()
        rebuilt=deployment.decode(latent,device='cuda',tiled=False).float().clamp(-1,1)
        pose,grip=val.ds.codec.decode(rebuilt,sample['pose'][0].cuda())
        error=metrics(pose[0,1:].cpu(),grip[0,1:].cpu(),sample['pose'][1:],sample['gripper'][1:])
        loss,_=core.compute_reconstruction_loss(rebuilt,sample['target'][None].cuda(),masks,loss_args)
        rows.append(dict(val_index=index,action_is_pad=sample['action_is_pad'].tolist(),
                         vae_vs_target=error,loss=float(loss)))
    gathered=[None]*world
    dist.all_gather_object(gathered,rows)
    if rank==0:
        rows=sorted([r for part in gathered for r in part],key=lambda r:r['val_index'])
        assert [r['val_index'] for r in rows]==expected_indices
        report=dict(step=step,windows=len(rows),val_loss=float(np.mean([r['loss'] for r in rows])),
                    first8=summarize(rows,'vae_vs_target',8),all16=summarize(rows,'vae_vs_target',16))
        atomic_json(output/f'eval_step{step:06d}.json',report)
        print('VALIDATION',step,json.dumps(report['first8']),flush=True)
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    c=json.loads(args.config.read_text())
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(2)
    dist.init_process_group('nccl')
    output=Path(c['output_dir']+('_smoke' if args.smoke else ''))
    if rank==0:
        if output.exists() and any(output.iterdir()) and not args.resume:
            raise FileExistsError(output)
        output.mkdir(parents=True,exist_ok=True)
    dist.barrier()
    random.seed(c['seed']+rank);np.random.seed(c['seed']+rank);torch.manual_seed(c['seed']+rank)
    config=OmegaConf.load(c['train_config'])
    train,val=Windows(config,'train'),Windows(config,'val')
    validation_indices=None
    subset_path=c.get('validation_window_manifest')
    if subset_path:
        reader=val.ds.windows.reader
        validation_indices=load_indices(subset_path, total_windows=len(val),
            episode_layout=[(ep,reader.lengths[ep]) for ep in reader.episodes],
            split_sha256=val.ds.episode_split_metadata['sha256'])
    assert not set(train.episode_ids)&set(val.ds.windows.reader.episodes)
    assert train.ds.codec.metadata()==val.ds.codec.metadata()
    assert train.ds.codec.norm_stats.fingerprint()==val.ds.codec.norm_stats.fingerprint()
    cache_meta=train.cache_metadata
    assert cache_meta['vae_identity']['sha256']==sha256_file(c['base_vae'])
    assert cache_meta['model_variant']=='wan2.1-t2v-1.3b'
    effective=c['batch_size']*c['grad_accum_steps']*world
    per_epoch=math.ceil(len(train)/effective)
    total=2 if args.smoke else per_epoch*c['epochs']
    metadata=dict(vae_variant='wan2.1-t2v-1.3b',config=c,smoke=args.smoke,
        world_size=world,effective_batch=effective,steps_per_epoch=per_epoch,max_steps=total,
        train_windows=len(train),val_windows=len(val),
        train_episodes=train.episode_ids,val_episodes=val.ds.windows.reader.episodes,
        dataset_contract=cache_meta['dataset_contract'],vae_identity=cache_meta['vae_identity'])
    if subset_path:
        metadata.update(validation_windows_used=len(validation_indices),
                        validation_window_manifest_sha256=sha256_file(subset_path),
                        validation_window_indices=validation_indices)
    if rank==0:
        atomic_json(output/'training_config.json',metadata)
    # FP32 master weights; BF16 autocast only for decoder computation.
    vae=_load_registered_model(c['base_vae'],'wan_video_vae',torch_dtype=torch.float32,device='cuda')
    trainable=core.prepare_decoder_finetune(vae)
    ddp=DDP(core.VaeDecodeWrapper(vae.model),device_ids=[local],gradient_as_bucket_view=True)
    optimizer=torch.optim.AdamW(trainable,lr=c['lr'],weight_decay=c['weight_decay'])
    scheduler=core.build_scheduler(optimizer,total,min(int(total*c['warmup_ratio']),total-1),
                                   c['min_lr_ratio'],'cosine')
    # Separate BF16 deployment copy gives eval the exact precision of rollout.
    deployment=_load_registered_model(c['base_vae'],'wan_video_vae',torch_dtype=torch.bfloat16,device='cuda')
    deployment.eval().requires_grad_(False)
    # Audit reused raymap latents against fresh original encoder output on each rank.
    audit_indices=([0]+train.ends[:-1]) if len(train.parts)>1 else list(range(min(world,len(train))))
    for index in audit_indices[rank::world]:
        check=train[index]
        with torch.no_grad():
            z=deployment.encode(check['target'][None].cuda().bfloat16(),device='cuda',tiled=False)[0].cpu()
        assert torch.equal(z,check['latent']), f'Cached target/latent mismatch at {index}'
        print(f'CACHE_AUDIT rank={rank} index={index} exact_match=True',flush=True)
        del z,check
    cfg=train.ds.codec.config
    center,_,direction,border=core.region_masks(cfg.tile_height,cfg.tile_width,cfg.center_frac,
        cfg.boundary_margin,cfg.outer_margin,torch.device('cuda'))
    masks={k:torch.cat([v]*3,-1) for k,v in dict(center=center,direction=direction,gripper=border).items()}
    loss_args=SimpleNamespace(**c)
    step=0;run_id=None
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        assert saved['metadata']==metadata,'Resume semantics changed'
        vae.model.decoder.load_state_dict(saved['decoder'])
        vae.model.conv2.load_state_dict(saved['conv2'])
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        step=saved['step'];run_id=saved.get('wandb_id')
        rng=saved['rng'][rank]
        random.setstate(rng['python']);np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
        del saved
    wandb_run=None
    if rank==0 and c['wandb'] and not args.smoke:
        import wandb
        wandb_run=wandb.init(project=c['wandb_project'],name=output.name,dir=str(output),
                            config=metadata,id=run_id,resume='must' if run_id else None)
    val_cache={}

    def validation():
        report=evaluate(vae,deployment,val,val_cache,masks,loss_args,step,output,rank,world,args.smoke,validation_indices)
        if wandb_run:
            flat={'eval/val_loss':report['val_loss']}
            for horizon in ('first8','all16'):
                for metric,values in report[horizon].items():
                    for stat,v in values.items(): flat[f'eval/{horizon}/{metric}_{stat}']=v
            wandb_run.log(flat,step=step)

    def save():
        local_rng=dict(python=random.getstate(),numpy=np.random.get_state(),
                       torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
        rng=[None]*world;dist.all_gather_object(rng,local_rng)
        if rank==0:
            path=output/f'checkpoint_step{step:06d}.pt'
            payload=dict(step=step,decoder=vae.model.decoder.state_dict(),conv2=vae.model.conv2.state_dict(),
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),metadata=metadata,rng=rng,
                wandb_id=wandb_run.id if wandb_run else None)
            tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
            export_stem=c.get('export_stem','Wan2.1_VAE_vlabench_select_book')
            core.export_complete_vae(str(output/f'{export_stem}_step{step:06d}.safetensors'),
                                     vae,step,metadata)
            print('SAVED',path,flush=True)
        dist.barrier()

    validation()
    start=time.monotonic();initial_step=step
    if rank==0: print(f'TRAIN windows={len(train)} effective_batch={effective} steps={total} FP32_master BF16_autocast decoder+conv2',flush=True)
    while step<total:
        epoch=step//per_epoch
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(c['seed']+epoch)).tolist()
        order+=order[:per_epoch*effective-len(order)]
        local_indices=[]
        for st in range(step%per_epoch, min(per_epoch,total-epoch*per_epoch)):
            for acc in range(c['grad_accum_steps']):
                offset=st*effective+acc*c['batch_size']*world+rank*c['batch_size']
                local_indices.extend(order[offset:offset+c['batch_size']])
        loader=DataLoader(Subset(train,local_indices),batch_size=c['batch_size'],shuffle=False,
            num_workers=c['num_workers'],worker_init_fn=init_worker,pin_memory=False,
            **({'prefetch_factor':1,'multiprocessing_context':'spawn'} if c['num_workers'] else {}))
        batches=iter(loader)
        while step<min((epoch+1)*per_epoch,total):
            optimizer.zero_grad(set_to_none=True)
            values=torch.zeros(4,device='cuda')
            for acc in range(c['grad_accum_steps']):
                sample=next(batches)
                target=sample['target'].cuda();latent=sample['latent'].cuda()
                with ddp.no_sync() if acc<c['grad_accum_steps']-1 else contextlib.nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        recon=ddp(latent,vae.scale)
                    loss,components=core.compute_reconstruction_loss(recon.float(),target,masks,loss_args)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                    (loss/c['grad_accum_steps']).backward()
                values+=torch.stack([loss.detach(),*[components[k].detach() for k in ('center','direction','gripper')]])/c['grad_accum_steps']
                del recon,loss,components,target,latent,sample
            grad=torch.nn.utils.clip_grad_norm_(trainable,c['grad_clip'])
            if not torch.isfinite(grad): raise RuntimeError('Nonfinite gradient')
            optimizer.step();scheduler.step();step+=1
            if step%c['log_every']==0 or args.smoke or step==total:
                dist.all_reduce(values);values/=world
                if rank==0:
                    elapsed=time.monotonic()-start
                    print(f'step={step}/{total} loss={values[0].item():.6f} lr={scheduler.get_last_lr()[0]:.3g} sec/step={elapsed/(step-initial_step):.2f} peak_GB={torch.cuda.max_memory_allocated()/2**30:.2f}',flush=True)
                    if wandb_run: wandb_run.log({**{f'train/{k}':float(v) for k,v in zip(('loss','center','direction','gripper'),values)},'train/lr':scheduler.get_last_lr()[0]},step=step)
            epoch_end=step%per_epoch==0
            do_eval=step%c['eval_every']==0 or step==total or (epoch_end and c.get('eval_at_epoch_end',False))
            do_save=step%c['save_every']==0 or step==total or (epoch_end and c.get('save_at_epoch_end',False))
            if c.get('save_before_eval',False):
                if do_save: save()
                if do_eval: validation()
            else:
                if do_eval: validation()
                if do_save: save()
        del batches,loader
    if wandb_run: wandb_run.finish()
    dist.destroy_process_group()


if __name__=='__main__':
    main()
