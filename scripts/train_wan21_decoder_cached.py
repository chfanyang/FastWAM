"""Fresh Wan2.1 decoder training with policy BF16 cache and global-batch logging."""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import finetune_rothko_vae_decoder as core
from fastwam.datasets.latent_cache import LatentCacheReader, build_dataset_contract, sha256_file
from fastwam.representations.libero_rothko import LiberoRothkoCodec
from compare_libero_rothko_decoders import errors, summarize


class CachedWindows:
    def __init__(self, args):
        self.args = args
        self.path = Path(args.latent_cache_dir)
        m = json.loads((self.path/'metadata.json').read_text())
        roots = [r['path'] for r in m['dataset_contract']['dataset_roots']]
        expected_roots = [str((Path(args.data_root)/core.SUITE_DIRS[s]).resolve()) for s in args.suites]
        assert roots == expected_roots, 'Dataset roots/order changed'
        core.load_norm_stats(args.norm_stats_path, args)  # Validate training representation metadata.
        self.codec = LiberoRothkoCodec(norm_stats=args.norm_stats_path, expected_action_horizon=16,
                                      decode_mode='robust_joint', decode_anchor_alpha=0, decode_block_grid=4)
        self.offsets = {}
        count = 0
        for root in roots:
            episodes = [json.loads(s) for s in (Path(root)/'meta/episodes.jsonl').read_text().splitlines()]
            episodes.sort(key=lambda x:x['episode_index'])
            assert [e['episode_index'] for e in episodes] == list(range(len(episodes)))
            root_count = 0
            for e in episodes:
                self.offsets[(root,e['episode_index'])] = (count, e['length'])
                count += e['length']
                root_count += e['length']
            assert root_count == json.loads((Path(root)/'meta/info.json').read_text())['total_frames']
        contract = build_dataset_contract(dataset_dirs=roots, dataset_length=count, num_frames=17,
                    video_size=[224,448], raymap_representation='libero_rothko',
                    raymap_codec_metadata=self.codec.metadata(), norm_stats_sha256=self.codec.norm_stats.fingerprint())
        self.reader = LatentCacheReader(self.path, expected_dataset_contract=contract)
        assert m['vae_identity']['sha256'] == sha256_file(args.base_vae)
        assert m['model_variant'] == args.vae_variant and m['modalities'] == ['rgb','raymap']
        train, heldout = core.discover_episodes(args, rank=int(os.environ.get('RANK','0')))
        self.train_episodes = set(train)
        assert not self.train_episodes.intersection(heldout)
        self.train_refs = core.enumerate_window_refs(train,16)
        indices = [self.index(ref) for ref in self.train_refs]
        assert len(indices) == len(set(indices)) == 239929
        self.manifest = {'cache_metadata_sha256':sha256_file(self.path/'metadata.json'),
                         'training_windows':len(indices),'training_episodes':len(train),
                         'heldout_episodes':len(heldout),'cache_samples':len(self.reader),
                         'train_indices_sha256':hashlib.sha256(json.dumps(indices).encode()).hexdigest()}

    def index(self, ref):
        offset, length = self.offsets[(ref.episode.dataset_root,ref.episode.episode_index)]
        assert length == ref.episode.length and 0 <= ref.start <= length-16
        return offset + ref.start

    def latent(self, ref):
        return self.reader[self.index(ref)][1]

    def batch(self, windows, refs, device):
        assert all(r.episode in self.train_episodes for r in refs), 'Heldout leakage'
        target = torch.stack([self.codec.encode(torch.from_numpy(w.pose),torch.from_numpy(w.gripper)) for w in windows])
        latent = torch.stack([self.latent(r) for r in refs])
        assert latent.dtype == torch.bfloat16 and target.dtype == torch.float32
        return target.to(device),latent.to(device)

    def epoch_batches(self, refs, start_step, steps, effective_batch, world_size, rank, device):
        # Exactly the original shuffle and rank/microbatch assignment; bounded CPU prefetch.
        store = core.EpisodeStore(self.args.episode_cache_size)
        def jobs():
            for step in range(start_step, steps):
                for acc in range(self.args.grad_accum_steps):
                    offset = step*effective_batch + acc*self.args.batch_size*world_size + rank*self.args.batch_size
                    yield refs[offset:offset+self.args.batch_size]
        def prepare(batch_refs):
            windows = core.materialize_windows(store,batch_refs,16)
            target,latent = self.batch(windows,batch_refs,torch.device('cpu'))
            return target.pin_memory(),latent.pin_memory()
        source = iter(jobs())
        with ThreadPoolExecutor(max_workers=1,thread_name_prefix='cache-prefetch') as pool:
            pending = deque()
            for _ in range(2):
                batch = next(source,None)
                if batch is not None: pending.append(pool.submit(prepare,batch))
            while pending:
                target,latent = pending.popleft().result()
                batch = next(source,None)
                if batch is not None: pending.append(pool.submit(prepare,batch))
                yield target.to(device,non_blocking=True),latent.to(device,non_blocking=True)


@torch.inference_mode()
def audit(cache, args):
    device=torch.device('cuda:0')
    vae=core.load_base_vae(args.base_vae,args.vae_variant,device,torch.bfloat16)
    store=core.EpisodeStore(32)
    selected=[]
    for root in sorted({r.episode.dataset_root for r in cache.train_refs}):
        refs=[r for r in cache.train_refs if r.episode.dataset_root==root]
        selected += [refs[0],refs[len(refs)//2],refs[-1]]
    reports=[]
    for ref in selected:
        w=core.materialize_windows(store,[ref],16)[0]
        target=cache.codec.encode(torch.from_numpy(w.pose),torch.from_numpy(w.gripper)).unsqueeze(0).to(device,torch.bfloat16)
        actual=vae.encode(target,device=device,tiled=False)
        expected=cache.latent(ref).unsqueeze(0).to(device)
        exact=torch.equal(actual,expected)
        row={'index':cache.index(ref),'episode':ref.episode.episode_index,'root':ref.episode.dataset_root,
             'start':ref.start,'exact':exact,'max_abs_diff':(actual.float()-expected.float()).abs().max().item()}
        print('CACHE_AUDIT',json.dumps(row),flush=True)
        reports.append(row)
        assert exact, 'Cache/window mapping or policy encoding mismatch'
    args.audit_report.parent.mkdir(parents=True,exist_ok=True)
    args.audit_report.write_text(json.dumps({'contract':cache.manifest,'probes':reports},indent=2))


def constant_warmup(optimizer,max_steps,warmup_steps,min_lr_ratio,scheduler_type):
    assert scheduler_type=='constant' and warmup_steps==100
    # Independent of max_steps, so extending epochs preserves schedule on resume.
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lambda n:min((n+1)/warmup_steps,1.0))


def cosine_hold_factor(step, warmup_steps, decay_steps, floor_ratio=0.2):
    """Absolute-step schedule; extending training never restarts decay/warmup."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(max((step - warmup_steps) / (decay_steps - warmup_steps), 0.0), 1.0)
    return floor_ratio + (1 - floor_ratio) * .5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def cached_evaluate(cache, vae, store, episodes, stats, args, device, dtype):
    windows=core.sample_uniform_windows_per_episode(store,episodes,10,16)
    manifest=[{'dataset':w.episode.dataset_root,'episode':w.episode.episode_index,
               'task':w.episode.task,'start':w.start} for w in windows]
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    assert digest=='8df3578f439b8762bdc16d7a61b835c563bba6f58261bbf90d359530646ee7c2'
    if args.cached_smoke:
        windows=[windows[i] for i in range(0,800,100)] if args.parallel_cached_eval else [windows[i] for i in (0,200,400,600)]
    expected = len(windows)
    rank = torch.distributed.get_rank() if args.parallel_cached_eval else 0
    world = torch.distributed.get_world_size() if args.parallel_cached_eval else 1
    selected = list(enumerate(windows))[rank::world]
    ev=core.load_base_vae(args.base_vae,args.vae_variant,device,torch.bfloat16)
    ev.model.decoder.load_state_dict(vae.model.decoder.state_dict(),strict=True)
    ev.model.conv2.load_state_dict(vae.model.conv2.state_dict(),strict=True)
    rows=[]
    grip_sum=0.0
    for i,w in selected:
        ref=core.WindowRef(w.episode,w.start)
        assert ref.episode not in cache.train_episodes
        z=cache.latent(ref).unsqueeze(0).to(device)
        recon=ev.model.decode(z,ev.scale).float().clamp(-1,1)
        truth=torch.from_numpy(w.pose).unsqueeze(0).to(device)
        pred,_=cache.codec.decode(recon,truth[:,0])
        pe,re=errors(pred[:,1:],truth[:,1:])
        assert torch.isfinite(pe).all() and torch.isfinite(re).all()
        rows.append({'index':i,'errors':{'joint':{'translation_mm':pe[0].cpu().tolist(),'rotation_deg':re[0].cpu().tolist()}}})
        gripper=core.read_gripper(recon,args)
        grip_sum+=(gripper[:,1:]-torch.from_numpy(w.gripper).unsqueeze(0).to(device)[:,1:]).abs().mean().item()
    if args.parallel_cached_eval:
        gathered=[None]*world
        torch.distributed.all_gather_object(gathered,{'rows':rows,'gripper_sum':grip_sum})
        rows=sorted([row for shard in gathered for row in shard['rows']],key=lambda r:r['index'])
        grip_sum=sum(shard['gripper_sum'] for shard in gathered)
        assert len(rows)==expected and {r['index'] for r in rows}==set(range(expected))
        core.log(f'Parallel joint validation complete: {expected} windows, {world} GPUs',rank)
    m=summarize(rows)['joint']
    result={'future_position_mae_m':m['translation_mm']['mean']/1000,'future_rotation_mean_deg':m['rotation_deg']['mean'],
            'future_gripper_mae':grip_sum/len(windows),'num_windows':len(windows),'manifest_sha256':digest,
            'decode_mode':'robust_joint','anchor_alpha':0,'precision':'BF16 cached latent and deployment decoder',
            'metrics':m}
    for metric,values in m.items():
        for key in ('mean','p95','first8_mean'):
            result[f'{metric}_{key}']=values[key]
    del ev,z,recon,pred
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--latent-cache-dir',default=str(ROOT/'data/libero_all4_rothko_centerfrac05_wan21_bf16_h16_latents'))
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--audit-report',type=Path,default=ROOT/'evaluate_results/libero_decoder_offline/wan21_cached_fresh_20260913/cache_audit.json')
    p.add_argument('--cached-smoke',action='store_true')
    p.add_argument('--micro-batch',type=int,choices=(2,4),default=2)
    p.add_argument('--accelerated',action='store_true')
    p.add_argument('--resume-constant-lr',type=float,default=None)
    p.add_argument('--batch64-cosine4-hold',action='store_true')
    p.add_argument('--cosine-peak-lr',type=float,default=None)
    custom,rest=p.parse_known_args()
    if custom.cosine_peak_lr is not None:
        assert custom.batch64_cosine4_hold and math.isfinite(custom.cosine_peak_lr) and custom.cosine_peak_lr > 0
    sys.argv=[sys.argv[0],*rest]
    args=core.parse_args()
    args.latent_cache_dir=custom.latent_cache_dir
    args.cached_smoke=custom.cached_smoke
    args.audit_report=custom.audit_report
    args.cached_decoder_training=True
    args.cached_prefetch=custom.accelerated
    args.parallel_cached_eval=custom.accelerated
    args.cpu_threads=4 if custom.accelerated else 2
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    if custom.accelerated:
        import pyarrow
        pyarrow.set_cpu_count(2)
        pyarrow.set_io_thread_count(2)
    args.batch_size=custom.micro_batch
    args.grad_accum_steps=16//custom.micro_batch
    args.lr=1e-5
    if custom.resume_constant_lr is not None:
        assert 0 < custom.resume_constant_lr < 1e-5
        args.lr=custom.resume_constant_lr
        args.resume_constant_lr=custom.resume_constant_lr
    args.lr_scheduler='constant'
    args.warmup_steps=100
    if custom.batch64_cosine4_hold:
        assert custom.resume_constant_lr is None and not custom.cached_smoke
        args.grad_accum_steps=8//custom.micro_batch
        args.lr=1.5e-5 if custom.cosine_peak_lr is None else custom.cosine_peak_lr
        args.lr_scheduler='cosine'
        args.min_lr_ratio=.2
        args.decay_epochs=4
        args.schedule_recipe='batch64_cosine4_hold_v1'
    args.independent_full_checkpoints=True
    args.save_every=args.step_checkpoint_every=500
    args.eval_every=args.export_every=300
    args.episode_cache_size=2048
    args.log_every=10
    args.no_progress_bar=True
    assert args.bf16 and args.vae_variant=='wan2.1-t2v-1.3b'
    assert not args.allow_weights_only_resume
    cache=CachedWindows(args)
    if custom.audit_only:
        audit(cache,args)
        return
    proof=json.loads(custom.audit_report.read_text())
    assert proof['contract']==cache.manifest and len(proof['probes'])==12 and all(x['exact'] for x in proof['probes'])
    args.audit_report=str(args.audit_report)
    args.cache_contract=cache.manifest
    if custom.batch64_cosine4_hold:
        args.decay_steps=4*math.ceil(len(cache.train_refs)/64)
        args.warmup_steps=int(.05*args.decay_steps)
        args.extra_full_eval_steps=[args.decay_steps]
    args.loss_logging=f'global mean across 8 ranks and {args.grad_accum_steps} accumulated microbatches; then mean over log interval optimizer steps'
    if custom.cached_smoke:
        args.wandb=False
        args.log_every=1
        args.save_every=args.eval_every=args.export_every=args.step_checkpoint_every=1
        # Two optimizer updates per epoch, retaining real data/cache samples.
        original=core.enumerate_window_refs
        core.enumerate_window_refs=lambda pool,h:original(pool,h)[:256]
    assert int(os.environ.get('WORLD_SIZE','1'))==8, 'This launch requires 8 GPUs'
    resume=core.resolve_resume(args)
    if custom.resume_constant_lr is not None:
        assert resume, 'LR branch requires a full checkpoint'
    if resume:
        old=torch.load(resume,map_location='cpu',weights_only=False,mmap=True)
        assert old.get('optimizer') is not None
        saved=old['metadata']['arguments']
        for key in ('latent_cache_dir','cache_contract','batch_size','grad_accum_steps','lr','warmup_steps','seed','cached_smoke'):
            if key=='lr' and custom.resume_constant_lr is not None:
                continue
            assert saved[key]==getattr(args,key), f'Resume contract changed: {key}'
        if custom.batch64_cosine4_hold:
            for key in ('schedule_recipe','decay_steps','min_lr_ratio'):
                assert saved[key]==getattr(args,key), f'Resume schedule changed: {key}'
        del old
    core.cached_training_batch=cache.batch
    core.cached_epoch_batches=cache.epoch_batches
    core.build_scheduler=constant_warmup
    if custom.batch64_cosine4_hold:
        def scheduler(optimizer,max_steps,warmup_steps,min_lr_ratio,scheduler_type):
            assert max_steps>=args.decay_steps and scheduler_type=='cosine'
            return torch.optim.lr_scheduler.LambdaLR(optimizer,
                lambda n:cosine_hold_factor(n,warmup_steps,args.decay_steps,min_lr_ratio))
        core.build_scheduler=scheduler
    core.evaluate=lambda *xs:cached_evaluate(cache,*xs)
    # No encoder invocation is permitted on the training path.
    core.vae_encode=lambda *xs:(_ for _ in ()).throw(RuntimeError('Unexpected online encoding in cached training'))
    core.train(args)


if __name__=='__main__':
    main()
