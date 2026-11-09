"""Isolated offline final-action comparison; never trains or starts a simulator."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
RAY_RUN = ROOT / 'runs/libero_goal_rothko_absolute_ray0_2cam224_full_wan21_1_3b_1e-4/2026-09-21_18-42-23'
MOT_RUN = ROOT / 'runs/libero_goal_matched_fastwam_wan21/2026-11-07_16-45-00'
VAE = ROOT / 'runs/libero_all4_absolute_ray0_decoder_wan21_bs2_ga4_lr1e-5_ep2/Wan2.1_VAE_libero_rothko_step008264.safetensors'


def write_json(path, data):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def prepare(out):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.libero_matched_fastwam import MatchedPoseCodec
    from fastwam.representations.rothko import quaternion_wxyz_to_matrix
    from fastwam.utils import misc

    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists() or (out/'inputs.pt').exists():
        raise FileExistsError('Prepared inputs already exist; choose a new output directory.')
    misc.register_work_dir(out)
    cfg = OmegaConf.load(RAY_RUN / 'config.yaml')
    cfg.data.train.latent_cache_only = False
    cfg.data.train.latent_cache_dir = None
    ds = instantiate(cfg.data.train)
    assert len(ds) == 52895
    codec = MatchedPoseCodec(ds.raymap_codec)
    data_root = ROOT / 'data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot'
    episodes = sorted([json.loads(x) for x in (data_root / 'meta/episodes.jsonl').read_text().splitlines()], key=lambda x:x['episode_index'])
    choices = {}
    index_meta = {}
    offset = 0
    for ep in episodes:
        assert len(ep['tasks']) == 1
        task = ep['tasks'][0]
        for start in range(ep['length']):
            idx = offset + start
            index_meta[idx] = dict(index=idx, episode=ep['episode_index'], frame_start=start,
                                   episode_length=ep['length'], task=task)
            if idx >= 4:
                choices.setdefault(task, []).append(idx)
        offset += ep['length']
    assert offset == len(ds) and len(choices) == 10
    rng = np.random.default_rng(42)
    records = [dict(index_meta[i], group='original_validation4') for i in range(4)]
    for task in sorted(choices):
        records += [dict(index_meta[int(i)], group='stratified100')
                    for i in sorted(rng.choice(choices[task], size=10, replace=False))]
    assert len(records) == len({r['index'] for r in records}) == 104
    inputs = []
    arrays = {}
    started = time.monotonic()
    for probe, record in enumerate(records):
        s = ds[record['index']]
        ep = record['episode']
        if ep not in arrays:
            path = next(data_root.glob(f'data/chunk-*/episode_{ep:06d}.parquet'))
            table = pq.read_table(path, columns=['observation.state.ee_pose_wxyz',
                'observation.state.gripper_open', 'action.osc_target_pose_wxyz', 'action'])
            arrays[ep] = {k:torch.tensor(table[k].to_pylist(), dtype=torch.float32) for k in table.column_names}
        raw = arrays[ep]
        start = record['frame_start']
        ix = torch.arange(start, start + 16).clamp_max(record['episode_length'] - 1)
        current = raw['observation.state.ee_pose_wxyz'][start]
        grip = raw['observation.state.gripper_open'][start].reshape(1)
        gt = raw['action.osc_target_pose_wxyz'][ix]
        gg = raw['action'][ix, -1:].clamp(0, 1)
        for a,b in ((current,s['current_endpose']), (gt,s['future_endpose']), (gg,s['future_gripper'])):
            torch.testing.assert_close(a,b,rtol=0,atol=1e-6)
        pads = torch.arange(start, start + 16) >= record['episode_length']
        assert torch.equal(pads, s['action_is_pad'])
        read_grip = ds.raymap_codec.read_gripper(s['raymap'].permute(1,0,2,3))[0].reshape(1)
        torch.testing.assert_close(grip, read_grip, rtol=0, atol=1e-6)
        numeric_state = codec.state_vector(current[None],grip[None])
        vectors = codec.vectors(torch.cat((current[None],gt))[None],torch.cat((grip[None],gg))[None])
        recovered, rg, bad = codec.recover(vectors[:,1:],current[None])
        assert not bad.any()
        represented, represented_grip = ds.raymap_codec.decode(s['raymap'], current)
        torch.testing.assert_close(recovered[0,:,:3],represented[1:,:3],rtol=0,atol=2e-6)
        agreement = errors(recovered[0], represented[1:], rg[0], represented_grip[1:])
        assert max(agreement['rotation_deg']) < .1, (record, agreement)
        torch.testing.assert_close(rg[0],represented_grip[1:],rtol=0,atol=2e-6)
        rel=(quaternion_wxyz_to_matrix(current[3:]).T @ (gt[:,:3]-current[:3]).T).T
        clipped=(rel.abs()>codec.bounds+1e-7).any(-1)
        record.update(probe=probe, seed=42+probe, padding_steps=int(pads.sum()),
            gt_translation_clipped=clipped.tolist(),
            gt_position_clip_l2_mm=((recovered[0,:,:3]-gt[:,:3]).norm(dim=-1)*1000).tolist(),
            direct_raymap_error=errors(represented[1:],gt,represented_grip[1:],gg),
            direct_numeric_error=errors(recovered[0],gt,rg[0],gg))
        inputs.append(dict(rgb=s['video'][:,0].contiguous().clone(), ray0=s['raymap'][:,0].contiguous().clone(),
            current_pose=current, numeric_state=numeric_state, current_gripper=grip,
            context=s['context'], context_mask=s['context_mask'],
            target_pose=gt, target_gripper=gg, action_is_pad=pads))
        if (probe+1)%10 == 0:
            print(f'PREPARED {probe+1}/104 elapsed={time.monotonic()-started:.1f}s',flush=True)
    torch.save(inputs, out/'inputs.pt')
    write_json(out/'manifest.json',dict(records=records, sampling_seed=42,
        scope='training-set diagnostic, not held-out; both runs trained on all 433 Goal episodes',
        num_steps=20, action_horizon=16, replan_reporting_horizon=8,
        raymap_checkpoint=str(RAY_RUN/'checkpoints/weights/step_004140.pt'),
        fastwam_checkpoint=str(MOT_RUN/'checkpoints/weights/step_004140.pt'),
        raymap_vae=str(VAE), raymap_decoder='legacy', raymap_anchor_alpha=0,
        stats_fingerprint=ds.raymap_codec.norm_stats.fingerprint(),
        data_alignment='All 104 current poses, future targets, grippers and padding verified against raw parquet. '
            'Numerical and direct Raymap XYZ/grippers agree after shared translation clipping; '
            'rotation agreement checked within 0.1 degree and direct round trips retained. '
            'Primary prediction errors use raw parquet targets, including any clipping floor.',
        seed_note='42 + probe index; shared seed does not imply identical diffusion noise across representations.'))
    print('PREPARATION COMPLETE',flush=True)


def errors(pose, gt, grip, gg):
    import torch
    p=pose.double(); t=gt.double()
    q=torch.nn.functional.normalize(p[:,3:],dim=-1)
    r=torch.nn.functional.normalize(t[:,3:],dim=-1)
    return dict(position_l2_mm=((p[:,:3]-t[:,:3]).norm(dim=-1)*1000).tolist(),
        rotation_deg=torch.rad2deg(2*torch.acos((q*r).sum(-1).abs().clamp(0,1))).tolist(),
        gripper_error_pct=((grip.reshape(-1)>=.5)!=(gg.reshape(-1)>=.5)).double().mul(100).tolist(),
        gripper_abs_error=(grip.reshape(-1)-gg.reshape(-1)).abs().tolist())


def worker(out, method, rank, shards):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.utils import misc
    misc.register_work_dir(out)
    path=out/f'{method}_shard{rank}.json'
    if path.exists(): raise FileExistsError(path)
    manifest=json.loads((out/'manifest.json').read_text())
    inputs=torch.load(out/'inputs.pt',map_location='cpu',weights_only=False,mmap=True)
    run=RAY_RUN if method=='raymap' else MOT_RUN
    cfg=OmegaConf.load(run/'config.yaml')
    cfg.model.load_text_encoder=False
    cfg.model.skip_dit_load_from_pretrain=True
    if method=='raymap':
        cfg.model.vae_safetensors_path=str(VAE)
        cfg.model.allow_vae_mismatch=True
        cfg.model.rothko_decode_mode='legacy'
        cfg.model.rothko_decode_anchor_alpha=0.
    else:
        cfg.model.action_dit_pretrained_path=None
    OmegaConf.save(cfg,out/f'{method}_config_shard{rank}.yaml')
    model=instantiate(cfg.model,device='cuda:0',model_dtype=torch.bfloat16)
    payload=model.load_checkpoint(str(run/'checkpoints/weights/step_004140.pt'))
    del payload
    assert model.raymap_codec.norm_stats.fingerprint()==manifest['stats_fingerprint']
    model.eval().requires_grad_(False)
    rows=[]
    started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for i in range(rank,len(inputs),shards):
            s=inputs[i]; record=manifest['records'][i]
            common=dict(prompt=None,input_image=s['rgb'][None],context=s['context'],
                context_mask=s['context_mask'],num_inference_steps=20,
                seed=record['seed'],rand_device='cpu',tiled=False)
            torch.cuda.synchronize(); before=time.monotonic()
            if method=='raymap':
                pred=model.infer(**common,input_raymap=s['ray0'][None],proprio=None,
                    current_endpose=s['current_pose'],num_frames=17,decode_future_rgb=False)
                pose=pred['pose'][0,1:].float().cpu()
                grip=pred['gripper'][0,1:].float().cpu()
                bad=None
            else:
                state=model.numeric_codec.state_vector(s['current_pose'][None],s['current_gripper'][None])
                torch.testing.assert_close(state,s['numeric_state'],rtol=0,atol=0)
                pred=model.infer_action(**common,proprio=state,action_horizon=16,text_cfg_scale=1.)
                pose,grip,bad=model.numeric_codec.recover(pred['action'][None],s['current_pose'][None])
                pose=pose[0].cpu();grip=grip[0].cpu()
            torch.cuda.synchronize(); duration=time.monotonic()-before
            assert pose.shape==(16,7) and grip.shape==(16,1)
            assert torch.isfinite(pose).all() and torch.isfinite(grip).all()
            rows.append(dict(record, action_is_pad=s['action_is_pad'].tolist(),
                predicted_pose=pose.tolist(),target_pose=s['target_pose'].tolist(),
                predicted_gripper=grip.tolist(),target_gripper=s['target_gripper'].tolist(),
                current_pose=s['current_pose'].tolist(),
                error=errors(pose,s['target_pose'],grip,s['target_gripper']),
                near_zero_quaternions=0 if bad is None else int(bad.sum()),inference_seconds=duration))
            payload=dict(method=method,rank=rank,shards=shards,complete=False,rows=rows,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            write_json(path,payload)
            print(f'{method} rank={rank} probe={i} done={len(rows)} inference={duration:.2f}s elapsed={time.monotonic()-started:.1f}s',flush=True)
            del pred
    payload.update(complete=True,elapsed_seconds=time.monotonic()-started)
    write_json(path,payload)


def summarize(rows,horizon,offset=None):
    import numpy as np
    result={}
    for metric in ('position_l2_mm','rotation_deg','gripper_error_pct','gripper_abs_error'):
        values=[v for r in rows for j,v in enumerate(r['error'][metric][:horizon])
                if not r['action_is_pad'][j] and (offset is None or offset==j)]
        if not values: continue
        result[metric]=dict(count=len(values),mean=float(np.mean(values)),median=float(np.median(values)),
            p95=float(np.percentile(values,95)),max=float(np.max(values)))
    return result


def aggregate(out,shards):
    manifest=json.loads((out/'manifest.json').read_text())
    result=dict(protocol=manifest,methods={})
    for method in ('raymap','fastwam'):
        rows=[]
        for rank in range(shards):
            part=json.loads((out/f'{method}_shard{rank}.json').read_text())
            assert part['complete'] and part['shards']==shards
            rows+=part['rows']
        rows.sort(key=lambda x:x['probe'])
        assert [r['index'] for r in rows]==[r['index'] for r in manifest['records']]
        groups={}
        for group in ('original_validation4','stratified100'):
            subset=[r for r in rows if r['group']==group]
            groups[group]=dict(windows=len(subset),**{f'first{h}':summarize(subset,h) for h in (8,16)},
                by_offset={str(j):summarize(subset,16,j) for j in range(16)},
                by_task={task:{f'first{h}':summarize([r for r in subset if r['task']==task],h)
                    for h in (8,16)} for task in sorted({r['task'] for r in subset})})
        result['methods'][method]=groups
    write_json(out/'summary.json',result)
    print(json.dumps({m:{g:{h:v[h] for h in ('first8','first16')} for g,v in d.items()}
        for m,d in result['methods'].items()},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['prepare','worker','aggregate'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=['raymap','fastwam'])
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--shards',type=int,default=2)
    a=p.parse_args()
    import torch
    from fastwam.utils.config_resolvers import register_default_resolvers
    torch.set_num_threads(2);register_default_resolvers()
    if a.mode=='prepare':prepare(a.output)
    elif a.mode=='worker':
        assert a.method is not None and 0 <= a.rank < a.shards
        worker(a.output,a.method,a.rank,a.shards)
    else:aggregate(a.output,a.shards)


if __name__=='__main__': main()
