"""Frozen-VAE round trips for the independent axis-angle codec and baseline.

Uses the exact historical 100 Goal windows. No DiT, simulator, training, or
cache mutation. Historical Flatten/Shuffled/fine-tuned-Raymap rows are retained
as explicitly labelled references after protocol and target checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
OLD=ROOT/'evaluate_results/libero/offline_audits/absolute_ray0_relative_repr100_20260923/results'
FT=ROOT/'evaluate_results/libero/offline_audits/absolute_ray0_relative_repr100_vae8264_rothko_only_20260924/results'
RUN=ROOT/'runs/libero_goal_rothko_absolute_ray0_2cam224_full_wan21_1_3b_1e-4/2026-09-21_18-42-23'
VAE=ROOT/'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'


def write_json(path,value):
    partial=path.with_suffix('.partial')
    partial.write_text(json.dumps(value,indent=2)+'\n');partial.replace(path)


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def initialize(out,shards):
    out.mkdir(parents=True,exist_ok=False)
    old=json.loads((OLD/'report.json').read_text())
    ft=json.loads((FT/'report.json').read_text())
    selection=json.loads((OLD/'selection.json').read_text())
    assert len(selection)==100 and selection==json.loads((FT/'selection.json').read_text())
    assert old['stats_fingerprint']==ft['stats_fingerprint']
    assert old['bounds']==ft['bounds']
    actual=checksum(VAE);assert actual==old['vae_sha256']
    (out/'selection.json').write_bytes((OLD/'selection.json').read_bytes())
    write_json(out/'protocol.json',dict(windows=100,shards=shards,seed=42,
        selection_sha256=checksum(out/'selection.json'),vae=str(VAE),vae_sha256=actual,
        codec_source_sha256=checksum(ROOT/'src/fastwam/representations/libero_axis_angle.py'),
        stats_fingerprint=old['stats_fingerprint'],bounds=old['bounds'],
        original_reference=str(OLD),finetuned_reference=str(FT),
        pose_semantics='absolute observed RAY0; future targets relative to current observed EE',
        pixel_frames=17,prediction_horizon=16,canvas=[224,448],tile=[224,224],
        vae_dtype='torch.bfloat16',vae_batch_size=1,tiled=False,
        padding='edge repeat before VAE; exclude padded future steps in metrics',
        quaternion_error='2 acos(abs(dot(normalize(pred),normalize(gt)))) in degrees',
        gripper_threshold='>0.5, identical to the historical reconstruction audit',
        fresh_representations=['rothko','constant_axis_angle'],
        reused_references=['flatten_chw/original','shuffled/original','rothko/finetuned8264'],
        stages=['direct','bf16_pixels','vae'],no_training=True,no_dit_or_rollout=True))
    print(out,flush=True)


def metrics(pose,grip,target,opening):
    import torch
    q=torch.nn.functional.normalize(pose[1:,3:].double(),dim=-1)
    gt=torch.nn.functional.normalize(target[1:,3:].double(),dim=-1)
    p=(pose[1:,:3].double()-target[1:,:3].double()).norm(dim=-1)*1000
    r=torch.rad2deg(2*torch.acos((q*gt).sum(-1).abs().clamp(0,1)))
    g=(grip[1:].reshape(-1)-opening[1:].reshape(-1)).abs()
    binary=((grip[1:].reshape(-1)>.5)!=(opening[1:].reshape(-1)>.5)).float()
    return dict(position_mm=p.tolist(),rotation_deg=r.tolist(),gripper_abs=g.tolist(),
                gripper_binary=binary.tolist())


def worker(out,rank,shards):
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from fastwam.representations.libero_rothko import LiberoRothkoCodec,LiberoRothkoCodecConfig
    from fastwam.representations.libero_axis_angle import LiberoAxisAngleCodec
    from fastwam.representations.rothko import quaternion_wxyz_to_matrix
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    torch.set_num_threads(2);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    torch.cuda.set_device(0)
    path=out/f'shard{rank}.json'
    if path.exists():raise FileExistsError(path)
    protocol=json.loads((out/'protocol.json').read_text())
    assert shards==protocol['shards']
    assert checksum(out/'selection.json')==protocol['selection_sha256']
    assert checksum(ROOT/'src/fastwam/representations/libero_axis_angle.py')==protocol['codec_source_sha256']
    selection=json.loads((out/'selection.json').read_text())
    cfg=OmegaConf.load(RUN/'config.yaml')
    base=LiberoRothkoCodec(LiberoRothkoCodecConfig(**OmegaConf.to_container(cfg.data.train.rothko_config)),ROOT/cfg.data.train.rothko_norm_stats)
    codec=LiberoAxisAngleCodec(base)
    assert base.norm_stats.fingerprint()==protocol['stats_fingerprint']
    lo=torch.tensor(protocol['bounds']['future_min']);hi=torch.tensor(protocol['bounds']['future_max'])
    vae=_load_registered_model(str(VAE),'wan_video_vae',torch_dtype=torch.bfloat16,device='cuda:0').eval().requires_grad_(False)
    root=ROOT/cfg.data.train.dataset_dirs[0]
    historical=json.loads((OLD/'rows.json').read_text())
    historical_direct={(r['episode'],r['start']):r for r in historical if r['representation']=='rothko' and r['stage']=='direct'}
    rows=[];windows=0;started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index in range(rank,len(selection),shards):
            sel=selection[index]
            table=pq.read_table(next(root.glob(f'data/chunk-*/episode_{sel["episode"]:06d}.parquet')),
                columns=['observation.state.ee_pose_wxyz','action.osc_target_pose_wxyz','observation.state.gripper_open','action'])
            data={k:torch.tensor(table[k].to_pylist(),dtype=torch.float32) for k in table.column_names}
            assert len(table)==sel['length']
            start=sel['start'];indices=torch.arange(start,start+16);valid=indices<len(table)
            indices=indices.clamp_max(len(table)-1)
            pose=torch.cat((data['observation.state.ee_pose_wxyz'][start:start+1],data['action.osc_target_pose_wxyz'][indices]))
            grip=torch.cat((data['observation.state.gripper_open'][start:start+1].reshape(1,1),data['action'][indices,-1:])).clamp(0,1)
            rotation=quaternion_wxyz_to_matrix(pose[:,3:])
            rel=(rotation[0].T @ (pose[1:,:3]-pose[0,:3]).T).T
            clipped=((rel<lo)|(rel>hi)).any(-1)
            old=historical_direct[(sel['episode'],start)]
            assert valid.tolist()==old['valid'] and clipped.tolist()==old['position_clipped']
            a=base.encode(pose,grip);b=codec.encode(pose,grip)
            center,_,_,border=codec.masks(a.device)
            for left,right in zip(a.split(224,-1),b.split(224,-1)):
                assert torch.equal(left[...,center|border],right[...,center|border])
            assert torch.equal(b[...,:224],b[...,224:])
            for name,pixels,decoder in [('rothko',a,base),('constant_axis_angle',b,codec)]:
                latent=vae.encode(pixels[None].to('cuda:0',torch.bfloat16),device='cuda:0',tiled=False)
                assert tuple(latent.shape)==(1,16,5,28,56) and torch.isfinite(latent).all()
                reconstruction=vae.decode(latent,device='cuda:0',tiled=False)[0].float().clamp(-1,1).cpu()
                assert torch.isfinite(reconstruction).all()
                for stage,value in [('direct',pixels),('bf16_pixels',pixels.bfloat16().float()),('vae',reconstruction)]:
                    decoded,opening=decoder.decode(value,current_pose7=pose[0])
                    error=metrics(decoded,opening,pose,grip)
                    if name=='rothko' and stage=='direct':
                        for key in ('position_mm','rotation_deg','gripper_binary'):
                            torch.testing.assert_close(torch.tensor(error[key]),torch.tensor(old[key]),rtol=0,atol=1e-7)
                    rows.append(dict(sel,selection_index=index,representation=name,stage=stage,
                        valid=valid.tolist(),position_clipped=clipped.tolist(),**error))
                del latent,reconstruction
            windows+=1
            payload=dict(rank=rank,shards=shards,complete=False,windows=windows,rows=rows,
                vae_sha256=protocol['vae_sha256'],stats_fingerprint=protocol['stats_fingerprint'],
                selection_sha256=protocol['selection_sha256'],axis_angle_metadata=codec.metadata(),
                peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)
            write_json(path,payload)
            print(f'rank={rank} index={index} windows={windows} elapsed={time.monotonic()-started:.1f}s peakGiB={payload["peak_gpu_GiB"]:.2f}',flush=True)
    payload.update(complete=True,elapsed_seconds=time.monotonic()-started)
    write_json(path,payload)


def stats(values):
    import numpy as np
    return dict(n=len(values),mean=float(np.mean(values)),p95=float(np.percentile(values,95)),max=float(np.max(values)))


def aggregate(out,shards):
    import numpy as np
    protocol=json.loads((out/'protocol.json').read_text())
    rows=[]
    for rank in range(shards):
        part=json.loads((out/f'shard{rank}.json').read_text())
        assert part['complete'] and part['shards']==shards
        for key in ('vae_sha256','stats_fingerprint','selection_sha256'):assert part[key]==protocol[key]
        rows+=part['rows']
    assert len(rows)==600
    rows.sort(key=lambda r:(r['selection_index'],r['representation'],r['stage']))
    assert len({(r['selection_index'],r['representation'],r['stage']) for r in rows})==600
    summary={}
    for rep in ('rothko','constant_axis_angle'):
        summary[rep]={}
        for stage in protocol['stages']:
            subset=[r for r in rows if r['representation']==rep and r['stage']==stage]
            assert len(subset)==100
            summary[rep][stage]={str(h):{k:stats([v for r in subset for v,ok in zip(r[k][:h],r['valid'][:h]) if ok])
                for k in ('position_mm','rotation_deg','gripper_abs','gripper_binary')} for h in (8,16)}
    old=json.loads((OLD/'report.json').read_text());ft=json.loads((FT/'report.json').read_text())
    # Recomputed baseline must reproduce the reference before reusing its other variants.
    reference_differences={}
    oldrows=json.loads((OLD/'rows.json').read_text())
    for stage in protocol['stages']:
        reference={(r['episode'],r['start']):r for r in oldrows if r['representation']=='rothko' and r['stage']==stage}
        current=[r for r in rows if r['representation']=='rothko' and r['stage']==stage]
        reference_differences[stage]={}
        for metric in ('position_mm','rotation_deg','gripper_abs','gripper_binary'):
            delta=max(abs(x-y) for r in current for x,y in zip(r[metric],reference[(r['episode'],r['start'])][metric]))
            reference_differences[stage][metric]=delta
        assert reference_differences[stage]['position_mm']<1e-3
        assert reference_differences[stage]['rotation_deg']<1e-3
        assert reference_differences[stage]['gripper_binary']==0
    report=dict(protocol=protocol,summary=summary,reference_max_per_step_differences=reference_differences,
        completed_windows=100,complete=True,
        references=dict(flatten_chw=old['summary']['flatten_chw'],shuffled=old['summary']['shuffled'],rothko_vae8264=ft['summary']['rothko']))
    write_json(out/'rows.json',rows);write_json(out/'report.json',report)
    lines=['# Goal 100窗口：冻结VAE动作重建消融','',
        '相同历史100窗口（10任务各10），绝对RAY0＋相对未来动作；17帧、224×448，原始Wan2.1 VAE，BF16、batch1、不tiled。尾部先edge-repeat再过VAE，统计排除padding。新轴角和原Raymap本次重跑；其余三组复用已校验同窗口/边界/归一化口径的历史记录。无DiT预测、训练或环境rollout。','']
    groups=[('Flatten-Resize CHW','original',old['summary']['flatten_chw'],'历史记录'),
            ('Shuffled Action Raymap','original',old['summary']['shuffled'],'历史记录'),
            ('Constant Axis-Angle','original',summary['constant_axis_angle'],'本次重跑'),
            ('Action Raymap','original',summary['rothko'],'本次重跑'),
            ('Action Raymap','fine-tuned8264',ft['summary']['rothko'],'历史记录')]
    table=[]
    for h in ('8','16'):
        lines += [f'## 前{h}步','','| 表示 | VAE | 平移均值/P95 (mm) | 旋转均值/P95 (°) | 夹爪错误率 | 来源 |','|---|---|---:|---:|---:|---|']
        for label,vae,s,origin in groups:
            m=s['vae'][h];p=m['position_mm'];r=m['rotation_deg'];g=m['gripper_binary']['mean']*100
            lines.append(f'| {label} | {vae} | {p["mean"]:.3f} / {p["p95"]:.3f} | {r["mean"]:.3f} / {r["p95"]:.3f} | {g:.3f}% | {origin} |')
            table.append(dict(representation=label,vae=vae,horizon=int(h),position_mean_mm=p['mean'],position_p95_mm=p['p95'],rotation_mean_deg=r['mean'],rotation_p95_deg=r['p95'],gripper_error_pct=g,source=origin))
        lines += ['']
    lines += ['## 本次直接编解码／BF16量化校验','', '```json',json.dumps({rep:{stage:summary[rep][stage] for stage in ('direct','bf16_pixels')} for rep in summary},indent=2),'```','',
              '统计按有效预测步聚合；保留共同GT位移裁剪造成的误差底。轴角与方向场同时改变旋转参数化和空间编码；此结果衡量与冻结VAE的适配性，不能代替训练或闭环成功率。微调8264仅在原Raymap上训练，作为原方案参考，不是轴角专门微调。']
    (out/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    import csv
    with (out/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print(json.dumps(table,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initialize','worker','aggregate'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=8)
    a=p.parse_args()
    if a.mode=='initialize':initialize(a.output,a.shards)
    elif a.mode=='worker':
        assert 0<=a.rank<a.shards
        worker(a.output,a.rank,a.shards)
    else:aggregate(a.output,a.shards)


if __name__=='__main__':main()
