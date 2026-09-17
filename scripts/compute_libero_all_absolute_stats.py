"""Exact all-four-suite state + OSC target bounds; CPU only, no window weighting."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from fastwam.representations.libero_rothko import LiberoRothkoCodecConfig
from fastwam.representations.libero_rothko_all_absolute import LiberoAllAbsoluteRothkoCodec
from fastwam.representations.rothko import RothkoNormStats

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 a.output_dir.mkdir(parents=True,exist_ok=True);assert not (a.output_dir/'bounds.json').exists()
 pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
 root=Path('data/libero_mujoco3.3.2');keys=['observation.state.ee_pose_wxyz','action.osc_target_pose_wxyz'];suites={};global_min=np.full(3,np.inf);global_max=-global_min;examples=[]
 for suite in ['spatial','object','goal','10']:
  d=root/f'libero_{suite}_no_noops_lerobot';info=json.loads((d/'meta/info.json').read_text());lo=np.full(3,np.inf);hi=-lo;n=0;mins=[None]*3;maxs=[None]*3;qmin=np.inf;qmax=0;files=sorted(d.glob('data/chunk-*/*.parquet'))
  assert len(files)==info['total_episodes']
  for file in files:
   table=pq.read_table(file,columns=keys);n+=len(table)
   for key in keys:
    x=np.asarray(table[key].combine_chunks().to_pylist(),dtype=np.float32)
    assert x.shape==(len(table),7) and np.isfinite(x).all(),file
    q=np.linalg.norm(x[:,3:].astype(float),axis=1);assert q.min()>1e-8
    qmin=min(qmin,float(q.min()));qmax=max(qmax,float(q.max()))
    for axis in range(3):
     i=int(x[:,axis].argmin());j=int(x[:,axis].argmax())
     if x[i,axis]<lo[axis]:lo[axis]=float(x[i,axis]);mins[axis]=dict(file=str(file),key=key,row=i,pose=x[i].tolist())
     if x[j,axis]>hi[axis]:hi[axis]=float(x[j,axis]);maxs[axis]=dict(file=str(file),key=key,row=j,pose=x[j].tolist())
  assert n==info['total_frames']
  suites[suite]=dict(episodes=len(files),frames=n,raw_min=lo.tolist(),raw_max=hi.tolist(),min_records=mins,max_records=maxs,quaternion_norm_range=[qmin,qmax],info_sha256=hashlib.sha256((d/'meta/info.json').read_bytes()).hexdigest())
  global_min=np.minimum(global_min,lo);global_max=np.maximum(global_max,hi);print(suite,n,lo,hi,flush=True)
 lo=(global_min-.01).tolist();hi=(global_max+.01).tolist()
 report=dict(suites=suites,raw_min=global_min.tolist(),raw_max=global_max.tolist(),position_min=lo,position_max=hi,margin_m=.01,frames=sum(s['frames'] for s in suites.values()),episodes=sum(s['episodes'] for s in suites.values()))
 (a.output_dir/'bounds.json').write_text(json.dumps(report,indent=2))
 cfg=LiberoRothkoCodecConfig(frame0_pose_mode='absolute',absolute_position_min=lo,absolute_position_max=hi)
 codec=LiberoAllAbsoluteRothkoCodec(config=cfg);lower,upper=codec._absolute_frame_bounds(torch.zeros(1,3,224,448))
 meta={**codec.metadata(),'stats_format_version':2,'image_size':[224,448],'tile_size':[224,224],'action_horizon':16,'pixel_frames':17,'method':'joint signed per-axis min/max of raw EE states and OSC targets','sampling':'each raw state and target row exactly once; no window repetition or padding','margin_m':.01,'num_frames':report['frames'],'num_episodes':report['episodes'],'num_xyz_positions':2*report['frames'],'fit_split':'all four suite training datasets; existing config has no held-out split','raw_position_min':global_min.tolist(),'raw_position_max':global_max.tolist(),'source_bounds_file':str((a.output_dir/'bounds.json').resolve()),'source_bounds_sha256':hashlib.sha256((a.output_dir/'bounds.json').read_bytes()).hexdigest()}
 dest=root/'libero_all4_rothko_all_absolute_minmax_margin01_h16_224x448_centerfrac05.pt'
 assert not dest.exists() and not dest.with_suffix('.json').exists()
 torch.save(dict(lo=lower,hi=upper,metadata=meta),dest);dest.with_suffix('.json').write_text(json.dumps(meta,indent=2))
 checked=LiberoAllAbsoluteRothkoCodec(config=cfg,norm_stats=dest,expected_action_horizon=16)
 (a.output_dir/'stats_identity.json').write_text(json.dumps(dict(path=str(dest),fingerprint=checked.norm_stats.fingerprint()),indent=2));print('stats',dest,checked.norm_stats.fingerprint(),flush=True)
if __name__=='__main__':main()
