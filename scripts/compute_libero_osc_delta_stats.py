"""Fit RAY0 bounds to original all4 states; future OSC uses controller limits."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from fastwam.representations.libero_rothko import LiberoRothkoCodecConfig
from fastwam.representations.libero_rothko_osc_delta import LiberoOSCDeltaRothkoCodec


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    root=Path('data/libero_mujoco3.3.2');dest=root/'libero_all4_raw_state_osc_delta_h16_224x448.pt'
    if args.output_dir.exists() or dest.exists() or dest.with_suffix('.json').exists():raise FileExistsError('Refusing overwrite')
    args.output_dir.mkdir(parents=True);torch.set_num_threads(1)
    suites={};lo=np.full(3,np.inf);hi=-lo;amin=np.full(7,np.inf);amax=-amin
    for suite in ['spatial','object','goal','10']:
        d=root/f'libero_{suite}_no_noops_lerobot';mn=np.full(3,np.inf);mx=-mn;n=0;files=sorted(d.glob('data/chunk-*/*.parquet'))
        for p in files:
            t=pq.read_table(p,columns=['observation.state','action']);state=np.asarray(t['observation.state'].to_pylist(),dtype=np.float64);action=np.asarray(t['action'].to_pylist(),dtype=np.float64)
            assert state.shape==(len(t),8) and action.shape==(len(t),7) and np.isfinite(state).all() and np.isfinite(action).all()
            assert np.abs(action[:,:6]).max()<=1 and action[:,6].min()>=0 and action[:,6].max()<=1
            mn=np.minimum(mn,state[:,:3].min(0));mx=np.maximum(mx,state[:,:3].max(0));amin=np.minimum(amin,action.min(0));amax=np.maximum(amax,action.max(0));n+=len(t)
        info=json.loads((d/'meta/info.json').read_text());assert n==info['total_frames'] and len(files)==info['total_episodes']
        suites[suite]=dict(frames=n,episodes=len(files),raw_state_min=mn.tolist(),raw_state_max=mx.tolist());lo=np.minimum(lo,mn);hi=np.maximum(hi,mx)
    cfg=LiberoRothkoCodecConfig(frame0_pose_mode='absolute',absolute_position_min=(lo-.01).tolist(),absolute_position_max=(hi+.01).tolist())
    codec=LiberoOSCDeltaRothkoCodec(cfg);lower,upper=codec.delta_bounds(torch.zeros(1,3,224,448))
    report=dict(suites=suites,state_min=(lo-.01).tolist(),state_max=(hi+.01).tolist(),raw_action_min=amin.tolist(),raw_action_max=amax.tolist(),note='Raw world-state xyz only for RAY0, all4 min/max plus 1cm; no support-plane shift. Future xyz fixed +/-0.05m; rotation directions fixed +/-1. No window weighting or per-suite normalization.')
    report_path=args.output_dir/'bounds.json';report_path.write_text(json.dumps(report,indent=2))
    meta={**codec.metadata(),'stats_format_version':2,'image_size':[224,448],'tile_size':[224,224],'action_horizon':16,'pixel_frames':17,'state_margin_m':.01,'fit_scope':'all4 original training state rows','num_frames':sum(x['frames'] for x in suites.values()),'source_report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest()}
    torch.save(dict(lo=lower,hi=upper,metadata=meta),dest);dest.with_suffix('.json').write_text(json.dumps(meta,indent=2));checked=LiberoOSCDeltaRothkoCodec(cfg,dest,expected_action_horizon=16)
    (args.output_dir/'stats_identity.json').write_text(json.dumps(dict(path=str(dest),fingerprint=checked.norm_stats.fingerprint()),indent=2));print(json.dumps(report));print('fingerprint',checked.norm_stats.fingerprint())

if __name__=='__main__':main()
