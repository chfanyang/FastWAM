"""Compare the two-step full-state smoke and its step1->2 resume numerically."""
import argparse
import json
import math
from pathlib import Path

import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--resumed',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--world-size',type=int,required=True)
    p.add_argument('--ga',type=int,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    x=torch.load(a.initial/'checkpoints/weights/step_000002.pt',map_location='cpu',weights_only=False,mmap=True)
    y=torch.load(a.resumed/'checkpoints/weights/step_000002.pt',map_location='cpu',weights_only=False,mmap=True)
    z=torch.load(a.initial/'checkpoints/weights/step_000001.pt',map_location='cpu',weights_only=False,mmap=True)
    assert x['matched_identity']==y['matched_identity']==z['matched_identity']
    metrics={}
    for group in ('mot','proprio_encoder'):
        assert x[group].keys()==y[group].keys()==z[group].keys()
        count=different=0;diff_ss=update_ss=0.;maximum=0.
        for key,value in x[group].items():
            left=value.reshape(-1);right=y[group][key].reshape(-1);previous=z[group][key].reshape(-1)
            for begin in range(0,left.numel(),1024*1024):
                xx=left[begin:begin+1024*1024].float()
                yy=right[begin:begin+1024*1024].float()
                zz=previous[begin:begin+1024*1024].float()
                assert torch.isfinite(xx).all() and torch.isfinite(yy).all() and torch.isfinite(zz).all()
                d=xx-yy;u=xx-zz
                count+=xx.numel();different+=int((xx!=yy).sum())
                diff_ss+=float(d.double().square().sum());update_ss+=float(u.double().square().sum())
                maximum=max(maximum,float(d.abs().max()))
        rms=math.sqrt(diff_ss/count);update_rms=math.sqrt(update_ss/count)
        metrics[group]=dict(elements=count,different=different,different_fraction=different/count,
            rms_diff=rms,update_rms=update_rms,diff_vs_update_rms=rms/max(update_rms,1e-30),max_diff=maximum)
    sa=a.initial/'checkpoints/state/step_000002';sb=a.resumed/'checkpoints/state/step_000002'
    tx=json.loads((sa/'trainer_state.json').read_text());ty=json.loads((sb/'trainer_state.json').read_text())
    assert tx==ty and ty['global_step']==2 and ty['batch_in_epoch']==2*a.ga
    assert torch.load(sa/'scheduler.bin',weights_only=False)==torch.load(sb/'scheduler.bin',weights_only=False)
    for rank in range(a.world_size):
        rx=torch.load(sa/f'random_states_{rank}.pkl',weights_only=False)
        ry=torch.load(sb/f'random_states_{rank}.pkl',weights_only=False)
        assert len(rx['torch_cuda_manual_seed'])==len(ry['torch_cuda_manual_seed'])
        assert all(torch.equal(i,j) for i,j in zip(rx['torch_cuda_manual_seed'],ry['torch_cuda_manual_seed']))
        opt=sb/'pytorch_model'/f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
        assert opt.is_file() and opt.stat().st_size>0
    # BF16 GPU kernels previously showed tiny non-bitwise differences. Report
    # them explicitly and require numerical agreement relative to the update.
    accepted=all(m['max_diff']<=1e-3 and m['diff_vs_update_rms']<.1 for m in metrics.values())
    result=dict(numerical_resume_accepted=accepted,exact_step2_weight_match=all(m['different']==0 for m in metrics.values()),
        metrics=metrics,world_size=a.world_size,effective_batch=4*a.ga*a.world_size,
        restored_from_step=1,continued_to_step=2,trainer_state_equal=True,scheduler_equal=True,
        cuda_rng_equal_all_ranks=True,optimizer_shards_complete=True,
        note='This is a numerical resume check, not a claim of bitwise GPU determinism.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not accepted:raise AssertionError('Resume numerical difference exceeds predeclared diagnostic thresholds')


if __name__=='__main__':main()
