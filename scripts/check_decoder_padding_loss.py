"""Loss invariance and gradient checks for the isolated decoder workflow."""
from types import SimpleNamespace
import torch
import os
from prepare_all4_absolute_ray0_decoder import reconstruction_loss_valid_frames

def main():
    args=SimpleNamespace(center_loss_weight=1.,direction_loss_weight=1.,gripper_loss_weight=1.)
    masks={'center':torch.tensor([[True,False],[False,False]]),'direction':torch.tensor([[False,True],[True,False]]),'gripper':torch.tensor([[False,False],[False,True]])}
    valid=torch.tensor([[True,True,False],[True,False,False]])
    pred=torch.ones(2,3,3,2,2,requires_grad=True);gt=torch.zeros_like(pred)
    loss,_=reconstruction_loss_valid_frames(pred,gt,masks,args,valid)
    torch.testing.assert_close(loss,torch.tensor(3.));loss.backward()
    padded=(~valid)[:,None,:,None,None].expand_as(pred)
    assert torch.count_nonzero(pred.grad[padded])==0
    assert (pred.grad[~padded]>0).all()
    changed=gt.detach().clone();changed[padded]=10000
    other,_=reconstruction_loss_valid_frames(pred.detach(),changed,masks,args,valid)
    torch.testing.assert_close(loss.detach(),other)
    full=torch.ones_like(valid)
    baseline=sum((pred.detach()-gt).abs()[...,m].mean() for m in masks.values())
    actual,_=reconstruction_loss_valid_frames(pred.detach(),gt,masks,args,full)
    torch.testing.assert_close(actual,baseline)
    if int(os.environ.get('WORLD_SIZE','1'))>1:
        torch.distributed.init_process_group('gloo')
        rank=torch.distributed.get_rank()
        local=pred.detach()[rank:rank+1].clone().requires_grad_(True)
        local_loss,_=reconstruction_loss_valid_frames(local,gt[rank:rank+1],masks,args,valid[rank:rank+1])
        local_loss.backward()
        torch.testing.assert_close(local.grad/2,pred.grad[rank:rank+1])
        mean_loss=local_loss.detach();torch.distributed.all_reduce(mean_loss);mean_loss/=2
        torch.testing.assert_close(mean_loss,loss.detach())
        torch.distributed.destroy_process_group()
        print('PASS: unequal valid-frame counts across two ranks match concatenated global batch loss and DDP-averaged gradients')
    print('PASS: padded targets cannot change loss; padded output gradient zero; valid/RAY0 gradients nonzero; unpadded loss equals baseline')
if __name__=='__main__':main()
