"""Isolated trainer entry point; validation uses the matched numerical codec."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import hydra
import torch
from fastwam.trainer import Wan22Trainer
from fastwam.utils.config_resolvers import register_default_resolvers
register_default_resolvers()


class MatchedTrainer(Wan22Trainer):
    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:return None
        model=self.accelerator.unwrap_model(self.model)
        was_training=model.dit.training;model.eval()
        device=self.accelerator.device
        totals=torch.zeros(2,device=device,dtype=torch.float64)
        devices=[device.index] if device.type=='cuda' else []
        try:
            for offset in range(self.accelerator.process_index,self.eval_num_samples,self.accelerator.num_processes):
                index=(self.eval_sample_index+offset)%len(self.val_dataset)
                raw=self.val_dataset[index]
                sample={k:v[None] if isinstance(v,torch.Tensor) else v for k,v in raw.items()}
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(self.eval_random_seed+offset)
                    loss,_=model.training_loss(sample)
                totals[0]+=loss.double();totals[1]+=1
        finally:
            if was_training:self._apply_dit_only_train_mode(model)
        totals=self.accelerator.gather(totals[None]).sum(0)
        return dict(val_loss=float(totals[0]/totals[1].clamp_min(1)))


@hydra.main(config_path='../configs',config_name='train',version_base='1.3')
def main(cfg):
    if cfg.model._target_!='fastwam.libero_matched_fastwam.create_model':
        raise ValueError('This entry point only accepts the isolated matched baseline')
    # The shared trainer seeds after model construction. Seed here as well so
    # the new action input/head and proprio projection are reproducible.
    from fastwam.utils.pytorch_utils import set_global_seed
    set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    # Local to this process; the shared runtime/trainer files remain unchanged.
    from fastwam import runtime
    runtime.Wan22Trainer=MatchedTrainer
    runtime.run_training(cfg)


if __name__=='__main__':main()
