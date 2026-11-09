"""Goal matched-target MoT evaluation using the existing LIBERO control loop."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import hydra
import torch
from omegaconf import open_dict
from experiments.libero import eval_libero_single as base


@torch.no_grad()
def predict(obs,task_description,model,processor,cfg,*,action_horizon,input_w,input_h,
            model_device,prompt_context=None,prompt_context_mask=None):
    if action_horizon!=16 or base._record_prediction_videos(cfg):
        raise ValueError('Matched baseline expects h16 action-only evaluation')
    if float(cfg.EVALUATION.text_cfg_scale)!=1. or str(cfg.EVALUATION.negative_prompt)!='':
        raise ValueError('Matched action-only baseline does not support text CFG or negative prompts')
    image,_,imgs=base._obs_to_model_input(obs,cfg=cfg,processor=processor,
        width=input_w,height=input_h,device=model_device,dtype=model.torch_dtype)
    pose=base._extract_absolute_pose_wxyz(obs)
    grip=base._extract_gripper_open(obs)
    state=model.numeric_codec.state_vector(pose[None],grip[None])
    text=dict(prompt=base.DEFAULT_PROMPT.format(task=task_description))
    if prompt_context is not None:
        text=dict(prompt=None,context=prompt_context,context_mask=prompt_context_mask)
    seed=cfg.EVALUATION.get('inference_seed',cfg.get('seed',42))
    pred=model.infer_action(**text,input_image=image,proprio=state,action_horizon=16,
        num_inference_steps=int(cfg.EVALUATION.num_inference_steps),
        seed=None if seed is None else int(seed),rand_device=str(cfg.EVALUATION.rand_device),
        sigma_shift=cfg.EVALUATION.get('sigma_shift'),
        tiled=bool(cfg.EVALUATION.tiled),text_cfg_scale=1.)
    targets,opening,_=model.numeric_codec.recover(pred['action'][None],pose[None])
    return dict(target_pose=targets[0].cpu().numpy(),gripper_open=opening[0].cpu().numpy()),imgs,None


def run_matched_evaluation(cfg):
    if cfg.model._target_!='fastwam.libero_matched_fastwam.create_model':
        raise ValueError('Only matched FastWAM checkpoints accepted')
    original_predict=base._predict_action_chunk
    original_instantiate=base.instantiate
    def instantiate(config,*args,**kwargs):
        if config.get('_target_')=='fastwam.libero_matched_fastwam.create_model':
            # The shared evaluator sees our LIBERO codec data contract and
            # applies its Raymap/LoRA pretrained-loading branch. Undo that
            # branch here: full matched checkpoints contain BOTH experts.
            # Neither base DiT nor the initialization artifact is needed.
            with open_dict(config):
                config.skip_dit_load_from_pretrain=True
                config.action_dit_pretrained_path=None
        return original_instantiate(config,*args,**kwargs)
    base._predict_action_chunk=predict
    base.instantiate=instantiate
    try:
        return base.eval_single_process.__wrapped__(cfg)
    finally:
        base._predict_action_chunk=original_predict
        base.instantiate=original_instantiate


@hydra.main(config_path='../../configs',config_name='sim_libero_matched_fastwam',version_base='1.3')
def main(cfg):
    return run_matched_evaluation(cfg)


if __name__=='__main__':main()
