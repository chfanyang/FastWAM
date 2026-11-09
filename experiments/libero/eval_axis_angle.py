"""Explicit entry point for axis-angle Goal checkpoints using the shared control loop."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import hydra
from experiments.libero.eval_libero_single import eval_single_process


@hydra.main(config_path='../../configs',config_name='sim_libero_goal_axis_angle',version_base='1.3')
def main(cfg):
    if cfg.model._target_!='fastwam.axis_angle_runtime.create_axis_angle_raymap':
        raise ValueError('This entry point accepts only the isolated axis-angle model.')
    if str(cfg.EVALUATION.task_suite_name)!='libero_goal':
        raise ValueError('This ablation is scoped to LIBERO Goal.')
    return eval_single_process.__wrapped__(cfg)


if __name__=='__main__':main()
