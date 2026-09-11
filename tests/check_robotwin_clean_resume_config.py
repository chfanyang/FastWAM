"""CPU-only check of the actual Hydra single-entry argument forwarding."""
import ast
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ('evaluate_results/robotwin/robotwin_c2r_7task_rothko_centerfrac05_full_wan22_5b_1e-4_2026-09-08_06-20-00/'
              'ckpt002730_vaeOriginal_replan8_ensembleOff_clean100_20260908')
sys.path.insert(0, str(ROOT / 'third_party/RoboTwin/script'))
from eval_resume import identity


def capture(cmd, **kwargs):
    import yaml
    options = yaml.safe_load((ROOT / 'experiments/robotwin/fastwam_policy/deploy_policy.yml').read_text())
    pairs = cmd[cmd.index('--overrides') + 1:]
    for key, value in zip(pairs[::2], pairs[1::2]):
        options[key.removeprefix('--')] = ast.literal_eval(value)
    saved = json.loads((OUT / 'click_alarmclock/progress_clean.json').read_text())
    assert identity(options) == saved['identity'], (identity(options), saved['identity'])
    assert options['resume'] is True
    print('PASS: actual single-entry forwarded resume identity matches imported progress; no simulator launched.')
    raise SystemExit(0)


if __name__ == '__main__':
    options = json.loads((OUT / 'resume_original_options.json').read_text())
    sys.argv = ['eval_robotwin_single.py',
                'task=' + options['sim_task'], 'ckpt=' + options['ckpt_setting'],
                'gpu_id=2', 'EVALUATION.task_name=click_alarmclock',
                'EVALUATION.task_config=demo_clean', 'EVALUATION.resume=true',
                'EVALUATION.eval_num_episodes=100', 'EVALUATION.replan_steps=8',
                'EVALUATION.num_inference_steps=20', '+EVALUATION.eval_step_limit=400',
                'EVALUATION.dataset_stats_path=' + options['dataset_stats_path'],
                'EVALUATION.output_dir=' + str(OUT)]
    with patch('subprocess.Popen', capture):
        runpy.run_path(str(ROOT / 'experiments/robotwin/eval_robotwin_single.py'), run_name='__main__')
