import importlib.util
import json
import tempfile
import unittest
import ast
import sys
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

spec = importlib.util.spec_from_file_location('eval_resume', Path(__file__).resolve().parents[1] /
                                             'third_party/RoboTwin/script/eval_resume.py')
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


class ResumeTests(unittest.TestCase):
    def test_actual_eval_loop_resumes_counts_seeds_and_model_episode(self):
        source = (Path(__file__).resolve().parents[1] /
                  'third_party/RoboTwin/script/eval_policy.py').read_text()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                    and n.name == 'eval_policy')
        class Env:
            eval_video_path = None
            render_freq = 0
            plan_success = True
            step_lim = 4
            def setup_demo(self, **kwargs):
                self.take_action_cnt = 0
                self.eval_success = False
                self.seed = kwargs['seed']
            def play_once(self): return {'info': {}}
            def close_env(self, **kwargs): pass
            def check_success(self): return True
            def set_instruction(self, **kwargs): pass
            def get_obs(self): return {}
        calls = []
        def reset(model): model.episode_count += 1
        def step(env, model, observation):
            calls.append((env.seed, model.episode_count - 1))
            env.take_action_cnt += 1
            env.eval_success = True
        namespace = dict(
            UnStableError=RuntimeError,
            eval_function_decorator=lambda policy, name: step if name == 'eval' else reset,
            generate_episode_descriptions=lambda *args: [{'unseen': ['test']}],
            np=SimpleNamespace(random=SimpleNamespace(choice=lambda a: a[0])),
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<eval_loop>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as d, patch.dict(sys.modules, {'eval_resume': resume}):
            state = dict(version=1, identity={}, completed=2, successes=1, next_seed=4300005)
            path = Path(d) / 'progress.json'
            result = namespace['eval_policy'](
                'test', Env(), dict(task_name='test', policy_name='test', render_freq=0,
                                   clear_cache_freq=1, task_config='demo_clean', ckpt_setting='x'),
                SimpleNamespace(episode_count=0), 4300000, test_num=4,
                instruction_type='unseen', resume_state=state, resume_file=path)
            self.assertEqual(calls, [(4300005, 2), (4300006, 3)])
            self.assertEqual(result, (4300007, 3))
            self.assertEqual(json.loads(path.read_text())['completed'], 4)

    def test_roundtrip_and_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'progress_clean.json'
            options = {'seed': 42, 'replan_steps': 8, 'eval_step_limit': 400}
            ident = resume.identity(options)
            state = dict(version=1, identity=ident, completed=2, successes=1, next_seed=4300005)
            resume.atomic_save(path, state)
            self.assertEqual(resume.load_progress(path, ident, 100, 4300000), state)
            for key, value in [('seed', 43), ('replan_steps', 16), ('eval_step_limit', 220),
                               ('vae_safetensors_path', 'different.safetensors')]:
                with self.assertRaises(ValueError):
                    resume.load_progress(path, resume.identity(dict(options, **{key: value})), 100, 4300000)
            with self.assertRaises(ValueError):
                resume.load_progress(path, ident, 1, 4300000)

    def test_import_and_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            log = root / 'eval.log'
            log.write_text('Success rate: \x1b[96m1/1\x1b[0m => 100%, current seed: 4300002\n'
                           'Success rate: 1/2 => 50%, current seed: 4300004\n')
            for i, success in [(0, 'true'), (1, 'false'), (2, 'true')]:
                (root / f'episode{i}_randomized-false_success-{success}.mp4').write_bytes(b'video')
            (root / 'episode2.mp4').write_bytes(b'partial')
            (root / 'predictions/demo_clean/episode_002').mkdir(parents=True)
            state = resume.import_legacy(log, root, 'clean', {'seed': 42})
            self.assertEqual((state['completed'], state['successes'], state['next_seed']), (2, 1, 4300005))
            with self.assertRaises(FileExistsError):
                resume.import_legacy(log, root, 'clean', {'seed': 42})
            resume.archive_uncommitted(root, 'clean', 2)
            self.assertTrue((root / 'episode0_randomized-false_success-true.mp4').exists())
            self.assertFalse((root / 'episode2.mp4').exists())
            self.assertFalse((root / 'predictions/demo_clean/episode_002').exists())
            self.assertEqual(len(list((root / 'interrupted').rglob('*.mp4'))), 2)


if __name__ == '__main__':
    unittest.main()
