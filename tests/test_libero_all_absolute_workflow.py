import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('absolute_workflow', Path(__file__).resolve().parents[1]/'scripts/libero_all_absolute_workflow.py')
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


class WorkflowTest(unittest.TestCase):
    def args(self, stage):
        return SimpleNamespace(stage=stage,output=Path('/example/new_output'),run=Path('/example/run'),
            batch_size=8,num_workers=4,indices_json=None)

    def test_benchmark_cannot_write_formal_cache(self):
        cmd=workflow.command(self.args('benchmark'))
        self.assertIn('--benchmark-max-samples',cmd)
        self.assertNotIn(str(workflow.CACHE),cmd)
        cmd=workflow.command(self.args('cache'))
        self.assertNotIn('--benchmark-max-samples',cmd)
        self.assertIn(str(workflow.CACHE),cmd)

    def test_eval_forwards_worker_protocol_explicitly(self):
        cmd=workflow.command(self.args('eval'))
        for setting in ['seed=42','EVALUATION.replan_steps=8','EVALUATION.num_trials=50',
                        'EVALUATION.num_inference_steps=20','model.vae_safetensors_path=null',
                        'model.rothko_decode_mode=legacy','model.rothko_decode_anchor_alpha=0',
                        'EVALUATION.use_action_ensembler=false','MULTIRUN.max_tasks_per_gpu=2']:
            self.assertIn(setting,cmd)
        self.assertIn('ckpt=/example/run/checkpoints/weights/step_021700.pt',cmd)

    def test_default_gpu_stages_are_side_effect_free(self):
        for stage in ['audit','benchmark','cache','train','eval']:
            args=['workflow',stage,'--output','/example/not_created','--run','/example/run']
            with patch('sys.argv',args),patch.object(workflow.subprocess,'check_output') as query, \
                    patch.object(workflow.subprocess,'Popen') as launch, \
                    patch.object(workflow,'check_inputs') as check, \
                    contextlib.redirect_stdout(io.StringIO()):
                workflow.main()
                query.assert_not_called();launch.assert_not_called();check.assert_not_called()

    def test_audit_manifest_forwarding(self):
        args=self.args('audit');args.indices_json=Path('/example/windows.json')
        self.assertEqual(workflow.command(args)[-2:],['--indices-json','/example/windows.json'])


if __name__=='__main__':
    unittest.main()
