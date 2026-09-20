"""CPU-only checks for persistent runtime reuse, routing and safe result reuse."""
import importlib.util
import json
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
def load(name, path):
    spec=importlib.util.spec_from_file_location(name, ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
manager=load('persistent_manager','experiments/vlabench/run_track1_persistent.py')
entry=load('persistent_entry','experiments/vlabench/eval_select_book.py')

class PersistentTests(unittest.TestCase):
    def test_exact_coverage_and_four_slots(self):
        from collections import Counter
        track={f'task{i}':list(range(50)) for i in range(10)}
        jobs=manager.jobs_for(list(track),32,track)
        self.assertEqual(len(jobs),320);self.assertEqual(len(set(jobs)),320)
        self.assertEqual(Counter(t for t,e in jobs),{t:32 for t in track})
        self.assertEqual(Counter(manager.slot_gpus(list(range(8)),4)),{g:4 for g in range(8)})
        with self.assertRaises(ValueError):manager.slot_gpus([0,0],4)
        with self.assertRaises(ValueError):manager.jobs_for(['bad'],32,track)

    def test_runtime_stays_same_across_different_tasks(self):
        jobs=queue.Queue()
        sequence=[('book',0),('fruit',1),('book',2)]
        for j in sequence:jobs.put(j)
        jobs.put(None)
        loads=[];seen=[];notifications=[]
        def runner(job,runtime):
            if runtime is None:
                runtime=object();loads.append(runtime)
            seen.append((job,runtime));return runtime
        manager.consume_jobs(jobs,runner,lambda kind,job:notifications.append((kind,job)))
        self.assertEqual(len(loads),1)
        self.assertEqual([j for j,r in seen],sequence)
        self.assertTrue(all(r is loads[0] for j,r in seen))
        self.assertEqual(notifications,[(kind,j) for j in sequence for kind in ('started','completed')])

    def test_error_does_not_report_completion_or_silently_reload(self):
        jobs=queue.Queue();jobs.put(('bad',0));jobs.put(('next',0));events=[]
        def runner(job,runtime):raise RuntimeError('simulator failure')
        with self.assertRaises(RuntimeError):manager.consume_jobs(jobs,runner,lambda *x:events.append(x))
        self.assertEqual(events,[('started',('bad',0))])
        self.assertEqual(jobs.get_nowait(),('next',0))

    def test_opt_in_recovery_keeps_runtime_and_does_not_complete_error(self):
        class PhysicsFailure(Exception): pass
        jobs=queue.Queue()
        for j in [('ok',0),('bad',0),('next',0),None]: jobs.put(j)
        runtime=object();seen=[];events=[];errors=[]
        def runner(job, previous):
            seen.append(previous)
            if job[0]=='bad': raise PhysicsFailure('invalid physics')
            return runtime
        manager.consume_jobs(jobs,runner,lambda *x:events.append(x),
                             (PhysicsFailure,),lambda job,error:errors.append(job))
        self.assertEqual(seen,[None,runtime,runtime])
        self.assertEqual(errors,[('bad',0)])
        self.assertNotIn(('completed',('bad',0)),events)
        self.assertIn(('completed',('next',0)),events)
        jobs.put(('unexpected',0))
        def broken(job, previous): raise RuntimeError('model failure')
        with self.assertRaises(RuntimeError):
            manager.consume_jobs(jobs,broken,lambda *x:None,(PhysicsFailure,),lambda *x:None)

    def test_runtime_identity_excludes_task_but_locks_model_and_policy(self):
        args=SimpleNamespace(checkpoint=Path('/model.pt'),vae_safetensors_path=None,
            decode_mode='legacy',allow_vae_mismatch=False,replan_steps=8,
            gripper_threshold=.5,seed=42,task='book')
        identity=entry.runtime_identity(args)
        args.task='fruit';self.assertEqual(identity,entry.runtime_identity(args))
        args.replan_steps=4;self.assertNotEqual(identity,entry.runtime_identity(args))

    def test_inherit_completed_only_and_reject_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);dest=folder/'book/episode_000';dest.mkdir(parents=True)
            args=SimpleNamespace(checkpoint=folder/'model.pt',seed=42,replan_steps=8,
                                 gripper_threshold=.5,decode_mode='legacy')
            identity=dict(task='book',seed=42,replan_steps=8,gripper_threshold=.5,
                          decoder='legacy',vae='original',max_substeps=1,checkpoint=str(args.checkpoint))
            (dest/'evaluation_config.json').write_text(json.dumps(identity))
            (dest/'episode_000.json').write_text(json.dumps(dict(task='book',episode=0,success=True)))
            inherited=manager.inherited_results(folder,[('book',0),('book',1)],args)
            self.assertEqual(set(inherited),{('book',0)})
            report=manager.summary(['book'],2,inherited)
            self.assertFalse(report['complete']);self.assertEqual(report['completed'],1)
            self.assertIsNone(report['by_task']['book']['success_rate'])
            args.vae_safetensors_path=folder/'finetuned.safetensors'
            with self.assertRaises(ValueError):manager.inherited_results(folder,[('book',0)],args)
            identity['vae']=str(args.vae_safetensors_path.resolve())
            (dest/'evaluation_config.json').write_text(json.dumps(identity))
            self.assertEqual(len(manager.inherited_results(folder,[('book',0)],args)),1)
            args.seed=43
            with self.assertRaises(ValueError):manager.inherited_results(folder,[('book',0)],args)

if __name__=='__main__':unittest.main()
