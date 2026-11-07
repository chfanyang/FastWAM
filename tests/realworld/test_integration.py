"""Optional CPU integration checks using installed websockets and scipy, without robot drivers."""
from functools import partial
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest

import numpy as np

from fastwam.realworld.common import xyzw_to_wxyz, wxyz_to_xyzw
from fastwam.realworld.execution import execute_chunk
from fastwam.realworld.protocol import pack, unpack

ROOT = Path(__file__).resolve().parents[2]


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(importlib.util.find_spec("websockets"), "Optional serving dependency not installed")
class WebSocketIntegrationTest(unittest.TestCase):
    def test_real_socket_metadata_arrays_errors_and_reconnect(self):
        from websockets.sync.server import serve
        from websockets.sync.client import connect
        transport = module_from_path("realworld_serve", ROOT / "scripts/realworld/serve.py")
        def infer(obs):
            if obs["request_id"] == -1:
                raise ValueError("invalid request")
            return dict(request_id=obs["request_id"], target_pose_wxyz=obs["pose"])
        policy = SimpleNamespace(metadata={"protocol_version": 1}, infer=infer)
        with serve(partial(transport.handle_connection, policy), "127.0.0.1", 0,
                   compression=None, max_size=8 * 1024 * 1024) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                port = server.socket.getsockname()[1]
                for request_id in (1, 2):
                    with connect(f"ws://127.0.0.1:{port}", compression=None) as client:
                        self.assertEqual(unpack(client.recv(timeout=3)), policy.metadata)
                        pose = np.arange(112, dtype=np.float32).reshape(16, 7)
                        client.send(pack(dict(request_id=request_id, pose=pose)))
                        response = unpack(client.recv(timeout=3))
                        self.assertEqual(response["request_id"], request_id)
                        np.testing.assert_array_equal(response["target_pose_wxyz"], pose)
                        client.send(pack(dict(request_id=-1)))
                        self.assertIn("invalid request", client.recv(timeout=3))
            finally:
                server.shutdown()
                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())


@unittest.skipUnless(importlib.util.find_spec("scipy"), "Optional YARP interpolation dependency not installed")
class YarpInterpolatorIntegrationTest(unittest.TestCase):
    def test_real_interpolator_arrives_at_gripper_deadlines(self):
        yarp = module_from_path("yarp_interpolator", ROOT / "third_party/yarp/src/yarp/common/trajectory_interpolator.py")
        for step_m in (.002, .01):  # Nominal 10 Hz, then a speed-limited trajectory.
            with self.subTest(step_m=step_m):
                initial = np.array([.5, 0, .25, 1., 0, 0, 0])
                targets = np.tile(initial, (4, 1))
                targets[:, 0] += step_m * np.arange(1, 5)
                time_now, queued, gripper_states = [0.], [], []
                interp = yarp.PoseTrajectoryInterpolator([0.], [wxyz_to_xyzw(initial)])
                last_waypoint = [0.]
                # Run exactly the schedule semantics used by FrankaController.run().
                def submit(pose, target_time):
                    nonlocal interp
                    consume_time = len(queued) * .01
                    interp = interp.schedule_waypoint(pose, target_time - 100,
                        curr_time=consume_time + .01, last_waypoint_time=last_waypoint[0],
                        max_pos_speed=.05, max_rot_speed=.2)
                    last_waypoint[0] = target_time - 100
                    queued.append(target_time - 100)
                class Stop:
                    def is_set(self): return False
                    def wait(self, seconds):
                        time_now[0] += seconds
                        return False
                def grip_event():
                    gripper_states.append((time_now[0], xyzw_to_wxyz(interp(time_now[0]))))
                arm = SimpleNamespace(schedule_waypoint=submit)
                gripper = SimpleNamespace(grasp=lambda **_: grip_event(), release=grip_event)
                execute_chunk(arm, gripper, targets, [0, 1, 0, 1], initial, True, Stop(),
                    clock=lambda: time_now[0], wall_clock=lambda: 100 + time_now[0])
                np.testing.assert_allclose([s[1][:3] for s in gripper_states], targets[:, :3], atol=1e-6)
                np.testing.assert_allclose(np.diff(queued), max(.1, step_m / .05), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
