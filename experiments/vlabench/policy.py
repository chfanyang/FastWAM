"""VLABench-only EE policy bridge; benchmark remains responsible for IK."""
from collections import deque
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from fastwam.datasets.libero_rgb import build_libero_rgb_canvas
from fastwam.datasets.vlabench_rgb import build_vlabench_rgb_canvas


def build_policy_rgb_canvas(observation, codec_config):
    """Checkpoint-controlled opt-in; old two-camera evaluations stay unchanged."""
    if getattr(codec_config, 'horizontal_copies', 2) == 3:
        if (codec_config.image_height, codec_config.image_width) != (192, 576):
            raise ValueError('Three-camera VLABench checkpoint must use 192x576')
        return build_vlabench_rgb_canvas(dict(image=observation['rgb'][2],
            second_image=observation['rgb'][0], wrist_image=observation['rgb'][3]))
    return build_libero_rgb_canvas(observation['rgb'][2], observation['rgb'][3])


class FastWAMVLABenchPolicy:
    name = "fastwam_rothko"
    control_mode = "ee"

    def __init__(self, model, *, replan_steps, gripper_threshold, seed=42):
        if not 1 <= replan_steps <= model.action_horizon:
            raise ValueError("replan_steps must be within the predicted horizon")
        if not 0 <= gripper_threshold <= 1:
            raise ValueError("gripper threshold must be within [0,1]")
        self.model, self.replan_steps = model, replan_steps
        self.gripper_threshold, self.seed = gripper_threshold, seed
        self.reset()

    def reset(self):
        self.pending = deque()
        self.language_history = []

    @torch.no_grad()
    def predict(self, observation, **kwargs):
        if not self.pending:
            state = np.array(observation['ee_state'], dtype=np.float32, copy=True)
            origin = np.asarray(observation['robot_frame'], dtype=np.float32)
            if state.shape != (8,) or origin.shape != (3,) or not np.isfinite(state).all():
                raise ValueError("Expected finite VLABench EE xyz,wxyz,gripper and 3D robot origin")
            pose = torch.from_numpy(state[:7].copy())
            pose[:3] -= torch.from_numpy(origin)
            opening = 1 - float(state[7])
            if not 0 <= opening <= 1:
                raise ValueError("Observed gripper must be in [0,1]")
            horizon = self.model.action_horizon + 1
            ray = self.model.raymap_codec.encode(
                pose[None].repeat(horizon, 1), torch.full((horizon, 1), opening))[:, 0]
            rgb = build_policy_rgb_canvas(observation, self.model.raymap_codec.config)
            language = str(observation['instruction'])
            self.language_history.append(language)
            prompt = "A video recorded from a robot's point of view executing the following instruction: " + language
            prediction = self.model.infer(prompt=prompt, input_image=rgb,
                input_raymap=ray, current_endpose=pose, num_frames=horizon,
                num_inference_steps=20, seed=self.seed, tiled=False,
                decode_future_rgb=False)
            # Frame zero is the observation anchor, never an action command.
            poses = prediction['pose'][0, 1:1+self.replan_steps].numpy()
            grippers = prediction['gripper'][0, 1:1+self.replan_steps].numpy()
            for target, grip in zip(poses, grippers):
                if not np.isfinite(target).all() or not np.isfinite(grip).all():
                    raise ValueError("Nonfinite model action")
                world_xyz = target[:3] + origin
                euler = Rotation.from_quat(target[[4,5,6,3]]).as_euler('xyz')
                fingers = np.full(2, .04 if float(grip[0]) >= self.gripper_threshold else 0.)
                self.pending.append((world_xyz, euler, fingers))
        return self.pending.popleft()
