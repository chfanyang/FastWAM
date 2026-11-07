"""Original FastWAM deployment using the established absolute-EE wire protocol."""
import json
import re
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .baseline_data import ActionCodec, REPRESENTATION
from .common import CAMERAS, PREPROCESSING, POSE_CONVENTION, rgb_canvas, validate_pose
from .dataset import load_context
from .protocol import VERSION, ACTION_SPACE


class OriginalFastWAMPolicy:
    def __init__(self, model, manifest_path, stats_path, text_cache_dir,
                 task_name=None, num_inference_steps=20, seed=42, save_rgb_dir=None):
        self.model = model
        manifest = json.loads(Path(manifest_path).read_text())
        if (tuple(manifest['camera_keys']) != CAMERAS or manifest['horizon'] != 16
                or manifest['preprocessing'] != PREPROCESSING
                or manifest['pose_convention'] != POSE_CONVENTION):
            raise ValueError('Unexpected deployment data contract')
        self.codec = ActionCodec(stats_path, manifest_path)
        if self.codec.fingerprint != model.action_codec.fingerprint:
            raise ValueError('Deployment model/action statistics mismatch')
        if self.codec.stats['gripper_open_threshold_m'] != manifest['gripper_open_threshold_m']:
            raise ValueError('Gripper convention mismatch')
        records = [r for r in manifest['episodes'] if task_name is None or r['task'] == task_name]
        if not records:
            raise ValueError('Unknown deployment task')
        self.contexts = {s: load_context(text_cache_dir, s)
                         for s in sorted({r['instruction'] for r in records})}
        self.steps, self.seed = int(num_inference_steps), seed
        if self.steps < 1:
            raise ValueError('Inference steps must be positive')
        self.lock = threading.Lock()
        self.save_rgb_dir = Path(save_rgb_dir) if save_rgb_dir else None
        self.metadata = dict(protocol_version=VERSION, action_space=ACTION_SPACE,
            model_family='original_fastwam_wan22_5b', task_name=task_name,
            horizon=16, action_dt=0.1, pose_quaternion_order='wxyz', ee_link='fr3_link8',
            camera_keys=list(CAMERAS), color_order='RGB', preprocessing=PREPROCESSING,
            gripper_open_threshold_m=manifest['gripper_open_threshold_m'],
            instructions=list(self.contexts), stats_fingerprint=self.codec.fingerprint,
            saves_prediction_rgb=self.save_rgb_dir is not None,
            codec=dict(representation=REPRESENTATION,
                       absolute_position_min=self.codec.lo.tolist(),
                       absolute_position_max=self.codec.hi.tolist()))

    def infer(self, obs):
        started = time.monotonic()
        if (obs.get('protocol_version') != VERSION or obs.get('ee_link') != 'fr3_link8'
                or obs.get('color_order') != 'RGB'):
            raise ValueError('Expected protocol v1, fr3_link8 pose and RGB images')
        if type(obs.get('request_id')) is not int or obs['request_id'] < 0:
            raise ValueError('Expected nonnegative integer request_id')
        if obs['prompt'] not in self.contexts:
            raise ValueError('Prompt does not match the deployed task')
        pose = validate_pose(obs['ee_pose_wxyz'])
        if pose.shape != (7,):
            raise ValueError('Expected one current pose')
        if np.any((pose[:3] < self.codec.lo) | (pose[:3] > self.codec.hi)):
            raise ValueError('Current position outside training bounds')
        grip = float(obs['gripper_open'])
        proprio = torch.from_numpy(self.codec.proprio(pose, grip))
        canvas = rgb_canvas(obs['images'], CAMERAS)
        rgb = torch.from_numpy(canvas).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0)
        context, mask = self.contexts[obs['prompt']]
        steps = obs.get('record_steps', 8)
        session = obs.get('record_session')
        if type(steps) is not int or not 1 <= steps <= 16:
            raise ValueError('Invalid recording horizon')
        if session is not None and (not isinstance(session, str) or re.fullmatch('[0-9a-f]{32}', session) is None):
            raise ValueError('Invalid recording session')
        kwargs = dict(prompt=None, input_image=rgb, action_horizon=16, proprio=proprio,
            context=context, context_mask=mask, num_inference_steps=self.steps,
            seed=self.seed, text_cfg_scale=1., tiled=False)
        with self.lock, torch.inference_mode():
            if self.save_rgb_dir is None:
                prediction = self.model.infer_action(**kwargs)
            else:
                prediction = self.model.infer_joint(**kwargs, num_video_frames=17,
                    test_action_with_infer_action=False)
        actions = prediction['action'].detach().float().cpu().numpy()
        if actions.shape != (16, 7):
            raise ValueError('Invalid model action horizon')
        targets, gripper = self.codec.decode(pose, actions)
        artifact = None
        if self.save_rgb_dir is not None:
            try:
                artifact = self.save_rgb(prediction['video'], canvas, obs, steps, session)
            except OSError as exc:
                print(f'Warning: predicted RGB was not saved: {exc}', flush=True)
        return dict(protocol_version=VERSION, request_id=obs['request_id'],
            action_space=ACTION_SPACE, action_dt=0.1, target_pose_wxyz=targets,
            gripper_open=gripper, prediction_artifact_key=artifact,
            server_latency_s=time.monotonic()-started)

    def save_rgb(self, frames, canvas, obs, steps, session):
        if len(frames) != 17:
            raise ValueError('Expected current plus 16 predicted RGB frames')
        self.save_rgb_dir.mkdir(parents=True, exist_ok=True)
        if session is None:
            folder = Path(tempfile.mkdtemp(prefix=f"request_{obs['request_id']:06d}_", dir=self.save_rgb_dir))
        else:
            folder = self.save_rgb_dir / session / f"request_{obs['request_id']:06d}"
            folder.mkdir(parents=True, exist_ok=False)
        Image.fromarray(canvas).save(folder / 'observation.png')
        frames[0].save(folder / 'reconstructed_current.png')
        for index, frame in enumerate(frames[1:steps+1], 1):
            frame.save(folder / f'step_{index:02d}_pred_rgb_full.png')
            frame.crop((0, 0, frame.width // 2, frame.height)).save(folder / f'step_{index:02d}_pred_rgb.png')
        (folder / 'metadata.json').write_text(json.dumps(dict(
            model_family=self.metadata['model_family'], session=session,
            request_id=obs['request_id'], prompt=obs['prompt'], saved_steps=steps,
            alignment='action_index_not_wall_clock', camera_keys=list(CAMERAS),
            action_dt=0.1, num_inference_steps=self.steps, seed=self.seed), indent=2))
        return str(folder.relative_to(self.save_rgb_dir))
