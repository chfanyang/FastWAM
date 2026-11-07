"""Realworld inference without importing any LIBERO simulator or robot driver."""
import json
from pathlib import Path
import threading
import time
import tempfile
import re

import numpy as np
import torch

from .common import PREPROCESSING, validate_pose, rgb_canvas, sha256_file
from .dataset import load_context, make_codec
from .protocol import VERSION, ACTION_SPACE


class RealWorldPolicy:
    def __init__(self, model, manifest_path, stats_path, text_cache_dir,
                 num_inference_steps=20, seed=42, save_rgb_dir=None):
        self.model = model
        self.manifest = json.loads(Path(manifest_path).read_text())
        codec = make_codec(stats_path)
        if codec.norm_stats.metadata["split_manifest_sha256"] != sha256_file(manifest_path):
            raise ValueError("Deployment manifest differs from normalization split")
        if (model.raymap_codec.metadata() != codec.metadata() or
            model.raymap_codec.norm_stats.fingerprint() != codec.norm_stats.fingerprint()):
            raise ValueError("Deployment model/codec mismatch")
        self.codec = codec
        self.contexts = {s: load_context(text_cache_dir, s) for s in
                         sorted({r["instruction"] for r in self.manifest["episodes"]})}
        self.steps, self.seed = int(num_inference_steps), seed
        self.lock = threading.Lock()
        self.save_rgb_dir = Path(save_rgb_dir) if save_rgb_dir else None
        if self.save_rgb_dir is not None:
            self.save_rgb_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(protocol_version=VERSION, action_space=ACTION_SPACE,
            horizon=16, action_dt=0.1, pose_quaternion_order="wxyz", ee_link="fr3_link8",
            camera_keys=self.manifest["camera_keys"], color_order="RGB",
            preprocessing=PREPROCESSING, gripper_open_threshold_m=self.manifest["gripper_open_threshold_m"],
            instructions=list(self.contexts), codec=codec.metadata(),
            stats_fingerprint=codec.norm_stats.fingerprint())

    def infer(self, obs):
        start = time.monotonic()
        if obs.get("protocol_version") != VERSION:
            raise ValueError("Protocol version mismatch")
        if obs.get("ee_link") != "fr3_link8" or obs.get("color_order") != "RGB":
            raise ValueError("Expected fr3_link8 pose and RGB images")
        if not isinstance(obs.get("request_id"), int):
            raise ValueError("request_id must be an integer")
        instruction = obs["prompt"]
        if instruction not in self.contexts:
            raise ValueError("Unknown prompt; prepare its text context before deployment")
        pose = validate_pose(obs["ee_pose_wxyz"])
        if pose.shape != (7,):
            raise ValueError("Expected one current pose")
        lo, hi = self.codec.config.absolute_position_min, self.codec.config.absolute_position_max
        if np.any((pose[:3] < lo) | (pose[:3] > hi)):
            raise ValueError("Current position is outside absolute RAY0 training bounds")
        g = float(obs["gripper_open"])
        if not np.isfinite(g) or not 0 <= g <= 1:
            raise ValueError("gripper_open must be in [0,1]")
        canvas = rgb_canvas(obs["images"], self.metadata["camera_keys"])
        rgb = torch.from_numpy(canvas).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0)
        pose_tensor = torch.from_numpy(pose)
        ray0 = self.codec.encode(pose_tensor[None], torch.tensor([[g]]))[:, 0].unsqueeze(0)
        context, mask = self.contexts[instruction]
        with self.lock, torch.inference_mode():
            pred = self.model.infer(prompt=None, input_image=rgb, input_raymap=ray0,
                current_endpose=pose_tensor, context=context, context_mask=mask,
                num_frames=17, num_inference_steps=self.steps, seed=self.seed,
                decode_future_rgb=self.save_rgb_dir is not None)
        targets = pred["pose"][0, 1:].float().cpu().numpy()
        gripper = pred["gripper"][0, 1:].float().cpu().numpy().reshape(-1)
        if targets.shape != (16, 7) or gripper.shape != (16,):
            raise ValueError("Invalid model action horizon")
        targets = validate_pose(targets)
        if not np.isfinite(gripper).all() or np.any((gripper < 0) | (gripper > 1)):
            raise ValueError("Invalid model gripper output")
        artifact_key = None
        if self.save_rgb_dir is not None:
            try:
                artifact_key = self.save_prediction_rgb(pred, obs, canvas)
            except OSError as exc:
                # Optional image recording must not abort a valid action response.
                print(f"Warning: predicted RGB was not saved: {exc}", flush=True)
        return dict(protocol_version=VERSION, request_id=obs["request_id"],
            action_space=ACTION_SPACE, action_dt=0.1, target_pose_wxyz=targets,
            gripper_open=gripper, server_latency_s=time.monotonic() - start,
            prediction_artifact_key=artifact_key)

    def save_prediction_rgb(self, pred, obs, canvas):
        """Save RGB PNGs in training camera order; frame zero is not a future frame."""
        from PIL import Image
        frames = pred["video"]
        if len(frames) != 17:
            raise ValueError(f"Expected 17 decoded RGB frames, got {len(frames)}")
        steps = int(obs.get("record_steps", 8))
        session = obs.get("record_session")
        if not 1 <= steps <= 16:
            raise ValueError("Invalid recording horizon")
        if session is not None and (not isinstance(session, str) or
                                    re.fullmatch(r"[0-9a-f]{32}", session) is None):
            raise ValueError("Invalid recording session")
        self.save_rgb_dir.mkdir(parents=True, exist_ok=True)
        if session is None:
            folder = Path(tempfile.mkdtemp(
                prefix=f"request_{obs['request_id']:06d}_", dir=self.save_rgb_dir))
        else:
            folder = self.save_rgb_dir / session / f"request_{obs['request_id']:06d}"
            folder.mkdir(parents=True, exist_ok=False)
        Image.fromarray(canvas).save(folder / "observation.png")
        frames[0].save(folder / "reconstructed_current.png")
        ray = pred["raymap"][0].float().cpu()
        if tuple(ray.shape) != (3, 17, canvas.shape[0], canvas.shape[1]):
            raise ValueError(f"Unexpected Rothko shape: {tuple(ray.shape)}")
        ray = ((ray + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
        left_index = list(self.metadata["camera_keys"]).index("left_camera")
        for index, frame in enumerate(frames[1:steps + 1], 1):
            frame.save(folder / f"step_{index:02d}_pred_rgb_full.png")
            width = frame.width // len(self.metadata["camera_keys"])
            frame.crop((left_index * width, 0, (left_index + 1) * width, frame.height)).save(
                folder / f"step_{index:02d}_pred_rgb.png")
            Image.fromarray(ray[:, index].permute(1, 2, 0).numpy()).save(
                folder / f"step_{index:02d}_pred_rothko.png")
        (folder / "metadata.json").write_text(json.dumps(dict(
            session=session, request_id=obs["request_id"], prompt=obs["prompt"],
            saved_steps=steps, alignment="action_index_not_wall_clock",
            camera_keys=list(self.metadata["camera_keys"]), action_dt=0.1,
            num_inference_steps=self.steps, seed=self.seed,
            ee_pose_wxyz=np.asarray(obs["ee_pose_wxyz"]).tolist()), indent=2))
        print(f"Saved predicted RGB: {folder}", flush=True)
        return str(folder.relative_to(self.save_rgb_dir))
