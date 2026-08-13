from collections import defaultdict
import numpy as np


class ActionEnsembler:
    def __init__(self):
        self.action_cache = defaultdict(list)

    def reset(self):
        self.action_cache.clear()

    def add_actions(self, action_chunk: np.ndarray, start_timestamp: int):
        if action_chunk.ndim == 3:
            action_chunk = action_chunk.squeeze(0)
        horizon, action_dim = action_chunk.shape

        for i in range(horizon):
            target_ts = start_timestamp + i
            self.action_cache[target_ts].append(action_chunk[i, :])

    def get_action(self, timestamp: int) -> np.ndarray:
        if timestamp not in self.action_cache:
            raise ValueError(f"No actions cached for timestamp {timestamp}")
        preds = self.action_cache[timestamp]
        stacked_preds = np.stack(preds, axis=0)
        averaged_action = np.mean(stacked_preds, axis=0)
        return averaged_action

    def _cleanup(self, current_timestamp: int):
        keys_to_delete = [ts for ts in self.action_cache.keys() if ts < current_timestamp]
        for ts in keys_to_delete:
            del self.action_cache[ts]


class AbsolutePoseActionEnsembler:
    """Temporally ensemble overlapping absolute EE-pose action chunks.

    Predictions are aligned by their target environment timestamp. Newer
    replans receive exponentially larger weights than older replans. Position
    and gripper values are averaged directly; quaternion signs are aligned to
    the newest prediction before weighted averaging and normalization.
    """

    def __init__(self, decay: float = 0.01):
        self.decay = float(decay)
        if self.decay < 0:
            raise ValueError(f"decay must be non-negative, got {self.decay}.")
        self.action_cache = defaultdict(list)

    def reset(self):
        self.action_cache.clear()

    def add_actions(
        self,
        target_pose_chunk: np.ndarray,
        gripper_open_chunk: np.ndarray,
        start_timestamp: int,
    ):
        target_pose_chunk = np.asarray(target_pose_chunk, dtype=np.float32)
        gripper_open_chunk = np.asarray(gripper_open_chunk, dtype=np.float32)
        if target_pose_chunk.ndim == 3:
            target_pose_chunk = target_pose_chunk.squeeze(0)
        if gripper_open_chunk.ndim == 3:
            gripper_open_chunk = gripper_open_chunk.squeeze(0)
        if target_pose_chunk.ndim != 2 or target_pose_chunk.shape[1] != 7:
            raise ValueError(
                "Expected target pose chunk [T,7] in xyz+wxyz format, got "
                f"{target_pose_chunk.shape}."
            )
        gripper_open_chunk = gripper_open_chunk.reshape(
            target_pose_chunk.shape[0], -1
        )

        for index in range(target_pose_chunk.shape[0]):
            target_timestamp = int(start_timestamp) + index
            self.action_cache[target_timestamp].append(
                (
                    int(start_timestamp),
                    target_pose_chunk[index].copy(),
                    gripper_open_chunk[index].copy(),
                )
            )

    def get_action(self, timestamp: int) -> tuple[np.ndarray, np.ndarray]:
        if timestamp not in self.action_cache:
            raise ValueError(f"No actions cached for timestamp {timestamp}")

        predictions = self.action_cache[timestamp]
        source_timestamps = np.asarray(
            [prediction[0] for prediction in predictions], dtype=np.float64
        )
        latest_source_timestamp = float(source_timestamps.max())
        ages = latest_source_timestamp - source_timestamps
        weights = np.exp(-self.decay * ages)
        weights /= weights.sum()

        poses = np.stack([prediction[1] for prediction in predictions], axis=0).astype(
            np.float64
        )
        positions = np.sum(poses[:, :3] * weights[:, None], axis=0)

        quaternions = poses[:, 3:7]
        quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
        reference_index = int(np.argmax(source_timestamps))
        reference_quaternion = quaternions[reference_index]
        signs = np.where(
            np.sum(quaternions * reference_quaternion[None], axis=1) < 0.0,
            -1.0,
            1.0,
        )
        quaternions *= signs[:, None]
        quaternion = np.sum(quaternions * weights[:, None], axis=0)
        quaternion /= np.linalg.norm(quaternion)

        grippers = np.stack(
            [prediction[2] for prediction in predictions], axis=0
        ).astype(np.float64)
        gripper = np.sum(grippers * weights[:, None], axis=0)

        pose = np.concatenate((positions, quaternion), axis=0).astype(np.float32)
        return pose, gripper.astype(np.float32)

    def cleanup(self, current_timestamp: int):
        expired_timestamps = [
            timestamp
            for timestamp in self.action_cache
            if timestamp < current_timestamp
        ]
        for timestamp in expired_timestamps:
            del self.action_cache[timestamp]
