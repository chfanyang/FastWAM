"""Numerical repeat baseline: absolute state, chunk-relative future targets.

Independent of visual VAE encoding and its latent mean/std. Only Wan2.1 h16
224x448 is supported by this deliberately narrow ablation.
"""
import torch
from .rothko import quaternion_wxyz_to_matrix, matrix_to_quaternion_wxyz


class LiberoLatentRepeat:
    identity = 'absolute_ray0_relative_targets_repeat_chw_v1'
    shape = (16, 5, 28, 56)
    frame_size = 16 * 28 * 56

    def __init__(self, raymap_codec):
        c = raymap_codec.config
        if c.frame0_pose_mode != 'absolute' or (c.image_height, c.image_width) != (224, 448):
            raise ValueError('Latent-Repeat requires absolute RAY0 and 224x448')
        self.low = torch.tensor(c.absolute_position_min, dtype=torch.float32)
        self.high = torch.tensor(c.absolute_position_max, dtype=torch.float32)
        self.bounds = torch.tensor(raymap_codec.norm_stats.metadata['translation_abs_bounds_xyz_m'])

    @staticmethod
    def canonical(q):
        norm = q.norm(dim=-1, keepdim=True)
        bad = norm < 1e-8
        q = q / norm.clamp_min(1e-8)
        identity = torch.zeros_like(q); identity[..., 0] = 1
        q = torch.where(bad, identity, q)
        first = (q != 0).long().argmax(-1, keepdim=True)
        q = q * torch.where(q.gather(-1, first) < 0, -1., 1.)
        return q, bad.squeeze(-1)

    def vectors(self, pose, gripper):
        if pose.ndim != 3 or pose.shape[1:] != (17, 7) or gripper.shape != (*pose.shape[:2], 1):
            raise ValueError('Expected pose[B,17,7], gripper[B,17,1]')
        if not torch.isfinite(pose).all() or not torch.isfinite(gripper).all():
            raise ValueError('Nonfinite target')
        pose = pose.float(); gripper = gripper.float()
        rotation = quaternion_wxyz_to_matrix(pose[..., 3:])
        base = rotation[:, :1].transpose(-1, -2)
        xyz = (base @ (pose[..., :3] - pose[:, :1, :3]).unsqueeze(-1)).squeeze(-1)
        xyz = xyz / self.bounds.to(pose)
        xyz[:, 0] = 2 * (pose[:, 0, :3] - self.low.to(pose)) / (self.high-self.low).to(pose) - 1
        q = matrix_to_quaternion_wxyz(base @ rotation)
        q[:, 0] = pose[:, 0, 3:]
        q, _ = self.canonical(q)
        return torch.cat((xyz.clamp(-1, 1), q, 2*gripper.clamp(0, 1)-1), -1)

    def pack(self, vectors):
        if vectors.ndim != 3 or vectors.shape[1:] != (17, 8):
            raise ValueError('Expected vectors[B,17,8]')
        b = vectors.shape[0]
        first = vectors[:, :1].repeat(1, 1, self.frame_size//8)
        future = vectors[:, 1:].reshape(b, 4, 32).repeat(1, 1, self.frame_size//32)
        return torch.cat((first, future), 1).reshape(b, 5, 16, 28, 56).permute(0, 2, 1, 3, 4).contiguous()

    def unpack(self, latents):
        if latents.ndim != 5 or tuple(latents.shape[1:]) != self.shape:
            raise ValueError('Expected action latents[B,16,5,28,56]')
        b = latents.shape[0]
        flat = latents.float().permute(0, 2, 1, 3, 4).reshape(b, 5, self.frame_size)
        first = flat[:, :1].reshape(b, 1, -1, 8).mean(2)
        future = flat[:, 1:].reshape(b, 4, -1, 32).mean(2).reshape(b, 16, 8)
        return torch.cat((first, future), 1)

    def encode(self, pose, gripper):
        return self.pack(self.vectors(pose, gripper))

    def decode(self, latents, current_pose):
        v = self.unpack(latents)
        q, bad = self.canonical(v[..., 3:7])
        xyz = v[..., :3].clamp(-1, 1) * self.bounds.to(v)
        base = current_pose.to(v)
        rot = quaternion_wxyz_to_matrix(base[..., 3:])[:, None]
        xyz = (rot @ xyz.unsqueeze(-1)).squeeze(-1) + base[:, None, :3]
        q = matrix_to_quaternion_wxyz(rot @ quaternion_wxyz_to_matrix(q))
        # State is observed, not predicted; never use noisy RAY0 as a reference.
        xyz[:, 0] = base[:, :3]; q[:, 0] = base[:, 3:]
        return torch.cat((xyz, q), -1), ((v[..., 7:8]+1)/2).clamp(0, 1), bad
