"""Opt-in Wan2.1 MoT baseline with the Goal Raymap experiment's pose targets.

Original FastWAM loss, attention, and action-only inference are inherited.
Nothing in the legacy factory or Raymap model is changed.
"""
import json

import torch
from omegaconf import DictConfig, OmegaConf

from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.helpers.loader import load_wan_video_components
from fastwam.representations.libero_rothko import LiberoRothkoCodec, LiberoRothkoCodecConfig
from fastwam.representations.libero_latent_repeat import LiberoLatentRepeat
from fastwam.representations.rothko import quaternion_wxyz_to_matrix, matrix_to_quaternion_wxyz
from fastwam.datasets.lerobot.libero_latent_repeat import LiberoLatentRepeatDataset


class MatchedPoseCodec(LiberoLatentRepeat):
    """Reuse numerical normalization only; no latent repetition or action VAE."""
    identity = 'libero_absolute_state_relative_pose8_mot_wan21_v1'

    def state_vector(self, pose, grip):
        p = pose[:, None].expand(-1, 17, -1)
        g = grip.reshape(-1, 1, 1).expand(-1, 17, -1)
        return self.vectors(p, g)[:, 0]

    def recover(self, action, current_pose):
        if action.ndim != 3 or action.shape[1:] != (16, 8):
            raise ValueError('Expected numerical actions[B,16,8]')
        if not torch.isfinite(action).all():
            raise ValueError('Nonfinite numerical action')
        a = action.float(); base = current_pose.to(a)
        q, bad = self.canonical(a[..., 3:7])
        r = quaternion_wxyz_to_matrix(base[..., 3:])[:, None]
        xyz = (r @ (a[..., :3].clamp(-1, 1)*self.bounds.to(a)).unsqueeze(-1)).squeeze(-1)
        pose = torch.cat((xyz+base[:, None, :3], matrix_to_quaternion_wxyz(r @ quaternion_wxyz_to_matrix(q))), -1)
        return pose, ((a[..., 7:8]+1)/2).clamp(0, 1), bad


class MatchedGoalDataset(LiberoLatentRepeatDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numeric_codec = MatchedPoseCodec(self.raymap_codec)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        v = self.numeric_codec.vectors(sample.pop('repeat_pose')[None], sample.pop('repeat_gripper')[None])[0]
        sample['action'] = v[1:].contiguous()
        sample['proprio'] = v[:1].expand(16, -1).clone()
        sample.pop('raymap', None)
        sample.pop('raymap_is_pad', None)
        return sample


class MatchedFastWAM(FastWAM):
    action_horizon = 16

    def validate_latent_cache_metadata(self, metadata):
        if (metadata.get('vae_identity', {}).get('sha256') != self.original_vae_sha256
                or tuple(metadata.get('latent_shape', [])) != (16,5,28,56)
                or metadata.get('encoding_torch_dtype') != 'torch.bfloat16'
                or metadata.get('model_variant') != 'wan2.1-t2v-1.3b'):
            raise ValueError('RGB cache VAE identity/shape/precision mismatch')

    def identity(self):
        value = dict(kind=MatchedPoseCodec.identity, horizon=16, rgb_frames=17,
                     codec=self.raymap_codec.metadata(),
                     original_vae_sha256=self.original_vae_sha256,
                     stats_sha256=self.raymap_codec.norm_stats.fingerprint(),
                     video=self.video_config, action=self.action_config,
                     losses=[self.loss_lambda_video, self.loss_lambda_action])
        return json.loads(json.dumps(value))

    def build_inputs(self, sample, tiled=False):
        if 'rgb_latents' not in sample:
            return super().build_inputs(sample, tiled=tiled)
        z = sample['rgb_latents'].to(self.device, self.torch_dtype)
        if tuple(z.shape[1:]) != (16, 5, 28, 56):
            raise ValueError('Matched Goal RGB cache must be [B,16,5,28,56]')
        action = sample['action'].to(self.device, self.torch_dtype)
        if tuple(action.shape[1:]) != (16, 8):
            raise ValueError('Matched action must be [B,16,8]')
        context = sample['context'].to(self.device, self.torch_dtype)
        mask = sample['context_mask'].to(self.device, torch.bool)
        context, mask = self._append_proprio_to_context(context, mask, sample['proprio'][:, 0].to(self.device, self.torch_dtype))
        return dict(context=context, context_mask=mask, input_latents=z,
                    first_frame_latents=z[:, :, :1], fuse_vae_embedding_in_latents=True,
                    action=action, action_is_pad=sample['action_is_pad'].to(self.device, torch.bool),
                    image_is_pad=sample['image_is_pad'].to(self.device, torch.bool))

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = dict(mot=self.mot.state_dict(), proprio_encoder=self.proprio_encoder.state_dict(),
                       matched_identity=self.identity(), step=step)
        if optimizer is not None: payload['optimizer'] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('matched_identity') != self.identity():
            raise ValueError('Matched FastWAM checkpoint identity/config/stats mismatch')
        self.mot.load_state_dict(payload['mot'], strict=True)
        self.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
        if optimizer is not None and 'optimizer' in payload: optimizer.load_state_dict(payload['optimizer'])
        return payload


def plain(x):
    return OmegaConf.to_container(x, resolve=True) if isinstance(x, DictConfig) else dict(x)


def create_model(rothko_config, rothko_norm_stats, video_dit_config, action_dit_config,
                 video_scheduler, action_scheduler, loss, action_dit_pretrained_path,
                 model_dtype=torch.bfloat16, device='cuda', load_text_encoder=False,
                 skip_dit_load_from_pretrain=False, **kwargs):
    if kwargs:
        raise ValueError(f'Unexpected matched-baseline options: {sorted(kwargs)}')
    vc, ac = plain(video_dit_config), plain(action_dit_config)
    if (vc['in_dim'], vc['hidden_dim'], vc['num_heads'], vc['num_layers'], ac['action_dim']) != (16,1536,12,30,8):
        raise ValueError('Expected Wan2.1 1.3B and 8D matched pose targets')
    if any(vc[k] != ac[k] for k in ('num_heads', 'attn_head_dim', 'num_layers')):
        raise ValueError('MoT expert heads/depth must match')
    if not skip_dit_load_from_pretrain:
        payload = torch.load(action_dit_pretrained_path, map_location='cpu', weights_only=False)
        if payload.get('source_model') != 'Wan-AI/Wan2.1-T2V-1.3B':
            raise ValueError('Generate a Wan2.1 ActionDiT backbone; Wan2.2 initialization is forbidden')
        del payload
    components = load_wan_video_components(device=device, torch_dtype=model_dtype,
        model_id='Wan-AI/Wan2.1-T2V-1.3B', model_variant='wan2.1-t2v-1.3b',
        tokenizer_model_id='Wan-AI/Wan2.1-T2V-1.3B', tokenizer_max_len=128,
        load_text_encoder=load_text_encoder, dit_config=vc,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain)
    action = ActionDiT.from_pretrained(ac, action_dit_pretrained_path,
        skip_dit_load_from_pretrain=skip_dit_load_from_pretrain, device=device, torch_dtype=model_dtype)
    mot = MoT({'video':components.dit, 'action':action}, mot_checkpoint_mixed_attn=False)
    sched = {}
    for prefix, value in [('video',video_scheduler),('action',action_scheduler)]:
        for key, v in plain(value).items(): sched[prefix+'_'+key] = v
    losses = plain(loss)
    model = MatchedFastWAM(components.dit, action, mot, components.vae,
        text_encoder=components.text_encoder, tokenizer=components.tokenizer,
        text_dim=4096, proprio_dim=8, device=device, torch_dtype=model_dtype,
        loss_lambda_video=losses['lambda_video'], loss_lambda_action=losses['lambda_action'], **sched)
    model.raymap_codec = LiberoRothkoCodec(LiberoRothkoCodecConfig(**plain(rothko_config)), rothko_norm_stats)
    model.numeric_codec = MatchedPoseCodec(model.raymap_codec)
    model.video_config, model.action_config = vc, ac
    from fastwam.datasets.latent_cache import sha256_file
    model.original_vae_sha256 = sha256_file(components.vae_path)
    return model
