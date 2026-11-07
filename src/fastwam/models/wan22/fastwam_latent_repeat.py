"""Isolated Wan adaptation of numerical Latent-Repeat. Baseline files unchanged."""
import torch
from fastwam.representations.libero_latent_repeat import LiberoLatentRepeat
from .fastwam_visual_action import FastWAMVideoOnlyRaymap


class FastWAMLatentRepeat(FastWAMVideoOnlyRaymap):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.raymap_representation != 'libero_rothko' or self.action_horizon != 16 or self.vae_latent_channels != 16:
            raise ValueError('This ablation supports only LIBERO Wan2.1 h16')
        if self.latent_layout != 'rgb_then_raymap' or self.future_rgb_mode != 'joint' or self.proprio_encoder is not None:
            raise ValueError('Latent-Repeat requires baseline joint temporal layout without numeric proprio token')
        self.repeat_codec = LiberoLatentRepeat(self.raymap_codec)
        # Explicit checkpoint identity prevents loading either representation as the other.
        self.raymap_representation = 'libero_latent_repeat'

    def repeat_targets(self, sample):
        if 'repeat_pose' in sample and 'repeat_gripper' in sample:
            return sample['repeat_pose'], sample['repeat_gripper']
        # The unchanged trainer batches a fixed whitelist of validation fields.
        # Rebuild from its original numeric targets, never from decoded pixels.
        pose = torch.cat((sample['current_endpose'][:, None], sample['future_endpose']), 1)
        state_image = sample['raymap'][:, :, :1].permute(0, 2, 1, 3, 4)
        current_gripper = self.raymap_codec.read_gripper(state_image.float())
        grip = torch.cat((current_gripper, sample['future_gripper']), 1)
        return pose, grip

    def build_inputs(self, sample, tiled=False):
        pose, grip = self.repeat_targets(sample)
        numerical = self.repeat_codec.encode(pose, grip).to(self.device, self.torch_dtype)
        rgb = sample.get('rgb_latents')
        if rgb is None:
            rgb = self._encode_video_latents(sample['video'].to(self.device,self.torch_dtype), tiled=tiled)
        proxy = dict(sample, rgb_latents=rgb, raymap_latents=numerical)
        proxy.pop('video', None); proxy.pop('raymap', None)
        return super().build_inputs(proxy, tiled=tiled)

    def _visual_action_checkpoint_config(self):
        return {**super()._visual_action_checkpoint_config(), 'action_latent_encoding':self.repeat_codec.identity}

    def _validate_visual_action_checkpoint_config(self, visual_config, *, checkpoint_path):
        if visual_config.get('action_latent_encoding') != self.repeat_codec.identity:
            raise ValueError('Latent-Repeat checkpoint layout/normalization identity mismatch')
        return super()._validate_visual_action_checkpoint_config(visual_config,checkpoint_path=checkpoint_path)

    @torch.no_grad()
    def infer(self, prompt, input_image, input_raymap, current_endpose=None,
              context=None, context_mask=None, num_frames=None, num_inference_steps=20,
              sigma_shift=None, seed=None, rand_device='cpu', tiled=False,
              decode_future_rgb=True, **kwargs):
        self.eval()
        if current_endpose is None or num_frames not in (None,17):
            raise ValueError('Observed current pose and h16 are required')
        if input_image.ndim == 3: input_image = input_image[None]
        if input_raymap.ndim == 3: input_raymap = input_raymap[None]
        if tuple(input_image.shape) != (1,3,224,448) or input_image.shape != input_raymap.shape:
            raise ValueError('Expected single 224x448 RGB/state condition')
        base = current_endpose.reshape(1,7).to(self.device,torch.float32)
        # Compatibility with baseline eval's current-state image; no action VAE.
        grip = self.raymap_codec.read_gripper(input_raymap[:, None].float()).reshape(1,1,1).to(base)
        numerical = self.repeat_codec.encode(base[:,None].expand(-1,17,-1),grip.expand(-1,17,-1))
        state_condition = numerical[:,:,:1].to(self.device,self.torch_dtype)
        rgb_condition = self._encode_video_latents(input_image.to(self.device,self.torch_dtype)[:,:,None],tiled=tiled)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents = torch.randn((1,16,10,28,56),generator=generator,device=rand_device,dtype=torch.float32).to(self.device,self.torch_dtype)
        self._set_clean_conditions(latents,rgb_condition,state_condition)
        if context is None or context_mask is None:
            if prompt is None: raise ValueError('Prompt or cached text required')
            context,context_mask = self.encode_prompt(prompt)
        else:
            if context.ndim == 2:context=context[None]
            if context_mask.ndim == 1:context_mask=context_mask[None]
            context=context.to(self.device,self.torch_dtype);context_mask=context_mask.to(self.device,torch.bool)
        times,deltas=self.infer_scheduler.build_inference_schedule(num_inference_steps=num_inference_steps,device=self.device,dtype=latents.dtype,shift_override=sigma_shift)
        for t,delta in zip(times,deltas):
            pred=self._model_fn(latents,t.reshape(1).to(self.device,latents.dtype),context,context_mask,
                                condition_latent_indices=self.condition_latent_indices)
            latents=self.infer_scheduler.step(pred,delta,latents)
            self._set_clean_conditions(latents,rgb_condition,state_condition)
        rgb,actions=self._split_rgb_raymap_latents(latents)
        pose,gripper,bad=self.repeat_codec.decode(actions,base)
        # Render only for existing visualization interfaces; action output is numerical.
        raymap=self.raymap_codec.encode(pose.float(),gripper.float())
        result={'pose':pose.cpu(),'gripper':gripper.cpu(),'raymap':raymap.cpu(),
                'action_latents':actions.cpu(),'degenerate_quaternions':int(bad.sum())}
        if decode_future_rgb:
            video=self._decode_video_tensor(rgb,tiled=tiled)
            result.update(video=self._video_tensor_to_pil(video[0]),video_tensor=video.cpu())
        return result
