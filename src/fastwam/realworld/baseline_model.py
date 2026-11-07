"""Original FastWAM math, with realworld checkpoint identity validation only."""
import torch
from omegaconf import DictConfig, OmegaConf

from fastwam.models.wan22.fastwam import FastWAM
from .baseline_data import ActionCodec


class OriginalRealWorldFastWAM(FastWAM):
    # training_loss, MoT attention, video/action schedulers and inference are
    # inherited unchanged. This subclass only binds robot data to checkpoints.
    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = dict(mot=self.mot.state_dict(), step=step, torch_dtype=str(self.torch_dtype),
            proprio_encoder=self.proprio_encoder.state_dict(),
            realworld_action_stats=self.action_codec.stats,
            realworld_action_stats_sha256=self.action_codec.fingerprint)
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload.get("realworld_action_stats") != self.action_codec.stats
                or payload.get("realworld_action_stats_sha256") != self.action_codec.fingerprint):
            raise ValueError("Original realworld checkpoint action statistics / split mismatch")
        self.mot.load_state_dict(payload["mot"], strict=True)
        self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload


def create_model(action_stats, split_manifest, video_scheduler, action_scheduler,
                 loss, model_dtype=torch.bfloat16, **kwargs):
    codec = ActionCodec(action_stats, split_manifest)
    def plain(value):
        return OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else dict(value)
    kwargs["video_dit_config"] = plain(kwargs["video_dit_config"])
    kwargs["action_dit_config"] = plain(kwargs["action_dit_config"])
    if (kwargs["model_id"] != "Wan-AI/Wan2.2-TI2V-5B" or kwargs["proprio_dim"] != 8
            or kwargs["action_dit_config"]["action_dim"] != 7
            or kwargs["video_dit_config"]["video_attention_mask_mode"] != "first_frame_causal"):
        raise ValueError("Expected original 5B FastWAM with 7D actions and 8D current state")
    extra = {}
    for prefix, scheduler in (("video", video_scheduler), ("action", action_scheduler)):
        scheduler = plain(scheduler)
        for key in ("train_shift", "infer_shift", "num_train_timesteps"):
            extra[f"{prefix}_{key}"] = scheduler[key]
    loss = plain(loss)
    model = OriginalRealWorldFastWAM.from_wan22_pretrained(torch_dtype=model_dtype,
        loss_lambda_video=float(loss["lambda_video"]), loss_lambda_action=float(loss["lambda_action"]),
        **extra, **kwargs)
    model.action_codec = codec
    return model
