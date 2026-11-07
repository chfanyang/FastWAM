"""Isolated original-FastWAM trainer: shared optimizer loop, native robot eval."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.runtime import build_datasets, _mixed_precision_to_model_dtype, _resolve_train_device
from fastwam.trainer import Wan22Trainer, _decoded_pose_metrics
from fastwam.utils import misc
from fastwam.utils.logging_config import setup_logging
from .common import write_json_checked


class OriginalRealWorldTrainer(Wan22Trainer):
    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None
        if self.fixed_validation_manifest is None:
            raise ValueError("Original realworld baseline requires fixed per-task validation windows")
        model = self.accelerator.unwrap_model(self.model)
        was_training = model.dit.training
        model.eval()
        keys = ("val_loss", "action_l1", "action_l2", "decoded_position_mae_m",
                "decoded_position_rmse_m", "decoded_rotation_geodesic_deg",
                "decoded_gripper_mae", "decoded_gripper_accuracy")
        totals = torch.zeros(len(keys) + 1, device=self.accelerator.device, dtype=torch.float64)
        records = self.fixed_validation_manifest["samples"]
        device = torch.device(model.device)
        rng_devices = [device] if device.type == "cuda" else []
        try:
            for offset in range(self.accelerator.process_index, len(records), self.accelerator.num_processes):
                record = records[offset]
                raw = self.val_dataset[record["val_dataset_index"]]
                sample = self._to_batched_eval_sample(raw)
                with torch.random.fork_rng(devices=rng_devices):
                    torch.manual_seed(record["diffusion_seed"])
                    with self.accelerator.autocast():
                        loss, _ = model.training_loss(sample)
                        prediction = model.infer_action(prompt=None, input_image=sample["video"][:, :, 0],
                            action_horizon=16, proprio=sample["proprio"][0, 0],
                            context=sample["context"][0], context_mask=sample["context_mask"][0],
                            num_inference_steps=self.eval_num_inference_steps,
                            seed=record["diffusion_seed"], text_cfg_scale=1., tiled=False)
                actions = prediction["action"].detach().float().cpu()
                if actions.shape != (16, 7) or not torch.isfinite(actions).all():
                    raise ValueError("Invalid baseline action prediction")
                poses, grip = self.val_dataset.codec.decode(raw["current_endpose"].numpy(), actions.numpy())
                metrics = _decoded_pose_metrics(torch.from_numpy(poses), raw["future_endpose"],
                    torch.from_numpy(grip[:, None]), raw["future_gripper"])
                error = actions - raw["action"]
                metrics.update(val_loss=float(loss), action_l1=float(error.abs().mean()),
                               action_l2=float(error.square().mean()))
                totals += totals.new_tensor([metrics[k] for k in keys] + [1.])
                output = Path(self.eval_dir) / f"step_{self.global_step:06d}_sample_{record['val_dataset_index']:06d}.npz"
                np.savez_compressed(output, normalized_action=actions.numpy(), target_pose_wxyz=poses,
                    gripper_open=grip, gt_pose_wxyz=raw["future_endpose"].numpy(),
                    gt_gripper_open=raw["future_gripper"].numpy())
            summed = self.accelerator.gather(totals[None]).sum(dim=0)
            if int(summed[-1]) != len(records):
                raise ValueError("Incomplete baseline validation group")
            result = {key: float(summed[i] / summed[-1]) for i, key in enumerate(keys)}
            if self.accelerator.is_main_process:
                path = Path(self.eval_dir) / f"step_{self.global_step:06d}_metrics.json"
                path.write_text(json.dumps(result, indent=2) + "\n")
            return result
        finally:
            if was_training:
                self._set_dit_only_train_mode()


def run_training(cfg):
    if cfg.model._target_ != "fastwam.realworld.baseline_model.create_model":
        raise ValueError("Use the isolated original-FastWAM realworld task")
    setup_logging(log_level="INFO", is_main_process=int(os.environ.get("RANK", "0")) == 0)
    misc.register_work_dir(cfg.output_dir)
    train, val = build_datasets(cfg.data)
    if train.codec.fingerprint != val.codec.fingerprint:
        raise ValueError("Training / validation action statistics differ")
    if int(os.environ.get("RANK", "0")) == 0:
        output = Path(cfg.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        destination = output / "config.yaml"
        if destination.exists():
            if not cfg.get("resume"):
                raise FileExistsError(f"Run already exists: {output}")
            destination = output / "resume_configs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".yaml")
            destination.parent.mkdir(exist_ok=True)
        OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), destination)
        write_json_checked(output / "dataset_stats.json", train.codec.stats)
    model = instantiate(cfg.model, device=_resolve_train_device(),
        model_dtype=_mixed_precision_to_model_dtype(cfg.mixed_precision))
    if model.action_codec.fingerprint != train.codec.fingerprint:
        raise ValueError("Model / dataset action statistics differ")
    if cfg.get("init_weights") and not cfg.get("resume"):
        model.load_checkpoint(cfg.init_weights)
    OriginalRealWorldTrainer(cfg=cfg, model=model, train_dataset=train, val_dataset=val).train()
