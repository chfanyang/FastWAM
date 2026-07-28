from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LORA_TARGET_MODULES = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",
    "cross_attn.v",
    "cross_attn.o",
)


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "LoRAConfig":
        values = {} if payload is None else dict(payload)
        target_modules = values.get(
            "target_modules", DEFAULT_LORA_TARGET_MODULES
        )
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        config = cls(
            rank=int(values.get("rank", 16)),
            alpha=float(values.get("alpha", 16.0)),
            dropout=float(values.get("dropout", 0.0)),
            target_modules=tuple(str(name) for name in target_modules),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}.")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"LoRA dropout must be in [0,1), got {self.dropout}."
            )
        if not self.target_modules:
            raise ValueError("LoRA target_modules must not be empty.")
        if any(not name.strip() for name in self.target_modules):
            raise ValueError("LoRA target_modules must contain non-empty names.")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError(
                f"LoRA target_modules contains duplicates: {self.target_modules}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
        }


class LoRALinear(nn.Module):
    """Frozen Linear layer plus a trainable low-rank residual."""

    def __init__(self, base_layer: nn.Linear, config: LoRAConfig):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                f"LoRALinear requires nn.Linear, got {type(base_layer)}."
            )
        config.validate()
        self.base_layer = base_layer
        self.rank = int(config.rank)
        self.alpha = float(config.alpha)
        self.scaling = self.alpha / self.rank
        self.dropout_p = float(config.dropout)
        self.dropout = (
            nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )

        weight = base_layer.weight
        init_a = torch.empty(
            (self.rank, base_layer.in_features),
            device=weight.device,
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(init_a, a=math.sqrt(5))
        self.lora_A = nn.Parameter(init_a.to(dtype=weight.dtype))
        self.lora_B = nn.Parameter(
            torch.zeros(
                (base_layer.out_features, self.rank),
                device=weight.device,
                dtype=weight.dtype,
            )
        )
        self.base_layer.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        adapter_inputs = self.dropout(inputs).to(dtype=self.lora_A.dtype)
        adapter = F.linear(
            F.linear(adapter_inputs, self.lora_A),
            self.lora_B,
        )
        return base + adapter.to(dtype=base.dtype) * self.scaling

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, dropout={self.dropout_p}"
        )


def _matches_target(module_name: str, target_name: str) -> bool:
    return module_name == target_name or module_name.endswith(f".{target_name}")


def _set_submodule(root: nn.Module, module_name: str, value: nn.Module) -> None:
    if not module_name:
        raise ValueError("Cannot replace the root module with LoRA.")
    parent_name, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, value)


def inject_lora(
    module: nn.Module,
    config: LoRAConfig | Mapping[str, Any],
) -> list[str]:
    """Replace matching Linear layers and return their fully qualified names."""

    if not isinstance(config, LoRAConfig):
        config = LoRAConfig.from_dict(config)
    existing_config = getattr(module, "_fastwam_lora_config", None)
    if existing_config is not None:
        existing = LoRAConfig.from_dict(existing_config)
        if existing != config:
            raise ValueError(
                "LoRA is already configured with different settings: "
                f"existing={existing.to_dict()}, requested={config.to_dict()}."
            )

    matched_by_target = {target: [] for target in config.target_modules}
    replacements: list[tuple[str, nn.Linear]] = []
    for name, child in list(module.named_modules()):
        matching_targets = [
            target
            for target in config.target_modules
            if _matches_target(name, target)
        ]
        if not matching_targets:
            continue
        if len(matching_targets) > 1:
            raise ValueError(
                f"Module {name!r} matches multiple LoRA targets: {matching_targets}."
            )
        target = matching_targets[0]
        if isinstance(child, LoRALinear):
            if (
                child.rank != config.rank
                or child.alpha != config.alpha
                or child.dropout_p != config.dropout
            ):
                raise ValueError(
                    f"Existing LoRA module {name!r} does not match requested config."
                )
            matched_by_target[target].append(name)
            continue
        if not isinstance(child, nn.Linear):
            raise TypeError(
                f"LoRA target {name!r} must be nn.Linear, got {type(child)}."
            )
        matched_by_target[target].append(name)
        replacements.append((name, child))

    missing_targets = [
        target for target, names in matched_by_target.items() if not names
    ]
    if missing_targets:
        raise ValueError(
            "LoRA target modules were not found in the model: "
            f"{missing_targets}."
        )

    for name, child in replacements:
        _set_submodule(module, name, LoRALinear(child, config))

    module._fastwam_lora_config = config.to_dict()
    module._fastwam_lora_module_names = [
        name
        for target in config.target_modules
        for name in matched_by_target[target]
    ]
    return list(module._fastwam_lora_module_names)


def iter_lora_modules(module: nn.Module):
    for name, child in module.named_modules():
        if isinstance(child, LoRALinear):
            yield name, child


def mark_only_lora_trainable(module: nn.Module) -> None:
    module.requires_grad_(False)
    found = 0
    for _, child in iter_lora_modules(module):
        child.lora_A.requires_grad_(True)
        child.lora_B.requires_grad_(True)
        found += 1
    if found == 0:
        raise ValueError("No LoRA modules are installed.")


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, child in iter_lora_modules(module):
        # DeepSpeed may expose parameters as views into larger flat buffers.
        # Clone after moving to CPU so torch.save serializes only this adapter
        # tensor instead of the complete backing storage.
        state[f"{name}.lora_A"] = child.lora_A.detach().cpu().clone()
        state[f"{name}.lora_B"] = child.lora_B.detach().cpu().clone()
    if not state:
        raise ValueError("Cannot save LoRA state because no adapters are installed.")
    return state


def load_lora_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    expected: dict[str, nn.Parameter] = {}
    for name, child in iter_lora_modules(module):
        expected[f"{name}.lora_A"] = child.lora_A
        expected[f"{name}.lora_B"] = child.lora_B

    provided = set(state_dict)
    missing = sorted(set(expected) - provided)
    unexpected = sorted(provided - set(expected))
    if missing or unexpected:
        raise ValueError(
            "LoRA state key mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}."
        )

    with torch.no_grad():
        for key, parameter in expected.items():
            value = torch.as_tensor(state_dict[key])
            if value.shape != parameter.shape:
                raise ValueError(
                    f"LoRA tensor shape mismatch for {key}: "
                    f"checkpoint={tuple(value.shape)}, model={tuple(parameter.shape)}."
                )
            parameter.copy_(
                value.to(device=parameter.device, dtype=parameter.dtype)
            )


def count_lora_parameters(module: nn.Module) -> tuple[int, int]:
    adapter_parameters = sum(
        child.lora_A.numel() + child.lora_B.numel()
        for _, child in iter_lora_modules(module)
    )
    adapter_modules = sum(1 for _ in iter_lora_modules(module))
    return adapter_parameters, adapter_modules


def lora_module_names(module: nn.Module) -> Sequence[str]:
    return tuple(name for name, _ in iter_lora_modules(module))
