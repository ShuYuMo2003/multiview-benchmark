"""Typed contracts shared by data, models, losses, training, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping

import torch


@dataclass
class PairBatch:
    """One head-to-wrist training batch.

    Images are float tensors in [0, 1]. Side uses 0 for left and 1 for right.
    Rotation residuals are axis-angle/log-SO(3) vectors in radians.
    Every supervision family owns an explicit mask so cross-episode or
    unobservable labels can be disabled independently.
    """

    head_image: torch.Tensor
    wrist_image: torch.Tensor
    side: torch.Tensor
    translation_residual: torch.Tensor
    rotation_residual: torch.Tensor
    joint_residual: torch.Tensor
    gripper_residual: torch.Tensor
    dino_residual: torch.Tensor
    consistency_label: torch.Tensor
    pose_mask: torch.Tensor
    joint_mask: torch.Tensor
    gripper_mask: torch.Tensor
    dino_mask: torch.Tensor
    consistency_mask: torch.Tensor

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PairBatch":
        required = {field.name for field in fields(cls)}
        missing = sorted(required.difference(values))
        if missing:
            raise KeyError(f"PairBatch is missing fields: {missing}")
        return cls(**{name: values[name] for name in required})

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "PairBatch":
        return PairBatch(
            **{
                field.name: getattr(self, field.name).to(device, non_blocking=non_blocking)
                for field in fields(self)
            }
        )

    def validate(self) -> None:
        batch_size = self.head_image.shape[0]
        if self.head_image.ndim != 4 or self.wrist_image.ndim != 4:
            raise ValueError("Images must have shape [B, C, H, W]")
        if self.head_image.shape[1] != 3 or self.wrist_image.shape[1] != 3:
            raise ValueError("Images must have three RGB channels")
        if self.wrist_image.shape[0] != batch_size:
            raise ValueError("Head and wrist batch sizes differ")
        for name in (
            "side", "translation_residual", "rotation_residual", "joint_residual", "gripper_residual",
            "dino_residual", "consistency_label", "pose_mask", "joint_mask", "gripper_mask",
            "dino_mask", "consistency_mask",
        ):
            if getattr(self, name).shape[0] != batch_size:
                raise ValueError(f"{name} has the wrong batch dimension")
        if self.translation_residual.shape[-1] != 3 or self.rotation_residual.shape[-1] != 3:
            raise ValueError("Translation and rotation residuals must have three components")
        if torch.any((self.side != 0) & (self.side != 1)):
            raise ValueError("side must contain only 0 (left) or 1 (right)")


@dataclass
class PairMetricOutput:
    """Predictions for one head-to-wrist pair."""

    translation_residual: torch.Tensor
    rotation_residual: torch.Tensor
    joint_residual: torch.Tensor
    gripper_residual: torch.Tensor
    dino_residual: torch.Tensor
    compatibility_energy: torch.Tensor
    validity_logit: torch.Tensor | None = None

    @property
    def consistency_logit(self) -> torch.Tensor:
        """Higher means more compatible; energy uses the opposite convention."""
        return -self.compatibility_energy

    def as_dict(self) -> dict[str, torch.Tensor | None]:
        return {field.name: getattr(self, field.name) for field in fields(self)}
