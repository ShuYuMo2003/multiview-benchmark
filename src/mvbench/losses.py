"""Masked multi-task objective for pairwise state consistency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import PairBatch, PairMetricOutput


def _flat_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.float().reshape(mask.shape[0], -1).amax(dim=1)


def _masked_average(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.reshape(values.shape[0], -1).mean(dim=1)
    mask = _flat_mask(mask).to(values.dtype)
    count = mask.sum()
    return (values * mask).sum() / count.clamp_min(1.0), count


@dataclass
class LossWeights:
    translation: float = 1.0
    rotation: float = 1.0
    joint: float = 0.5
    gripper: float = 1.0
    dino_vector: float = 1.0
    dino_norm: float = 0.25
    dino_direction: float = 0.1
    energy: float = 1.0


class MultitaskConsistencyLoss(nn.Module):
    def __init__(self, weights: LossWeights | None = None, huber_delta: float = 0.1):
        super().__init__()
        self.weights = weights or LossWeights()
        self.huber_delta = huber_delta

    def forward(
        self, output: PairMetricOutput, batch: PairBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        translation, pose_count = _masked_average(
            F.huber_loss(
                output.translation_residual, batch.translation_residual,
                delta=self.huber_delta, reduction="none",
            ),
            batch.pose_mask,
        )
        rotation, _ = _masked_average(
            F.huber_loss(
                output.rotation_residual, batch.rotation_residual,
                delta=self.huber_delta, reduction="none",
            ),
            batch.pose_mask,
        )
        joint, joint_count = _masked_average(
            F.huber_loss(
                output.joint_residual, batch.joint_residual,
                delta=self.huber_delta, reduction="none",
            ),
            batch.joint_mask,
        )
        gripper, gripper_count = _masked_average(
            F.huber_loss(
                output.gripper_residual, batch.gripper_residual,
                delta=self.huber_delta, reduction="none",
            ),
            batch.gripper_mask,
        )
        if output.dino_residual.shape != batch.dino_residual.shape:
            raise ValueError(
                f"DINO residual shape mismatch: {output.dino_residual.shape} vs {batch.dino_residual.shape}"
            )
        dino_vector, dino_count = _masked_average(
            F.huber_loss(
                output.dino_residual, batch.dino_residual,
                delta=self.huber_delta, reduction="none",
            ),
            batch.dino_mask,
        )
        predicted_dino_norm = torch.linalg.vector_norm(output.dino_residual, dim=-1)
        target_dino_norm = torch.linalg.vector_norm(batch.dino_residual, dim=-1)
        dino_norm, _ = _masked_average(
            F.huber_loss(
                predicted_dino_norm, target_dino_norm,
                delta=self.huber_delta, reduction="none",
            ),
            batch.dino_mask,
        )
        target_nonzero = target_dino_norm > 1e-6
        direction_mask = _flat_mask(batch.dino_mask).bool() & target_nonzero
        direction_values = 1.0 - F.cosine_similarity(
            output.dino_residual, batch.dino_residual, dim=-1, eps=1e-8
        )
        dino_direction, direction_count = _masked_average(
            direction_values, direction_mask.float()
        )
        energy, energy_count = _masked_average(
            F.binary_cross_entropy_with_logits(
                output.consistency_logit.reshape(-1),
                batch.consistency_label.float().reshape(-1),
                reduction="none",
            ),
            batch.consistency_mask,
        )
        weighted = {
            "translation": translation * self.weights.translation,
            "rotation": rotation * self.weights.rotation,
            "joint": joint * self.weights.joint,
            "gripper": gripper * self.weights.gripper,
            "dino_vector": dino_vector * self.weights.dino_vector,
            "dino_norm": dino_norm * self.weights.dino_norm,
            "dino_direction": dino_direction * self.weights.dino_direction,
            "energy": energy * self.weights.energy,
        }
        total = sum(weighted.values())
        if output.validity_logit is not None:
            # Keep the optional head in the DDP graph until explicit
            # observability labels are added to the data contract.
            total = total + output.validity_logit.sum() * 0.0
        diagnostics = {
            "loss": total.detach(),
            **{f"loss/{name}": value.detach() for name, value in weighted.items()},
            "count/pose": pose_count.detach(),
            "count/joint": joint_count.detach(),
            "count/gripper": gripper_count.detach(),
            "count/dino": dino_count.detach(),
            "count/dino_direction": direction_count.detach(),
            "count/energy": energy_count.detach(),
        }
        return total, diagnostics

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MultitaskConsistencyLoss":
        weights = config.get("weights", {})
        return cls(
            weights=LossWeights(**{
                name: float(weights.get(name, getattr(LossWeights(), name)))
                for name in LossWeights.__dataclass_fields__
            }),
            huber_delta=float(config.get("huber_delta", 0.1)),
        )
