"""Config-driven adapter from LeRobot rows to head/wrist pair supervision.

The adapter deliberately consumes a precomputed pair plan.  Pair mining over a
very large LeRobot dataset should be an offline, versioned data job rather than
hidden random logic inside DataLoader workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from mvbench.geometry import as_rotation_matrix, relative_rotation_log


@dataclass(frozen=True)
class PairPlan:
    anchor_index: np.ndarray
    candidate_index: np.ndarray
    side: np.ndarray
    consistency_label: np.ndarray
    same_episode: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "PairPlan":
        values = np.load(path, allow_pickle=False)
        plan = cls(**{
            name: np.asarray(values[name])
            for name in ("anchor_index", "candidate_index", "side", "consistency_label", "same_episode")
        })
        sizes = {getattr(plan, name).shape[0] for name in plan.__dataclass_fields__}
        if len(sizes) != 1:
            raise ValueError(f"Pair-plan columns have inconsistent lengths: {sizes}")
        return plan

    def __len__(self) -> int:
        return int(self.anchor_index.shape[0])


@dataclass(frozen=True)
class LeRobotSchema:
    head_image_key: str
    left_wrist_image_key: str
    right_wrist_image_key: str
    left_eef_position_key: str | None = None
    right_eef_position_key: str | None = None
    left_eef_rotation_key: str | None = None
    right_eef_rotation_key: str | None = None
    rotation_representation: str = "quat_xyzw"
    left_joint_key: str | None = None
    right_joint_key: str | None = None
    left_gripper_key: str | None = None
    right_gripper_key: str | None = None
    left_dino_key: str | None = None
    right_dino_key: str | None = None
    physical_cross_episode_valid: bool = True
    image_size: int | None = None

    def side_key(self, side: int, field: str) -> str | None:
        prefix = "left" if side == 0 else "right"
        return getattr(self, f"{prefix}_{field}_key")


def _tensor(row: Mapping[str, Any], key: str | None) -> torch.Tensor | None:
    if key is None or key not in row:
        return None
    return torch.as_tensor(row[key], dtype=torch.float32)


def _image(row: Mapping[str, Any], key: str, image_size: int | None) -> torch.Tensor:
    image = torch.as_tensor(row[key])
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor for {key}, got {image.shape}")
    if image.shape[-1] == 3 and image.shape[0] != 3:
        image = image.permute(2, 0, 1)
    image = image.float()
    if image.max() > 1.5:
        image = image / 255.0
    image = image.clamp(0.0, 1.0)
    if image_size is None or image.shape[-2:] == (image_size, image_size):
        return image
    height, width = image.shape[-2:]
    scale = min(image_size / height, image_size / width)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    resized = F.interpolate(
        image.unsqueeze(0), size=(resized_height, resized_width),
        mode="bilinear", align_corners=False, antialias=True,
    ).squeeze(0)
    canvas = image.new_empty((3, image_size, image_size))
    canvas[:] = image.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    canvas[:, top : top + resized_height, left : left + resized_width] = resized
    return canvas


class LeRobotPairDataset(Dataset[dict[str, torch.Tensor]]):
    """Map a LeRobot-compatible row dataset and offline pair plan to PairBatch rows."""

    def __init__(
        self,
        dataset: Sequence[Mapping[str, Any]],
        pair_plan: PairPlan,
        schema: LeRobotSchema,
        joint_dim: int,
        dino_dim: int,
    ):
        self.dataset = dataset
        self.plan = pair_plan
        self.schema = schema
        self.joint_dim = joint_dim
        self.dino_dim = dino_dim

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        anchor = self.dataset[int(self.plan.anchor_index[index])]
        candidate = self.dataset[int(self.plan.candidate_index[index])]
        side = int(self.plan.side[index])
        wrist_image_key = self.schema.left_wrist_image_key if side == 0 else self.schema.right_wrist_image_key
        physical_valid = bool(self.plan.same_episode[index]) or self.schema.physical_cross_episode_valid

        position_t = _tensor(anchor, self.schema.side_key(side, "eef_position"))
        position_s = _tensor(candidate, self.schema.side_key(side, "eef_position"))
        rotation_t = _tensor(anchor, self.schema.side_key(side, "eef_rotation"))
        rotation_s = _tensor(candidate, self.schema.side_key(side, "eef_rotation"))
        pose_valid = physical_valid and position_t is not None and position_s is not None and rotation_t is not None and rotation_s is not None
        if pose_valid:
            translation = position_t - position_s
            source_rotation = as_rotation_matrix(rotation_s, self.schema.rotation_representation)
            target_rotation = as_rotation_matrix(rotation_t, self.schema.rotation_representation)
            rotation = relative_rotation_log(source_rotation, target_rotation)
        else:
            translation = torch.zeros(3)
            rotation = torch.zeros(3)

        joint_t = _tensor(anchor, self.schema.side_key(side, "joint"))
        joint_s = _tensor(candidate, self.schema.side_key(side, "joint"))
        joint_valid = physical_valid and joint_t is not None and joint_s is not None
        joint = joint_t - joint_s if joint_valid else torch.zeros(self.joint_dim)
        if joint.numel() != self.joint_dim:
            raise ValueError(f"Expected {self.joint_dim} joint values for side {side}, got {joint.numel()}")

        gripper_t = _tensor(anchor, self.schema.side_key(side, "gripper"))
        gripper_s = _tensor(candidate, self.schema.side_key(side, "gripper"))
        gripper_valid = physical_valid and gripper_t is not None and gripper_s is not None
        gripper = (gripper_t - gripper_s).reshape(1) if gripper_valid else torch.zeros(1)

        dino_t = _tensor(anchor, self.schema.side_key(side, "dino"))
        dino_s = _tensor(candidate, self.schema.side_key(side, "dino"))
        dino_valid = dino_t is not None and dino_s is not None
        dino = dino_t - dino_s if dino_valid else torch.zeros(self.dino_dim)
        if dino.numel() != self.dino_dim:
            raise ValueError(f"Expected DINO dim {self.dino_dim}, got {dino.numel()}")

        return {
            "head_image": _image(anchor, self.schema.head_image_key, self.schema.image_size),
            "wrist_image": _image(candidate, wrist_image_key, self.schema.image_size),
            "side": torch.tensor(side, dtype=torch.long),
            "translation_residual": translation.reshape(3),
            "rotation_residual": rotation.reshape(3),
            "joint_residual": joint.reshape(self.joint_dim),
            "gripper_residual": gripper,
            "dino_residual": dino.reshape(self.dino_dim),
            "consistency_label": torch.tensor([float(self.plan.consistency_label[index])]),
            "pose_mask": torch.tensor([float(pose_valid)]),
            "joint_mask": torch.tensor([float(joint_valid)]),
            "gripper_mask": torch.tensor([float(gripper_valid)]),
            "dino_mask": torch.tensor([float(dino_valid)]),
            "consistency_mask": torch.ones(1),
        }
