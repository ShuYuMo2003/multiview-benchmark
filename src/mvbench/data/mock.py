"""Deterministic mock data for trainer, DDP, checkpoint, and metric tests."""

from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset


class MockPairDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        episodes: int = 8,
        frames_per_episode: int = 32,
        image_size: int = 56,
        joint_dim: int = 7,
        dino_dim: int = 16,
        seed: int = 17,
    ):
        self.episodes = episodes
        self.frames_per_episode = frames_per_episode
        self.image_size = image_size
        self.joint_dim = joint_dim
        self.dino_dim = dino_dim
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        state_dim = 3 + 3 + joint_dim + 1
        self.dino_projection = torch.randn(state_dim + 4, dino_dim, generator=generator) / math.sqrt(state_dim + 4)
        coordinates = torch.linspace(-1.0, 1.0, image_size)
        self.grid_y, self.grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")

    def __len__(self) -> int:
        return self.episodes * self.frames_per_episode * 2

    def _state(self, episode: int, frame: int, side: int) -> tuple[torch.Tensor, ...]:
        phase = 2.0 * math.pi * frame / self.frames_per_episode
        side_sign = -1.0 if side == 0 else 1.0
        position = torch.tensor([
            0.35 * math.sin(phase) + 0.15 * side_sign,
            0.25 * math.cos(phase * 0.7),
            0.45 + 0.08 * math.sin(phase * 1.3),
        ])
        rotation = torch.tensor([
            0.25 * math.sin(phase * 0.5),
            0.2 * math.cos(phase * 0.8),
            0.3 * math.sin(phase),
        ])
        joint = torch.stack([
            torch.tensor(math.sin(phase * (index + 1) / self.joint_dim + 0.2 * side_sign))
            for index in range(self.joint_dim)
        ])
        gripper = torch.tensor([1.0 if math.sin(phase * 1.5 + side) > 0 else 0.0])
        scene = torch.tensor([
            math.sin(episode * 0.37), math.cos(episode * 0.23), side_sign, episode / max(self.episodes, 1)
        ])
        feature = torch.cat([position, rotation, joint, gripper, scene]) @ self.dino_projection
        feature = torch.nn.functional.normalize(feature, dim=0)
        return position, rotation, joint, gripper, feature

    def _image(self, state: tuple[torch.Tensor, ...], view: str, episode: int) -> torch.Tensor:
        position, rotation, joint, gripper, _ = state
        view_bias = 0.15 if view == "head" else -0.1
        first = torch.sigmoid(2.0 * (self.grid_x * position[0] + self.grid_y * position[1] + view_bias))
        second = torch.sigmoid(2.0 * (self.grid_x * rotation[2] - self.grid_y * joint[0]))
        third = torch.clamp(
            0.3 + 0.25 * torch.sin((episode + 1) * self.grid_x) + 0.35 * gripper[0], 0.0, 1.0
        )
        return torch.stack([first, second, third], dim=0).float()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        side = index % 2
        row = index // 2
        episode = row // self.frames_per_episode
        frame = row % self.frames_per_episode
        mode = row % 5
        if mode == 0:
            candidate_episode, candidate_frame = episode, frame
        elif mode == 4:
            candidate_episode = (episode + 1) % self.episodes
            candidate_frame = (frame + 7) % self.frames_per_episode
        else:
            candidate_episode = episode
            candidate_frame = (frame + (1, 3, 8)[mode - 1]) % self.frames_per_episode
        target = self._state(episode, frame, side)
        candidate = self._state(candidate_episode, candidate_frame, side)
        position_t, rotation_t, joint_t, gripper_t, dino_t = target
        position_s, rotation_s, joint_s, gripper_s, dino_s = candidate
        return {
            "head_image": self._image(target, "head", episode),
            "wrist_image": self._image(candidate, "wrist", candidate_episode),
            "side": torch.tensor(side, dtype=torch.long),
            "translation_residual": position_t - position_s,
            "rotation_residual": rotation_t - rotation_s,
            "joint_residual": joint_t - joint_s,
            "gripper_residual": gripper_t - gripper_s,
            "dino_residual": dino_t - dino_s,
            "consistency_label": torch.tensor([float(mode == 0)]),
            "pose_mask": torch.ones(1),
            "joint_mask": torch.ones(1),
            "gripper_mask": torch.ones(1),
            "dino_mask": torch.ones(1),
            "consistency_mask": torch.ones(1),
        }
