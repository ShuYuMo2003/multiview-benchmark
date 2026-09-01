"""Foundation-model adapters with a common pairwise token contract."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class PairBackboneOutput:
    head_tokens: torch.Tensor
    wrist_tokens: torch.Tensor
    head_pooled: torch.Tensor
    wrist_pooled: torch.Tensor
    head_padding_mask: torch.Tensor | None = None
    wrist_padding_mask: torch.Tensor | None = None


class PairVisualBackbone(nn.Module, ABC):
    output_dim: int

    @abstractmethod
    def forward_pair(self, head: torch.Tensor, wrist: torch.Tensor) -> PairBackboneOutput:
        raise NotImplementedError

    def forward(self, head: torch.Tensor, wrist: torch.Tensor) -> PairBackboneOutput:
        return self.forward_pair(head, wrist)


class TinyPairBackbone(PairVisualBackbone):
    """Small convolutional token encoder used only for pipeline/DDP tests."""

    def __init__(self, output_dim: int = 64, patch_size: int = 7):
        super().__init__()
        self.output_dim = output_dim
        self.patch_embed = nn.Conv2d(3, output_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(output_dim)

    def _encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.patch_embed(image).flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)
        return tokens, tokens.mean(dim=1)

    def forward_pair(self, head: torch.Tensor, wrist: torch.Tensor) -> PairBackboneOutput:
        head_tokens, head_pooled = self._encode(head)
        wrist_tokens, wrist_pooled = self._encode(wrist)
        return PairBackboneOutput(head_tokens, wrist_tokens, head_pooled, wrist_pooled)


def _set_trainable(module: nn.Module, train_mode: str, last_n_blocks: int = 0) -> None:
    if train_mode not in {"frozen", "full", "last_n"}:
        raise ValueError(f"Unknown backbone train mode: {train_mode}")
    module.requires_grad_(train_mode == "full")
    if train_mode == "last_n":
        module.requires_grad_(False)
        block_lists = []
        for name in ("blocks", "frame_blocks", "global_blocks"):
            blocks = getattr(module, name, None)
            if blocks is not None:
                block_lists.append(blocks)
        if not block_lists:
            raise ValueError("last_n requested but backbone exposes no recognized transformer blocks")
        for blocks in block_lists:
            for block in list(blocks)[-last_n_blocks:]:
                block.requires_grad_(True)


class DINOv2PairBackbone(PairVisualBackbone):
    """Encode the two images independently with an official DINOv2 model."""

    def __init__(
        self,
        repo: str | Path,
        model_name: str = "dinov2_vitl14_reg",
        pretrained: bool = True,
        train_mode: str = "frozen",
        last_n_blocks: int = 0,
    ):
        super().__init__()
        self.encoder = torch.hub.load(str(Path(repo).resolve()), model_name, source="local", pretrained=pretrained)
        self.output_dim = int(self.encoder.embed_dim)
        self.train_mode = train_mode
        _set_trainable(self.encoder, train_mode, last_n_blocks)
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )

    def train(self, mode: bool = True) -> "DINOv2PairBackbone":
        super().train(mode)
        if self.train_mode == "frozen":
            self.encoder.eval()
        return self

    def forward_pair(self, head: torch.Tensor, wrist: torch.Tensor) -> PairBackboneOutput:
        if head.shape != wrist.shape:
            raise ValueError(f"DINO pair images must have identical shapes, got {head.shape} and {wrist.shape}")
        images = torch.cat([head, wrist], dim=0)
        images = (images - self.image_mean) / self.image_std
        output = self.encoder.forward_features(images)
        batch_size = head.shape[0]
        tokens = output["x_norm_patchtokens"]
        pooled = output["x_norm_clstoken"]
        return PairBackboneOutput(
            head_tokens=tokens[:batch_size],
            wrist_tokens=tokens[batch_size:],
            head_pooled=pooled[:batch_size],
            wrist_pooled=pooled[batch_size:],
        )


class VGGTPairBackbone(PairVisualBackbone):
    """Jointly encode a head/wrist pair with VGGT alternating attention."""

    def __init__(
        self,
        repo: str | Path,
        checkpoint: str | Path,
        train_mode: str = "frozen",
        last_n_blocks: int = 0,
    ):
        super().__init__()
        repo = Path(repo).resolve()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from safetensors.torch import load_file
        from vggt.models.vggt import VGGT

        full_model = VGGT()
        state = load_file(str(Path(checkpoint).resolve()))
        full_model.load_state_dict(state, strict=True)
        self.aggregator = full_model.aggregator
        self.output_dim = int(self.aggregator.frame_blocks[0].norm1.normalized_shape[0]) * 2
        self.train_mode = train_mode
        _set_trainable(self.aggregator, train_mode, last_n_blocks)

    def train(self, mode: bool = True) -> "VGGTPairBackbone":
        super().train(mode)
        if self.train_mode == "frozen":
            self.aggregator.eval()
        return self

    def forward_pair(self, head: torch.Tensor, wrist: torch.Tensor) -> PairBackboneOutput:
        if head.shape != wrist.shape:
            raise ValueError(f"VGGT pair images must have identical shapes, got {head.shape} and {wrist.shape}")
        images = torch.stack([head, wrist], dim=1)
        cached, patch_start = self.aggregator(images)
        tokens = next(value for value in reversed(cached) if value is not None)
        patch_tokens = tokens[:, :, patch_start:, :]
        pooled = patch_tokens.mean(dim=2)
        return PairBackboneOutput(
            head_tokens=patch_tokens[:, 0],
            wrist_tokens=patch_tokens[:, 1],
            head_pooled=pooled[:, 0],
            wrist_pooled=pooled[:, 1],
        )


def build_backbone(config: dict[str, Any]) -> PairVisualBackbone:
    name = config["name"]
    if name == "tiny":
        return TinyPairBackbone(
            output_dim=int(config.get("output_dim", 64)),
            patch_size=int(config.get("patch_size", 7)),
        )
    common = {
        "train_mode": config.get("train_mode", "frozen"),
        "last_n_blocks": int(config.get("last_n_blocks", 0)),
    }
    if name == "dinov2":
        return DINOv2PairBackbone(
            repo=config["repo"],
            model_name=config.get("model_name", "dinov2_vitl14_reg"),
            pretrained=bool(config.get("pretrained", True)),
            **common,
        )
    if name == "vggt":
        return VGGTPairBackbone(
            repo=config["repo"], checkpoint=config["checkpoint"], **common
        )
    raise ValueError(f"Unknown backbone: {name}")
