"""Pairwise head-to-wrist metric with task-specific readout tokens."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mvbench.contracts import PairMetricOutput
from .backbones import PairVisualBackbone, build_backbone


READOUT_NAMES = ("energy", "translation", "rotation", "joint", "gripper", "dino", "validity")


class CrossAttentionReadoutLayer(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        hidden = int(round(dim * mlp_ratio))
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim)
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        context_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        attended, _ = self.cross_attention(
            normalized_queries,
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        queries = queries + attended
        normalized_queries = self.self_norm(queries)
        attended, _ = self.self_attention(
            normalized_queries, normalized_queries, normalized_queries, need_weights=False
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class PairwiseStateConsistencyModel(nn.Module):
    """Shared model run independently for head-left and head-right pairs."""

    def __init__(
        self,
        backbone: PairVisualBackbone,
        fusion_dim: int = 512,
        fusion_depth: int = 4,
        fusion_heads: int = 8,
        fusion_mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        joint_residual_dim: int = 7,
        dino_residual_dim: int = 128,
    ):
        super().__init__()
        self.backbone = backbone
        self.dino_residual_dim = dino_residual_dim
        self.joint_residual_dim = joint_residual_dim
        self.input_projection = nn.Linear(backbone.output_dim, fusion_dim)
        self.readout_tokens = nn.Parameter(torch.empty(1, len(READOUT_NAMES), fusion_dim))
        self.view_embedding = nn.Parameter(torch.empty(2, fusion_dim))
        self.side_embedding = nn.Embedding(2, fusion_dim)
        self.layers = nn.ModuleList(
            [
                CrossAttentionReadoutLayer(
                    fusion_dim, fusion_heads, fusion_mlp_ratio, dropout
                )
                for _ in range(fusion_depth)
            ]
        )
        self.output_norm = nn.LayerNorm(fusion_dim)
        self.energy_head = nn.Linear(fusion_dim, 1)
        self.translation_head = nn.Linear(fusion_dim, 3)
        self.rotation_head = nn.Linear(fusion_dim, 3)
        self.joint_head = nn.Linear(fusion_dim, joint_residual_dim)
        self.gripper_head = nn.Linear(fusion_dim, 1)
        self.dino_head = nn.Linear(fusion_dim, dino_residual_dim)
        self.validity_head = nn.Linear(fusion_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.readout_tokens, std=0.02)
        nn.init.trunc_normal_(self.view_embedding, std=0.02)
        nn.init.normal_(self.side_embedding.weight, std=0.02)
        # Exact aligned pairs have zero residuals; a zero-biased initialization
        # makes this behavior explicit without freezing any head.
        for head in (
            self.energy_head, self.translation_head, self.rotation_head, self.joint_head,
            self.gripper_head, self.dino_head, self.validity_head,
        ):
            nn.init.zeros_(head.bias)

    def forward(self, head_image: torch.Tensor, wrist_image: torch.Tensor, side: torch.Tensor) -> PairMetricOutput:
        if side.ndim != 1:
            side = side.reshape(-1)
        side = side.long()
        if torch.any((side < 0) | (side > 1)):
            raise ValueError("side must be 0 (left) or 1 (right)")
        encoded = self.backbone.forward_pair(head_image, wrist_image)
        head_tokens = self.input_projection(encoded.head_tokens) + self.view_embedding[0]
        wrist_tokens = (
            self.input_projection(encoded.wrist_tokens)
            + self.view_embedding[1]
            + self.side_embedding(side).unsqueeze(1)
        )
        context = torch.cat([head_tokens, wrist_tokens], dim=1)
        padding_mask = None
        if encoded.head_padding_mask is not None or encoded.wrist_padding_mask is not None:
            if encoded.head_padding_mask is None or encoded.wrist_padding_mask is None:
                raise ValueError("Both views must provide padding masks or neither may provide one")
            padding_mask = torch.cat(
                [encoded.head_padding_mask, encoded.wrist_padding_mask], dim=1
            )
        queries = self.readout_tokens.expand(head_image.shape[0], -1, -1)
        queries = queries + self.side_embedding(side).unsqueeze(1)
        for layer in self.layers:
            queries = layer(queries, context, padding_mask)
        queries = self.output_norm(queries)
        token = {name: queries[:, index] for index, name in enumerate(READOUT_NAMES)}
        return PairMetricOutput(
            translation_residual=self.translation_head(token["translation"]),
            rotation_residual=self.rotation_head(token["rotation"]),
            joint_residual=self.joint_head(token["joint"]),
            gripper_residual=self.gripper_head(token["gripper"]),
            dino_residual=self.dino_head(token["dino"]),
            compatibility_energy=self.energy_head(token["energy"]),
            validity_logit=self.validity_head(token["validity"]),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PairwiseStateConsistencyModel":
        backbone = build_backbone(config["backbone"])
        fusion = config.get("fusion", {})
        return cls(
            backbone=backbone,
            fusion_dim=int(fusion.get("dim", 512)),
            fusion_depth=int(fusion.get("depth", 4)),
            fusion_heads=int(fusion.get("heads", 8)),
            fusion_mlp_ratio=float(fusion.get("mlp_ratio", 4.0)),
            dropout=float(fusion.get("dropout", 0.0)),
            joint_residual_dim=int(config.get("joint_residual_dim", 7)),
            dino_residual_dim=int(config.get("dino_residual_dim", 128)),
        )
