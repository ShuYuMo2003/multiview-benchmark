"""Evaluation metrics for residual accuracy, ranking, and calibration."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from .contracts import PairBatch, PairMetricOutput


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(values: np.ndarray, targets: np.ndarray) -> float:
    values = np.asarray(values).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    if values.size < 2 or np.std(values) < 1e-12 or np.std(targets) < 1e-12:
        return float("nan")
    return float(np.corrcoef(_rankdata(values), _rankdata(targets))[0, 1])


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float(
        (ranks[labels].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    probabilities = np.asarray(probabilities).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if np.any(mask):
            result += float(mask.mean()) * abs(float(probabilities[mask].mean() - labels[mask].mean()))
    return result


class EvaluationAccumulator:
    """Collect scalar per-sample statistics and merge them across DDP ranks."""

    def __init__(self, max_samples: int = 200_000, seed: int = 17):
        self.max_samples = max_samples
        self.seed = seed
        self.values: dict[str, list[np.ndarray]] = defaultdict(list)

    @torch.inference_mode()
    def update(self, output: PairMetricOutput, batch: PairBatch) -> None:
        pose_mask = batch.pose_mask.reshape(batch.pose_mask.shape[0], -1).amax(dim=1).bool()
        gripper_mask = batch.gripper_mask.reshape(batch.gripper_mask.shape[0], -1).amax(dim=1).bool()
        joint_mask = batch.joint_mask.reshape(batch.joint_mask.shape[0], -1).amax(dim=1).bool()
        dino_mask = batch.dino_mask.reshape(batch.dino_mask.shape[0], -1).amax(dim=1).bool()
        energy_mask = batch.consistency_mask.reshape(batch.consistency_mask.shape[0], -1).amax(dim=1).bool()
        self._append("side", batch.side)
        self._append("translation_valid", pose_mask)
        self._append("translation_error", torch.linalg.vector_norm(
            output.translation_residual - batch.translation_residual, dim=-1
        ))
        self._append("translation_pred_gap", torch.linalg.vector_norm(output.translation_residual, dim=-1))
        self._append("translation_true_gap", torch.linalg.vector_norm(batch.translation_residual, dim=-1))
        self._append("rotation_valid", pose_mask)
        self._append("rotation_error_deg", torch.rad2deg(torch.linalg.vector_norm(
            output.rotation_residual - batch.rotation_residual, dim=-1
        )))
        self._append("rotation_pred_gap_deg", torch.rad2deg(torch.linalg.vector_norm(
            output.rotation_residual, dim=-1
        )))
        self._append("rotation_true_gap_deg", torch.rad2deg(torch.linalg.vector_norm(
            batch.rotation_residual, dim=-1
        )))
        self._append("joint_valid", joint_mask)
        self._append("joint_error", torch.linalg.vector_norm(
            output.joint_residual - batch.joint_residual, dim=-1
        ))
        self._append("joint_pred_gap", torch.linalg.vector_norm(output.joint_residual, dim=-1))
        self._append("joint_true_gap", torch.linalg.vector_norm(batch.joint_residual, dim=-1))
        self._append("gripper_valid", gripper_mask)
        self._append("gripper_abs_error", torch.abs(
            output.gripper_residual.reshape(-1) - batch.gripper_residual.reshape(-1)
        ))
        self._append("gripper_pred_gap", torch.abs(output.gripper_residual.reshape(-1)))
        self._append("gripper_true_gap", torch.abs(batch.gripper_residual.reshape(-1)))
        self._append("dino_valid", dino_mask)
        self._append("dino_mse", (output.dino_residual - batch.dino_residual).square().mean(dim=-1))
        self._append("dino_direction_cosine", F.cosine_similarity(
            output.dino_residual, batch.dino_residual, dim=-1, eps=1e-8
        ))
        self._append("dino_pred_gap", torch.linalg.vector_norm(output.dino_residual, dim=-1))
        self._append("dino_true_gap", torch.linalg.vector_norm(batch.dino_residual, dim=-1))
        self._append("energy_valid", energy_mask)
        self._append("consistency_label", batch.consistency_label.reshape(-1))
        self._append("consistency_probability", torch.sigmoid(output.consistency_logit.reshape(-1)))
        if output.validity_logit is not None:
            self._append("validity_probability", torch.sigmoid(output.validity_logit.reshape(-1)))

    def _append(self, name: str, tensor: torch.Tensor) -> None:
        self.values[name].append(tensor.detach().float().cpu().numpy())

    def _payload(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        payload = {
            name: np.concatenate(chunks, axis=0) if chunks else np.empty((0,), dtype=np.float32)
            for name, chunks in self.values.items()
        }
        count = next(iter(payload.values())).shape[0] if payload else 0
        if count > self.max_samples:
            ids = np.sort(rng.choice(count, size=self.max_samples, replace=False))
            payload = {name: values[ids] for name, values in payload.items()}
        return payload

    def compute(self, distributed: bool = True) -> dict[str, float]:
        payloads = [self._payload()]
        if distributed and dist.is_available() and dist.is_initialized():
            gathered: list[dict[str, np.ndarray] | None] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, payloads[0])
            payloads = [item for item in gathered if item is not None]
        keys = payloads[0].keys() if payloads else []
        values = {key: np.concatenate([payload[key] for payload in payloads], axis=0) for key in keys}
        return compute_metrics(values)


def compute_metrics(values: dict[str, np.ndarray]) -> dict[str, float]:
    result: dict[str, float] = {"sample_count": float(values["side"].size)}

    def masked(name: str, mask_name: str) -> np.ndarray:
        return values[name][values[mask_name].astype(bool)]

    for prefix, error_name, pred_name, true_name, mask_name in (
        ("translation", "translation_error", "translation_pred_gap", "translation_true_gap", "translation_valid"),
        ("rotation", "rotation_error_deg", "rotation_pred_gap_deg", "rotation_true_gap_deg", "rotation_valid"),
        ("joint", "joint_error", "joint_pred_gap", "joint_true_gap", "joint_valid"),
        ("gripper", "gripper_abs_error", "gripper_pred_gap", "gripper_true_gap", "gripper_valid"),
        ("dino", "dino_mse", "dino_pred_gap", "dino_true_gap", "dino_valid"),
    ):
        valid = values[mask_name].astype(bool)
        result[f"{prefix}/count"] = float(valid.sum())
        if np.any(valid):
            result[f"{prefix}/error_mean"] = float(values[error_name][valid].mean())
            result[f"{prefix}/gap_mae"] = float(
                np.abs(values[pred_name][valid] - values[true_name][valid]).mean()
            )
            result[f"{prefix}/gap_spearman"] = spearman(
                values[pred_name][valid], values[true_name][valid]
            )
    dino_valid = values["dino_valid"].astype(bool)
    dino_nonzero = dino_valid & (values["dino_true_gap"] > 1e-6)
    if np.any(dino_nonzero):
        result["dino/direction_cosine"] = float(values["dino_direction_cosine"][dino_nonzero].mean())
    energy_valid = values["energy_valid"].astype(bool)
    if np.any(energy_valid):
        probability = values["consistency_probability"][energy_valid]
        labels = values["consistency_label"][energy_valid]
        result["energy/count"] = float(energy_valid.sum())
        result["energy/auroc"] = binary_auroc(probability, labels > 0.5)
        result["energy/brier"] = float(np.mean((probability - labels) ** 2))
        result["energy/ece_15"] = expected_calibration_error(probability, labels, bins=15)
        result["energy/accuracy_0.5"] = float(np.mean((probability >= 0.5) == (labels >= 0.5)))
    for side_id, side_name in ((0, "left"), (1, "right")):
        side_mask = values["side"] == side_id
        result[f"side/{side_name}_count"] = float(side_mask.sum())
        for gap_name in ("translation_pred_gap", "rotation_pred_gap_deg", "joint_pred_gap", "gripper_pred_gap", "dino_pred_gap"):
            task = gap_name.split("_pred_gap")[0]
            valid_name = f"{task}_valid"
            mask = side_mask & values[valid_name].astype(bool)
            if np.any(mask):
                result[f"side/{side_name}_{task}_pred_gap_mean"] = float(values[gap_name][mask].mean())
    return result
