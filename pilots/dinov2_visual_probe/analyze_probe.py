#!/usr/bin/env python3
"""Run episode-held-out probes on synchronized DINOv2 multiview features.

This script answers four concrete feasibility questions:

1. How many PCA dimensions preserve wrist-view DINO distances?
2. How well can a head frame predict the aligned wrist feature?
3. Can a pairwise probe predict the full wrist feature residual or its norm?
4. Does a direct compatibility-energy head separate aligned and mismatched pairs?

No robot telemetry is used here.  Temporal mismatches are therefore candidate
negatives rather than definitive physical-state negatives; the report keeps
their offset types separate so this limitation remains visible.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PAIR_TYPES = ("aligned", "offset_1", "offset_3", "offset_8", "random_within", "cross_episode")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("scratch/dinov2_visual_probe/features_vitb14_reg_336_3fps.npz"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("scratch/dinov2_visual_probe/formal_pilot")
    )
    parser.add_argument(
        "--pca-feature-modes",
        nargs="+",
        choices=("cls", "mean_patch", "cls_mean"),
        default=("cls", "mean_patch", "cls_mean"),
    )
    parser.add_argument(
        "--probe-feature-mode",
        choices=("cls", "mean_patch", "cls_mean"),
        default="mean_patch",
    )
    parser.add_argument("--pca-dims", type=int, nargs="+", default=(32, 64, 128, 256))
    parser.add_argument("--probe-dim", type=int, default=128)
    parser.add_argument("--train-episodes", type=int, default=50)
    parser.add_argument("--val-episodes", type=int, default=10)
    parser.add_argument("--max-frames-per-episode", type=int, default=160)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


def feature_matrix(cache: Any, view: str, mode: str) -> np.ndarray:
    if mode == "cls":
        return normalize_rows(cache[f"{view}_cls"])
    if mode == "mean_patch":
        return normalize_rows(cache[f"{view}_mean_patch"])
    if mode == "cls_mean":
        cls = normalize_rows(cache[f"{view}_cls"])
        patch = normalize_rows(cache[f"{view}_mean_patch"])
        return normalize_rows(np.concatenate([cls, patch], axis=1))
    raise ValueError(mode)


def split_episodes(
    episode_ids: np.ndarray, train_count: int, val_count: int, seed: int
) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    unique = np.asarray(sorted(set(episode_ids.tolist())))
    if train_count + val_count >= unique.size:
        raise ValueError(
            f"Need at least one test episode, got {unique.size} total and "
            f"{train_count}+{val_count} requested"
        )
    rng = np.random.default_rng(seed)
    shuffled = unique[rng.permutation(unique.size)]
    episode_split = {
        "train": shuffled[:train_count].tolist(),
        "val": shuffled[train_count : train_count + val_count].tolist(),
        "test": shuffled[train_count + val_count :].tolist(),
    }
    frame_split = {
        name: np.flatnonzero(np.isin(episode_ids, values)).astype(np.int64)
        for name, values in episode_split.items()
    }
    return episode_split, frame_split


def cap_frames_per_episode(
    frame_split: dict[str, np.ndarray],
    episode_ids: np.ndarray,
    frame_indices: np.ndarray,
    maximum: int,
) -> dict[str, np.ndarray]:
    if maximum <= 0:
        return frame_split
    capped: dict[str, np.ndarray] = {}
    for split_name, split_rows in frame_split.items():
        selected: list[np.ndarray] = []
        for episode_id in sorted(set(episode_ids[split_rows].tolist())):
            rows = split_rows[episode_ids[split_rows] == episode_id]
            rows = rows[np.argsort(frame_indices[rows])]
            if rows.size > maximum:
                positions = np.rint(np.linspace(0, rows.size - 1, maximum)).astype(np.int64)
                rows = rows[positions]
            selected.append(rows)
        capped[split_name] = np.sort(np.concatenate(selected)).astype(np.int64)
    return capped


@dataclass
class PCAProjection:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, values: np.ndarray, dim: int, normalize: bool = True) -> np.ndarray:
        projected = (np.asarray(values, dtype=np.float32) - self.mean) @ self.components[:, :dim]
        return normalize_rows(projected) if normalize else projected.astype(np.float32)


def fit_pca(values: np.ndarray, max_dim: int, device: torch.device, seed: int) -> PCAProjection:
    values = np.asarray(values, dtype=np.float32)
    q = min(max_dim, values.shape[0] - 1, values.shape[1])
    if q < 1:
        raise ValueError(f"Not enough data for PCA: {values.shape}")
    torch.manual_seed(seed)
    tensor = torch.from_numpy(values).to(device)
    mean = tensor.mean(dim=0, keepdim=True)
    centered = tensor - mean
    total_variance = centered.square().sum() / max(1, tensor.shape[0] - 1)
    _, singular_values, components = torch.pca_lowrank(
        centered, q=q, center=False, niter=6
    )
    explained = singular_values.square() / max(1, tensor.shape[0] - 1)
    ratios = explained / total_variance.clamp_min(1e-12)
    return PCAProjection(
        mean=mean.squeeze(0).cpu().numpy().astype(np.float32),
        components=components.cpu().numpy().astype(np.float32),
        explained_variance_ratio=ratios.cpu().numpy().astype(np.float64),
    )


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
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


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = rankdata(scores)
    numerator = ranks[labels].sum() - positive_count * (positive_count + 1) / 2.0
    return float(numerator / (positive_count * negative_count))


def pca_preservation(
    raw: np.ndarray,
    projection: PCAProjection,
    dims: list[int],
    seed: int,
    pair_count: int = 50000,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    count = raw.shape[0]
    first = rng.integers(0, count, size=pair_count)
    second = rng.integers(0, count, size=pair_count)
    different = first != second
    first, second = first[different], second[different]
    raw_distance = np.linalg.norm(raw[first] - raw[second], axis=1)
    subset_size = min(512, count)
    query_ids = rng.choice(count, size=subset_size, replace=False)
    raw_similarity = raw[query_ids] @ raw.T
    raw_similarity[np.arange(subset_size), query_ids] = -np.inf
    raw_neighbor = np.argmax(raw_similarity, axis=1)
    results: list[dict[str, float]] = []
    for dim in dims:
        reduced = projection.transform(raw, dim)
        reduced_distance = np.linalg.norm(reduced[first] - reduced[second], axis=1)
        reduced_similarity = reduced[query_ids] @ reduced.T
        reduced_similarity[np.arange(subset_size), query_ids] = -np.inf
        reduced_neighbor = np.argmax(reduced_similarity, axis=1)
        results.append(
            {
                "dim": int(dim),
                "cumulative_explained_variance": float(
                    projection.explained_variance_ratio[:dim].sum()
                ),
                "distance_spearman": spearman(raw_distance, reduced_distance),
                "top1_neighbor_retention": float(np.mean(raw_neighbor == reduced_neighbor)),
            }
        )
    return results


@dataclass
class PairSet:
    anchor: np.ndarray
    candidate: np.ndarray
    side: np.ndarray
    pair_type: np.ndarray
    consistent: np.ndarray


def build_pairs(
    frame_indices: np.ndarray,
    episode_ids: np.ndarray,
    split_indices: np.ndarray,
    seed: int,
) -> PairSet:
    rng = np.random.default_rng(seed)
    split_indices = np.asarray(split_indices, dtype=np.int64)
    groups: dict[str, np.ndarray] = {}
    for episode_id in sorted(set(episode_ids[split_indices].tolist())):
        ids = split_indices[episode_ids[split_indices] == episode_id]
        ids = ids[np.argsort(frame_indices[ids])]
        groups[episode_id] = ids
    episode_names = sorted(groups)
    anchor_rows: list[int] = []
    candidate_rows: list[int] = []
    sides: list[int] = []
    pair_types: list[int] = []
    consistent: list[int] = []

    def append(anchor: int, candidate: int, side: int, type_index: int, is_consistent: int) -> None:
        anchor_rows.append(anchor)
        candidate_rows.append(candidate)
        sides.append(side)
        pair_types.append(type_index)
        consistent.append(is_consistent)

    for episode_id in episode_names:
        ids = groups[episode_id]
        other_episodes = [name for name in episode_names if name != episode_id]
        for position, anchor in enumerate(ids.tolist()):
            for side in (0, 1):
                append(anchor, anchor, side, 0, 1)
                for type_index, offset in enumerate((1, 3, 8), start=1):
                    sign = -1 if rng.random() < 0.5 else 1
                    candidate_position = position + sign * offset
                    if candidate_position < 0 or candidate_position >= ids.size:
                        candidate_position = position - sign * offset
                    candidate_position = int(np.clip(candidate_position, 0, ids.size - 1))
                    if candidate_position == position and ids.size > 1:
                        candidate_position = (position + 1) % ids.size
                    append(anchor, int(ids[candidate_position]), side, type_index, 0)
                if ids.size > 1:
                    valid = np.flatnonzero(np.abs(np.arange(ids.size) - position) >= min(5, ids.size - 1))
                    if valid.size == 0:
                        valid = np.flatnonzero(np.arange(ids.size) != position)
                    candidate_position = int(rng.choice(valid))
                    append(anchor, int(ids[candidate_position]), side, 4, 0)
                else:
                    append(anchor, anchor, side, 4, 1)
                if other_episodes:
                    other_name = str(rng.choice(other_episodes))
                    append(anchor, int(rng.choice(groups[other_name])), side, 5, 0)
    return PairSet(
        anchor=np.asarray(anchor_rows, dtype=np.int64),
        candidate=np.asarray(candidate_rows, dtype=np.int64),
        side=np.asarray(sides, dtype=np.int64),
        pair_type=np.asarray(pair_types, dtype=np.int64),
        consistent=np.asarray(consistent, dtype=np.int64),
    )


def pair_arrays(
    pairs: PairSet,
    head_z: np.ndarray,
    left_z: np.ndarray,
    right_z: np.ndarray,
) -> dict[str, np.ndarray]:
    wrists = np.stack([left_z, right_z], axis=0)
    candidate_z = wrists[pairs.side, pairs.candidate]
    target_z = wrists[pairs.side, pairs.anchor]
    side_one_hot = np.eye(2, dtype=np.float32)[pairs.side]
    return {
        "pair_input": np.concatenate([head_z[pairs.anchor], candidate_z, side_one_hot], axis=1),
        "candidate_input": np.concatenate([candidate_z, side_one_hot], axis=1),
        "target_residual": (target_z - candidate_z).astype(np.float32),
        "target_distance": np.linalg.norm(target_z - candidate_z, axis=1).astype(np.float32),
        "candidate_z": candidate_z.astype(np.float32),
        "target_z": target_z.astype(np.float32),
    }


@dataclass
class RidgeModel:
    weights: np.ndarray
    input_mean: np.ndarray
    input_std: np.ndarray
    alpha: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        standardized = (values - self.input_mean) / self.input_std
        design = np.concatenate(
            [standardized, np.ones((standardized.shape[0], 1), dtype=np.float32)], axis=1
        )
        return design @ self.weights


def fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    alphas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
) -> RidgeModel:
    mean = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-5)
    train_standard = (train_x - mean) / std
    val_standard = (val_x - mean) / std
    train_design = np.concatenate(
        [train_standard, np.ones((train_standard.shape[0], 1), dtype=np.float32)], axis=1
    ).astype(np.float64)
    val_design = np.concatenate(
        [val_standard, np.ones((val_standard.shape[0], 1), dtype=np.float32)], axis=1
    ).astype(np.float64)
    xtx = train_design.T @ train_design
    xty = train_design.T @ train_y.astype(np.float64)
    # The two side one-hot columns and explicit bias are linearly dependent.
    # Regularizing every column keeps the closed-form system well-conditioned.
    identity = np.eye(xtx.shape[0], dtype=np.float64)
    best: tuple[float, np.ndarray, float] | None = None
    for alpha in alphas:
        weights = np.linalg.solve(xtx + alpha * identity, xty)
        loss = float(np.mean((val_design @ weights - val_y) ** 2))
        if best is None or loss < best[0]:
            best = (loss, weights.astype(np.float32), alpha)
    assert best is not None
    return RidgeModel(best[1], mean, std, best[2])


class ProbeMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def train_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_consistent: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_consistent: np.ndarray,
    task: str,
    device: torch.device,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
) -> tuple[ProbeMLP, dict[str, Any]]:
    torch.manual_seed(seed)
    output_dim = train_y.shape[1] if train_y.ndim == 2 else 1
    model = ProbeMLP(train_x.shape[1], output_dim, hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    x_train = torch.from_numpy(train_x).float().to(device)
    y_train = torch.from_numpy(train_y).float().to(device)
    c_train = torch.from_numpy(train_consistent).float().to(device)
    x_val = torch.from_numpy(val_x).float().to(device)
    y_val = torch.from_numpy(val_y).float().to(device)
    c_val = torch.from_numpy(val_consistent).float().to(device)
    positive_weight = float(max(1.0, (train_consistent == 0).sum() / max(1, (train_consistent == 1).sum())))
    sample_weight = torch.where(c_train > 0.5, positive_weight, 1.0)
    val_sample_weight = torch.where(c_val > 0.5, positive_weight, 1.0)

    def loss_fn(prediction: torch.Tensor, target: torch.Tensor, consistency: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if task == "energy":
            return F.binary_cross_entropy_with_logits(
                prediction.reshape(-1), consistency.reshape(-1),
                pos_weight=torch.tensor(positive_weight, device=device),
            )
        if task == "scalar":
            prediction = F.softplus(prediction.reshape(-1))
            per_sample = F.smooth_l1_loss(prediction, target.reshape(-1), reduction="none")
        elif task == "vector":
            per_sample = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=1)
        else:
            raise ValueError(task)
        return (per_sample * weights).sum() / weights.sum().clamp_min(1.0)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        generator = torch.Generator(device=device).manual_seed(seed + epoch)
        permutation = torch.randperm(x_train.shape[0], generator=generator, device=device)
        epoch_loss = 0.0
        epoch_weight = 0
        for start in range(0, permutation.numel(), batch_size):
            ids = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_train[ids])
            loss = loss_fn(prediction, y_train[ids], c_train[ids], sample_weight[ids])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * ids.numel()
            epoch_weight += ids.numel()
        model.eval()
        with torch.inference_mode():
            val_prediction = model(x_val)
            val_loss = float(loss_fn(val_prediction, y_val, c_val, val_sample_weight))
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss / epoch_weight, "val_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"No valid checkpoint for {task} probe")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_val_loss": best_loss, "history": history}


@torch.inference_mode()
def predict_mlp(model: ProbeMLP, values: np.ndarray, device: torch.device, task: str) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, values.shape[0], 2048):
        batch = torch.from_numpy(values[start : start + 2048]).float().to(device)
        prediction = model(batch)
        if task == "scalar":
            prediction = F.softplus(prediction)
        outputs.append(prediction.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def cosine_mean(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    pred = prediction[mask]
    true = target[mask]
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(true, axis=1)
    valid = denom > 1e-8
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.sum(pred[valid] * true[valid], axis=1) / denom[valid]))


def vector_metrics(prediction: np.ndarray, target: np.ndarray, pairs: PairSet) -> dict[str, Any]:
    pred_norm = np.linalg.norm(prediction, axis=1)
    true_norm = np.linalg.norm(target, axis=1)
    mismatch = pairs.consistent == 0
    mse = float(np.mean((prediction - target) ** 2))
    zero_mse = float(np.mean(target**2))
    centered_total = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - float(np.sum((prediction - target) ** 2)) / max(centered_total, 1e-12)
    return {
        "mse": mse,
        "zero_baseline_mse": zero_mse,
        "mse_reduction_vs_zero": 1.0 - mse / max(zero_mse, 1e-12),
        "r2": r2,
        "nonzero_direction_cosine": cosine_mean(prediction, target, mismatch),
        "norm_mae": float(np.mean(np.abs(pred_norm - true_norm))),
        "norm_spearman": spearman(pred_norm, true_norm),
        "mismatch_auroc": binary_auroc(pred_norm, mismatch),
        "aligned_predicted_norm_mean": float(np.mean(pred_norm[~mismatch])),
        "mismatch_predicted_norm_mean": float(np.mean(pred_norm[mismatch])),
        "by_pair_type": pair_type_metrics(pred_norm, true_norm, pairs),
    }


def scalar_metrics(prediction: np.ndarray, target: np.ndarray, pairs: PairSet) -> dict[str, Any]:
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    mismatch = pairs.consistent == 0
    return {
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "spearman": spearman(prediction, target),
        "mismatch_auroc": binary_auroc(prediction, mismatch),
        "aligned_prediction_mean": float(np.mean(prediction[~mismatch])),
        "mismatch_prediction_mean": float(np.mean(prediction[mismatch])),
        "by_pair_type": pair_type_metrics(prediction, target, pairs),
    }


def energy_metrics(logits: np.ndarray, pairs: PairSet) -> dict[str, Any]:
    logits = logits.reshape(-1)
    consistent = pairs.consistent.astype(np.float32)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    mismatch_score = -logits
    return {
        "mismatch_auroc": binary_auroc(mismatch_score, consistent == 0),
        "brier": float(np.mean((probability - consistent) ** 2)),
        "aligned_probability_mean": float(np.mean(probability[consistent == 1])),
        "mismatch_probability_mean": float(np.mean(probability[consistent == 0])),
        "by_pair_type": {
            PAIR_TYPES[type_id]: {
                "count": int(np.sum(pairs.pair_type == type_id)),
                "consistent_probability_mean": float(np.mean(probability[pairs.pair_type == type_id])),
            }
            for type_id in range(len(PAIR_TYPES))
            if np.any(pairs.pair_type == type_id)
        },
    }


def pair_type_metrics(prediction: np.ndarray, target: np.ndarray, pairs: PairSet) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for type_id, name in enumerate(PAIR_TYPES):
        mask = pairs.pair_type == type_id
        if not np.any(mask):
            continue
        result[name] = {
            "count": int(mask.sum()),
            "true_mean": float(np.mean(target[mask])),
            "prediction_mean": float(np.mean(prediction[mask])),
            "mae": float(np.mean(np.abs(prediction[mask] - target[mask]))),
        }
    return result


def exact_head_to_wrist_data(
    indices: np.ndarray,
    head_z: np.ndarray,
    left_z: np.ndarray,
    right_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    head = np.concatenate([head_z[indices], head_z[indices]], axis=0)
    side = np.concatenate([
        np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (indices.size, 1)),
        np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (indices.size, 1)),
    ], axis=0)
    target = np.concatenate([left_z[indices], right_z[indices]], axis=0)
    side_ids = np.concatenate([
        np.zeros(indices.size, dtype=np.int64), np.ones(indices.size, dtype=np.int64)
    ])
    return np.concatenate([head, side], axis=1), target, side_ids


def head_recovery_metrics(prediction: np.ndarray, target: np.ndarray, side_ids: np.ndarray) -> dict[str, Any]:
    prediction_normalized = normalize_rows(prediction)
    cosine = np.sum(prediction_normalized * target, axis=1)
    centered = target - target.mean(axis=0, keepdims=True)
    r2 = 1.0 - float(np.sum((prediction - target) ** 2)) / max(float(np.sum(centered**2)), 1e-12)
    return {
        "cosine_mean": float(np.mean(cosine)),
        "r2": r2,
        "mse": float(np.mean((prediction - target) ** 2)),
        "left_cosine_mean": float(np.mean(cosine[side_ids == 0])),
        "right_cosine_mean": float(np.mean(cosine[side_ids == 1])),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def markdown_report(results: dict[str, Any]) -> str:
    lines = [
        "# DINOv2 multiview visual feasibility pilot",
        "",
        "This is an episode-held-out RGB-only pilot. Temporal offsets are candidate mismatches; "
        "without robot telemetry they are not guaranteed physical-state mismatches.",
        "",
        "## Dataset and split",
        "",
        f"- Synchronized samples: {results['dataset']['sample_count']}",
        f"- Complete episodes: {results['dataset']['episode_count']}",
        f"- Episode split: {results['dataset']['split_episode_counts']}",
        f"- Probe feature: `{results['config']['probe_feature_mode']}`, PCA dim {results['config']['probe_dim']}",
        "",
        "## PCA preservation",
        "",
        "| Feature | Dim | Explained variance | Distance Spearman | Top-1 NN retained |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, rows in results["pca_preservation"].items():
        for row in rows:
            lines.append(
                f"| {mode} | {row['dim']} | {row['cumulative_explained_variance']:.3f} | "
                f"{row['distance_spearman']:.3f} | {row['top1_neighbor_retention']:.3f} |"
            )
    recovery = results["head_to_wrist_recovery"]
    lines.extend([
        "",
        "## Head-only aligned wrist feature recovery",
        "",
        f"- Test cosine: **{recovery['cosine_mean']:.3f}** "
        f"(left {recovery['left_cosine_mean']:.3f}, right {recovery['right_cosine_mean']:.3f})",
        f"- Test R2: **{recovery['r2']:.3f}**",
        "",
        "## Residual and compatibility probes",
        "",
        "| Probe | Main error | Rank correlation | Mismatch AUROC |",
        "|---|---:|---:|---:|",
    ])
    for name in ("zero_residual", "head_factorized_residual", "candidate_only_ridge", "pair_vector_mlp"):
        row = results["vector_probes"][name]
        lines.append(
            f"| {name} | MSE {row['mse']:.5f} | {row['norm_spearman']:.3f} | {row['mismatch_auroc']:.3f} |"
        )
    scalar = results["scalar_probe"]
    energy = results["energy_probe"]
    lines.append(
        f"| pair_scalar_mlp | MAE {scalar['mae']:.4f} | {scalar['spearman']:.3f} | {scalar['mismatch_auroc']:.3f} |"
    )
    lines.append(
        f"| compatibility_energy | Brier {energy['brier']:.4f} | n/a | {energy['mismatch_auroc']:.3f} |"
    )
    lines.extend([
        "",
        "## Pair-type breakdown",
        "",
        "| Type | True DINO gap | Vector predicted gap | Scalar predicted gap | Consistency probability |",
        "|---|---:|---:|---:|---:|",
    ])
    vector_types = results["vector_probes"]["pair_vector_mlp"]["by_pair_type"]
    scalar_types = scalar["by_pair_type"]
    energy_types = energy["by_pair_type"]
    for name in PAIR_TYPES:
        if name not in vector_types:
            continue
        lines.append(
            f"| {name} | {vector_types[name]['true_mean']:.3f} | "
            f"{vector_types[name]['prediction_mean']:.3f} | "
            f"{scalar_types[name]['prediction_mean']:.3f} | "
            f"{energy_types[name]['consistent_probability_mean']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- Exact synchronized frames are the only certain positives in this RGB-only pilot.",
        "- Nearby temporal offsets may still express the same physical state; per-offset rows should not be read as hard ground truth.",
        "- The global-feature MLP probes test information content, not the final patch-token cross-attention architecture.",
        "- EEF/gripper telemetry is required before selecting the benchmark's physical thresholds or final aggregate score.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    start_time = time.time()
    cache = np.load(args.features, allow_pickle=False)
    episode_ids = cache["episode_id"].astype(str)
    frame_indices = cache["frame_index"].astype(np.int64)
    episode_split, full_frame_split = split_episodes(
        episode_ids, args.train_episodes, args.val_episodes, args.seed
    )
    frame_split = cap_frames_per_episode(
        full_frame_split, episode_ids, frame_indices, args.max_frames_per_episode
    )
    max_dim = max(max(args.pca_dims), args.probe_dim)

    pca_results: dict[str, list[dict[str, float]]] = {}
    pca_models: dict[str, dict[str, PCAProjection]] = {}
    for mode_number, mode in enumerate(args.pca_feature_modes):
        head = feature_matrix(cache, "head", mode)
        left = feature_matrix(cache, "left", mode)
        right = feature_matrix(cache, "right", mode)
        train_wrist = np.concatenate([left[frame_split["train"]], right[frame_split["train"]]], axis=0)
        test_wrist = np.concatenate([left[frame_split["test"]], right[frame_split["test"]]], axis=0)
        wrist_pca = fit_pca(train_wrist, max_dim, device, args.seed + mode_number)
        train_head = head[frame_split["train"]]
        head_pca = fit_pca(train_head, max_dim, device, args.seed + 100 + mode_number)
        pca_models[mode] = {"head": head_pca, "wrist": wrist_pca}
        pca_results[mode] = pca_preservation(
            test_wrist, wrist_pca, list(args.pca_dims), args.seed + 200 + mode_number
        )
        print(f"PCA complete: {mode}", flush=True)

    mode = args.probe_feature_mode
    head_raw = feature_matrix(cache, "head", mode)
    left_raw = feature_matrix(cache, "left", mode)
    right_raw = feature_matrix(cache, "right", mode)
    head_z = pca_models[mode]["head"].transform(head_raw, args.probe_dim)
    left_z = pca_models[mode]["wrist"].transform(left_raw, args.probe_dim)
    right_z = pca_models[mode]["wrist"].transform(right_raw, args.probe_dim)

    pairs = {
        name: build_pairs(frame_indices, episode_ids, indices, args.seed + 300 + number)
        for number, (name, indices) in enumerate(frame_split.items())
    }
    pair_data = {
        name: pair_arrays(pair_set, head_z, left_z, right_z)
        for name, pair_set in pairs.items()
    }
    print(
        "Pair counts:", {name: int(value.anchor.size) for name, value in pairs.items()}, flush=True
    )

    recovery_data = {
        name: exact_head_to_wrist_data(indices, head_z, left_z, right_z)
        for name, indices in frame_split.items()
    }
    recovery_ridge = fit_ridge(
        recovery_data["train"][0], recovery_data["train"][1],
        recovery_data["val"][0], recovery_data["val"][1],
    )
    recovery_prediction = recovery_ridge.predict(recovery_data["test"][0])
    recovery_metrics = head_recovery_metrics(
        recovery_prediction, recovery_data["test"][1], recovery_data["test"][2]
    )
    recovery_metrics["ridge_alpha"] = recovery_ridge.alpha

    candidate_ridge = fit_ridge(
        pair_data["train"]["candidate_input"], pair_data["train"]["target_residual"],
        pair_data["val"]["candidate_input"], pair_data["val"]["target_residual"],
    )
    candidate_prediction = candidate_ridge.predict(pair_data["test"]["candidate_input"])

    vector_model, vector_training = train_mlp(
        pair_data["train"]["pair_input"], pair_data["train"]["target_residual"], pairs["train"].consistent,
        pair_data["val"]["pair_input"], pair_data["val"]["target_residual"], pairs["val"].consistent,
        "vector", device, args.hidden_dim, args.epochs, args.batch_size,
        args.learning_rate, args.patience, args.seed + 400,
    )
    vector_prediction = predict_mlp(vector_model, pair_data["test"]["pair_input"], device, "vector")
    print(f"Vector probe complete at epoch {vector_training['best_epoch']}", flush=True)

    scalar_model, scalar_training = train_mlp(
        pair_data["train"]["pair_input"], pair_data["train"]["target_distance"], pairs["train"].consistent,
        pair_data["val"]["pair_input"], pair_data["val"]["target_distance"], pairs["val"].consistent,
        "scalar", device, args.hidden_dim, args.epochs, args.batch_size,
        args.learning_rate, args.patience, args.seed + 500,
    )
    scalar_prediction = predict_mlp(scalar_model, pair_data["test"]["pair_input"], device, "scalar")
    print(f"Scalar probe complete at epoch {scalar_training['best_epoch']}", flush=True)

    energy_model, energy_training = train_mlp(
        pair_data["train"]["pair_input"], pairs["train"].consistent.astype(np.float32), pairs["train"].consistent,
        pair_data["val"]["pair_input"], pairs["val"].consistent.astype(np.float32), pairs["val"].consistent,
        "energy", device, args.hidden_dim, args.epochs, args.batch_size,
        args.learning_rate, args.patience, args.seed + 600,
    )
    energy_prediction = predict_mlp(energy_model, pair_data["test"]["pair_input"], device, "energy")
    print(f"Energy probe complete at epoch {energy_training['best_epoch']}", flush=True)

    test_pairs = pairs["test"]
    test_target = pair_data["test"]["target_residual"]
    test_candidate = pair_data["test"]["candidate_z"]
    test_side_one_hot = np.eye(2, dtype=np.float32)[test_pairs.side]
    factorized_input = np.concatenate([head_z[test_pairs.anchor], test_side_one_hot], axis=1)
    factorized_target_prediction = normalize_rows(recovery_ridge.predict(factorized_input))
    factorized_residual_prediction = factorized_target_prediction - test_candidate
    zero_prediction = np.zeros_like(test_target)

    results: dict[str, Any] = {
        "config": {
            "feature_cache": str(args.features.resolve()),
            "probe_feature_mode": mode,
            "probe_dim": args.probe_dim,
            "pca_dims": list(args.pca_dims),
            "seed": args.seed,
            "device": str(device),
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "max_frames_per_episode": args.max_frames_per_episode,
        },
        "dataset": {
            "sample_count": int(episode_ids.size),
            "episode_count": len(set(episode_ids.tolist())),
            "split_episode_counts": {name: len(values) for name, values in episode_split.items()},
            "split_frame_counts": {name: int(values.size) for name, values in frame_split.items()},
            "uncapped_split_frame_counts": {name: int(values.size) for name, values in full_frame_split.items()},
            "pair_counts": {name: int(value.anchor.size) for name, value in pairs.items()},
            "episode_split": episode_split,
        },
        "pca_preservation": pca_results,
        "head_to_wrist_recovery": recovery_metrics,
        "vector_probes": {
            "zero_residual": vector_metrics(zero_prediction, test_target, test_pairs),
            "head_factorized_residual": vector_metrics(factorized_residual_prediction, test_target, test_pairs),
            "candidate_only_ridge": vector_metrics(candidate_prediction, test_target, test_pairs),
            "pair_vector_mlp": vector_metrics(vector_prediction, test_target, test_pairs),
        },
        "scalar_probe": scalar_metrics(
            scalar_prediction, pair_data["test"]["target_distance"], test_pairs
        ),
        "energy_probe": energy_metrics(energy_prediction, test_pairs),
        "training": {
            "vector": vector_training,
            "scalar": scalar_training,
            "energy": energy_training,
            "head_ridge_alpha": recovery_ridge.alpha,
            "candidate_ridge_alpha": candidate_ridge.alpha,
        },
        "elapsed_sec": time.time() - start_time,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    json_path.write_text(json.dumps(json_ready(results), indent=2, ensure_ascii=False) + "\n")
    report_path = args.output_dir / "REPORT.md"
    report_path.write_text(markdown_report(results))
    torch.save(
        {
            "config": results["config"],
            "episode_split": episode_split,
            "head_pca": {
                "mean": pca_models[mode]["head"].mean,
                "components": pca_models[mode]["head"].components[:, : args.probe_dim],
            },
            "wrist_pca": {
                "mean": pca_models[mode]["wrist"].mean,
                "components": pca_models[mode]["wrist"].components[:, : args.probe_dim],
            },
            "head_ridge": recovery_ridge,
            "candidate_ridge": candidate_ridge,
            "vector_mlp": vector_model.cpu().state_dict(),
            "scalar_mlp": scalar_model.cpu().state_dict(),
            "energy_mlp": energy_model.cpu().state_dict(),
        },
        args.output_dir / "probe_artifacts.pt",
    )
    print(json.dumps({
        "results": str(json_path.resolve()),
        "report": str(report_path.resolve()),
        "elapsed_sec": results["elapsed_sec"],
        "pair_vector_mismatch_auroc": results["vector_probes"]["pair_vector_mlp"]["mismatch_auroc"],
        "scalar_mismatch_auroc": results["scalar_probe"]["mismatch_auroc"],
        "energy_mismatch_auroc": results["energy_probe"]["mismatch_auroc"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
