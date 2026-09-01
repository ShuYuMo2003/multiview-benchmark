#!/usr/bin/env python3
"""Reconstruct every synchronized three-camera frame with VGGT."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = PROJECT_ROOT / "third_party" / "vggt"
if str(VGGT_ROOT) not in sys.path:
    sys.path.insert(0, str(VGGT_ROOT))

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


VIEW_NAMES = ("front_head", "left_wrist", "right_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, nargs=3, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-sec", type=float, required=True)
    parser.add_argument("--end-sec", type=float, required=True)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--point-stride", type=int, default=3)
    parser.add_argument("--confidence-quantile", type=float, default=0.30)
    parser.add_argument("--distance-quantile", type=float, default=0.995)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--disable-temporal-alignment", action="store_true")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _video_properties(paths: list[Path]) -> tuple[float, int]:
    fps_values: list[float] = []
    frame_counts: list[int] = []
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        fps_values.append(float(cap.get(cv2.CAP_PROP_FPS)))
        frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    if max(fps_values) - min(fps_values) > 0.05:
        raise RuntimeError(f"Camera FPS mismatch: {fps_values}")
    return float(np.median(fps_values)), min(frame_counts)


def _decode_triplet(paths: list[Path], frame_index: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for path in paths:
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot decode frame {frame_index} from {path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return frames


def _preprocess_triplet(
    frames_rgb: list[np.ndarray], target_width: int = 518
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    processed: list[np.ndarray] = []
    for frame in frames_rgb:
        height, width = frame.shape[:2]
        new_height = max(14, round(height * target_width / width / 14) * 14)
        resized = Image.fromarray(frame).resize((target_width, new_height), Image.Resampling.BICUBIC)
        processed.append(np.asarray(resized, dtype=np.uint8))
    max_height = max(image.shape[0] for image in processed)
    max_width = max(image.shape[1] for image in processed)
    padded: list[np.ndarray] = []
    content_masks: list[np.ndarray] = []
    for image in processed:
        height, width = image.shape[:2]
        pad_top = (max_height - height) // 2
        pad_bottom = max_height - height - pad_top
        pad_left = (max_width - width) // 2
        pad_right = max_width - width - pad_left
        padded.append(
            np.pad(
                image,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=255,
            )
        )
        mask = np.zeros((max_height, max_width), dtype=bool)
        mask[pad_top : pad_top + height, pad_left : pad_left + width] = True
        content_masks.append(mask)
    rgb = np.stack(padded, axis=0)
    tensor = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).float() / 255.0
    return tensor, rgb, np.stack(content_masks, axis=0)


def _umeyama_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[0] < 3:
        return None
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if variance < 1e-12:
        return None
    covariance = (target_centered.T @ source_centered) / float(source.shape[0])
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    translation = target_mean - scale * (rotation @ source_mean)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _estimate_temporal_alignment(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    previous_points: np.ndarray,
    current_points: np.ndarray,
    previous_confidence: np.ndarray,
    current_confidence: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    identity = np.eye(4, dtype=np.float64)
    previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    features = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=1600,
        qualityLevel=0.008,
        minDistance=5,
        blockSize=7,
    )
    if features is None or len(features) < 24:
        return identity, {"method": "identity", "reason": "too_few_features", "matches": 0, "inliers": 0}
    forward, status_f, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, features, None, winSize=(25, 25), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    backward, status_b, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, forward, None, winSize=(25, 25), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    p0 = features.reshape(-1, 2)
    p1 = forward.reshape(-1, 2)
    p0_back = backward.reshape(-1, 2)
    height, width = previous_gray.shape
    valid = status_f.reshape(-1).astype(bool) & status_b.reshape(-1).astype(bool)
    valid &= np.linalg.norm(p0 - p0_back, axis=1) < 1.5
    valid &= (p1[:, 0] >= 0) & (p1[:, 0] < width) & (p1[:, 1] >= 0) & (p1[:, 1] < height)
    p0 = p0[valid]
    p1 = p1[valid]
    if p0.shape[0] < 24:
        return identity, {"method": "identity", "reason": "too_few_flow_matches", "matches": int(p0.shape[0]), "inliers": 0}

    # A valid sub-pixel coordinate can round from width/height-epsilon to the
    # exclusive upper bound. Clamp after rounding before indexing point maps.
    x0 = np.clip(np.rint(p0[:, 0]).astype(np.int64), 0, width - 1)
    y0 = np.clip(np.rint(p0[:, 1]).astype(np.int64), 0, height - 1)
    x1 = np.clip(np.rint(p1[:, 0]).astype(np.int64), 0, width - 1)
    y1 = np.clip(np.rint(p1[:, 1]).astype(np.int64), 0, height - 1)
    target_xyz = previous_points[y0, x0]
    source_xyz = current_points[y1, x1]
    conf0 = previous_confidence[y0, x0]
    conf1 = current_confidence[y1, x1]
    conf0_cut = float(np.quantile(previous_confidence[np.isfinite(previous_confidence)], 0.20))
    conf1_cut = float(np.quantile(current_confidence[np.isfinite(current_confidence)], 0.20))
    valid_3d = np.isfinite(source_xyz).all(axis=1) & np.isfinite(target_xyz).all(axis=1)
    valid_3d &= np.isfinite(conf0) & np.isfinite(conf1) & (conf0 >= conf0_cut) & (conf1 >= conf1_cut)
    source_xyz = source_xyz[valid_3d]
    target_xyz = target_xyz[valid_3d]
    match_count = int(source_xyz.shape[0])
    if match_count < 24:
        return identity, {"method": "identity", "reason": "too_few_3d_matches", "matches": match_count, "inliers": 0}

    lower = np.quantile(target_xyz, 0.10, axis=0)
    upper = np.quantile(target_xyz, 0.90, axis=0)
    scene_scale = max(float(np.linalg.norm(upper - lower)), 1e-5)
    threshold = 0.06 * scene_scale
    best_mask: np.ndarray | None = None
    best_median = float("inf")
    for _ in range(320):
        sample = rng.choice(match_count, size=4, replace=False)
        candidate = _umeyama_similarity(source_xyz[sample], target_xyz[sample])
        if candidate is None:
            continue
        step_scale = float(np.cbrt(abs(np.linalg.det(candidate[:3, :3]))))
        if not 0.65 <= step_scale <= 1.55:
            continue
        residual = np.linalg.norm(_transform_points(source_xyz, candidate) - target_xyz, axis=1)
        inliers = residual < threshold
        median = float(np.median(residual[inliers])) if np.any(inliers) else float("inf")
        if best_mask is None or np.count_nonzero(inliers) > np.count_nonzero(best_mask) or (
            np.count_nonzero(inliers) == np.count_nonzero(best_mask) and median < best_median
        ):
            best_mask = inliers
            best_median = median
    if best_mask is None or np.count_nonzero(best_mask) < max(20, int(0.20 * match_count)):
        return identity, {"method": "identity", "reason": "ransac_failed", "matches": match_count, "inliers": 0}
    refined = _umeyama_similarity(source_xyz[best_mask], target_xyz[best_mask])
    if refined is None:
        return identity, {"method": "identity", "reason": "refit_failed", "matches": match_count, "inliers": 0}
    residual = np.linalg.norm(_transform_points(source_xyz, refined) - target_xyz, axis=1)
    final_inliers = residual < threshold
    step_scale = float(np.cbrt(abs(np.linalg.det(refined[:3, :3]))))
    return refined, {
        "method": "flow_ransac_sim3",
        "matches": match_count,
        "inliers": int(np.count_nonzero(final_inliers)),
        "inlier_ratio": float(np.mean(final_inliers)),
        "median_residual": float(np.median(residual[final_inliers])),
        "threshold": threshold,
        "step_scale": step_scale,
    }


def _camera_to_sequence(extrinsics: np.ndarray, local_to_sequence: np.ndarray) -> np.ndarray:
    camera_to_local = closed_form_inverse_se3(extrinsics)
    linear = local_to_sequence[:3, :3]
    sim_scale = max(float(np.cbrt(abs(np.linalg.det(linear)))), 1e-12)
    sim_rotation = linear / sim_scale
    result = np.tile(np.eye(4, dtype=np.float32), (extrinsics.shape[0], 1, 1))
    for view_idx in range(extrinsics.shape[0]):
        result[view_idx, :3, :3] = sim_rotation @ camera_to_local[view_idx, :3, :3]
        result[view_idx, :3, 3] = (
            linear @ camera_to_local[view_idx, :3, 3] + local_to_sequence[:3, 3]
        )
    return result


def main() -> int:
    args = parse_args()
    videos = list(args.videos)
    for path in [*videos, args.checkpoint]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    fps, total_frames = _video_properties(videos)
    start_frame = int(np.clip(round(args.start_sec * fps), 0, total_frames - 1))
    end_frame = int(np.clip(round(args.end_sec * fps), start_frame, total_frames - 1))
    frame_indices = np.arange(start_frame, end_frame + 1, max(1, int(args.frame_step)), dtype=np.int64)

    print("Building VGGT camera+depth model", flush=True)
    model = VGGT(enable_point=False, enable_track=False).eval()
    state = load_file(str(args.checkpoint), device="cpu")
    model_keys = set(model.state_dict())
    filtered = {name: tensor for name, tensor in state.items() if name in model_keys}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    del state, filtered
    gc.collect()
    device = torch.device(args.device)
    precision = _dtype(args.precision)
    # Keep model parameters and inputs in FP32. VGGT intentionally disables
    # autocast inside its camera/depth heads, while the aggregator is run under
    # the outer mixed-precision context (matching the official inference path).
    model = model.to(device=device).eval()
    print(f"Loaded {len(model_keys)} tensors; processing {len(frame_indices)} synchronized triplets", flush=True)

    rng = np.random.default_rng(20260831)
    points_sequence: list[np.ndarray] = []
    confidence_sequence: list[np.ndarray] = []
    colors_sequence: list[np.ndarray] = []
    valid_sequence: list[np.ndarray] = []
    rgb_sequence: list[np.ndarray] = []
    extrinsics_sequence: list[np.ndarray] = []
    intrinsics_sequence: list[np.ndarray] = []
    cameras_sequence: list[np.ndarray] = []
    transforms_sequence: list[np.ndarray] = []
    alignment_stats: list[dict[str, float | int | str]] = []
    inference_seconds: list[float] = []
    previous_front_rgb: np.ndarray | None = None
    previous_front_points: np.ndarray | None = None
    previous_front_conf: np.ndarray | None = None
    previous_to_sequence = np.eye(4, dtype=np.float64)

    for sequence_idx, frame_index in enumerate(frame_indices.tolist()):
        frames = _decode_triplet(videos, frame_index)
        image_tensor, processed_rgb, content_mask = _preprocess_triplet(frames)
        image_tensor = image_tensor.to(device=device, dtype=torch.float32)
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=precision,
            enabled=device.type == "cuda" and precision != torch.float32,
        ):
            prediction = model(image_tensor)
        extrinsics_t, intrinsics_t = pose_encoding_to_extri_intri(
            prediction["pose_enc"], image_tensor.shape[-2:]
        )
        inference_seconds.append(time.perf_counter() - started)
        depth = prediction["depth"].squeeze(0).float().cpu().numpy()
        depth_conf = prediction["depth_conf"].squeeze(0).float().cpu().numpy()
        extrinsics = extrinsics_t.squeeze(0).float().cpu().numpy()
        intrinsics = intrinsics_t.squeeze(0).float().cpu().numpy()
        world_points = unproject_depth_map_to_point_map(depth, extrinsics, intrinsics)
        del prediction, image_tensor, extrinsics_t, intrinsics_t

        if sequence_idx == 0 or args.disable_temporal_alignment:
            current_to_previous = np.eye(4, dtype=np.float64)
            stat: dict[str, float | int | str] = {
                "method": "identity_first_frame" if sequence_idx == 0 else "disabled",
                "matches": 0,
                "inliers": 0,
            }
        else:
            assert previous_front_rgb is not None and previous_front_points is not None and previous_front_conf is not None
            current_to_previous, stat = _estimate_temporal_alignment(
                previous_front_rgb,
                processed_rgb[0],
                previous_front_points,
                world_points[0],
                previous_front_conf,
                depth_conf[0],
                rng,
            )
        local_to_sequence = previous_to_sequence @ current_to_previous
        previous_to_sequence = local_to_sequence
        stat = {"sequence_index": sequence_idx, "source_frame": frame_index, **stat}

        stride = max(1, int(args.point_stride))
        sampled_local = world_points[:, ::stride, ::stride].astype(np.float32)
        sampled_conf = depth_conf[:, ::stride, ::stride].astype(np.float32)
        sampled_colors = processed_rgb[:, ::stride, ::stride].astype(np.uint8)
        sampled_global = _transform_points(sampled_local.reshape(-1, 3), local_to_sequence).reshape(sampled_local.shape)
        sampled_content = content_mask[:, ::stride, ::stride]
        sampled_valid = (
            np.isfinite(sampled_global).all(axis=-1) & np.isfinite(sampled_conf) & sampled_content
        )
        for view_idx in range(len(VIEW_NAMES)):
            values = sampled_conf[view_idx][sampled_valid[view_idx]]
            if values.size:
                cutoff = float(np.quantile(values, np.clip(args.confidence_quantile, 0.0, 0.95)))
                sampled_valid[view_idx] &= sampled_conf[view_idx] >= cutoff
        if np.count_nonzero(sampled_valid) > 16:
            valid_points = sampled_global[sampled_valid]
            center = np.median(valid_points, axis=0)
            distances = np.linalg.norm(sampled_global - center, axis=-1)
            distance_cutoff = float(
                np.quantile(distances[sampled_valid], np.clip(args.distance_quantile, 0.5, 1.0))
            )
            sampled_valid &= distances <= distance_cutoff

        points_sequence.append(sampled_global.astype(np.float32))
        confidence_sequence.append(sampled_conf)
        colors_sequence.append(sampled_colors)
        valid_sequence.append(sampled_valid)
        rgb_sequence.append(processed_rgb)
        extrinsics_sequence.append(extrinsics.astype(np.float32))
        intrinsics_sequence.append(intrinsics.astype(np.float32))
        cameras_sequence.append(_camera_to_sequence(extrinsics, local_to_sequence))
        transforms_sequence.append(local_to_sequence.astype(np.float32))
        alignment_stats.append(stat)
        previous_front_rgb = processed_rgb[0]
        previous_front_points = world_points[0]
        previous_front_conf = depth_conf[0]
        kept = int(np.count_nonzero(sampled_valid))
        print(
            f"[{sequence_idx + 1}/{len(frame_indices)}] source={frame_index} "
            f"infer={inference_seconds[-1]:.3f}s kept={kept} align={stat['method']}",
            flush=True,
        )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        view_names=np.asarray(VIEW_NAMES),
        frame_indices=frame_indices,
        timestamps=frame_indices.astype(np.float64) / fps,
        rgb=np.stack(rgb_sequence),
        xyz_sequence=np.stack(points_sequence),
        confidence=np.stack(confidence_sequence),
        colors_rgb=np.stack(colors_sequence),
        valid=np.stack(valid_sequence),
        extrinsics_local=np.stack(extrinsics_sequence),
        intrinsics=np.stack(intrinsics_sequence),
        camera_to_sequence=np.stack(cameras_sequence),
        local_to_sequence=np.stack(transforms_sequence),
    )
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "model": "facebook/VGGT-1B",
                "vggt_commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
                "checkpoint": str(args.checkpoint.resolve()),
                "videos": {name: str(path.resolve()) for name, path in zip(VIEW_NAMES, videos, strict=True)},
                "source_fps": fps,
                "source_total_synchronized_frames": total_frames,
                "frame_indices": frame_indices.tolist(),
                "timestamps_seconds": (frame_indices.astype(np.float64) / fps).tolist(),
                "processed_image_size": list(rgb_sequence[0].shape[1:3]),
                "point_stride": int(args.point_stride),
                "sampled_grid_size": list(points_sequence[0].shape[1:3]),
                "confidence_quantile": float(args.confidence_quantile),
                "distance_quantile": float(args.distance_quantile),
                "precision": str(precision),
                "device": str(device),
                "inference_seconds_per_triplet": inference_seconds,
                "alignment": alignment_stats,
                "coordinate_frame": "first VGGT triplet world frame; later frames chained with front-head flow + 3D Sim(3)",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved reconstruction: {output_path}", flush=True)
    print(f"Saved metadata:       {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
