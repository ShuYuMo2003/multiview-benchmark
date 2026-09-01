#!/usr/bin/env python3
"""Run OpenD4RT on one wrist-camera video and save a portable reconstruction."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
D4RT_ROOT = PROJECT_ROOT / "third_party" / "Open-d4rt"
if str(D4RT_ROOT) not in sys.path:
    sys.path.insert(0, str(D4RT_ROOT))

from infer_track_3d import _unwrap_state_dict  # noqa: E402
from src.core import load_yaml_config  # noqa: E402
from src.model import build_model  # noqa: E402
from vis.build_like_demo import _infer_point_cloud_ref0  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a time-varying wrist-view point cloud with OpenD4RT."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path.")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument(
        "--end-sec",
        type=float,
        default=-1.0,
        help="End time in seconds; <=0 samples through the end of the video.",
    )
    parser.add_argument("--grid-cols", type=int, default=64)
    parser.add_argument("--grid-rows", type=int, default=36)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("float16", "bfloat16", "float32"), default="float16")
    return parser.parse_args()


def _sample_video(
    path: Path,
    num_frames: int,
    start_sec: float,
    end_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    if total_frames <= 0:
        raise RuntimeError(f"Video reports no frames: {path}")

    start_idx = int(np.clip(round(max(0.0, start_sec) * fps), 0, total_frames - 1))
    if end_sec <= 0.0:
        end_idx = total_frames - 1
    else:
        end_idx = int(np.clip(round(end_sec * fps), start_idx, total_frames - 1))
    sample_count = max(2, min(int(num_frames), end_idx - start_idx + 1))
    frame_indices = np.rint(np.linspace(start_idx, end_idx, sample_count)).astype(np.int64)

    frames: list[np.ndarray] = []
    for frame_idx in frame_indices.tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to decode frame {frame_idx} from {path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    video_rgb = np.stack(frames, axis=0).astype(np.uint8)
    timestamps = frame_indices.astype(np.float64) / fps
    return video_rgb, frame_indices, timestamps, fps, total_frames


def _normalized_grid(cols: int, rows: int, margin: float = 0.02) -> np.ndarray:
    cols = max(2, int(cols))
    rows = max(2, int(rows))
    xs = np.linspace(margin, 1.0 - margin, cols, dtype=np.float32)
    ys = np.linspace(margin, 1.0 - margin, rows, dtype=np.float32)
    return np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)


def _sample_grid_colors(video_rgb: np.ndarray, uv_norm: np.ndarray) -> np.ndarray:
    height, width = int(video_rgb.shape[1]), int(video_rgb.shape[2])
    x = np.clip(np.rint(uv_norm[:, 0] * (width - 1)), 0, width - 1).astype(np.int64)
    y = np.clip(np.rint(uv_norm[:, 1] * (height - 1)), 0, height - 1).astype(np.int64)
    return video_rgb[:, y, x, :].copy()


def _disable_encoder_pretraining(config: Any) -> None:
    pretrained = config.get_path("model.encoder.pretrained", None)
    if isinstance(pretrained, dict):
        pretrained["enabled"] = False
        pretrained["path"] = ""
        pretrained["strict"] = False
        pretrained["must_succeed"] = False


def _load_state_dict_safely(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state_dict = _unwrap_state_dict(payload)
    if not state_dict:
        raise RuntimeError(f"Checkpoint has no usable model state_dict: {path}")
    return state_dict


def _precision_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def main() -> int:
    args = parse_args()
    for path in (args.video, args.config, args.checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available.")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".json")

    print(f"Loading sampled frames from {args.video}", flush=True)
    video_rgb, frame_indices, timestamps, source_fps, total_frames = _sample_video(
        args.video,
        num_frames=args.num_frames,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
    )

    config = load_yaml_config(args.config)
    _disable_encoder_pretraining(config)
    expected_frames = int(config.get_path("model.input.clip_frames", args.num_frames))
    if int(video_rgb.shape[0]) != expected_frames:
        raise ValueError(
            f"Checkpoint expects exactly {expected_frames} frames, but {video_rgb.shape[0]} were sampled."
        )
    model_h, model_w = [int(v) for v in config.get_path("model.input.image_size", [256, 256])]
    video_model_rgb = np.stack(
        [cv2.resize(frame, (model_w, model_h), interpolation=cv2.INTER_AREA) for frame in video_rgb], axis=0
    )

    print("Building the OpenD4RT model on CPU", flush=True)
    model = build_model(config["model"]).eval()
    state_dict = _load_state_dict_safely(args.checkpoint)
    model_keys = set(model.state_dict().keys())
    matched_keys = model_keys.intersection(state_dict.keys())
    matched_ratio = len(matched_keys) / max(1, len(model_keys))
    if matched_ratio < 0.95:
        raise RuntimeError(
            f"Only {len(matched_keys)}/{len(model_keys)} model tensors match the checkpoint; refusing inference."
        )
    incompatible = model.load_state_dict(state_dict, strict=False)
    del state_dict
    gc.collect()
    print(
        f"Checkpoint loaded ({len(matched_keys)}/{len(model_keys)} tensors; "
        f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})",
        flush=True,
    )

    device = torch.device(args.device)
    dtype = _precision_dtype(args.precision)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"Moving model to {device} as {dtype}", flush=True)
    model = model.to(device=device, dtype=dtype).eval()

    uv_norm = _normalized_grid(args.grid_cols, args.grid_rows)
    colors_rgb = _sample_grid_colors(video_rgb, uv_norm)
    native_aspect = float(video_rgb.shape[2]) / float(video_rgb.shape[1])

    started = time.perf_counter()
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=dtype if autocast_enabled else torch.float32,
        enabled=autocast_enabled,
    ):
        xyz_ref0, finite_mask, confidence_logits, _ = _infer_point_cloud_ref0(
            model=model,
            video_model_rgb=video_model_rgb,
            native_aspect_ratio=native_aspect,
            point_query_uv_norm=uv_norm,
            query_chunk_size=int(args.query_chunk_size),
            umeyama_slide_window=False,
        )
    inference_seconds = time.perf_counter() - started
    confidence = 1.0 / (1.0 + np.exp(-np.clip(confidence_logits, -30.0, 30.0)))
    finite_mask &= np.isfinite(xyz_ref0).all(axis=-1)

    np.savez_compressed(
        output_path,
        video_rgb=video_rgb,
        frame_indices=frame_indices,
        timestamps=timestamps,
        uv_norm=uv_norm,
        xyz_ref0=xyz_ref0.astype(np.float32),
        finite_mask=finite_mask.astype(np.bool_),
        confidence=confidence.astype(np.float32),
        colors_rgb=colors_rgb.astype(np.uint8),
    )

    metadata = {
        "source_video": str(args.video.resolve()),
        "source_fps": source_fps,
        "source_total_frames": total_frames,
        "sampled_frame_indices": frame_indices.tolist(),
        "sampled_timestamps_seconds": timestamps.tolist(),
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "open_d4rt_commit": "403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3",
        "coordinate_frame": "camera frame at sampled t=0 (ref0); learned, non-metric scale",
        "grid": {"cols": int(args.grid_cols), "rows": int(args.grid_rows)},
        "model_image_size": [model_h, model_w],
        "native_video_size": [int(video_rgb.shape[1]), int(video_rgb.shape[2])],
        "precision": str(dtype),
        "device": str(device),
        "inference_seconds": inference_seconds,
        "finite_points_per_frame": finite_mask.sum(axis=1).astype(int).tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved reconstruction: {output_path}", flush=True)
    print(f"Saved metadata:       {metadata_path}", flush=True)
    print(f"Inference time:       {inference_seconds:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
