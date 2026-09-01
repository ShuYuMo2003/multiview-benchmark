#!/usr/bin/env python3
"""Run one loaded OpenD4RT model over multiple synchronized camera videos."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from run_d4rt_wrist import (
    _disable_encoder_pretraining,
    _infer_point_cloud_ref0,
    _load_state_dict_safely,
    _normalized_grid,
    _precision_dtype,
    _sample_grid_colors,
    _sample_video,
    build_model,
    load_yaml_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("front_head", "left_wrist", "right_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--grid-cols", type=int, default=160)
    parser.add_argument("--grid-rows", type=int, default=90)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("float16", "bfloat16", "float32"), default="float16")
    return parser.parse_args()


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = manifest.get("episodes", [])
    if not episodes:
        raise RuntimeError("Manifest contains no episodes.")

    config = load_yaml_config(args.config)
    _disable_encoder_pretraining(config)
    expected_frames = int(config.get_path("model.input.clip_frames", args.num_frames))
    if int(args.num_frames) != expected_frames:
        raise ValueError(f"Checkpoint expects {expected_frames} frames, got {args.num_frames}.")
    model_h, model_w = [int(v) for v in config.get_path("model.input.image_size", [256, 256])]

    print("Building one shared OpenD4RT model on CPU", flush=True)
    model = build_model(config["model"]).eval()
    state_dict = _load_state_dict_safely(args.checkpoint)
    model_keys = set(model.state_dict().keys())
    matched_keys = model_keys.intersection(state_dict.keys())
    if len(matched_keys) / max(1, len(model_keys)) < 0.95:
        raise RuntimeError(f"Only {len(matched_keys)}/{len(model_keys)} model tensors match.")
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
    model = model.to(device=device, dtype=dtype).eval()
    print(f"Shared model ready on {device} as {dtype}", flush=True)

    uv_norm = _normalized_grid(args.grid_cols, args.grid_rows)
    output_root = args.output_root.resolve()
    completed: list[dict[str, object]] = []
    jobs = [(episode, view) for episode in episodes for view in VIEWS]

    for job_index, (episode, view) in enumerate(jobs, start=1):
        episode_root = _resolve_project_path(episode["episode_root"])
        video_path = episode_root / f"observation.images.{view}" / episode["episode_file"]
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        output_dir = output_root / episode["id"] / view
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "reconstruction.npz"
        metadata_path = output_dir / "reconstruction.json"
        print(f"[{job_index}/{len(jobs)}] {episode['id']} / {view}", flush=True)

        video_rgb, frame_indices, timestamps, source_fps, total_frames = _sample_video(
            video_path,
            num_frames=int(args.num_frames),
            start_sec=float(episode["start_sec"]),
            end_sec=float(episode["end_sec"]),
        )
        video_model_rgb = np.stack(
            [cv2.resize(frame, (model_w, model_h), interpolation=cv2.INTER_AREA) for frame in video_rgb], axis=0
        )
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
            "episode_id": episode["id"],
            "task": episode["task"],
            "view": view,
            "source_video": str(video_path.resolve()),
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
        completed.append(
            {
                "episode_id": episode["id"],
                "task": episode["task"],
                "view": view,
                "output": str(output_path),
                "metadata": str(metadata_path),
                "inference_seconds": inference_seconds,
            }
        )
        print(f"    saved {output_path} ({inference_seconds:.2f}s)", flush=True)

    index_path = output_root / "batch_reconstruction.json"
    index_path.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "jobs": completed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Completed {len(completed)} reconstructions; index: {index_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
