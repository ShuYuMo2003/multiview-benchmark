#!/usr/bin/env python3
"""Extract synchronized DINOv2 features from head/left/right robot videos.

The output is a frame-aligned NPZ cache.  It intentionally stores both the
normalized CLS token and the normalized mean patch token so downstream probes
can compare feature definitions without decoding the videos again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


VIEW_DIRS = {
    "head": "observation.images.front_head",
    "left": "observation.images.left_wrist",
    "right": "observation.images.right_wrist",
}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Episode:
    episode_id: str
    head: Path
    left: Path
    right: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("real_data/Multiview_Benchmark/source_videos"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scratch/dinov2_visual_probe/features_vitb14_reg_336_3fps.npz"),
    )
    parser.add_argument("--model-repo", type=Path, default=Path("third_party/dinov2"))
    parser.add_argument("--model-name", default="dinov2_vitb14_reg")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def discover_episodes(data_root: Path) -> tuple[list[Episode], list[str]]:
    episodes: list[Episode] = []
    incomplete: list[str] = []
    head_dir = VIEW_DIRS["head"]
    for head in sorted(data_root.glob(f"**/{head_dir}/episode_*.mp4")):
        view_parent = head.parent.parent
        paths = {
            view: view_parent / dirname / head.name
            for view, dirname in VIEW_DIRS.items()
        }
        rel = head.relative_to(data_root)
        parts = list(rel.parts)
        parts[-2] = "observation.images.VIEW"
        episode_id = "/".join(parts)
        missing = [view for view, path in paths.items() if not path.is_file()]
        if missing:
            incomplete.append(f"{episode_id}: missing {','.join(missing)}")
            continue
        episodes.append(Episode(episode_id, paths["head"], paths["left"], paths["right"]))
    return episodes, incomplete


def _video_metadata(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok_first, _ = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames - 1))
    ok_last, _ = cap.read()
    cap.release()
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid FPS {fps} for {path}")
    if frames <= 0:
        raise RuntimeError(f"Invalid frame count {frames} for {path}")
    if not ok_first or not ok_last:
        raise RuntimeError(f"Cannot decode first/last frame from {path}")
    return fps, frames, width, height


def validate_episodes(
    episodes: list[Episode], max_episodes: int | None
) -> tuple[list[tuple[Episode, dict[str, tuple[float, int, int, int]]]], list[str]]:
    valid: list[tuple[Episode, dict[str, tuple[float, int, int, int]]]] = []
    invalid: list[str] = []
    for episode in episodes:
        try:
            metadata = {view: _video_metadata(getattr(episode, view)) for view in VIEW_DIRS}
            fps_values = np.asarray([value[0] for value in metadata.values()], dtype=np.float64)
            if float(np.max(fps_values) - np.min(fps_values)) > 0.05:
                raise RuntimeError(f"FPS mismatch: {fps_values.tolist()}")
            valid.append((episode, metadata))
        except RuntimeError as exc:
            invalid.append(f"{episode.episode_id}: {exc}")
        if max_episodes is not None and len(valid) >= max_episodes:
            break
    return valid, invalid


def _letterbox_rgb(frame_bgr: np.ndarray, image_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = min(image_size / width, image_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    pad_color = np.rint(IMAGENET_MEAN * 255.0).astype(np.uint8)
    canvas = np.empty((image_size, image_size, 3), dtype=np.uint8)
    canvas[...] = pad_color
    x0 = (image_size - new_width) // 2
    y0 = (image_size - new_height) // 2
    canvas[y0 : y0 + new_height, x0 : x0 + new_width] = resized
    return canvas


def _preprocess(frames: list[np.ndarray], image_size: int) -> torch.Tensor:
    images = np.stack([_letterbox_rgb(frame, image_size) for frame in frames], axis=0)
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.from_numpy(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.from_numpy(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_cached_checkpoint(model_name: str) -> Path | None:
    torch_home = Path(torch.hub.get_dir()).parent
    candidates = sorted((torch_home / "hub" / "checkpoints").glob(f"{model_name.replace('_reg', '')}*reg*pretrain.pth"))
    return candidates[0] if candidates else None


class FeatureAccumulator:
    def __init__(self, model: torch.nn.Module, device: torch.device, image_size: int, batch_size: int):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.batch_size = batch_size
        self.frames: list[np.ndarray] = []
        self.keys: list[tuple[int, str]] = []
        self.cls: dict[tuple[int, str], np.ndarray] = {}
        self.patch: dict[tuple[int, str], np.ndarray] = {}

    def add(self, key: tuple[int, str], frame: np.ndarray) -> None:
        self.keys.append(key)
        self.frames.append(frame)
        if len(self.frames) >= self.batch_size:
            self.flush()

    @torch.inference_mode()
    def flush(self) -> None:
        if not self.frames:
            return
        batch = _preprocess(self.frames, self.image_size).to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            output = self.model.forward_features(batch)
        cls = F.normalize(output["x_norm_clstoken"].float(), dim=-1).cpu().numpy().astype(np.float16)
        patch = F.normalize(
            output["x_norm_patchtokens"].float().mean(dim=1), dim=-1
        ).cpu().numpy().astype(np.float16)
        for key, cls_row, patch_row in zip(self.keys, cls, patch, strict=True):
            self.cls[key] = cls_row
            self.patch[key] = patch_row
        self.frames.clear()
        self.keys.clear()


def _selected_indices(frame_count: int, source_fps: float, sample_fps: float) -> np.ndarray:
    step = max(1, int(round(source_fps / sample_fps)))
    start = step // 2
    return np.arange(start, frame_count, step, dtype=np.int32)


def _iter_synchronized_frames(
    episode: Episode, selected: np.ndarray
) -> Iterable[tuple[int, dict[str, np.ndarray]]]:
    captures = {
        view: cv2.VideoCapture(str(getattr(episode, view)))
        for view in VIEW_DIRS
    }
    if not all(cap.isOpened() for cap in captures.values()):
        for cap in captures.values():
            cap.release()
        raise RuntimeError(f"Cannot open all views for {episode.episode_id}")
    selected_set = set(selected.tolist())
    max_index = int(selected[-1]) if selected.size else -1
    try:
        for frame_index in range(max_index + 1):
            decoded: dict[str, np.ndarray] = {}
            for view, cap in captures.items():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"Decode failed at frame {frame_index} ({view}) for {episode.episode_id}"
                    )
                decoded[view] = frame
            if frame_index in selected_set:
                yield frame_index, decoded
    finally:
        for cap in captures.values():
            cap.release()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.force:
        print(f"Feature cache already exists: {args.output}")
        return 0
    if args.image_size % 14 != 0:
        raise ValueError("--image-size must be divisible by DINOv2's patch size 14")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_root = args.data_root.resolve()
    model_repo = args.model_repo.resolve()
    discovered_episodes, incomplete = discover_episodes(data_root)
    validated_episodes, invalid = validate_episodes(discovered_episodes, args.max_episodes)
    if not validated_episodes:
        raise RuntimeError(f"No complete synchronized episodes found under {data_root}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = torch.hub.load(str(model_repo), args.model_name, source="local", pretrained=True)
    model.eval().requires_grad_(False).to(device)
    accumulator = FeatureAccumulator(model, device, args.image_size, args.batch_size)

    episode_ids: list[str] = []
    frame_indices: list[int] = []
    timestamps: list[float] = []
    source_fps_values: list[float] = []
    episode_manifest: list[dict[str, object]] = []
    start_time = time.time()
    sample_index = 0

    for episode_number, (episode, metadata) in enumerate(validated_episodes, start=1):
        fps_values = np.asarray([value[0] for value in metadata.values()], dtype=np.float64)
        if float(np.max(fps_values) - np.min(fps_values)) > 0.05:
            raise RuntimeError(f"FPS mismatch in {episode.episode_id}: {fps_values.tolist()}")
        source_fps = float(np.median(fps_values))
        frame_count = min(value[1] for value in metadata.values())
        selected = _selected_indices(frame_count, source_fps, args.sample_fps)
        episode_manifest.append(
            {
                "episode_id": episode.episode_id,
                "source_fps": source_fps,
                "common_frame_count": frame_count,
                "sample_count": int(selected.size),
                "paths": {view: str(getattr(episode, view).resolve()) for view in VIEW_DIRS},
                "resolutions": {
                    view: [metadata[view][2], metadata[view][3]] for view in VIEW_DIRS
                },
            }
        )
        for frame_index, decoded in _iter_synchronized_frames(episode, selected):
            for view in VIEW_DIRS:
                accumulator.add((sample_index, view), decoded[view])
            episode_ids.append(episode.episode_id)
            frame_indices.append(frame_index)
            timestamps.append(frame_index / source_fps)
            source_fps_values.append(source_fps)
            sample_index += 1
        accumulator.flush()
        elapsed = time.time() - start_time
        print(
            f"[{episode_number:02d}/{len(validated_episodes):02d}] {episode.episode_id} "
            f"samples={selected.size} total={sample_index} elapsed={elapsed:.1f}s",
            flush=True,
        )

    accumulator.flush()
    ordered_keys = [(index, view) for index in range(sample_index) for view in VIEW_DIRS]
    missing_keys = [key for key in ordered_keys if key not in accumulator.cls]
    if missing_keys:
        raise RuntimeError(f"Missing extracted features, first keys: {missing_keys[:5]}")

    arrays: dict[str, np.ndarray] = {
        "episode_id": np.asarray(episode_ids),
        "frame_index": np.asarray(frame_indices, dtype=np.int32),
        "timestamp_sec": np.asarray(timestamps, dtype=np.float32),
        "source_fps": np.asarray(source_fps_values, dtype=np.float32),
    }
    for view in VIEW_DIRS:
        arrays[f"{view}_cls"] = np.stack(
            [accumulator.cls[(index, view)] for index in range(sample_index)], axis=0
        )
        arrays[f"{view}_mean_patch"] = np.stack(
            [accumulator.patch[(index, view)] for index in range(sample_index)], axis=0
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    checkpoint = _find_cached_checkpoint(args.model_name)
    manifest = {
        "feature_cache": str(args.output.resolve()),
        "created_unix": time.time(),
        "data_root": str(data_root),
        "discovered_complete_path_count": len(discovered_episodes),
        "complete_episode_count": len(validated_episodes),
        "incomplete_episodes": incomplete,
        "invalid_episodes": invalid,
        "sample_count": sample_index,
        "image_count": sample_index * len(VIEW_DIRS),
        "sample_fps": args.sample_fps,
        "image_size": args.image_size,
        "model_name": args.model_name,
        "model_repo": str(model_repo),
        "model_repo_commit": _git_commit(model_repo),
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint else None,
        "feature_dtype": "float16",
        "feature_types": ["normalized_cls", "normalized_mean_patch"],
        "elapsed_sec": time.time() - start_time,
        "episodes": episode_manifest,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "feature_cache", "complete_episode_count", "sample_count", "image_count", "elapsed_sec"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
