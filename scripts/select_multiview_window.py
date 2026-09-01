#!/usr/bin/env python3
"""Select a synchronized, motion-rich window from three camera videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


VIEW_NAMES = ("front_head", "left_wrist", "right_wrist")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, nargs=3, required=True)
    parser.add_argument("--window-sec", type=float, default=2.0666667)
    parser.add_argument("--edge-margin-sec", type=float, default=0.5)
    parser.add_argument("--contact-sheet", type=Path)
    return parser.parse_args()


def _read_signal(path: Path) -> tuple[float, int, np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    advertised_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    motion: list[float] = []
    brightness: list[float] = []
    previous: np.ndarray | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        brightness.append(float(np.mean(gray)))
        motion.append(0.0 if previous is None else float(np.mean(cv2.absdiff(gray, previous))))
        previous = gray
    cap.release()
    if not motion:
        raise RuntimeError(f"No frames decoded from {path}")
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    return fps, advertised_frames, np.asarray(motion), np.asarray(brightness)


def _rolling_mean(values: np.ndarray, length: int) -> np.ndarray:
    prefix = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(values, dtype=np.float64)])
    return (prefix[length:] - prefix[:-length]) / float(length)


def _decode_frame(path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot decode frame {frame_index} from {path}")
    return frame


def _write_contact_sheet(
    path: Path,
    videos: list[Path],
    fps: float,
    start_frame: int,
    end_frame: int,
) -> None:
    sample_ids = np.rint(np.linspace(start_frame, end_frame, 5)).astype(int)
    cell_w, cell_h = 384, 216
    canvas = np.full((cell_h * 3, cell_w * 5, 3), 245, dtype=np.uint8)
    for row, (view, video) in enumerate(zip(VIEW_NAMES, videos, strict=True)):
        for col, frame_idx in enumerate(sample_ids.tolist()):
            frame = cv2.resize(_decode_frame(video, frame_idx), (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            cv2.putText(
                frame,
                f"{view}  {frame_idx / fps:.2f}s",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{view}  {frame_idx / fps:.2f}s",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            canvas[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = frame
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Failed to write {path}")


def main() -> int:
    args = parse_args()
    videos = list(args.videos)
    signals = [_read_signal(path) for path in videos]
    fps_values = np.asarray([item[0] for item in signals], dtype=np.float64)
    if float(np.max(fps_values) - np.min(fps_values)) > 0.05:
        raise RuntimeError(f"Camera FPS mismatch: {fps_values.tolist()}")
    fps = float(np.median(fps_values))
    num_frames = min(int(item[2].shape[0]) for item in signals)
    window_frames = int(round(float(args.window_sec) * fps)) + 1
    if num_frames < window_frames:
        raise RuntimeError(f"Only {num_frames} synchronized frames; need {window_frames}")

    normalized_motion: list[np.ndarray] = []
    brightness: list[np.ndarray] = []
    for _, _, motion, light in signals:
        motion = motion[:num_frames]
        scale = max(float(np.quantile(motion[1:], 0.90)), 1e-6)
        normalized_motion.append(np.clip(motion / scale, 0.0, 2.0))
        brightness.append(light[:num_frames])
    combined_motion = np.mean(np.stack(normalized_motion, axis=0), axis=0)
    combined_light = np.mean(np.stack(brightness, axis=0), axis=0)
    score = _rolling_mean(combined_motion, window_frames)
    light = _rolling_mean(combined_light, window_frames)
    score *= np.clip((light - 8.0) / 24.0, 0.0, 1.0)

    margin = int(round(float(args.edge_margin_sec) * fps))
    valid_start = margin
    valid_end = max(valid_start + 1, score.shape[0] - margin)
    start_frame = valid_start + int(np.argmax(score[valid_start:valid_end]))
    end_frame = start_frame + window_frames - 1
    if args.contact_sheet is not None:
        _write_contact_sheet(args.contact_sheet, videos, fps, start_frame, end_frame)

    result = {
        "videos": {name: str(path.resolve()) for name, path in zip(VIEW_NAMES, videos, strict=True)},
        "fps": fps,
        "synchronized_frames": num_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_sec": start_frame / fps,
        "end_sec": end_frame / fps,
        "motion_score": float(score[start_frame]),
        "mean_brightness": float(light[start_frame]),
        "contact_sheet": str(args.contact_sheet.resolve()) if args.contact_sheet is not None else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
