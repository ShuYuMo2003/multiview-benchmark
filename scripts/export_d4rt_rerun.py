#!/usr/bin/env python3
"""Convert a D4RT reconstruction NPZ into an interactive Rerun recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rerun as rr
import rerun.blueprint as rrb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export D4RT point clouds to .rrd.")
    parser.add_argument("--input", type=Path, required=True, help="Input reconstruction .npz.")
    parser.add_argument("--output", type=Path, required=True, help="Output Rerun .rrd.")
    parser.add_argument("--preview", type=Path, default=None, help="Optional PNG preview.")
    parser.add_argument("--view-name", default="camera", help="Camera/view label used in the Rerun blueprint.")
    parser.add_argument(
        "--confidence-quantile",
        type=float,
        default=0.10,
        help="Per-frame confidence quantile removed from the displayed cloud.",
    )
    parser.add_argument("--distance-quantile", type=float, default=0.995)
    parser.add_argument(
        "--point-radius",
        type=float,
        default=3.0,
        help="Displayed point radius in Rerun UI points.",
    )
    return parser.parse_args()


def _display_mask(
    points: np.ndarray,
    finite: np.ndarray,
    confidence: np.ndarray,
    confidence_quantile: float,
    distance_quantile: float,
) -> np.ndarray:
    mask = finite & np.isfinite(points).all(axis=-1) & np.isfinite(confidence)
    if not np.any(mask):
        return mask
    conf_cutoff = float(np.quantile(confidence[mask], np.clip(confidence_quantile, 0.0, 0.95)))
    mask &= confidence >= conf_cutoff
    if np.count_nonzero(mask) < 8:
        return mask
    center = np.median(points[mask], axis=0)
    distance = np.linalg.norm(points - center[None, :], axis=-1)
    distance_cutoff = float(np.quantile(distance[mask], np.clip(distance_quantile, 0.5, 1.0)))
    return mask & (distance <= distance_cutoff)


def _set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    lo = np.quantile(points, 0.01, axis=0)
    hi = np.quantile(points, 0.99, axis=0)
    center = (lo + hi) * 0.5
    radius = max(float(np.max(hi - lo)) * 0.55, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] + radius, center[1] - radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _save_preview(
    path: Path,
    frame: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    frame_index: int,
    timestamp: float,
    view_name: str,
) -> None:
    fig = plt.figure(figsize=(14, 6), dpi=150)
    ax_image = fig.add_subplot(1, 2, 1)
    ax_image.imshow(frame)
    ax_image.set_title(f"{view_name} RGB · sampled frame {frame_index} · {timestamp:.2f}s")
    ax_image.axis("off")

    ax_cloud = fig.add_subplot(1, 2, 2, projection="3d")
    ax_cloud.scatter(
        points[:, 0], points[:, 2], -points[:, 1], c=colors.astype(np.float32) / 255.0, s=2.5, linewidths=0
    )
    plot_points = np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=-1)
    _set_equal_axes(ax_cloud, plot_points)
    ax_cloud.view_init(elev=18, azim=-70)
    ax_cloud.set_title("OpenD4RT point cloud · common ref0 frame")
    ax_cloud.set_xlabel("x")
    ax_cloud.set_ylabel("z")
    ax_cloud.set_zlabel("-y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(args.input)
    video_rgb = np.asarray(data["video_rgb"], dtype=np.uint8)
    frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    xyz = np.asarray(data["xyz_ref0"], dtype=np.float32)
    finite = np.asarray(data["finite_mask"], dtype=bool)
    confidence = np.asarray(data["confidence"], dtype=np.float32)
    colors = np.asarray(data["colors_rgb"], dtype=np.uint8)

    recording = rr.RecordingStream(f"open_d4rt_{args.view_name}_4d")
    recording.save(str(args.output.resolve()))
    recording.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name=f"{args.view_name} RGB", origin="/camera"),
                rrb.Spatial3DView(name="D4RT 4D point cloud", origin="/world"),
                column_shares=[1, 1],
            ),
            collapse_panels=True,
        )
    )
    recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    kept_counts: list[int] = []
    masks: list[np.ndarray] = []
    for i in range(xyz.shape[0]):
        mask = _display_mask(
            xyz[i],
            finite[i],
            confidence[i],
            confidence_quantile=float(args.confidence_quantile),
            distance_quantile=float(args.distance_quantile),
        )
        masks.append(mask)
        kept_counts.append(int(np.count_nonzero(mask)))
        recording.set_time("sample_frame", sequence=i)
        recording.set_time("video_time", timestamp=float(timestamps[i]))
        recording.log("camera/rgb", rr.Image(video_rgb[i]))
        recording.log(
            "world/points",
            rr.Points3D(
                positions=xyz[i, mask],
                colors=colors[i, mask],
                radii=rr.Radius.ui_points(float(args.point_radius)),
            ),
        )
        recording.log("stats/confidence", rr.Scalars(float(np.median(confidence[i, mask])) if np.any(mask) else 0.0))
        recording.log("stats/source_frame", rr.Scalars(int(frame_indices[i])))
    recording.disconnect()

    if args.preview is not None:
        preview_idx = len(masks) // 2
        preview_mask = masks[preview_idx]
        _save_preview(
            args.preview,
            video_rgb[preview_idx],
            xyz[preview_idx, preview_mask],
            colors[preview_idx, preview_mask],
            int(frame_indices[preview_idx]),
            float(timestamps[preview_idx]),
            str(args.view_name),
        )

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "rrd": str(args.output.resolve()),
                "rerun_version": rr.__version__,
                "frames": int(xyz.shape[0]),
                "raw_points_per_frame": int(xyz.shape[1]),
                "displayed_points_per_frame": kept_counts,
                "confidence_quantile": float(args.confidence_quantile),
                "distance_quantile": float(args.distance_quantile),
                "point_radius_ui": float(args.point_radius),
                "view_name": str(args.view_name),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved Rerun recording: {args.output.resolve()}")
    print(f"Saved export summary:  {summary_path.resolve()}")
    if args.preview is not None:
        print(f"Saved PNG preview:     {args.preview.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
