#!/usr/bin/env python3
"""Export a synchronized VGGT sequence reconstruction to Rerun 0.26."""

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


VIEW_COLORS = {
    "front_head": [80, 180, 255],
    "left_wrist": [100, 230, 130],
    "right_wrist": [255, 150, 80],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--point-radius", type=float, default=2.0)
    return parser.parse_args()


def _set_equal_axes(ax: plt.Axes, points: np.ndarray) -> None:
    lower = np.quantile(points, 0.01, axis=0)
    upper = np.quantile(points, 0.99, axis=0)
    center = (lower + upper) * 0.5
    radius = max(float(np.max(upper - lower)) * 0.55, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] + radius, center[2] - radius)
    ax.set_zlim(-center[1] - radius, -center[1] + radius)
    ax.set_box_aspect((1, 1, 1))


def _save_preview(
    path: Path,
    rgb: np.ndarray,
    xyz: np.ndarray,
    colors: np.ndarray,
    valid: np.ndarray,
    view_names: list[str],
    source_frame: int,
    timestamp: float,
) -> None:
    figure = plt.figure(figsize=(15, 9), dpi=140)
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 1.45])
    for view_idx, view_name in enumerate(view_names):
        axis = figure.add_subplot(grid[0, view_idx])
        axis.imshow(rgb[view_idx])
        axis.set_title(view_name)
        axis.axis("off")
    axis_3d = figure.add_subplot(grid[1, :], projection="3d")
    all_points: list[np.ndarray] = []
    for view_idx in range(len(view_names)):
        mask = valid[view_idx]
        points = xyz[view_idx][mask]
        point_colors = colors[view_idx][mask].astype(np.float32) / 255.0
        all_points.append(points)
        axis_3d.scatter(
            points[:, 0], points[:, 2], -points[:, 1], c=point_colors, s=0.45, linewidths=0, alpha=0.9
        )
    combined = np.concatenate(all_points, axis=0)
    _set_equal_axes(axis_3d, combined)
    axis_3d.view_init(elev=18, azim=-68)
    axis_3d.set_xlabel("x")
    axis_3d.set_ylabel("z")
    axis_3d.set_zlabel("-y")
    axis_3d.set_title(
        f"VGGT fused three-view cloud · source frame {source_frame} · {timestamp:.2f}s"
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = np.load(args.input)
    view_names = [str(value) for value in data["view_names"].tolist()]
    frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    rgb = np.asarray(data["rgb"], dtype=np.uint8)
    xyz = np.asarray(data["xyz_sequence"], dtype=np.float32)
    colors = np.asarray(data["colors_rgb"] if "colors_rgb" in data else data["rgb"][:, :, ::3, ::3], dtype=np.uint8)
    confidence = np.asarray(data["confidence"], dtype=np.float32)
    valid = np.asarray(data["valid"], dtype=bool)
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
    camera_to_sequence = np.asarray(data["camera_to_sequence"], dtype=np.float32)

    if colors.shape[:4] != xyz.shape[:4]:
        raise RuntimeError(f"Point color shape {colors.shape} does not match XYZ {xyz.shape}")

    recording = rr.RecordingStream("vggt_synchronized_three_view_sequence")
    recording.save(str(args.output.resolve()))
    image_views = [
        rrb.Spatial2DView(name=view_name, origin=f"/inputs/{view_name}") for view_name in view_names
    ]
    recording.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(*image_views, column_shares=[1.0] * len(image_views)),
                rrb.Spatial3DView(name="VGGT three-view reconstruction", origin="/world"),
                row_shares=[0.8, 1.5],
            ),
            collapse_panels=True,
        )
    )
    recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    displayed_counts: list[list[int]] = []
    for time_idx in range(xyz.shape[0]):
        recording.set_time("source_frame", sequence=int(frame_indices[time_idx]))
        recording.set_time("video_time", timestamp=float(timestamps[time_idx]))
        frame_counts: list[int] = []
        for view_idx, view_name in enumerate(view_names):
            recording.log(f"inputs/{view_name}", rr.Image(rgb[time_idx, view_idx]).compress(jpeg_quality=90))
            mask = valid[time_idx, view_idx]
            frame_counts.append(int(np.count_nonzero(mask)))
            recording.log(
                f"world/points/{view_name}",
                rr.Points3D(
                    positions=xyz[time_idx, view_idx][mask],
                    colors=colors[time_idx, view_idx][mask],
                    radii=rr.Radius.ui_points(float(args.point_radius)),
                ),
            )
            camera = camera_to_sequence[time_idx, view_idx]
            camera_path = f"world/cameras/{view_name}"
            recording.log(
                camera_path,
                rr.Transform3D(translation=camera[:3, 3], mat3x3=camera[:3, :3]),
            )
            recording.log(
                camera_path,
                rr.Pinhole(
                    image_from_camera=intrinsics[time_idx, view_idx],
                    resolution=[rgb.shape[3], rgb.shape[2]],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=0.25,
                    color=VIEW_COLORS.get(view_name, [220, 220, 220]),
                    line_width=1.5,
                ),
            )
            median_conf = float(np.median(confidence[time_idx, view_idx][mask])) if np.any(mask) else 0.0
            recording.log(f"stats/confidence/{view_name}", rr.Scalars(median_conf))
        displayed_counts.append(frame_counts)
        recording.log("stats/points_total", rr.Scalars(sum(frame_counts)))
    recording.disconnect()

    if args.preview is not None:
        preview_idx = xyz.shape[0] // 2
        _save_preview(
            args.preview,
            rgb[preview_idx],
            xyz[preview_idx],
            colors[preview_idx],
            valid[preview_idx],
            view_names,
            int(frame_indices[preview_idx]),
            float(timestamps[preview_idx]),
        )

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "rrd": str(args.output.resolve()),
                "rerun_version": rr.__version__,
                "frames": int(xyz.shape[0]),
                "views": view_names,
                "sampled_points_per_view": int(np.prod(xyz.shape[2:4])),
                "displayed_points_per_frame_per_view": displayed_counts,
                "point_radius_ui": float(args.point_radius),
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
