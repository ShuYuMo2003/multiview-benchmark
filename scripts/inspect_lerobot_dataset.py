#!/usr/bin/env python3
"""Print the authoritative schema needed to configure the LeRobot adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--revision")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from lerobot.datasets import LeRobotDatasetMetadata
    except ImportError as exc:
        raise RuntimeError("Install the LeRobot version matching the dataset first") from exc
    kwargs = {"repo_id": args.repo_id, "root": args.root, "revision": args.revision}
    metadata = LeRobotDatasetMetadata(**{key: value for key, value in kwargs.items() if value is not None})
    result = {
        "repo_id": args.repo_id,
        "root": str(metadata.root),
        "codebase_version": metadata.info.get("codebase_version"),
        "robot_type": metadata.robot_type,
        "fps": metadata.fps,
        "total_episodes": metadata.total_episodes,
        "total_frames": metadata.total_frames,
        "video_keys": list(metadata.video_keys),
        "features": metadata.features,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
