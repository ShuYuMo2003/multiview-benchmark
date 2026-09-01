"""Dataset-factory entrypoint used by JSON training configurations."""

from __future__ import annotations

from typing import Any

from .lerobot_pair import LeRobotPairDataset, LeRobotSchema, PairPlan


def build_datasets(config: dict[str, Any]):
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not installed in this environment. Install the version "
            "matching the dataset metadata before launching formal training."
        ) from exc
    dataset_args = {
        "repo_id": config["repo_id"],
        "root": config.get("root"),
        "revision": config.get("revision"),
        "download_videos": bool(config.get("download_videos", True)),
        "video_backend": config.get("video_backend"),
    }
    dataset_args = {key: value for key, value in dataset_args.items() if value is not None}
    base_dataset = LeRobotDataset(**dataset_args)
    schema = LeRobotSchema(**config["schema"])
    common = {
        "dataset": base_dataset,
        "schema": schema,
        "joint_dim": int(config["joint_dim"]),
        "dino_dim": int(config["dino_dim"]),
    }
    train = LeRobotPairDataset(
        pair_plan=PairPlan.load(config["train_pair_plan"]), **common
    )
    validation = LeRobotPairDataset(
        pair_plan=PairPlan.load(config["val_pair_plan"]), **common
    )
    return train, validation
