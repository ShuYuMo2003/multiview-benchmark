#!/usr/bin/env python3
"""torchrun entrypoint for pairwise multiview metric training."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvbench.checkpoint import load_checkpoint
from mvbench.data import MockPairDataset
from mvbench.distributed import finalize_distributed, initialize_distributed, wrap_ddp
from mvbench.losses import MultitaskConsistencyLoss
from mvbench.models import PairwiseStateConsistencyModel
from mvbench.trainer import Trainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def import_object(path: str):
    module_name, object_name = path.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), object_name)


def build_datasets(config: dict[str, Any]):
    name = config["name"]
    if name == "mock":
        common = {
            "frames_per_episode": int(config.get("frames_per_episode", 32)),
            "image_size": int(config.get("image_size", 56)),
            "joint_dim": int(config.get("joint_dim", 7)),
            "dino_dim": int(config.get("dino_dim", 16)),
        }
        return (
            MockPairDataset(
                episodes=int(config.get("train_episodes", 8)),
                seed=int(config.get("seed", 17)),
                **common,
            ),
            MockPairDataset(
                episodes=int(config.get("val_episodes", 2)),
                seed=int(config.get("seed", 17)) + 10000,
                **common,
            ),
        )
    if name == "factory":
        factory = import_object(config["factory"])
        return factory(config)
    raise ValueError(f"Unknown dataset configuration: {name}")


def worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def parameter_groups(model: torch.nn.Module, config: dict[str, Any]):
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters() if parameter.requires_grad}
    backbone = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) in backbone_ids]
    heads = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in backbone_ids]
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": float(config.get("backbone_lr", config["lr"])), "name": "backbone"})
    if heads:
        groups.append({"params": heads, "lr": float(config["lr"]), "name": "fusion_and_heads"})
    return groups


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if args.output_dir is not None:
        config["trainer"]["output_dir"] = str(args.output_dir)
    if args.epochs is not None:
        config["trainer"]["epochs"] = args.epochs
    context = initialize_distributed(config.get("distributed", {}).get("backend"))
    seed = int(config.get("seed", 17)) + context.rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_dataset, val_dataset = build_datasets(config["data"])
    loader_config = config.get("loader", {})
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=context.world_size, rank=context.rank,
        shuffle=True, seed=int(config.get("seed", 17)), drop_last=True,
    ) if context.distributed else None
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=context.world_size, rank=context.rank,
        shuffle=False, drop_last=False,
    ) if context.distributed else None
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": int(loader_config.get("batch_size", 8)),
        "num_workers": int(loader_config.get("workers", 4)),
        "pin_memory": bool(loader_config.get("pin_memory", True)),
        "persistent_workers": bool(loader_config.get("persistent_workers", True)) and int(loader_config.get("workers", 4)) > 0,
        "worker_init_fn": worker_init,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, shuffle=train_sampler is None,
        drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, sampler=val_sampler, shuffle=False,
        drop_last=False, **loader_kwargs,
    )

    base_model = PairwiseStateConsistencyModel.from_config(config["model"])
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        parameter_groups(base_model, optimizer_config),
        lr=float(optimizer_config["lr"]),
        betas=tuple(optimizer_config.get("betas", (0.9, 0.95))),
        weight_decay=float(optimizer_config.get("weight_decay", 0.05)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["trainer"].get("epochs", 10)),
        eta_min=float(optimizer_config.get("min_lr", 0.0)),
    )
    model = wrap_ddp(
        base_model, context,
        find_unused_parameters=bool(config.get("distributed", {}).get("find_unused_parameters", False)),
        static_graph=bool(config.get("distributed", {}).get("static_graph", False)),
    )
    loss_fn = MultitaskConsistencyLoss.from_config(config.get("loss", {}))
    trainer_config = TrainerConfig(**config["trainer"])
    trainer = Trainer(
        model, loss_fn, optimizer, scheduler, train_loader, val_loader,
        context, trainer_config, config,
    )
    if args.resume is not None:
        payload = load_checkpoint(
            args.resume, model, optimizer, scheduler, trainer.scaler, strict=True, restore_rng=True
        )
        trainer.start_epoch = int(payload["epoch"])
        trainer.global_step = int(payload["global_step"])
        if context.is_main:
            print(json.dumps({"resumed": str(args.resume), "epoch": trainer.start_epoch, "step": trainer.global_step}))
    try:
        trainer.fit()
    finally:
        finalize_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
