"""Minimal, explicit torchrun/DDP lifecycle helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(backend: str | None = None) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        selected_backend = backend or "nccl"
    else:
        device = torch.device("cpu")
        selected_backend = backend or "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=selected_backend, init_method="env://")
    return DistributedContext(rank, local_rank, world_size, device)


def wrap_ddp(
    model: torch.nn.Module,
    context: DistributedContext,
    find_unused_parameters: bool = False,
    static_graph: bool = False,
) -> torch.nn.Module:
    model = model.to(context.device)
    if not context.distributed:
        return model
    device_ids = [context.local_rank] if context.device.type == "cuda" else None
    return DistributedDataParallel(
        model,
        device_ids=device_ids,
        output_device=context.local_rank if device_ids else None,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
        gradient_as_bucket_view=True,
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_scalars(values: dict[str, torch.Tensor], context: DistributedContext) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in values.items():
        tensor = value.detach().float().to(context.device)
        if tensor.numel() != 1:
            tensor = tensor.mean()
        if context.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= context.world_size
        result[name] = float(tensor.cpu())
    return result


def barrier(context: DistributedContext) -> None:
    if context.distributed:
        dist.barrier()


def finalize_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
