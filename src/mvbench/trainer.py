"""BF16/FP16 trainer with gradient accumulation, DDP, eval, and resume."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import save_checkpoint
from .contracts import PairBatch
from .distributed import DistributedContext, reduce_scalars
from .metrics import EvaluationAccumulator


@dataclass
class TrainerConfig:
    epochs: int = 10
    precision: str = "bf16"
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.0
    log_every_steps: int = 20
    eval_every_epochs: int = 1
    checkpoint_every_epochs: int = 1
    output_dir: str = "runs/default"
    max_eval_samples: int = 200_000


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: Iterable[dict[str, torch.Tensor]],
        val_loader: Iterable[dict[str, torch.Tensor]],
        context: DistributedContext,
        config: TrainerConfig,
        raw_config: dict[str, Any],
        start_epoch: int = 0,
        global_step: int = 0,
    ):
        if config.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported precision: {config.precision}")
        self.model = model
        self.loss_fn = loss_fn.to(context.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.context = context
        self.config = config
        self.raw_config = raw_config
        self.start_epoch = start_epoch
        self.global_step = global_step
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.precision == "fp16" and context.device.type == "cuda"
        )
        self.output_dir = Path(config.output_dir)
        if context.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _autocast(self):
        if self.config.precision == "fp32" or self.context.device.type != "cuda":
            return nullcontext()
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def fit(self) -> dict[str, float]:
        final_metrics: dict[str, float] = {}
        for epoch in range(self.start_epoch, self.config.epochs):
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            train_metrics = self.train_epoch(epoch)
            if self.context.is_main:
                print(json.dumps({"epoch": epoch + 1, "train": train_metrics}), flush=True)
            if (epoch + 1) % self.config.eval_every_epochs == 0:
                final_metrics = self.evaluate()
                if self.context.is_main:
                    print(json.dumps({"epoch": epoch + 1, "validation": final_metrics}), flush=True)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.context.is_main and (epoch + 1) % self.config.checkpoint_every_epochs == 0:
                save_checkpoint(
                    self.output_dir / f"epoch_{epoch + 1:04d}.pt",
                    self.model, self.optimizer, self.scheduler, self.scaler,
                    epoch=epoch + 1, global_step=self.global_step,
                    config=self.raw_config, extra={"validation": final_metrics},
                )
                save_checkpoint(
                    self.output_dir / "latest.pt",
                    self.model, self.optimizer, self.scheduler, self.scaler,
                    epoch=epoch + 1, global_step=self.global_step,
                    config=self.raw_config, extra={"validation": final_metrics},
                )
        return final_metrics

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        batches = 0
        started = time.time()
        accumulation = self.config.gradient_accumulation_steps
        loader_length = len(self.train_loader) if hasattr(self.train_loader, "__len__") else None
        final_window = loader_length % accumulation if loader_length is not None else 0
        for batch_index, mapping in enumerate(self.train_loader):
            batch = PairBatch.from_mapping(mapping).to(self.context.device)
            batch.validate()
            is_last_batch = loader_length is not None and batch_index + 1 == loader_length
            should_step = (batch_index + 1) % accumulation == 0 or is_last_batch
            loss_divisor = (
                final_window
                if final_window and loader_length is not None and batch_index >= loader_length - final_window
                else accumulation
            )
            synchronization = (
                self.model.no_sync()
                if isinstance(self.model, DistributedDataParallel) and not should_step
                else nullcontext()
            )
            with synchronization, self._autocast():
                output = self.model(batch.head_image, batch.wrist_image, batch.side)
                loss, diagnostics = self.loss_fn(output, batch)
                scaled_loss = loss / loss_divisor
            self.scaler.scale(scaled_loss).backward()
            if should_step:
                if self.config.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
            reduced = reduce_scalars(diagnostics, self.context)
            for name, value in reduced.items():
                totals[name] = totals.get(name, 0.0) + value
            batches += 1
            if self.context.is_main and self.global_step > 0 and self.global_step % self.config.log_every_steps == 0 and should_step:
                print(json.dumps({
                    "epoch": epoch + 1, "step": self.global_step,
                    "loss": reduced.get("loss"), "elapsed_sec": time.time() - started,
                }), flush=True)
        # Iterables without __len__ cannot identify their last partial window in
        # advance, so flush it here. Sized DataLoaders step on their final batch.
        if loader_length is None and batches % accumulation != 0:
            if self.config.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
        return {name: value / max(1, batches) for name, value in totals.items()}

    @torch.inference_mode()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        accumulator = EvaluationAccumulator(
            max_samples=self.config.max_eval_samples,
            seed=int(self.raw_config.get("seed", 17)),
        )
        for mapping in self.val_loader:
            batch = PairBatch.from_mapping(mapping).to(self.context.device)
            with self._autocast():
                output = self.model(batch.head_image, batch.wrist_image, batch.side)
            accumulator.update(output, batch)
        return accumulator.compute(distributed=True)
