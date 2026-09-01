#!/usr/bin/env python3
"""Extract model-only OpenD4RT weights and optionally convert them to FP16."""

from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip OpenD4RT training state and create a compact inference checkpoint."
    )
    parser.add_argument("--input", type=Path, required=True, help="Released OpenD4RT checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Compact model-only checkpoint.")
    parser.add_argument(
        "--precision",
        choices=("float16", "float32"),
        default="float16",
        help="Floating-point tensor precision in the compact checkpoint.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    payload = torch.load(args.input, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError("Expected the released checkpoint to contain a 'model' state dict.")

    dtype = torch.float16 if args.precision == "float16" else torch.float32
    compact: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in payload["model"].items():
        if not torch.is_tensor(tensor):
            raise TypeError(f"Non-tensor entry in model state dict: {name}")
        compact[name] = tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, args.output)
    digest = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    total_bytes = sum(t.numel() * t.element_size() for t in compact.values())
    print(f"Saved {len(compact)} tensors ({total_bytes / 1024**3:.2f} GiB): {args.output}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
