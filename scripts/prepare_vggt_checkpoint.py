#!/usr/bin/env python3
"""Convert the public VGGT-1B safetensors checkpoint to compact BF16 weights."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    source = load_file(str(args.input), device="cpu")
    compact: dict[str, torch.Tensor] = {}
    for name, tensor in source.items():
        compact[name] = tensor.to(torch.bfloat16).contiguous() if tensor.is_floating_point() else tensor.contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(compact, str(args.output), metadata={"precision": "bfloat16", "source": "facebook/VGGT-1B"})
    digest = _sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    total_bytes = sum(t.numel() * t.element_size() for t in compact.values())
    print(f"Saved {len(compact)} tensors ({total_bytes / 1024**3:.2f} GiB): {args.output}")
    print(f"SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
