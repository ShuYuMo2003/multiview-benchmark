#!/usr/bin/env python3
"""Build a compact 3×3 overview from per-view D4RT preview images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


EPISODES = ("place_nameplate", "adjust_chair", "open_washer")
VIEWS = ("front_head", "left_wrist", "right_wrist")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cell_w, cell_h = 640, 300
    canvas = np.full((cell_h * len(EPISODES), cell_w * len(VIEWS), 3), 245, dtype=np.uint8)
    for row, episode in enumerate(EPISODES):
        for col, view in enumerate(VIEWS):
            source = args.root / episode / view / "preview.png"
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(source)
            image = cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            label = f"{episode} / {view}"
            cv2.rectangle(image, (0, 0), (cell_w, 34), (15, 15, 15), thickness=-1)
            cv2.putText(
                image,
                label,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            canvas[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = image
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise RuntimeError(f"Failed to write {args.output}")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
