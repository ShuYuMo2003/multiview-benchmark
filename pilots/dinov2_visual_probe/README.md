# DINOv2 visual probe

This directory contains the deliberately limited RGB-only feasibility pilot.
It is not the production training pipeline under `src/mvbench`.

The pilot uses frozen global DINOv2 features to test:

- PCA distance preservation;
- head-only aligned-wrist feature recovery;
- vector residual prediction versus a zero baseline;
- scalar visual-gap prediction;
- direct pair-compatibility classification.

All default artifacts go to `scratch/dinov2_visual_probe/`, keeping temporary
features, probe weights, and reports separate from formal model runs.

Run feature extraction from the repository root:

```bash
TORCH_HOME=$PWD/.cache/torch .venv-vggt/bin/python \
  pilots/dinov2_visual_probe/extract_features.py
```

Then run the held-out probe:

```bash
.venv-vggt/bin/python pilots/dinov2_visual_probe/analyze_probe.py
```

The probe has no EEF/gripper telemetry. Temporal offsets are candidate
mismatches, not guaranteed physical-state mismatches. Its result can guide
feature and architecture choices but cannot validate the final benchmark.
