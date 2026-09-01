# Experiment log

Append new entries; do not rewrite previous results.

## 2026-09-01 — DINOv2-B pipeline smoke test

- Purpose: validate video discovery, synchronized sampling, official DINOv2
  loading, feature caching, PCA, vector/scalar/energy probe code.
- Data: four decodable example episodes, 258 synchronized samples at 1 FPS.
- Temporary artifacts: `/tmp/dino_multiview_smoke.npz` and
  `/tmp/dino_multiview_probe_smoke/`.
- Outcome: end-to-end code completed. Metrics are intentionally not interpreted
  because the smoke split had only one test episode and eight training epochs.
- Data finding: at least one path-complete MP4 triplet contained a corrupt video
  (`moov atom not found`). The extractor now validates first/last decode and logs
  invalid episodes.

## 2026-09-01 — Production training-stack validation

- Five core unit tests passed: model/loss backward, perfect-metric sanity,
  SO(3), LeRobot row mapping, and checkpoint round-trip.
- One-GPU BF16 mock training completed two epochs with evaluation and atomic
  checkpoints. Mean training loss changed from 0.7902 to 0.7015.
- Two-GPU torchrun/DDP completed one epoch, including DistributedSampler,
  gradient accumulation with `no_sync`, static graph, and cross-rank metrics.
- Two-GPU resume loaded epoch 1 / global step 4 and completed epoch 2.
- Production DINOv2-B patch-token model forward passed with all seven outputs.
- Production VGGT-1B pair forward passed at 518x518. Model size was 909,581,488
  parameters and peak allocated CUDA memory for the frozen B=1 inference smoke
  was approximately 3.58 GiB.

## 2026-09-01 — 69-episode example-video DINOv2 visual pilot

- Scope: example MP4 data only; not formal LeRobot benchmark evidence.
- Extraction: 69 decodable synchronized episodes, 8,578 triplets / 25,734
  images at 336 px and 2 FPS. One corrupt path-complete episode was excluded.
- Split: 50 train, 10 validation, 9 test episodes; at most 160 evenly spaced
  frames per episode for PCA/probes.
- At 128 PCA dimensions, wrist distance Spearman was 0.702 for CLS, 0.731 for
  mean-patch, and 0.727 for their concatenation. Raising to 256 dimensions only
  reached 0.708, 0.739, and 0.733 respectively.
- Mean-patch was the best probe representation: head-only aligned-wrist cosine
  0.404 and R2 0.042; vector norm Spearman 0.406 / mismatch AUROC 0.589; scalar
  Spearman 0.439 / AUROC 0.592; energy AUROC 0.607.
- CLS and CLS+mean were weaker: energy AUROC 0.597 and 0.604 respectively.
- All probes generalized poorly across episodes and selected epoch 1 or 2;
  training loss continued improving while validation degraded. Predictions
  separated cross-episode candidates much more strongly than within-episode
  temporal offsets.
- Conclusion: a global frozen feature plus small MLP is an inadequate final
  model. The visual residual remains useful, but production work should use
  patch-token fusion, larger foundation backbones, real robot-state supervision,
  and hard-negative/state-equivalent pair plans.
- Artifacts: `scratch/dinov2_visual_probe/example_69ep_pilot*`.

## 2026-09-01 — Initial repository snapshot

The reproducible engineering state was prepared for publication at
`https://github.com/ShuYuMo2003/multiview-benchmark.git`. The snapshot includes
the production package, DDP trainer, mock and LeRobot data adapters, model
templates, tests, pilot scripts, and the complete `memory/` record.

Local datasets, MP4 examples, model weights, caches, virtual environments,
third-party source trees, and generated experiment outputs are deliberately
excluded. Exact validated third-party revisions are recorded in
`THIRD_PARTY.md` so they can be restored without committing nested repositories
or multi-gigabyte artifacts.
