# Project state

Last updated: 2026-09-01 UTC

## Objective

Build a frame-level pairwise metric for whether a generated head view and one
wrist view express the same robot/environment state. The shared model is run
once for the left wrist and once for the right wrist.

## Current implementation

- Typed `PairBatch` and `PairMetricOutput` contracts.
- Pluggable pairwise foundation backbones: DINOv2, VGGT, and a tiny test model.
- Task-specific cross-attention readouts for translation, rotation, joints,
  gripper, DINO residual, global compatibility energy, and validity.
- Masked multi-task loss.
- Episode/pair-oriented LeRobot adapter boundary using an offline pair plan.
- BF16/FP16 trainer with torchrun DDP, gradient accumulation, checkpoint/resume,
  distributed evaluation, and left/right diagnostics.
- Deterministic mock dataset for pipeline and DDP tests.
- An isolated DINOv2-B RGB-only pilot under `pilots/dinov2_visual_probe`.

## Not yet complete

- Real LeRobot feature-key mapping must be validated against the incoming data.
- Pair-plan mining/preprocessing for large LeRobot datasets is not implemented.
- DINO teacher feature preprocessing and train-only PCA projection are not yet
  integrated with LeRobot shards.
- Validity/observability has a model head but no labels or loss yet.
- FSDP is not implemented; DDP is ready. Full fine-tuning of billion-parameter
  backbones may require FSDP depending on per-GPU memory.
- Physical-state tests await real EEF/joint/gripper observations.

## Next executable steps

1. When LeRobot data arrives, audit `meta.info.features`, fix schema keys, build
   episode-level splits and versioned pair plans, then run label statistics.
2. Precompute train-only projected DINO teacher features in LeRobot-compatible
   shards and validate state-conditioned negative strata.
3. Launch DINOv2-L/G and VGGT-1B patch-token ablations with identical plans.
