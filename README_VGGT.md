# VGGT synchronized three-view reconstruction

This pipeline reconstructs one 3D scene for every synchronized source frame from:

- `front_head`
- `left_wrist`
- `right_wrist`

The model input is `B=1, S=3`: the three camera images at the same source-frame index are one
multi-view sample. They are not treated as three unrelated batch elements. The implementation uses
VGGT's camera and depth heads, then unprojects the predicted depth with the predicted intrinsics and
extrinsics. This follows the upstream recommendation that depth + camera unprojection is generally
more accurate than the direct point-map head.

## Installed components

- Upstream source: `third_party/vggt`
- Pinned commit: `a288dd0f14786c93483e45524328726ab7b1b4ce`
- Environment: `.venv-vggt`
- Checkpoint: `third_party/vggt/checkpoints/vggt-1b-bf16.safetensors`
- Checkpoint SHA-256: `3ec7f6343257df297e6856942ac803adf82530181da3be69d8dbfe48fce76f08`
- Rerun writer/viewer: `0.26.2` from the `transform_env` Conda environment

The local checkpoint is a BF16 disk conversion of the public `facebook/VGGT-1B` safetensors file.
Model parameters and input tensors remain FP32 in memory where VGGT disables autocast; the upstream
aggregator runs under BF16 autocast.

## Data handling

For the `open_washer` sample, the source videos are 30 FPS and frames 112 through 174 inclusive are
processed (63 synchronized triplets, 3.7333 s through 5.8000 s).

Every image is resized to width 518 with dimensions divisible by 14. `front_head` becomes 392x518;
the wrist views become 294x518 and are center-padded with white to 392x518 as in upstream VGGT.
Padding is explicitly excluded from the point cloud.

Each triplet has its own arbitrary VGGT world frame. To make the result playable as a 4D sequence,
the implementation tracks static-looking features in consecutive `front_head` frames, lifts the
matches with predicted depth, robustly estimates a 3D Sim(3) transform with RANSAC, and chains each
frame into the first triplet's coordinate frame.

## Reproduce the dense result

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv-vggt/bin/python scripts/run_vggt_multiview_sequence.py \
  --videos \
  real_data/Multiview_Benchmark/source_videos/FoundationModel/sportsmeeting4_3_2/open_washing_machine_door/34730/jiangyifufangruxiyiji_R001ABDD22AA0020_20260527_sportsmeeting4_3_2_34730_37486/videos/chunk-000/observation.images.front_head/episode_000020.mp4 \
  real_data/Multiview_Benchmark/source_videos/FoundationModel/sportsmeeting4_3_2/open_washing_machine_door/34730/jiangyifufangruxiyiji_R001ABDD22AA0020_20260527_sportsmeeting4_3_2_34730_37486/videos/chunk-000/observation.images.left_wrist/episode_000020.mp4 \
  real_data/Multiview_Benchmark/source_videos/FoundationModel/sportsmeeting4_3_2/open_washing_machine_door/34730/jiangyifufangruxiyiji_R001ABDD22AA0020_20260527_sportsmeeting4_3_2_34730_37486/videos/chunk-000/observation.images.right_wrist/episode_000020.mp4 \
  --checkpoint third_party/vggt/checkpoints/vggt-1b-bf16.safetensors \
  --output outputs/vggt_open_washer/vggt_three_view_sequence_dense.npz \
  --start-sec 3.7333333333 \
  --end-sec 5.8 \
  --frame-step 1 \
  --point-stride 2 \
  --device cuda:0 \
  --precision bfloat16

MPLCONFIGDIR=/tmp/mpl_vggt \
  /mnt/home/sujiayi/miniconda3/envs/transform_env/bin/python \
  scripts/export_vggt_rerun.py \
  --input outputs/vggt_open_washer/vggt_three_view_sequence_dense.npz \
  --output outputs/vggt_open_washer/vggt_three_view_sequence_dense.rrd \
  --preview outputs/vggt_open_washer/preview_dense.png \
  --point-radius 1.5
```

Open the compatible recording with:

```bash
/mnt/home/sujiayi/miniconda3/envs/transform_env/bin/rerun \
  outputs/vggt_open_washer/vggt_three_view_sequence_dense.rrd
```

The recording contains three synchronized RGB panels, three separately selectable colored point
clouds, predicted camera frustums, per-view confidence, and both `source_frame` and `video_time`
timelines.

## Current quality check

- Dense export: about 111k to 126k retained points per frame after confidence, padding, and far-outlier filtering.
- Temporal registration: 62/62 non-initial frames used the robust Sim(3) path.
- Median / minimum RANSAC inlier ratio: 98.6% / 96.5%.
- Per-step scale range: 0.969 to 1.031; first-to-last cumulative scale is about 0.952.
- First-to-last accumulated translation is about 0.072 in the model's arbitrary scene units.
- The complete `.rrd` was successfully decoded by Rerun 0.26.2.

These are internal consistency checks, not metric accuracy against calibrated ground truth. The
current VGGT result has recognizable washer, door, floor, and robot geometry without pose jumps, so
VGGT-Omega was not activated. Omega is the next fallback when visible double surfaces, dynamic-object
tearing, or camera jumps remain after filtering; its 512px checkpoint requires approved Hugging Face
access.

## Relevant files

- `scripts/run_vggt_multiview_sequence.py`: synchronized reconstruction and temporal alignment
- `scripts/export_vggt_rerun.py`: Rerun 0.26 writer and PNG preview
- `scripts/prepare_vggt_checkpoint.py`: deterministic FP32-to-BF16 checkpoint conversion
- `outputs/vggt_open_washer/vggt_three_view_sequence_dense.json`: inference and alignment metadata
- `outputs/vggt_open_washer/vggt_three_view_sequence_dense.summary.json`: Rerun export summary

Upstream references: [VGGT](https://github.com/facebookresearch/vggt),
[VGGT-1B checkpoint](https://huggingface.co/facebook/VGGT-1B), and
[VGGT-Omega](https://github.com/facebookresearch/vggt-omega).
