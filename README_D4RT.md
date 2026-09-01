# Wrist-view 4D reconstruction with OpenD4RT

## What is in `real_data`

The local collection follows a LeRobot-style video layout. One selected episode contains synchronized MP4 files under:

```text
videos/chunk-000/
  observation.images.front_head/episode_XXXXXX.mp4
  observation.images.left_wrist/episode_XXXXXX.mp4
  observation.images.right_wrist/episode_XXXXXX.mp4
```

The files at `real_data/Multiview_Benchmark/` serve these roles:

- `dataset_groups.tsv`: maps each dataset/environment group to its selected source episode.
- `downloaded_source_files.txt`: manifest of downloaded per-view MP4 paths.
- `output_durations.tsv`: duration of the assembled three-view episode for each group.
- `select_smallest_episodes.awk`: selects the smallest episode for which all three views exist.

In the current local snapshot, `dataset_groups.tsv` has 124 selected dataset groups and the download manifest lists 385 MP4 objects (130 front, 127 left wrist, 128 right wrist). Of those, 213 MP4s are currently materialized locally: 71 per camera view, representing 70 complete three-view episode triplets plus three isolated single-view files. No camera intrinsics or inter-camera extrinsics are present in this directory, so this run uses video-only learned reconstruction rather than calibrated metric multiview fusion.

The first OpenD4RT run uses the left-wrist `pour_water` video because it contains a stable textured background plus clear cup/arm motion:

```text
real_data/Multiview_Benchmark/source_videos/FoundationModel/sz_hys3.0/
  pour_water/10377/unknown-23858_23858/videos/chunk-000/
  observation.images.left_wrist/episode_000105.mp4
```

It is H.264, 640×360, 30 FPS, and about 17.43 seconds long. The demo samples 32 frames from 5.8–7.87 seconds (roughly stride 2), where the cup moves through the wrist camera view. This is close to OpenD4RT's training-time temporal stride of 1–2. Only the model copy is resized to 256×256; original-resolution RGB frames are retained for visualization.

## Model choice and caveat

Google DeepMind has published the [D4RT paper and project page](https://d4rt-paper.github.io/), but not the original implementation/checkpoint. This workspace therefore deploys [OpenD4RT](https://github.com/Lijiaxin0111/Open-d4rt), an unofficial PyTorch reproduction with released pretrained weights. Do not interpret its output as the unreleased official D4RT model's output.

The predicted coordinates share the camera frame of the first sampled image (`ref0`) but have learned, non-metric scale. The output is suitable for qualitative geometry/motion inspection, not metric measurement without calibration or alignment.

## Installed layout

```text
third_party/Open-d4rt/       # source at commit 403290a6...
.venv-d4rt/                  # local venv reusing PyTorch 2.6 + CUDA 12.4
scripts/prepare_opend4rt_checkpoint.py  # training checkpoint -> model-only FP16 weights
scripts/run_d4rt_wrist.py    # MP4 -> compressed reconstruction NPZ
scripts/export_d4rt_rerun.py # NPZ -> Rerun RRD + PNG preview
outputs/d4rt_wrist_pour/     # generated artifacts
```

The released checkpoint is 13.95 GB because it also contains optimizer, scheduler, and scaler state. The workspace quota cannot hold that training checkpoint. For deployment, extract only the 627 model tensors and convert them to the FP16 precision used by inference:

```bash
HF_XET_HIGH_PERFORMANCE=1 HF_HOME=/tmp/hf_home_d4rt HF_XET_CACHE=/tmp/hf_xet_d4rt \
  .venv-d4rt/bin/python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='Lijiaxin0111/OpenD4RT', filename='checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt', local_dir='/tmp/opend4rt-local'))"
```

The released file is 13,950,006,682 bytes; its verified SHA-256 is `1f63305422fdc2000b057fbbc1d37459ac1a8063bbfcd0e3b7d473f5485943f5`.

```bash
.venv-d4rt/bin/python scripts/prepare_opend4rt_checkpoint.py \
  --input /tmp/opend4rt-local/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt \
  --output third_party/Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.fp16.ckpt
```

The compact checkpoint is model-only, about 2.17 GiB, and produces the same FP16 parameters as loading the FP32 training checkpoint and then casting the model for inference.
Its local SHA-256 is recorded beside it in `opend4rt.fp16.ckpt.sha256`. A repeat inference check produced byte-identical XYZ/confidence arrays (`max_abs_diff=0`).

## Reproduce

Run model inference on a CUDA-visible shell:

```bash
.venv-d4rt/bin/python scripts/run_d4rt_wrist.py \
  --video real_data/Multiview_Benchmark/source_videos/FoundationModel/sz_hys3.0/pour_water/10377/unknown-23858_23858/videos/chunk-000/observation.images.left_wrist/episode_000105.mp4 \
  --config third_party/Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --checkpoint third_party/Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.fp16.ckpt \
  --output outputs/d4rt_wrist_pour/reconstruction.npz \
  --num-frames 32 --start-sec 5.8 --end-sec 7.87 \
  --grid-cols 160 --grid-rows 90 --query-chunk-size 512
```

Export the interactive recording with the local Rerun 0.26.2 SDK. Newer viewers no longer accept the legacy 0.21 `.rrd` encoding, so the writer and viewer versions must stay compatible:

```bash
MPLCONFIGDIR=/tmp/mpl_d4rt \
  /mnt/home/sujiayi/miniconda3/envs/transform_env/bin/python scripts/export_d4rt_rerun.py \
  --input outputs/d4rt_wrist_pour/reconstruction.npz \
  --output outputs/d4rt_wrist_pour/open_d4rt_wrist_4d.rrd \
  --preview outputs/d4rt_wrist_pour/preview.png \
  --point-radius 3.0
```

Open it with:

```bash
/mnt/home/sujiayi/miniconda3/envs/transform_env/bin/rerun \
  outputs/d4rt_wrist_pour/open_d4rt_wrist_4d.rrd
```

Use the `video_time` or `sample_frame` timeline to play the time-varying point cloud next to the wrist RGB image.

## Three-view batch

Three additional synchronized episodes are configured in `configs/d4rt_multiview_jobs.json`. Run all nine episode/view combinations while loading the model only once:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv-d4rt/bin/python scripts/run_d4rt_multiview_batch.py \
  --manifest configs/d4rt_multiview_jobs.json \
  --config third_party/Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/model.yaml \
  --checkpoint third_party/Open-d4rt/checkpoints/OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.fp16.ckpt \
  --output-root outputs/d4rt_multiview \
  --grid-cols 160 --grid-rows 90 --query-chunk-size 512
```

The generated results and per-view Rerun links are indexed in `outputs/d4rt_multiview/README.md`.

## Produced example

- `reconstruction.npz`: 32 frames × 14,400 points in the first sampled camera frame.
- `open_d4rt_wrist_4d.rrd`: Rerun 0.26 recording with synchronized RGB, colored points, confidence, and source-frame timelines.
- `preview.png`: static middle-frame sanity-check preview.
- `*.json`: inference and export metadata.
