# Multiview robot state-consistency benchmark

This project learns a frame-level metric for whether a generated head view and
a generated wrist view express the same robot and local scene state. The shared
pairwise model is evaluated independently for the left and right wrist.

For each pair it predicts:

- EEF translation residual;
- EEF rotation residual in log-SO(3);
- arm joint residual;
- gripper residual;
- projected frozen-DINO wrist-feature residual;
- learned compatibility energy;
- optional validity/observability confidence.

## Repository layout

```text
src/mvbench/                 production package
  data/                      LeRobot adapter and deterministic mock data
  models/                    DINOv2/VGGT adapters and pairwise readouts
  checkpoint.py              atomic save/resume
  distributed.py             torchrun/DDP lifecycle
  losses.py                  masked multi-task objective
  metrics.py                 residual, ranking, and calibration metrics
  trainer.py                 BF16/FP16 DDP trainer
scripts/train_multiview_metric.py
configs/                     mock and large-model templates
pilots/dinov2_visual_probe/  isolated RGB-only example-data experiment
memory/                      persistent project state and decisions
scratch/                     disposable local smoke-test artifacts
tests/                       core contract and pipeline tests
```

The MP4 files in `real_data/` are examples used by the isolated visual pilot.
Formal training is designed for a LeRobot dataset and accesses it through the
public `LeRobotDataset` API.

Large upstream models and their checkpoints are kept outside Git. See
[THIRD_PARTY.md](THIRD_PARTY.md) for the validated DINOv2, VGGT, and Open-D4RT
revisions and expected local paths.

## Test the training stack

Run unit tests:

```bash
PYTHONPATH=src .venv-vggt/bin/python -m unittest discover -s tests -v
```

Run one-GPU mock training:

```bash
PYTHONPATH=src .venv-vggt/bin/python scripts/train_multiview_metric.py \
  --config configs/mock_ddp_smoke.json
```

Run two-GPU DDP training:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
  .venv-vggt/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  scripts/train_multiview_metric.py \
  --config configs/mock_ddp_smoke.json \
  --output-dir scratch/mock_ddp_two_gpu
```

Resume by adding `--resume RUN_DIR/latest.pt` and increasing `--epochs`.

## Formal model templates

- `configs/train_dinov2_large_template.json`: DINOv2-G with trainable final
  blocks and patch-token readouts.
- `configs/train_vggt_1b_template.json`: VGGT-1B joint pair encoding with a
  frozen-backbone starting point.

Template LeRobot feature keys are illustrative. Before a real run, inspect the
incoming dataset metadata and update `data.schema`. `observation.state` is the
default physical label source; `action` is used only after its absolute/delta
semantics are explicitly verified.

## Persistent engineering context

Start with the project purpose in
[memory/PROJECT_CHARTER.md](memory/PROJECT_CHARTER.md), then read the current
implementation status in [memory/PROJECT_STATE.md](memory/PROJECT_STATE.md).
The design, data, evaluation, and experiment records are indexed by
[memory/README.md](memory/README.md).
