# Evaluation protocol

## Residual prediction accuracy

- translation vector error, predicted/true gap MAE and Spearman;
- rotation log-vector error in degrees, gap MAE and Spearman;
- joint vector error and gap correlation;
- gripper absolute error and gap correlation;
- DINO feature MSE, non-zero direction cosine, norm MAE and Spearman.

## Compatibility and calibration

- aligned/mismatch AUROC;
- Brier score;
- 15-bin expected calibration error;
- accuracy at 0.5 only as a secondary, threshold-dependent number.

Metrics are reported separately for left/right wrists. Evaluation gathering is
DDP-aware and stores scalar per-sample statistics up to a configurable cap.

## Dataset-level requirements

- split by episode, never randomly by frame;
- add task/object/environment/generator-family holdouts when metadata permits;
- stratify negative types and physical distances;
- treat different timestamps with equivalent state as positives;
- retain blur/compression/lighting controls to detect quality leakage.

## Video-level benchmark aggregation

After frame calibration is fixed, report mean, low percentile, failure-frame
fraction, and longest consecutive failure duration. Do not let a mean score hide
short but severe cross-view failures.
