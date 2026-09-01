# Design decisions

## Pairwise rather than three-view input

The same model evaluates `(head, left_wrist)` and `(head, right_wrist)` in two
forward passes. A learned side embedding identifies the arm/camera. This keeps
attribution clear and prevents the two wrist views from becoming a shortcut.

## Foundation model is pluggable

The production model consumes patch tokens through `PairVisualBackbone`.
DINOv2 encodes each image independently; VGGT jointly encodes the pair with its
alternating frame/global attention. The pilot's global-feature MLP is not the
production architecture.

## Residual heads plus compatibility energy

Energy is a learned unified compatibility output; it does not replace the
physical and visual residual heads. Benchmark releases should retain all
submetrics. Whether energy or a calibrated transparent aggregation becomes the
headline score will be decided from held-out generated-video/human agreement.

## DINO feature residual remains a core target

The visual residual captures object/contact/local-scene state not represented
by robot telemetry. It will use a train-only fixed projection of frozen teacher
features (initially PCA). Vector, norm, and direction losses are separate.
The example-video pilot favored mean-pooled patch features over CLS or their
concatenation, but the production model retains dense patch tokens rather than
committing to global mean pooling.

## Joint residual is an auxiliary physical target

EEF pose alone can miss different configurations of a kinematically redundant
arm. Joint residual is therefore supervised when available and reported as a
diagnostic submetric.

## Observation is state; action is not assumed to be state

`observation.state.*` is the default source of EEF/joint/gripper labels.
`action` may be delta control, velocity, or an absolute target depending on the
dataset. It is never substituted for state unless dataset metadata explicitly
defines compatible absolute semantics.

## Offline pair plans

Positive, hard-negative, cross-episode, and state-equivalent sampling is
materialized as a versioned plan. This makes experiments reproducible and keeps
expensive mining out of DataLoader workers.
