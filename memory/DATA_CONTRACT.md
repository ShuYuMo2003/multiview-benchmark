# Data contract

The formal source dataset is LeRobot. The adapter uses the public dataset API;
it does not infer episode boundaries from video filenames. LeRobot v3 stores
low-dimensional signals in Parquet, camera streams in video shards, and episode
offsets/schema in metadata.

## Required pair inputs

- synchronized anchor head RGB at state `t`;
- candidate left or right wrist RGB at state `s`;
- side id: `0=left`, `1=right`;
- episode ids and frame/timestamp indices used by the offline pair plan.

## Preferred observation labels per side

- EEF position in a common robot-base frame, meters;
- EEF orientation with an explicit representation/order;
- arm joint positions in radians;
- gripper width/state with a fixed normalization convention;
- precomputed projected frozen-DINO wrist feature.

Targets use:

```text
translation = p_t - p_s
rotation = Log(R_s^T R_t)
joint = q_t - q_s
gripper = g_t - g_s
dino = z(W_t) - z(W_s)
```

Each family owns a mask. Missing or semantically invalid values are zero-filled
and masked, never silently treated as zero residual.

## LeRobot integration boundary

`LeRobotPairDataset` accepts a base `LeRobotDataset`, a `PairPlan`, and a
config-driven `LeRobotSchema`. Exact feature keys will be updated after reading
the incoming dataset's `meta.info.features`. The current JSON templates contain
illustrative keys only.

## Pair-plan columns

```text
anchor_index: int64
candidate_index: int64
side: uint8
consistency_label: float32
same_episode: bool
```

Future plans should also record negative type, physical distances, source
episode/task/domain, observability, and mining version.
