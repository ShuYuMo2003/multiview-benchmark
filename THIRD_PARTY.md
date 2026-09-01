# Third-party dependencies

Upstream repositories and model checkpoints are intentionally not committed to
this repository. The current validated local revisions are:

| Dependency | Revision | Expected path |
| --- | --- | --- |
| DINOv2 | `7764ea0f912e53c92e82eb78a2a1631e92725fc8` | `third_party/dinov2` |
| VGGT | `a288dd0f14786c93483e45524328726ab7b1b4ce` | `third_party/vggt` |
| Open-D4RT | `403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3` | `third_party/Open-d4rt` |

Clone and pin them from the repository root:

```bash
mkdir -p third_party
git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
git -C third_party/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8

git clone https://github.com/facebookresearch/vggt.git third_party/vggt
git -C third_party/vggt checkout a288dd0f14786c93483e45524328726ab7b1b4ce

git clone https://github.com/Lijiaxin0111/Open-d4rt.git third_party/Open-d4rt
git -C third_party/Open-d4rt checkout 403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3
```

Download pretrained weights according to each upstream project's instructions.
Keep all weights under ignored cache or checkpoint directories; none are needed
for the tiny-backbone unit tests and mock DDP smoke tests.
