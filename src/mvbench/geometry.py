"""Small, dependency-free SO(3) utilities used by dataset adapters."""

from __future__ import annotations

import torch


def quaternion_to_matrix(quaternion: torch.Tensor, order: str = "xyzw") -> torch.Tensor:
    quaternion = torch.as_tensor(quaternion, dtype=torch.float32)
    if quaternion.shape[-1] != 4:
        raise ValueError("Quaternion must have four components")
    if order == "xyzw":
        x, y, z, w = quaternion.unbind(dim=-1)
    elif order == "wxyz":
        w, x, y, z = quaternion.unbind(dim=-1)
    else:
        raise ValueError(f"Unknown quaternion order: {order}")
    norm = torch.sqrt(w * w + x * x + y * y + z * z).clamp_min(1e-8)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def axis_angle_to_matrix(vector: torch.Tensor) -> torch.Tensor:
    vector = torch.as_tensor(vector, dtype=torch.float32)
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    axis = vector / angle.clamp_min(1e-8)
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(
        vector.shape[:-1] + (3, 3)
    )
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(skew.shape)
    sin = torch.sin(angle).unsqueeze(-1)
    cos = torch.cos(angle).unsqueeze(-1)
    matrix = identity + sin * skew + (1.0 - cos) * (skew @ skew)
    small = angle.squeeze(-1) < 1e-7
    if torch.any(small):
        matrix = torch.where(small[..., None, None], identity + skew, matrix)
    return matrix


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=torch.float32)
    if value.shape[-1] != 6:
        raise ValueError("6D rotation must have six components")
    first = torch.nn.functional.normalize(value[..., :3], dim=-1)
    second_raw = value[..., 3:]
    second = torch.nn.functional.normalize(
        second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first, dim=-1
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def as_rotation_matrix(value: torch.Tensor, representation: str) -> torch.Tensor:
    if representation == "matrix":
        return torch.as_tensor(value, dtype=torch.float32).reshape(3, 3)
    if representation == "quat_xyzw":
        return quaternion_to_matrix(value, "xyzw")
    if representation == "quat_wxyz":
        return quaternion_to_matrix(value, "wxyz")
    if representation == "axis_angle":
        return axis_angle_to_matrix(value)
    if representation == "rotation_6d":
        return rotation_6d_to_matrix(value)
    raise ValueError(f"Unknown rotation representation: {representation}")


def so3_log_map(matrix: torch.Tensor) -> torch.Tensor:
    matrix = torch.as_tensor(matrix, dtype=torch.float32)
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    vee = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    scale = angle / (2.0 * torch.sin(angle)).clamp_min(1e-7)
    result = scale.unsqueeze(-1) * vee
    small = angle < 1e-5
    return torch.where(small.unsqueeze(-1), 0.5 * vee, result)


def relative_rotation_log(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return so3_log_map(source.transpose(-1, -2) @ target)
