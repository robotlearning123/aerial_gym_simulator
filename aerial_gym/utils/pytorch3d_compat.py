"""PyTorch3D compatibility module.

Provides the pytorch3d.transforms functions used by aerial_gym without
requiring the pytorch3d package. All functions use the same API as pytorch3d.
"""

import torch
import math


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (wxyz convention)."""
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    quat = torch.zeros(m.shape[0], 4, device=matrix.device, dtype=matrix.dtype)

    s = torch.sqrt(trace + 1.0) * 2
    w = 0.25 * s
    x = (m[:, 2, 1] - m[:, 1, 2]) / s
    y = (m[:, 0, 2] - m[:, 2, 0]) / s
    z = (m[:, 1, 0] - m[:, 0, 1]) / s

    s2 = torch.sqrt(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]) * 2
    w2 = (m[:, 2, 1] - m[:, 1, 2]) / s2
    x2 = 0.25 * s2
    y2 = (m[:, 0, 1] + m[:, 1, 0]) / s2
    z2 = (m[:, 0, 2] + m[:, 2, 0]) / s2

    s3 = torch.sqrt(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2]) * 2
    w3 = (m[:, 0, 2] - m[:, 2, 0]) / s3
    x3 = (m[:, 0, 1] + m[:, 1, 0]) / s3
    y3 = 0.25 * s3
    z3 = (m[:, 1, 2] + m[:, 2, 1]) / s3

    s4 = torch.sqrt(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1]) * 2
    w4 = (m[:, 1, 0] - m[:, 0, 1]) / s4
    x4 = (m[:, 0, 2] + m[:, 2, 0]) / s4
    y4 = (m[:, 1, 2] + m[:, 2, 1]) / s4
    z4 = 0.25 * s4

    mask1 = trace > 0
    mask2 = (~mask1) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    mask3 = (~mask1) & (~mask2) & (m[:, 1, 1] > m[:, 2, 2])
    mask4 = (~mask1) & (~mask2) & (~mask3)

    quat[mask1] = torch.stack([w, x, y, z], dim=-1)[mask1]
    quat[mask2] = torch.stack([w2, x2, y2, z2], dim=-1)[mask2]
    quat[mask3] = torch.stack([w3, x3, y3, z3], dim=-1)[mask3]
    quat[mask4] = torch.stack([w4, x4, y4, z4], dim=-1)[mask4]

    return quat.reshape(*batch_shape, 4)


def quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz) to rotation matrix."""
    batch_shape = quat.shape[:-1]
    q = quat.reshape(-1, 4)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    matrix = torch.zeros(q.shape[0], 3, 3, device=quat.device, dtype=quat.dtype)

    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - w * z)
    matrix[:, 0, 2] = 2 * (x * z + w * y)
    matrix[:, 1, 0] = 2 * (x * y + w * z)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - w * x)
    matrix[:, 2, 0] = 2 * (x * z - w * y)
    matrix[:, 2, 1] = 2 * (y * z + w * x)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return matrix.reshape(*batch_shape, 3, 3)


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    """Convert Euler angles to rotation matrix."""
    batch_shape = euler_angles.shape[:-1]
    angles = euler_angles.reshape(-1, 3)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    c1, c2, c3 = cos[:, 0], cos[:, 1], cos[:, 2]
    s1, s2, s3 = sin[:, 0], sin[:, 1], sin[:, 2]

    matrix = torch.zeros(angles.shape[0], 3, 3, device=euler_angles.device, dtype=euler_angles.dtype)

    if convention == "XYZ":
        matrix[:, 0, 0] = c2 * c3
        matrix[:, 0, 1] = -c2 * s3
        matrix[:, 0, 2] = s2
        matrix[:, 1, 0] = c1 * s3 + c3 * s1 * s2
        matrix[:, 1, 1] = c1 * c3 - s1 * s2 * s3
        matrix[:, 1, 2] = -c2 * s1
        matrix[:, 2, 0] = s1 * s3 - c1 * c3 * s2
        matrix[:, 2, 1] = c3 * s1 + c1 * s2 * s3
        matrix[:, 2, 2] = c1 * c2
    elif convention == "ZYX":
        matrix[:, 0, 0] = c1 * c2
        matrix[:, 0, 1] = c1 * s2 * s3 - c3 * s1
        matrix[:, 0, 2] = s1 * s3 + c1 * c3 * s2
        matrix[:, 1, 0] = c2 * s1
        matrix[:, 1, 1] = c1 * c3 + s1 * s2 * s3
        matrix[:, 1, 2] = c3 * s1 * s2 - c1 * s3
        matrix[:, 2, 0] = -s2
        matrix[:, 2, 1] = c2 * s3
        matrix[:, 2, 2] = c2 * c3
    else:
        raise ValueError(f"Unsupported convention: {convention}")

    return matrix.reshape(*batch_shape, 3, 3)


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    """Convert rotation matrix to Euler angles."""
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)

    if convention == "XYZ":
        sy = torch.sqrt(m[:, 0, 0] ** 2 + m[:, 0, 1] ** 2)
        singular = sy < 1e-6

        x = torch.atan2(-m[:, 1, 2], m[:, 2, 2])
        y = torch.atan2(m[:, 0, 2], sy)
        z = torch.atan2(-m[:, 0, 1], m[:, 0, 0])

        x2 = torch.atan2(m[:, 2, 1], m[:, 1, 1])
        y2 = torch.atan2(m[:, 0, 2], sy)
        z2 = torch.zeros_like(z)

        x = torch.where(singular, x2, x)
        z = torch.where(singular, z2, z)
    elif convention == "ZYX":
        sy = torch.sqrt(m[:, 0, 0] ** 2 + m[:, 1, 0] ** 2)
        singular = sy < 1e-6

        x = torch.atan2(m[:, 2, 1], m[:, 2, 2])
        y = torch.atan2(-m[:, 2, 0], sy)
        z = torch.atan2(m[:, 1, 0], m[:, 0, 0])

        x2 = torch.atan2(-m[:, 1, 2], m[:, 1, 1])
        y2 = torch.atan2(-m[:, 2, 0], sy)
        z2 = torch.zeros_like(z)

        x = torch.where(singular, x2, x)
        z = torch.where(singular, z2, z)
    else:
        raise ValueError(f"Unsupported convention: {convention}")

    return torch.stack([x, y, z], dim=-1).reshape(*batch_shape, 3)


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrix."""
    batch_shape = rotation_6d.shape[:-1]
    a1 = rotation_6d[..., :3]
    a2 = rotation_6d[..., 3:6]

    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    matrix = torch.stack([b1, b2, b3], dim=-2)
    return matrix.reshape(*batch_shape, 3, 3)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6D rotation representation."""
    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    return m[:, :2, :].reshape(*batch_shape, 6)


# Create a module-like namespace for pytorch3d.transforms compatibility
class _TransformsModule:
    matrix_to_quaternion = staticmethod(matrix_to_quaternion)
    quaternion_to_matrix = staticmethod(quaternion_to_matrix)
    euler_angles_to_matrix = staticmethod(euler_angles_to_matrix)
    matrix_to_euler_angles = staticmethod(matrix_to_euler_angles)
    rotation_6d_to_matrix = staticmethod(rotation_6d_to_matrix)
    matrix_to_rotation_6d = staticmethod(matrix_to_rotation_6d)


transforms = _TransformsModule()
