#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common rigid transform helpers used across calibration modules."""

from typing import Sequence, Union
import numpy as np
from scipy.spatial.transform import Rotation


def vec_to_skew(v: np.ndarray) -> np.ndarray:
    """Convert a 3D vector to a 3x3 skew-symmetric matrix."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)


def skew_to_vec(skew: np.ndarray) -> np.ndarray:
    """Convert a 3x3 skew-symmetric matrix to its 3D vector form."""
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)


def rotvec_to_matrix(rotvec: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert a rotation vector to a 3x3 rotation matrix."""
    v = np.asarray(rotvec, dtype=np.float64).reshape(3)
    return Rotation.from_rotvec(v).as_matrix()


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a rotation vector."""
    Rn = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return Rotation.from_matrix(Rn).as_rotvec()


def validate_transform(T: np.ndarray, name: str = 'transform') -> np.ndarray:
    """Reject malformed SE(3) data instead of silently projecting it onto SO(3)."""
    value = np.asarray(T, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f'{name}: expected a finite 4x4 transform')
    R = value[:3, :3]
    if (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1.0, atol=1e-5)):
        raise ValueError(f'{name}: invalid rigid transform')
    return value


def average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Arithmetic translation mean and chordal rotation mean; never average matrices."""
    values = np.asarray(transforms, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or len(values) == 0:
        raise ValueError('At least one 4x4 transform is required')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_matrix(values[:, :3, :3]).mean().as_matrix()
    result[:3, 3] = values[:, :3, 3].mean(axis=0)
    return result


def target_poses(mode: str, robot_poses: Sequence[np.ndarray],
                 camera_poses: Sequence[np.ndarray], X: np.ndarray) -> np.ndarray:
    """Target poses in their stationary frame: base for EOH, TCP for ETH."""
    if len(robot_poses) != len(camera_poses) or len(robot_poses) == 0:
        raise ValueError('Robot and camera poses must be nonempty and paired')
    if mode == 'eye_on_hand':
        left = np.asarray(robot_poses)
    elif mode == 'eye_to_hand':
        left = np.asarray([invert_transform(t) for t in robot_poses])
    else:
        raise ValueError(f'Unknown calibration mode: {mode}')
    return left @ X @ np.asarray(camera_poses)


def pose_to_mat(pose: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert [x, y, z, rx, ry, rz] pose to a 4x4 homogeneous transform."""
    pose_arr = np.asarray(pose, dtype=np.float64)
    if pose_arr.shape == (4, 4):
        return pose_arr.copy()

    if pose_arr.shape != (6,):
        raise ValueError(f"pose must be shape (6,) or (4,4), got {pose_arr.shape}")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotvec_to_matrix(pose_arr[3:6])
    T[:3, 3] = pose_arr[:3]
    return T


def mat_to_pose(T: np.ndarray) -> np.ndarray:
    """Convert a 4x4 homogeneous transform to [x, y, z, rx, ry, rz]."""
    Tn = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return np.concatenate([Tn[:3, 3], matrix_to_rotvec(Tn[:3, :3])])


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transform."""
    Tn = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = Tn[:3, :3]
    t = Tn[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv
