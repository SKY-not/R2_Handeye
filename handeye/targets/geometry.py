"""Shared target geometry and ordered AprilTag correspondences."""
from copy import deepcopy

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from handeye.calibration.transforms import invert_transform, pose_to_mat, validate_transform
from handeye.config import target_definition


def target_config(backend: str) -> dict:
    return target_definition(backend)


def checkerboard_points(config: dict) -> np.ndarray:
    cols, rows = config['size']
    square = float(config['square_size'])
    if cols < 2 or rows < 2 or not np.isfinite(square) or square <= 0:
        raise ValueError('Invalid checkerboard dimensions')
    points = np.zeros((cols * rows, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return points


def tag_transform(config: dict, tag_id: int) -> np.ndarray:
    """T_board_tag from the configured tag center and rotation."""
    if tag_id not in [int(i) for i in config['tag_ids']]:
        raise ValueError(f'Unexpected board tag ID: {tag_id}')
    centers = config['tag_centers']
    rotations = config.get('tag_rotvecs', {})
    center = centers.get(tag_id, centers.get(str(tag_id)))
    rotation = rotations.get(tag_id, rotations.get(str(tag_id), [0, 0, 0]))
    if center is None:
        raise ValueError(f'Missing center for tag {tag_id}')
    return validate_transform(pose_to_mat([*center, *rotation]))


def tag_points(config: dict, tag_id: int) -> np.ndarray:
    """Corners in pyapriltags order, expressed in the four-tag board frame."""
    size = float(config['tag_size'])
    if not np.isfinite(size) or size <= 0:
        raise ValueError('Tag size must be positive and finite')
    half = size / 2
    points = np.array([[-half, half, 0], [half, half, 0],
                       [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
    transform = tag_transform(config, tag_id)
    return points @ transform[:3, :3].T + transform[:3, 3]


def tag_correspondences(config: dict, ids: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=int).reshape(-1)
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if len(ids) == 0 or len(set(ids)) != len(ids) or len(corners) != 4 * len(ids):
        raise ValueError('Each distinct tag ID must have exactly four ordered corners')
    if not np.isfinite(corners).all():
        raise ValueError('Non-finite tag corners')
    return np.vstack([tag_points(config, int(i)) for i in ids]), corners


def reprojection_errors(T: np.ndarray, points: np.ndarray, corners: np.ndarray,
                        K: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    camera_points = points @ T[:3, :3].T + T[:3, 3]
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError('Target points lie behind the camera')
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    projected, _ = cv2.projectPoints(points, rvec, T[:3, 3], K, distortion)
    return np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1)


def solve_board_pose(config: dict, ids: np.ndarray, corners: np.ndarray,
                     K: np.ndarray, tag_poses: np.ndarray, margins: np.ndarray) -> tuple[np.ndarray, float]:
    """Fuse direct poses with decision-margin weights, as in commit 7d9d24c.

    Reprojection only validates the result; no OpenCV PnP or pose refinement.
    """
    if len(ids) < int(config.get('min_visible_tags', 2)):
        raise ValueError('Too few visible tags for a board pose')
    points, corners = tag_correspondences(config, ids, corners)
    poses = np.asarray(tag_poses, dtype=np.float64)
    if poses.shape != (len(ids), 4, 4):
        raise ValueError('Each visible tag must have one 4x4 camera-frame pose')
    board_poses = []
    for tag_id, pose in zip(ids, poses):
        validate_transform(pose, 'pyapriltags camera-to-tag pose')
        board_poses.append(pose @ invert_transform(tag_transform(config, int(tag_id))))
    # Preserve the translation/rotation weighting from commit 7d9d24c.
    weights = np.asarray(margins, dtype=np.float64).reshape(-1)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('Invalid AprilTag decision margins')
    if len(weights) != len(board_poses) or float(weights.sum()) <= 1e-9:
        weights = np.ones(len(board_poses), dtype=np.float64)
    weights = weights / float(weights.sum())
    translations = np.asarray([pose[:3, 3] for pose in board_poses])
    rotations = Rotation.from_matrix([pose[:3, :3] for pose in board_poses])
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.average(translations, axis=0, weights=weights)
    T[:3, :3] = rotations.mean(weights=weights).as_matrix()
    errors = reprojection_errors(T, points, corners, K, np.zeros(5))
    if np.any(errors.reshape(-1, 4).mean(axis=1) > float(config.get('max_reprojection_error_px', 2.0))):
        raise ValueError('Inconsistent tag geometry or excessive reprojection error')
    return T, float(errors.mean())
