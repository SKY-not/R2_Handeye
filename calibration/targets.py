"""Shared target geometry and ordered AprilTag correspondences."""
from copy import deepcopy

import cv2
import numpy as np

from calibration.transforms import pose_to_mat, validate_transform
from config import APRILTAG_BOARD_CONFIG, APRILTAG_CONFIG, CHECKERBOARD_CONFIG


def target_config(backend: str) -> dict:
    configs = {'checkerboard': CHECKERBOARD_CONFIG, 'apriltag': APRILTAG_CONFIG,
               'apriltag_board': APRILTAG_BOARD_CONFIG}
    if backend not in configs:
        raise ValueError(f'Unknown target backend: {backend}')
    return deepcopy(configs[backend])


def checkerboard_points(config: dict) -> np.ndarray:
    cols, rows = config['size']
    square = float(config['square_size'])
    if cols < 2 or rows < 2 or not np.isfinite(square) or square <= 0:
        raise ValueError('Invalid checkerboard dimensions')
    points = np.zeros((cols * rows, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return points


def tag_points(config: dict, tag_id: int, board: bool = False) -> np.ndarray:
    """Corners in pyapriltags order, in the tag or board coordinate frame."""
    size = float(config['tag_size'])
    if not np.isfinite(size) or size <= 0:
        raise ValueError('Tag size must be positive and finite')
    half = size / 2
    points = np.array([[-half, half, 0], [half, half, 0],
                       [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
    if not board:
        if tag_id != int(config['target_tag_id']):
            raise ValueError(f'Unexpected tag ID: {tag_id}')
        return points
    if tag_id not in [int(i) for i in config['tag_ids']]:
        raise ValueError(f'Unexpected board tag ID: {tag_id}')
    centers = config['tag_centers']
    rotations = config.get('tag_rotvecs', {})
    center = centers.get(tag_id, centers.get(str(tag_id)))
    rotation = rotations.get(tag_id, rotations.get(str(tag_id), [0, 0, 0]))
    if center is None:
        raise ValueError(f'Missing center for tag {tag_id}')
    transform = pose_to_mat([*center, *rotation])
    validate_transform(transform)
    return points @ transform[:3, :3].T + transform[:3, 3]


def tag_correspondences(config: dict, ids: np.ndarray, corners: np.ndarray,
                        board: bool) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ids, dtype=int).reshape(-1)
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if len(ids) == 0 or len(set(ids)) != len(ids) or len(corners) != 4 * len(ids):
        raise ValueError('Each distinct tag ID must have exactly four ordered corners')
    if not np.isfinite(corners).all():
        raise ValueError('Non-finite tag corners')
    return np.vstack([tag_points(config, int(i), board) for i in ids]), corners


def reprojection_errors(T: np.ndarray, points: np.ndarray, corners: np.ndarray,
                        K: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    camera_points = points @ T[:3, :3].T + T[:3, 3]
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError('Target points lie behind the camera')
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    projected, _ = cv2.projectPoints(points, rvec, T[:3, 3], K, distortion)
    return np.linalg.norm(projected.reshape(-1, 2) - corners, axis=1)


def solve_board_pose(config: dict, ids: np.ndarray, corners: np.ndarray,
                     K: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit all visible tags jointly; reject a frame if any tag disagrees."""
    if len(ids) < int(config.get('min_visible_tags', 2)):
        raise ValueError('Too few visible tags for a board pose')
    points, corners = tag_correspondences(config, ids, corners, board=True)
    distortion = np.zeros(5)
    planar = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)[-1] < 1e-8
    flag = cv2.SOLVEPNP_IPPE if planar else cv2.SOLVEPNP_ITERATIVE
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(points, corners, K, distortion, flags=flag)
    if not ok:
        raise ValueError('Board PnP failed')
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs):
        T = pose_to_mat(np.r_[tvec.ravel(), rvec.ravel()])
        try:
            errors = reprojection_errors(T, points, corners, K, distortion)
        except ValueError:
            continue
        candidates.append((float(np.mean(errors ** 2)), rvec, tvec))
    if not candidates:
        raise ValueError('No positive-depth board pose')
    _, rvec, tvec = min(candidates, key=lambda item: item[0])
    rvec, tvec = cv2.solvePnPRefineLM(points, corners, K, distortion, rvec, tvec)
    T = pose_to_mat(np.r_[tvec.ravel(), rvec.ravel()])
    errors = reprojection_errors(T, points, corners, K, distortion)
    if np.any(errors.reshape(-1, 4).mean(axis=1) > float(config.get('max_reprojection_error_px', 2.0))):
        raise ValueError('Inconsistent tag geometry or excessive reprojection error')
    return T, float(errors.mean())
