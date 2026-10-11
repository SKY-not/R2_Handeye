"""Vectorized, dimensionless hand-eye least-squares refinement."""
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, least_squares
from scipy.spatial.transform import Rotation

from calibration.solver_axxb import HandEyeSolver
from calibration.transforms import average_transforms, invert_transform, mat_to_pose, pose_to_mat, validate_transform


class HandEyeOptimizer:
    def __init__(self, mode: str = 'eye_on_hand') -> None:
        self.mode = mode
        self.solver = HandEyeSolver(mode)

    def optimize(self, robot_poses: List[np.ndarray], camera_data: List[np.ndarray],
                 intrinsics: Optional[np.ndarray] = None, initial_X: Optional[np.ndarray] = None,
                 position_weight: float = 0.5, rotation_weight: float = 0.5,
                 position_scale_m: float = 0.001,
                 rotation_scale_rad: float = np.pi / 180) -> Tuple[np.ndarray, float, OptimizeResult]:
        # intrinsics is retained for compatibility; this objective uses metric poses.
        if len(robot_poses) != len(camera_data) or len(robot_poses) < 3:
            raise ValueError('At least three paired poses are required')
        weights = np.asarray([position_weight, rotation_weight])
        scales = np.asarray([position_scale_m, rotation_scale_rad])
        if (not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0)
                or not np.isfinite(scales).all() or np.any(scales <= 0)):
            raise ValueError('Weights must be finite/nonnegative and scales finite/positive')
        robot = np.stack([validate_transform(t) for t in robot_poses])
        camera = np.stack([validate_transform(t) for t in camera_data])
        left = robot if self.mode == 'eye_on_hand' else np.stack([invert_transform(t) for t in robot])
        initial = self.solver.solve_axxb_svd(robot_poses, camera_data) if initial_X is None else validate_transform(initial_X)
        factors = np.sqrt(weights) / scales

        def residual(params: np.ndarray) -> np.ndarray:
            targets = left @ pose_to_mat(params) @ camera
            reference = average_transforms(targets)
            errors = invert_transform(reference) @ targets
            translation = errors[:, :3, 3] * factors[0]
            rotation = Rotation.from_matrix(errors[:, :3, :3]).as_rotvec() * factors[1]
            return np.concatenate((translation, rotation), axis=1).ravel()

        x0 = mat_to_pose(initial)
        initial_residual = residual(x0)
        initial_cost = float(initial_residual @ initial_residual)
        fit = least_squares(residual, x0, method='trf', x_scale='jac', max_nfev=1000,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        cost = float(fit.fun @ fit.fun)
        accepted = bool(fit.success and np.isfinite(fit.x).all() and np.isfinite(cost)
                        and cost <= initial_cost + max(1e-12, initial_cost * 1e-9))
        result = OptimizeResult(success=accepted, x=fit.x if accepted else x0,
                                fun=cost if accepted else initial_cost, initial_fun=initial_cost,
                                candidate_fun=cost, message=str(fit.message), nfev=fit.nfev,
                                position_scale_m=position_scale_m, rotation_scale_rad=rotation_scale_rad)
        return (pose_to_mat(result.x) if accepted else initial.copy()), 1.0, result


def calibrate(robot_poses: List[np.ndarray], camera_data: List[np.ndarray],
              mode: str = 'eye_on_hand', use_optimization: bool = True) -> Tuple[np.ndarray, float]:
    solver = HandEyeSolver(mode)
    initial = solver.solve_axxb_svd(robot_poses, camera_data)
    if not use_optimization:
        return initial, 1.0
    X, scale, _ = HandEyeOptimizer(mode).optimize(robot_poses, camera_data, initial_X=initial)
    return X, scale
