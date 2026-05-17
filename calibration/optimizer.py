#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手眼标定优化器
带深度缩放因子的非线性优化
"""

import numpy as np
from scipy import optimize
from scipy.optimize import OptimizeResult
from scipy.spatial.transform import Rotation
from typing import List, Optional, Tuple

from calibration.solver_axxb import HandEyeSolver


class HandEyeOptimizer:
    """手眼标定非线性优化器"""

    def __init__(self, mode: str = 'eye_on_hand') -> None:
        """
        初始化优化器

        Args:
            mode: 'eye_on_hand' 或 'eye_to_hand'
        """
        self.mode = mode
        self.solver = HandEyeSolver(mode)

    def optimize(
        self,
        robot_poses: List[np.ndarray],
        camera_data: List[np.ndarray],
        intrinsics: Optional[np.ndarray],
        initial_X: Optional[np.ndarray] = None,
        position_weight: float = 0.5,
        rotation_weight: float = 0.5
    ) -> Tuple[np.ndarray, float, OptimizeResult]:
        """
        优化求解手眼变换

        Args:
            robot_poses: 机器人末端位姿列表 (4x4)
            camera_data: 相机观测到的标定目标位姿列表 (4x4)
            intrinsics: 相机内参
            initial_X: 初始手眼矩阵
        Returns:
            X: 优化后的手眼矩阵
            z_scale: fixed at 1.0 for result-file compatibility
            result: 优化结果
        """
        n = len(robot_poses)
        if n != len(camera_data):
            raise ValueError("robot_poses and camera_data must have the same length")

        for idx, cam_pose in enumerate(camera_data):
            if not isinstance(cam_pose, np.ndarray) or cam_pose.shape != (4, 4):
                raise ValueError(f"camera_data[{idx}] must be a 4x4 pose matrix")

        if position_weight < 0 or rotation_weight < 0:
            raise ValueError("position_weight and rotation_weight must be non-negative")
        if position_weight == 0 and rotation_weight == 0:
            raise ValueError("at least one of position_weight or rotation_weight must be positive")

        # 初始估计
        if initial_X is None:
            initial_X = self.solver.solve_axxb_svd(robot_poses, camera_data)

        # 参数向量: [x, y, z, rx, ry, rz]
        x0 = np.zeros(6)
        x0[:3] = initial_X[:3, 3]
        x0[3:6] = self.solver.log_rot(initial_X[:3, :3])

        # 定义目标函数
        def average_transforms(transforms: List[np.ndarray]) -> np.ndarray:
            translations = np.asarray([T[:3, 3] for T in transforms], dtype=np.float64)
            rotations = Rotation.from_matrix(np.asarray([T[:3, :3] for T in transforms], dtype=np.float64))

            T_avg = np.eye(4, dtype=np.float64)
            T_avg[:3, :3] = rotations.mean().as_matrix()
            T_avg[:3, 3] = np.mean(translations, axis=0)
            return T_avg

        def objective(params: np.ndarray) -> float:
            X = self.solver.pose_to_mat(params[:6])

            target_poses: List[np.ndarray] = []

            for i in range(n):
                T_cam = camera_data[i].copy()

                # Eye-on-Hand: target should be stationary in base.
                # Eye-to-Hand: target should be stationary in TCP.
                if self.mode == 'eye_on_hand':
                    T_world = robot_poses[i] @ X @ T_cam
                else:
                    T_world = self.solver.invert_transform(robot_poses[i]) @ X @ T_cam

                target_poses.append(T_world)

            if not target_poses:
                return 0.0

            T_ref = average_transforms(target_poses)
            T_ref_inv = self.solver.invert_transform(T_ref)
            position_cost = 0.0
            rotation_cost = 0.0

            for T_world in target_poses:
                E = T_ref_inv @ T_world
                error_pose = self.solver.mat_to_pose(E)
                position_cost += float(np.dot(error_pose[:3], error_pose[:3]))
                rotation_cost += float(np.dot(error_pose[3:6], error_pose[3:6]))

            return float(position_weight * position_cost + rotation_weight * rotation_cost)

        # 优化
        result = optimize.minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )

        # 提取结果
        X_opt = self.solver.pose_to_mat(result.x[:6])
        z_scale_opt = 1.0

        return X_opt, z_scale_opt, result

def calibrate(
    robot_poses: List[np.ndarray],
    camera_data: List[np.ndarray],
    mode: str = 'eye_on_hand',
    use_optimization: bool = True
) -> Tuple[np.ndarray, float]:
    """
    便捷标定函数

    Args:
        robot_poses: 机器人位姿列表
        camera_data: 相机数据
        mode: 标定模式
        use_optimization: 是否使用优化

    Returns:
        X: 手眼变换矩阵
        z_scale: 深度缩放因子
    """
    optimizer = HandEyeOptimizer(mode)

    if use_optimization:
        X, z_scale, result = optimizer.optimize(
            robot_poses, camera_data, None
        )
    else:
        solver = HandEyeSolver(mode)
        X = solver.solve_axxb_svd(robot_poses, camera_data)
        z_scale = 1.0

    return X, z_scale
