#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手眼标定求解器
基于 AX=XB 模型
支持 Eye-on-Hand 和 Eye-to-Hand 两种模式
"""

import numpy as np
from scipy import optimize
from scipy.linalg import expm, logm
from typing import Any, List, Optional, Tuple


class HandEyeSolver:
    """手眼标定求解器"""

    def __init__(self, mode: str = 'eye_on_hand') -> None:
        """
        初始化求解器

        Args:
            mode: 'eye_on_hand' 或 'eye_to_hand'
        """
        self.mode = mode

    # ==================== 刚体变换辅助函数 ====================

    @staticmethod
    def vec_to_skew(v: np.ndarray) -> np.ndarray:
        """
        向量转反对称矩阵

        Args:
            v: 3维向量

        Returns:
            skew: 3x3 反对称矩阵
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    @staticmethod
    def skew_to_vec(skew: np.ndarray) -> np.ndarray:
        """
        反对称矩阵转向量

        Args:
            skew: 3x3 反对称矩阵

        Returns:
            v: 3维向量
        """
        return np.array([skew[2, 1], skew[1, 0], skew[0, 1]])

    @staticmethod
    def log_rot(R: np.ndarray) -> np.ndarray:
        """
        旋转矩阵的对数映射 (旋转向量)

        Args:
            R: 3x3 旋转矩阵

        Returns:
            rotvec: 3维旋转向量
        """
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

        if np.abs(theta) < 1e-6:
            return np.zeros(3)

        log_R = (theta / (2 * np.sin(theta))) * (R - R.T)
        return HandEyeSolver.skew_to_vec(log_R)

    @staticmethod
    def exp_rot(rotvec: np.ndarray) -> np.ndarray:
        """
        旋转向量的指数映射 (旋转矩阵)

        Args:
            rotvec: 3维旋转向量

        Returns:
            R: 3x3 旋转矩阵
        """
        theta = np.linalg.norm(rotvec)

        if theta < 1e-6:
            return np.eye(3)

        axis = rotvec / theta
        K = HandEyeSolver.vec_to_skew(axis)

        return np.asarray(np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K), dtype=np.float64)

    @staticmethod
    def mat_to_pose(T: np.ndarray) -> np.ndarray:
        """
        齐次变换矩阵转位姿数组

        Args:
            T: 4x4 齐次变换矩阵

        Returns:
            pose: [x, y, z, rx, ry, rz]
        """
        pos = T[:3, 3]
        rotvec = HandEyeSolver.log_rot(T[:3, :3])
        return np.concatenate([pos, rotvec])

    @staticmethod
    def pose_to_mat(pose: np.ndarray) -> np.ndarray:
        """
        位姿数组转齐次变换矩阵

        Args:
            pose: [x, y, z, rx, ry, rz]

        Returns:
            T: 4x4 齐次变换矩阵
        """
        x, y, z, rx, ry, rz = pose
        R = HandEyeSolver.exp_rot(np.array([rx, ry, rz]))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def invert_transform(T: np.ndarray) -> np.ndarray:
        """
        求逆变换

        Args:
            T: 4x4 齐次变换矩阵

        Returns:
            T_inv: 逆变换矩阵
        """
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv

    # ==================== AX=XB 求解 ====================

    def compute_motion(self, T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """
        计算两次位姿之间的相对变换

        Args:
            T1: 初始位姿 (4x4)
            T2: 目标位姿 (4x4)

        Returns:
            delta_T: 相对变换 (4x4), 使得 T2 = delta_T @ T1
        """
        return np.asarray(self.invert_transform(T1) @ T2, dtype=np.float64)

    def solve_axxb_svd(self, robot_poses: List[np.ndarray], camera_poses: List[np.ndarray]) -> np.ndarray:
        """
        使用SVD方法求解 AX=XB

        Args:
            robot_poses: 机器人末端位姿列表 (eye_on_hand: 基坐标系下; eye_to_hand: 相机坐标系下)
            camera_poses: 相机观测到的标定板位姿列表

        Returns:
            X: 手眼变换矩阵 (4x4)
        """
        n = len(robot_poses)
        if n < 2:
            raise ValueError("至少需要2组数据")

        # 计算相对运动
        A_list = []  # 机器人相对运动
        B_list = []  # 相机相对运动

        for i in range(n - 1):
            if self.mode == 'eye_to_hand':
                # Eye-to-Hand 直接求解 X = T_base_cam:
                #   inv(T_base_tcp_i) * X * T_cam_board_i = inv(T_base_tcp_j) * X * T_cam_board_j
                # => (T_base_tcp_j * inv(T_base_tcp_i)) * X = X * (T_cam_board_j * inv(T_cam_board_i))
                A = np.asarray(robot_poses[i + 1] @ self.invert_transform(robot_poses[i]), dtype=np.float64)
                B = np.asarray(camera_poses[i + 1] @ self.invert_transform(camera_poses[i]), dtype=np.float64)
            else:
                A = self.compute_motion(robot_poses[i], robot_poses[i + 1])
                B = self.compute_motion(camera_poses[i], camera_poses[i + 1])
            A_list.append(A)
            B_list.append(B)

        # 构建求解方程
        # 对于每个相对运动: A_i * X = X * B_i
        # 转化为: (I - A_i) * T_cg = T_cg * (I - B_i) 的线性化形式
        # 或者使用对数方法

        # 使用 Tsai-Lenz 方法的变体
        rotations = []
        translations = []

        for A, B in zip(A_list, B_list):
            R_A = A[:3, :3]
            t_A = A[:3, 3]
            R_B = B[:3, :3]
            t_B = B[:3, 3]

            # 旋转求解
            R_X = self.solve_rotation(R_A, R_B)

            # 平移求解
            t_X = self.solve_translation(R_A, t_A, R_B, t_B, R_X)

            rotations.append(R_X)
            translations.append(t_X)

        # 融合所有估计 (取中值)
        R_X = np.median(rotations, axis=0)
        # 确保是合法旋转矩阵
        U, _, Vt = np.linalg.svd(R_X)
        R_X = U @ Vt
        if np.linalg.det(R_X) < 0:
            U[:, -1] *= -1
            R_X = U @ Vt

        t_X = np.median(translations, axis=0)

        # 构建齐次变换矩阵
        X = np.eye(4)
        X[:3, :3] = R_X
        X[:3, 3] = t_X

        return np.asarray(X, dtype=np.float64)

    def solve_rotation(self, R_A: np.ndarray, R_B: np.ndarray) -> np.ndarray:
        """
        求解旋转部分

        Args:
            R_A: 机器人相对旋转
            R_B: 相机相对旋转

        Returns:
            R_X: 手眼旋转矩阵
        """
        # 使用SVD求解
        M = R_A @ R_B.T
        U, _, Vt = np.linalg.svd(M)
        R_X = U @ Vt

        # 处理反射情况
        if np.linalg.det(R_X) < 0:
            Vt[-1, :] *= -1
            R_X = U @ Vt

        return np.asarray(R_X, dtype=np.float64)

    def solve_translation(
        self,
        R_A: np.ndarray,
        t_A: np.ndarray,
        R_B: np.ndarray,
        t_B: np.ndarray,
        R_X: np.ndarray
    ) -> np.ndarray:
        """
        求解平移部分

        Args:
            R_A, t_A: 机器人相对运动
            R_B, t_B: 相机相对运动
            R_X: 已求解的旋转矩阵

        Returns:
            t_X: 手眼平移向量
        """
        # (I - R_A) * t_X = R_X * t_B - t_A
        M = np.eye(3) - R_A
        rhs = R_X @ t_B - t_A

        # 最小二乘求解
        # 由于M可能是奇异的，我们使用伪逆
        t_X = np.linalg.lstsq(M, rhs, rcond=None)[0]

        return np.asarray(t_X, dtype=np.float64)

    # ==================== 带 z_scale 的优化 ====================

    def solve_with_zscale(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        camera_intrinsics: np.ndarray,
        depth_scales: Optional[List[float]] = None,
        initial_X: Optional[np.ndarray] = None,
        z_scale_init: float = 1.0
    ) -> Tuple[np.ndarray, float, Any]:
        """
        带深度缩放因子的优化求解

        Args:
            robot_poses: 机器人位姿列表 (4x4)
            camera_poses: 相机观测到的标定板位姿列表 (4x4) 或 3D点列表
            camera_intrinsics: 相机内参
            depth_scales: 深度缩放因子列表 (可选)
            initial_X: 初始手眼矩阵
            z_scale_init: 初始深度缩放因子

        Returns:
            X: 优化后的手眼矩阵
            z_scale: 优化后的深度缩放因子
            result: 优化结果
        """
        n = len(robot_poses)

        if initial_X is None:
            # 先用SVD求解初始值
            initial_X = self.solve_axxb_svd(robot_poses, camera_poses)

        # 构建优化参数向量
        # [x, y, z, rx, ry, rz, z_scale]
        initial_params = np.zeros(7)
        initial_params[:3] = initial_X[:3, 3]
        initial_params[3:6] = self.log_rot(initial_X[:3, :3])
        initial_params[6] = z_scale_init

        # 优化目标函数
        def objective(params: np.ndarray) -> float:
            X = self.pose_to_mat(params[:6])
            z_scale = params[6]

            residual = []

            for i in range(n):
                # 使用中心点
                if isinstance(camera_poses[i], np.ndarray) and camera_poses[i].shape[0] == 3:
                    # 3D点，应用深度缩放
                    p_cam = camera_poses[i].copy()
                    if len(p_cam.shape) == 1:
                        p_cam[2] *= z_scale
                    else:
                        p_cam[:, 2] *= z_scale

                    # 转换到世界坐标
                    p_world = X @ np.append(p_cam, 1)

                    # 理论上的世界坐标点 (从机器人位姿)
                    p_theory = robot_poses[i] @ np.append(
                        self.get_checkerboard_center(), 1
                    )

                    # 残差
                    residual.append(p_world[:3] - p_theory[:3])
                else:
                    # 使用位姿
                    # 计算重投影误差
                    H_cam = camera_poses[i]
                    H_cam_scaled = self.apply_zscale(H_cam, z_scale)

                    # 计算残差: A * X * B - X
                    AXB = robot_poses[i] @ X @ H_cam_scaled
                    error = self.mat_to_pose(AXB) - self.mat_to_pose(X)
                    residual.append(error)

            residual = np.concatenate(residual)
            return float(np.sum(residual ** 2))

        # 优化
        result = optimize.minimize(
            objective,
            initial_params,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
        )

        # 提取结果
        X_opt = self.pose_to_mat(result.x[:6])
        z_scale_opt = result.x[6]

        return X_opt, z_scale_opt, result

    @staticmethod
    def apply_zscale(T: np.ndarray, z_scale: float) -> np.ndarray:
        """应用深度缩放"""
        T_scaled = T.copy()
        T_scaled[2, 3] *= z_scale
        return T_scaled

    @staticmethod
    def get_checkerboard_center() -> np.ndarray:
        """获取棋盘格中心的世界坐标 (需要根据实际标定板设置)"""
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    # ==================== 简化的求解方法 ====================

    def solve_point_to_point(self, robot_positions: np.ndarray, camera_points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        简化方法: 点到点匹配求解

        Args:
            robot_positions: 机器人末端位置 (N, 3)
            camera_points_3d: 相机坐标系下的3D点 (N, 3)

        Returns:
            X: 手眼变换矩阵
            z_scale: 深度缩放因子
        """
        # 使用SVD求解刚体变换
        return self.solve_rigid_transform(camera_points_3d, robot_positions)

    @staticmethod
    def solve_rigid_transform(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用SVD求解刚体变换: B = R @ A + t

        Args:
            A: 源点集 (N, 3)
            B: 目标点集 (N, 3)

        Returns:
            R: 旋转矩阵
            t: 平移向量
            z_scale: 缩放因子 (可选)
        """
        n = A.shape[0]

        # 中心化
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        AA = A - centroid_A
        BB = B - centroid_B

        # SVD分解
        H = AA.T @ BB
        U, _, Vt = np.linalg.svd(H)

        R = Vt.T @ U.T

        # 处理反射
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_B - R @ centroid_A

        return R, t


def solve_hand_eye(robot_poses: List[np.ndarray], camera_poses: List[np.ndarray], mode: str = 'eye_on_hand') -> np.ndarray:
    """
    便捷函数: 求解手眼标定

    Args:
        robot_poses: 机器人位姿列表
        camera_poses: 相机位姿列表
        mode: 'eye_on_hand' 或 'eye_to_hand'

    Returns:
        X: 手眼变换矩阵
    """
    solver = HandEyeSolver(mode)
    return solver.solve_axxb_svd(robot_poses, camera_poses)


if __name__ == "__main__":
    # 测试代码
    solver = HandEyeSolver()

    # 测试旋转向量转换
    rotvec = np.array([0.1, 0.2, 0.3])
    R = solver.exp_rot(rotvec)
    rotvec_back = solver.log_rot(R)

    print("旋转向量转换测试:")
    print(f"  原始: {rotvec}")
    print(f"  恢复: {rotvec_back}")
    print(f"  误差: {np.linalg.norm(rotvec - rotvec_back)}")
