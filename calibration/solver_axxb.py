#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手眼标定求解器
基于 AX=XB 模型
支持 Eye-on-Hand 和 Eye-to-Hand 两种模式
"""

import numpy as np
from scipy import optimize
from scipy.optimize import OptimizeResult
from typing import List, Optional, Tuple
from calibration.transforms import mat_to_pose, pose_to_mat, invert_transform


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
        return mat_to_pose(T)

    @staticmethod
    def pose_to_mat(pose: np.ndarray) -> np.ndarray:
        """
        位姿数组转齐次变换矩阵

        Args:
            pose: [x, y, z, rx, ry, rz]

        Returns:
            T: 4x4 齐次变换矩阵
        """
        return pose_to_mat(pose)

    @staticmethod
    def invert_transform(T: np.ndarray) -> np.ndarray:
        """
        求逆变换

        Args:
            T: 4x4 齐次变换矩阵

        Returns:
            T_inv: 逆变换矩阵
        """
        return invert_transform(T)

    # AX=XB 求解
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

        for i in range(n):
            for j in range(i + 1, n):
                if self.mode == 'eye_to_hand':
                    A = np.asarray(robot_poses[j] @ self.invert_transform(robot_poses[i]), dtype=np.float64)
                    B = np.asarray(camera_poses[j] @ self.invert_transform(camera_poses[i]), dtype=np.float64)
                else:
                    A = np.asarray(self.invert_transform(robot_poses[i]) @ robot_poses[j], dtype=np.float64)
                    B = np.asarray(camera_poses[i] @ self.invert_transform(camera_poses[j]), dtype=np.float64)
                A_list.append(A)
                B_list.append(B)

        # 使用 Tsai-Lenz 方法的变体
        if not A_list:
            raise ValueError("No valid relative motions for AX=XB solving")

        R_X = self.solve_rotation(A_list, B_list)
        t_X = self.solve_translation(A_list, B_list, R_X)

        # 构建齐次变换矩阵
        X = np.eye(4)
        X[:3, :3] = R_X
        X[:3, 3] = t_X

        return np.asarray(X, dtype=np.float64)

    def solve_rotation(self, A_list: List[np.ndarray], B_list: List[np.ndarray]) -> np.ndarray:
        """
        求解旋转部分

        Args:
            A_list: 机器人相对运动列表
            B_list: 相机相对运动列表

        Returns:
            R_X: 手眼旋转矩阵
        """
        if len(A_list) != len(B_list):
            raise ValueError("A_list and B_list must have the same length")

        I = np.eye(3, dtype=np.float64)
        rows = []

        for A, B in zip(A_list, B_list):
            R_A = np.asarray(A[:3, :3], dtype=np.float64)
            R_B = np.asarray(B[:3, :3], dtype=np.float64)
            rows.append(np.kron(I, R_A) - np.kron(R_B.T, I))

        M = np.vstack(rows)
        _, _, Vt = np.linalg.svd(M)
        R_raw = Vt[-1].reshape(3, 3, order='F')
        if np.linalg.det(R_raw) < 0:
            R_raw = -R_raw

        # Project the linear solution back onto SO(3).
        U, _, Vt = np.linalg.svd(R_raw)
        R_X = U @ Vt
        if np.linalg.det(R_X) < 0:
            U[:, -1] *= -1
            R_X = U @ Vt

        return np.asarray(R_X, dtype=np.float64)

    def solve_translation(
        self,
        A_list: List[np.ndarray],
        B_list: List[np.ndarray],
        R_X: np.ndarray
    ) -> np.ndarray:
        """
        求解平移部分

        Args:
            A_list: 机器人相对运动列表
            B_list: 相机相对运动列表
            R_X: 已求解的旋转矩阵

        Returns:
            t_X: 手眼平移向量
        """
        if len(A_list) != len(B_list):
            raise ValueError("A_list and B_list must have the same length")

        lhs_rows = []
        rhs_rows = []
        I = np.eye(3, dtype=np.float64)

        for A, B in zip(A_list, B_list):
            R_A = np.asarray(A[:3, :3], dtype=np.float64)
            t_A = np.asarray(A[:3, 3], dtype=np.float64)
            t_B = np.asarray(B[:3, 3], dtype=np.float64)

            lhs_rows.append(R_A - I)
            rhs_rows.append(R_X @ t_B - t_A)

        M = np.vstack(lhs_rows)
        rhs = np.concatenate(rhs_rows)
        t_X = np.linalg.lstsq(M, rhs, rcond=None)[0]

        return np.asarray(t_X, dtype=np.float64)

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
