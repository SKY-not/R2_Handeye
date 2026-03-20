#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
误差计算模块 - 计算重投影误差
"""

import numpy as np
import sys
import os
from typing import Dict, List, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CHECKERBOARD_CONFIG, REALSENSE_CONFIG


class ErrorCalculator:
    """重投影误差计算器"""

    def __init__(self, mode: str) -> None:
        """
        初始化误差计算器

        Args:
            mode: 'eye_on_hand' 或 'eye_to_hand'
        """
        self.mode = mode

        # 相机内参
        intr = REALSENSE_CONFIG['default_intrinsics']
        self.intrinsics = np.array([
            [intr['fx'], 0, intr['cx']],
            [0, intr['fy'], intr['cy']],
            [0, 0, 1]
        ])

        # 棋盘格参数
        self.cb_size = CHECKERBOARD_CONFIG['size']
        self.square_size = CHECKERBOARD_CONFIG['square_size']

    @staticmethod
    def _pose_to_mat(pose: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Convert [x,y,z,rx,ry,rz] pose to 4x4 matrix."""
        if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
            return pose

        x, y, z, rx, ry, rz = pose
        theta = np.linalg.norm([rx, ry, rz])
        if theta > 1e-8:
            axis = np.array([rx, ry, rz]) / theta
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        else:
            R = np.eye(3)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def _build_checkerboard_object_points(self) -> np.ndarray:
        """Build checkerboard corner points in board coordinate frame."""
        cb_cols, cb_rows = self.cb_size
        objp = np.zeros((cb_cols * cb_rows, 3), dtype=np.float64)
        objp[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    @staticmethod
    def _invert(T: np.ndarray) -> np.ndarray:
        """Invert homogeneous transform."""
        R = T[:3, :3]
        t = T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t
        return T_inv

    def calculate_reprojection_error(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        corners_2d_list: List[np.ndarray],
        X: np.ndarray,
        z_scale: float = 1.0
    ) -> np.ndarray:
        """
        计算重投影误差

        对于每帧数据:
        1. 将标定板角点从相机坐标系转换到世界(基座)坐标系
        2. 将世界坐标投影到像素坐标系
        3. 与检测到的角点比较

        Args:
            robot_poses: 机器人位姿列表
            camera_poses: 相机观测位姿列表 (T_cam_board)
            corners_2d_list: 每帧检测角点 (N, 2)
            X: 手眼变换矩阵
            z_scale: 深度缩放因子

        Returns:
            errors: 每帧的重投影误差 (像素)
        """
        if not (len(robot_poses) == len(camera_poses) == len(corners_2d_list)):
            raise ValueError("robot_poses/camera_poses/corners_2d_list 长度不一致")

        objp = self._build_checkerboard_object_points()
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        cx = self.intrinsics[0, 2]
        cy = self.intrinsics[1, 2]

        frame_errors = []

        if self.mode == 'eye_on_hand':
            base_boards = [tcp @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
            T_base_board_ref = np.mean(np.stack(base_boards), axis=0)

            for tcp, corners in zip(robot_poses, corners_2d_list):
                T_cam_board_pred = self._invert(X) @ self._invert(tcp) @ T_base_board_ref
                R = T_cam_board_pred[:3, :3]
                t = T_cam_board_pred[:3, 3]
                reproj = (R @ objp.T + t.reshape(3, 1)).T
                valid = reproj[:, 2] > 1e-8
                if not np.any(valid):
                    continue
                u_proj = fx * reproj[valid, 0] / reproj[valid, 2] + cx
                v_proj = fy * reproj[valid, 1] / reproj[valid, 2] + cy
                det = corners[valid]
                err = np.sqrt((det[:, 0] - u_proj) ** 2 + (det[:, 1] - v_proj) ** 2)
                frame_errors.append(float(np.mean(err)))
        else:
            base_to_tcp_board_list = [self._invert(tcp) @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
            T_tcp_board_ref = np.mean(np.stack(base_to_tcp_board_list), axis=0)

            for tcp, corners in zip(robot_poses, corners_2d_list):
                T_cam_board_pred = self._invert(X) @ tcp @ T_tcp_board_ref
                R = T_cam_board_pred[:3, :3]
                t = T_cam_board_pred[:3, 3]
                reproj = (R @ objp.T + t.reshape(3, 1)).T
                valid = reproj[:, 2] > 1e-8
                if not np.any(valid):
                    continue
                u_proj = fx * reproj[valid, 0] / reproj[valid, 2] + cx
                v_proj = fy * reproj[valid, 1] / reproj[valid, 2] + cy
                det = corners[valid]
                err = np.sqrt((det[:, 0] - u_proj) ** 2 + (det[:, 1] - v_proj) ** 2)
                frame_errors.append(float(np.mean(err)))

        return np.array(frame_errors, dtype=np.float64)

    def calculate_position_error(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        X: np.ndarray,
        z_scale: float = 1.0,
        board_to_base: Optional[List[float]] = None,
        board_to_tcp: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        计算位置误差 (3D误差)

        将标定板3D点转换到基座坐标系，与:
        - Eye-on-Hand: 粗略估计的标定板位置比较
        - Eye-to-Hand: TCP @ board_to_tcp 比较

        Args:
            robot_poses: 机器人位姿列表
            camera_poses: 相机观测位姿列表 (T_cam_board)
            X: 手眼变换矩阵
            z_scale: 深度缩放因子
            board_to_base: 标定板相对于基座的位姿 (Eye-on-Hand用)
            board_to_tcp: 标定板相对于TCP的位姿 (Eye-to-Hand用)

        Returns:
            errors: 位置误差列表 (米)
        """
        errors = []

        for tcp, cam_pose in zip(robot_poses, camera_poses):
            if self.mode == 'eye_on_hand':
                T_measured = tcp @ X @ cam_pose
                p_measured = T_measured[:3, 3]

                # 与粗略估计比较
                if board_to_base is not None:
                    T_expected = self._pose_to_mat(board_to_base)
                    p_expected = T_expected[:3, 3]
                    error = np.linalg.norm(p_measured - p_expected)
                    errors.append(error)
                else:
                    errors.append(0)
            else:
                T_measured = X @ cam_pose
                p_measured = T_measured[:3, 3]

                # 与粗略估计比较
                if board_to_tcp is not None:
                    T_board_tcp = self._pose_to_mat(board_to_tcp)
                    T_expected = tcp @ T_board_tcp
                    p_expected = T_expected[:3, 3]
                    error = np.linalg.norm(p_measured - p_expected)
                    errors.append(error)
                else:
                    errors.append(0)

        return np.array(errors)

    def compute_statistics(self, errors: np.ndarray) -> Dict[str, float]:
        """
        计算误差统计信息

        Args:
            errors: 误差数组

        Returns:
            dict: 统计信息
        """
        if len(errors) == 0:
            return {'mean': 0, 'max': 0, 'min': 0, 'std': 0}

        return {
            'mean': np.mean(errors),
            'max': np.max(errors),
            'min': np.min(errors),
            'std': np.std(errors),
            'median': np.median(errors)
        }

    def print_error_report(self, errors: np.ndarray, title: str = "重投影误差报告") -> None:
        """
        打印误差报告

        Args:
            errors: 误差数组
            title: 报告标题
        """
        stats = self.compute_statistics(errors)

        print("\n" + "=" * 50)
        print(title)
        print("=" * 50)
        print(f"数据点数: {len(errors)}")
        # Heuristic unit display: pixel-scale metrics are usually > 0.1,
        # while meter-scale metrics (position) are much smaller.
        if stats['mean'] > 0.1:
            print(f"平均误差: {stats['mean']:.3f} px")
            print(f"最大误差: {stats['max']:.3f} px")
            print(f"最小误差: {stats['min']:.3f} px")
            print(f"标准差:   {stats['std']:.3f} px")
            print(f"中位数:   {stats['median']:.3f} px")
            mean_mm = stats['mean']
        else:
            print(f"平均误差: {stats['mean']*1000:.3f} mm")
            print(f"最大误差: {stats['max']*1000:.3f} mm")
            print(f"最小误差: {stats['min']*1000:.3f} mm")
            print(f"标准差:   {stats['std']*1000:.3f} mm")
            print(f"中位数:   {stats['median']*1000:.3f} mm")
            mean_mm = stats['mean'] * 1000

        # 评级
        if mean_mm < 2:
            rating = "Excellent (优秀)"
        elif mean_mm < 5:
            rating = "Good (良好)"
        elif mean_mm < 10:
            rating = "Fair (一般)"
        else:
            rating = "Poor (较差)"

        print(f"评级: {rating}")
        print("=" * 50)
