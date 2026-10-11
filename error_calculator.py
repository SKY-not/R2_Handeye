#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
误差计算模块 - 计算重投影误差
"""

import numpy as np
import cv2
import sys
import os
from typing import Dict, List, Optional, cast
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import REALSENSE_CONFIG
from calibration.transforms import invert_transform, pose_to_mat, average_transforms, target_poses
from calibration.targets import target_config, checkerboard_points, tag_correspondences, reprojection_errors
from camera.model import Rectifier


class ErrorCalculator:
    """重投影误差计算器"""

    def __init__(
        self,
        mode: str,
        intrinsics: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        backend: str = 'checkerboard',
        target: Optional[dict] = None,
    ) -> None:
        """
        初始化误差计算器

        Args:
            mode: 'eye_on_hand' 或 'eye_to_hand'
            intrinsics: 相机内参 (3x3)
            dist_coeffs: 相机畸变参数
        """
        self.mode = mode
        self.backend = backend
        self.target_config = target_config(backend) if target is None else target

        # 相机内参（优先使用外部传入真实内参）
        if intrinsics is not None:
            self.intrinsics = np.asarray(intrinsics, dtype=np.float64)
        else:
            intr_any = REALSENSE_CONFIG.get('default_intrinsics')
            intr = cast(Dict[str, float], intr_any)
            self.intrinsics = np.array([
                [float(intr['fx']), 0, float(intr['cx'])],
                [0, float(intr['fy']), float(intr['cy'])],
                [0, 0, 1]
            ], dtype=np.float64)
            
        # 相机畸变参数（如果未提供，则默认为0）
        if dist_coeffs is not None:
            self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        # 棋盘格参数
        self.rectifier = Rectifier(self.intrinsics, self.dist_coeffs)

    def _build_checkerboard_object_points(self) -> np.ndarray:
        return checkerboard_points(self.target_config)

    @staticmethod
    def _rotation_angle_error_rad(R1: np.ndarray, R2: np.ndarray) -> float:
        """Compute relative rotation angle between two rotation matrices in radians."""
        R_rel = R1 @ R2.T
        cos_theta = (np.trace(R_rel) - 1.0) / 2.0
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        return float(np.arccos(cos_theta))

    @staticmethod
    def _average_transforms(transforms: List[np.ndarray]) -> np.ndarray:
        """Average rigid transforms by averaging translations and rotations separately."""
        return average_transforms(transforms)

    @staticmethod
    def _project_points(T_camera_target: np.ndarray, object_points: np.ndarray, intrinsics: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
        rvec, _ = cv2.Rodrigues(T_camera_target[:3, :3])
        tvec = T_camera_target[:3, 3].reshape(3, 1)
        image_points, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsics, dist_coeffs)
        return image_points.reshape(-1, 2)

    def _apriltag_projection_params(self, image: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the camera model used by the AprilTag detector.

        AprilTag detection is performed on undistorted images with alpha=0 optimal intrinsics,
        so reprojection must use the same camera matrix and zero distortion.
        """
        if image is None or image.size == 0:
            raise ValueError('Image dimensions are required for rectified-pixel reprojection')
        h, w = image.shape[:2]
        return self.rectifier.prepare((w, h)), np.zeros(5)

    def _apriltag_frame_reprojection_error(
        self, T_camera_target: np.ndarray, tag_corners: np.ndarray, tag_ids: np.ndarray,
        intrinsics: np.ndarray, dist_coeffs: np.ndarray,
    ) -> float:
        points, corners = tag_correspondences(self.target_config, tag_ids, tag_corners,
                                              self.backend == 'apriltag_board')
        return float(reprojection_errors(T_camera_target, points, corners, intrinsics, dist_coeffs).mean())

    def calculate_reprojection_error(self, robot_poses, camera_poses, corners_2d_list, X,
                                     board_to_base=None, board_to_tcp=None) -> np.ndarray:
        if self.backend != 'checkerboard':
            raise ValueError('Use the AprilTag reprojection evaluator for this backend')
        if not (len(robot_poses) == len(camera_poses) == len(corners_2d_list)):
            raise ValueError('Robot/camera/corner sample counts differ')
        targets = target_poses(self.mode, robot_poses, camera_poses, X)
        provided = board_to_base if self.mode == 'eye_on_hand' else board_to_tcp
        reference = average_transforms(targets) if provided is None else pose_to_mat(provided)
        left = [invert_transform(t) for t in robot_poses] if self.mode == 'eye_on_hand' else robot_poses
        predictions = invert_transform(X) @ np.asarray(left) @ reference
        points = self._build_checkerboard_object_points()
        errors = []
        for pose, corners in zip(predictions, corners_2d_list):
            corners = np.asarray(corners).reshape(-1, 2)
            if len(corners) != len(points):
                raise ValueError('Checkerboard corner count differs from the recorded geometry')
            try:
                errors.append(float(reprojection_errors(pose, points, corners, self.intrinsics, self.dist_coeffs).mean()))
            except ValueError:
                errors.append(float('nan'))
        return np.asarray(errors)

    def calculate_pose_error_components(
        self, robot_poses: List[np.ndarray], camera_poses: List[np.ndarray], X: np.ndarray,
        board_to_base: Optional[List[float]] = None, board_to_tcp: Optional[List[float]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Residuals in the reference target frame, including ETH's stationary TCP frame."""
        targets = target_poses(self.mode, robot_poses, camera_poses, X)
        provided = board_to_base if self.mode == 'eye_on_hand' else board_to_tcp
        reference = average_transforms(targets) if provided is None else pose_to_mat(provided)
        errors = invert_transform(reference) @ targets
        return errors[:, :3, 3], Rotation.from_matrix(errors[:, :3, :3]).as_rotvec()

    def calculate_position_error(self, robot_poses, camera_poses, X,
                                 board_to_base=None, board_to_tcp=None) -> np.ndarray:
        positions, _ = self.calculate_pose_error_components(robot_poses, camera_poses, X,
                                                             board_to_base, board_to_tcp)
        return np.linalg.norm(positions, axis=1)

    def calculate_rotation_error(self, robot_poses, camera_poses, X,
                                 board_to_base=None, board_to_tcp=None) -> np.ndarray:
        _, rotations = self.calculate_pose_error_components(robot_poses, camera_poses, X,
                                                             board_to_base, board_to_tcp)
        return np.linalg.norm(rotations, axis=1)

    def calculate_spatial_consistency(self, robot_poses, camera_poses, X):
        positions, rotations = self.calculate_pose_error_components(robot_poses, camera_poses, X)
        return np.linalg.norm(positions, axis=1), np.linalg.norm(rotations, axis=1), positions, rotations

    def calculate_apriltag_observed_reprojection_error(
        self,
        camera_poses: List[np.ndarray],
        tag_corners_list: List[np.ndarray],
        tag_ids_list: List[np.ndarray],
        images: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """Evaluate AprilTag projection using each frame's observed tag/board pose."""
        if self.backend not in ('apriltag', 'apriltag_board'):
            return np.array([], dtype=np.float64)

        if not (len(camera_poses) == len(tag_corners_list) == len(tag_ids_list)):
            raise ValueError('Pose/corner/ID sample counts differ')
        if images is not None and len(images) != len(camera_poses):
            raise ValueError('Image/pose sample counts differ')
        frame_errors: List[float] = []
        if images is None:
            images = [None] * len(camera_poses)  # type: ignore[list-item]
        for camera_pose, tag_corners, tag_ids, image in zip(camera_poses, tag_corners_list, tag_ids_list, images):
            proj_intr, proj_dist = self._apriltag_projection_params(image)
            err = self._apriltag_frame_reprojection_error(camera_pose, tag_corners, tag_ids, proj_intr, proj_dist)
            if err is not None:
                frame_errors.append(err)
        return np.asarray(frame_errors, dtype=np.float64)

    def calculate_apriltag_chain_reprojection_error(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        tag_corners_list: List[np.ndarray],
        tag_ids_list: List[np.ndarray],
        X: np.ndarray,
        images: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Evaluate AprilTag reprojection through the full hand-eye chain.

        The target reference is the spatial-consistency mean pose, so this does not depend on
        the manually supplied rough target pose.
        """
        if self.backend not in ('apriltag', 'apriltag_board'):
            return np.array([], dtype=np.float64)

        if len(robot_poses) != len(camera_poses):
            raise ValueError('Robot/camera sample counts differ')
        if self.mode == 'eye_on_hand':
            target_poses = [tcp @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
            T_ref = self._average_transforms(target_poses)
            predicted_camera_poses = [invert_transform(X) @ invert_transform(tcp) @ T_ref for tcp in robot_poses]
        else:
            target_poses = [invert_transform(tcp) @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
            T_ref = self._average_transforms(target_poses)
            predicted_camera_poses = [invert_transform(X) @ tcp @ T_ref for tcp in robot_poses]

        if not (len(camera_poses) == len(tag_corners_list) == len(tag_ids_list)):
            raise ValueError('Pose/corner/ID sample counts differ')
        if images is not None and len(images) != len(camera_poses):
            raise ValueError('Image/pose sample counts differ')
        frame_errors: List[float] = []
        if images is None:
            images = [None] * len(predicted_camera_poses)  # type: ignore[list-item]
        for camera_pose, tag_corners, tag_ids, image in zip(predicted_camera_poses, tag_corners_list, tag_ids_list, images):
            proj_intr, proj_dist = self._apriltag_projection_params(image)
            err = self._apriltag_frame_reprojection_error(camera_pose, tag_corners, tag_ids, proj_intr, proj_dist)
            if err is not None:
                frame_errors.append(err)
        return np.asarray(frame_errors, dtype=np.float64)

    def visualize_reprojection_frames(
        self,
        images: List[np.ndarray],
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        corners_2d_list: List[np.ndarray],
        X: np.ndarray,
        board_to_base: Optional[List[float]] = None,
        board_to_tcp: Optional[List[float]] = None,
        window_name: str = 'Reprojection Frame Viewer'
    ) -> None:
        """
        逐帧显示重投影结果。

        - 绿色圆点: 检测到的角点
        - 红色圆点: 根据标定结果重投影得到的角点
        - Eye-on-Hand: 若给定 board_to_base，使用其作为固定参考；否则使用观测均值参考
        - Eye-to-Hand: 若给定 board_to_tcp，使用其作为 TCP 系固定参考；否则使用观测均值参考
        - 左/右方向键（兼容 Shift+方向键）: 切换帧
        - Esc: 退出
        """
        if self.backend in ('apriltag', 'apriltag_board'):
            print('AprilTag 后端跳过逐帧重投影显示')
            return

        n = min(len(images), len(robot_poses), len(camera_poses), len(corners_2d_list))
        if n == 0:
            print('无可视化数据，跳过逐帧重投影显示')
            return

        objp = self._build_checkerboard_object_points()

        if self.mode == 'eye_on_hand':
            if board_to_base is not None:
                T_base_board_ref = pose_to_mat(board_to_base)
            else:
                base_boards = [tcp @ X @ cam_pose for tcp, cam_pose in zip(robot_poses[:n], camera_poses[:n])]
                T_base_board_ref = average_transforms(base_boards)
        else:
            if board_to_tcp is not None:
                T_tcp_board_ref = pose_to_mat(board_to_tcp)
            else:
                tcp_board_list = [invert_transform(tcp) @ X @ cam_pose for tcp, cam_pose in zip(robot_poses[:n], camera_poses[:n])]
                T_tcp_board_ref = average_transforms(tcp_board_list)

        idx = 0
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            img_src = images[idx]
            if img_src is None:
                canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                if img_src.ndim == 2:
                    canvas = cv2.cvtColor(img_src.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                else:
                    canvas = img_src.copy().astype(np.uint8)

            if self.mode == 'eye_on_hand':
                T_cam_board_pred = invert_transform(X) @ invert_transform(robot_poses[idx]) @ T_base_board_ref
            else:
                T_cam_board_pred = invert_transform(X) @ robot_poses[idx] @ T_tcp_board_ref

            R = T_cam_board_pred[:3, :3]
            t = T_cam_board_pred[:3, 3]
            
            pcam = (R @ objp.T + t.reshape(3, 1)).T
            valid = pcam[:, 2] > 1e-8

            if np.any(valid):
                rvec, _ = cv2.Rodrigues(R)
                img_pts, _ = cv2.projectPoints(objp, rvec, t, self.intrinsics, self.dist_coeffs)
                reproj_pts = img_pts.reshape(-1, 2)
                reproj_pts = reproj_pts[valid]
            else:
                reproj_pts = np.zeros((0, 2), dtype=np.float64)

            det = corners_2d_list[idx]
            det_pts = det.reshape(-1, 2)

            for pt in det_pts:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)

            if len(det_pts) > 0:
                det_start = tuple(det_pts[0].astype(int))
                det_end = tuple(det_pts[-1].astype(int))
                cv2.circle(canvas, det_start, 8, (0, 255, 255), -1)
                cv2.putText(canvas, 'DET_START', (det_start[0] + 8, det_start[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.circle(canvas, det_end, 8, (255, 0, 255), -1)
                cv2.putText(canvas, 'DET_END', (det_end[0] + 8, det_end[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

            for pt in reproj_pts:
                x_i, y_i = int(pt[0]), int(pt[1])
                if 0 <= x_i < canvas.shape[1] and 0 <= y_i < canvas.shape[0]:
                    cv2.circle(canvas, (x_i, y_i), 4, (0, 0, 255), -1)

            if len(reproj_pts) > 0:
                rep_start = tuple(reproj_pts[0].astype(int))
                rep_end = tuple(reproj_pts[-1].astype(int))
                if 0 <= rep_start[0] < canvas.shape[1] and 0 <= rep_start[1] < canvas.shape[0]:
                    cv2.circle(canvas, rep_start, 8, (0, 165, 255), -1)
                    cv2.putText(canvas, 'REPROJ_START', (rep_start[0] + 8, rep_start[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
                if 0 <= rep_end[0] < canvas.shape[1] and 0 <= rep_end[1] < canvas.shape[0]:
                    cv2.circle(canvas, rep_end, 8, (255, 255, 0), -1)
                    cv2.putText(canvas, 'REPROJ_END', (rep_end[0] + 8, rep_end[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            if len(reproj_pts) > 0:
                frame_err = np.linalg.norm(det_pts[valid] - reproj_pts, axis=1)
                frame_err_text = f'mean err: {float(np.mean(frame_err)):.3f}px'
            else:
                frame_err_text = 'mean err: N/A'

            cv2.putText(
                canvas,
                f'Frame {idx + 1}/{n} | Green: detected | Red: reprojected | {frame_err_text}',
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            cv2.putText(
                canvas,
                'Left/Right: switch frame (Shift compatible) | Esc: exit',
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            cv2.imshow(window_name, canvas)
            key_raw = cv2.waitKeyEx(0)
            key = key_raw & 0xFF

            if key == 27:
                break

            # Linux/X11 often reports 81/83 for left/right after mask; Windows uses large codes.
            is_left = key in (81,) or key_raw in (2424832,)
            is_right = key in (83,) or key_raw in (2555904,)

            if is_left:
                idx = (idx - 1) % n
            elif is_right:
                idx = (idx + 1) % n

        cv2.destroyWindow(window_name)

    def compute_statistics(self, errors: np.ndarray) -> Dict[str, float]:
        """
        计算误差统计信息

        Args:
            errors: 误差数组

        Returns:
            dict: 统计信息
        """
        if len(errors) == 0:
            return dict.fromkeys(('mean', 'max', 'min', 'std', 'median'), float('nan'))

        return {
            'mean': np.mean(errors),
            'max': np.max(errors),
            'min': np.min(errors),
            'std': np.std(errors),
            'median': np.median(errors)
        }

    def print_error_report(
        self,
        errors: np.ndarray,
        title: str = "重投影误差报告",
        unit: str = "auto"
    ) -> None:
        """
        打印误差报告

        Args:
            errors: 误差数组
            title: 报告标题
            unit: 输出单位, 可选 'auto'/'px'/'m'/'deg'
        """
        if len(errors) == 0 or not np.isfinite(errors).all():
            print(f'\n{title}: N/A (no complete valid evaluation data)')
            return
        stats = self.compute_statistics(errors)

        print("\n" + "=" * 50)
        print(title)
        print("=" * 50)
        print(f"数据点数: {len(errors)}")
        if unit == 'deg':
            print(f"平均误差: {stats['mean']:.3f} deg")
            print(f"最大误差: {stats['max']:.3f} deg")
            print(f"最小误差: {stats['min']:.3f} deg")
            print(f"标准差:   {stats['std']:.3f} deg")
            print(f"中位数:   {stats['median']:.3f} deg")
            mean_metric = stats['mean']
        elif unit == 'px':
            print(f"平均误差: {stats['mean']:.3f} px")
            print(f"最大误差: {stats['max']:.3f} px")
            print(f"最小误差: {stats['min']:.3f} px")
            print(f"标准差:   {stats['std']:.3f} px")
            print(f"中位数:   {stats['median']:.3f} px")
            mean_metric = stats['mean']
        elif unit == 'm':
            print(f"平均误差: {stats['mean']*1000:.3f} mm")
            print(f"最大误差: {stats['max']*1000:.3f} mm")
            print(f"最小误差: {stats['min']*1000:.3f} mm")
            print(f"标准差:   {stats['std']*1000:.3f} mm")
            print(f"中位数:   {stats['median']*1000:.3f} mm")
            mean_metric = stats['mean'] * 1000
        elif stats['mean'] > 0.1:
            print(f"平均误差: {stats['mean']:.3f} px")
            print(f"最大误差: {stats['max']:.3f} px")
            print(f"最小误差: {stats['min']:.3f} px")
            print(f"标准差:   {stats['std']:.3f} px")
            print(f"中位数:   {stats['median']:.3f} px")
            mean_metric = stats['mean']
        else:
            print(f"平均误差: {stats['mean']*1000:.3f} mm")
            print(f"最大误差: {stats['max']*1000:.3f} mm")
            print(f"最小误差: {stats['min']*1000:.3f} mm")
            print(f"标准差:   {stats['std']*1000:.3f} mm")
            print(f"中位数:   {stats['median']*1000:.3f} mm")
            mean_metric = stats['mean'] * 1000

        # 评级
        if mean_metric < 2:
            rating = "Excellent (优秀)"
        elif mean_metric < 5:
            rating = "Good (良好)"
        elif mean_metric < 10:
            rating = "Fair (一般)"
        else:
            rating = "Poor (较差)"

        print(f"评级: {rating}")
        print("=" * 50)
