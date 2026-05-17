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

from config import APRILTAG_BOARD_CONFIG, APRILTAG_CONFIG, CHECKERBOARD_CONFIG, REALSENSE_CONFIG
from calibration.transforms import invert_transform, matrix_to_rotvec, pose_to_mat


class ErrorCalculator:
    """重投影误差计算器"""

    def __init__(
        self,
        mode: str,
        intrinsics: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        backend: str = 'checkerboard'
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
        cb_size_any = CHECKERBOARD_CONFIG['size']
        cb_size_tuple = cast(tuple[int, int], cb_size_any)
        self.cb_size: tuple[int, int] = (int(cb_size_tuple[0]), int(cb_size_tuple[1]))
        self.square_size: float = float(cast(float, CHECKERBOARD_CONFIG['square_size']))

    def _build_checkerboard_object_points(self) -> np.ndarray:
        """Build checkerboard corner points in board coordinate frame."""
        cb_cols, cb_rows = self.cb_size
        objp = np.zeros((cb_cols * cb_rows, 3), dtype=np.float64)
        objp[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2)
        objp *= self.square_size
        return objp

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
        if not transforms:
            return np.eye(4, dtype=np.float64)
        translations = np.asarray([T[:3, 3] for T in transforms], dtype=np.float64)
        rotations = Rotation.from_matrix([T[:3, :3] for T in transforms])
        T_avg = np.eye(4, dtype=np.float64)
        T_avg[:3, 3] = np.mean(translations, axis=0)
        T_avg[:3, :3] = rotations.mean().as_matrix()
        return T_avg

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
            return self.intrinsics, np.zeros((5, 1), dtype=np.float64)

        h, w = image.shape[:2]
        new_k, _ = cv2.getOptimalNewCameraMatrix(
            self.intrinsics,
            self.dist_coeffs,
            (w, h),
            alpha=0,
            newImgSize=(w, h)
        )
        return np.asarray(new_k, dtype=np.float64), np.zeros((5, 1), dtype=np.float64)

    @staticmethod
    def _best_square_corner_error(projected: np.ndarray, detected: np.ndarray) -> float:
        """Compare one projected square to detected corners, allowing cyclic/reversed ordering."""
        projected_4 = np.asarray(projected, dtype=np.float64).reshape(4, 2)
        detected_4 = np.asarray(detected, dtype=np.float64).reshape(4, 2)
        candidates = []
        for shift in range(4):
            candidates.append(np.roll(detected_4, shift, axis=0))
            candidates.append(np.roll(detected_4[::-1], shift, axis=0))
        return float(min(np.mean(np.linalg.norm(projected_4 - cand, axis=1)) for cand in candidates))

    def _single_tag_object_points(self) -> np.ndarray:
        s = float(cast(float, APRILTAG_CONFIG['tag_size']))
        h = 0.5 * s
        return np.array([
            [-h, -h, 0.0],
            [h, -h, 0.0],
            [h, h, 0.0],
            [-h, h, 0.0],
        ], dtype=np.float64)

    def _board_tag_object_points(self, tag_id: int) -> np.ndarray:
        s = float(cast(float, APRILTAG_BOARD_CONFIG['tag_size']))
        h = 0.5 * s
        centers = cast(Dict[int, List[float]], APRILTAG_BOARD_CONFIG['tag_centers'])
        center = np.asarray(centers[int(tag_id)], dtype=np.float64).reshape(3)
        local = np.array([
            [-h, -h, 0.0],
            [h, -h, 0.0],
            [h, h, 0.0],
            [-h, h, 0.0],
        ], dtype=np.float64)
        return local + center.reshape(1, 3)

    def _apriltag_frame_reprojection_error(
        self,
        T_camera_target: np.ndarray,
        tag_corners: np.ndarray,
        tag_ids: np.ndarray,
        intrinsics: np.ndarray,
        dist_coeffs: np.ndarray
    ) -> Optional[float]:
        corners = np.asarray(tag_corners, dtype=np.float64).reshape(-1, 2)
        if corners.shape[0] < 4:
            return None

        if self.backend == 'apriltag':
            object_points = self._single_tag_object_points()
            detected = corners[:4]
            projected = self._project_points(T_camera_target, object_points, intrinsics, dist_coeffs)
            return self._best_square_corner_error(projected, detected)

        if self.backend != 'apriltag_board':
            return None

        ids = np.asarray(tag_ids, dtype=np.int32).reshape(-1)
        if ids.size == 0:
            return None

        per_tag_errors: List[float] = []
        for idx, tag_id in enumerate(ids):
            start = idx * 4
            end = start + 4
            if end > corners.shape[0]:
                break
            object_points = self._board_tag_object_points(int(tag_id))
            detected = corners[start:end]
            projected = self._project_points(T_camera_target, object_points, intrinsics, dist_coeffs)
            per_tag_errors.append(self._best_square_corner_error(projected, detected))

        if not per_tag_errors:
            return None
        return float(np.mean(per_tag_errors))

    def calculate_reprojection_error(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        corners_2d_list: List[np.ndarray],
        X: np.ndarray,
        board_to_base: Optional[List[float]] = None,
        board_to_tcp: Optional[List[float]] = None
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
            board_to_base: 标定板相对于基座的位姿 (Eye-on-Hand用)
            board_to_tcp: 标定板相对于TCP的位姿 (Eye-to-Hand用)

        Returns:
            errors: 每帧的重投影误差 (像素)
        """
        if self.backend in ('apriltag', 'apriltag_board'):
            print("AprilTag 后端跳过重投影误差计算")
            return np.array([], dtype=np.float64)

        if not (len(robot_poses) == len(camera_poses) == len(corners_2d_list)):
            raise ValueError("robot_poses/camera_poses/corners_2d_list 长度不一致")

        objp = self._build_checkerboard_object_points()

        frame_errors: List[float] = []

        if self.mode == 'eye_on_hand':
            if board_to_base is not None:
                T_base_board_ref = pose_to_mat(board_to_base)
            else:
                base_boards = [tcp @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
                T_base_board_ref = np.mean(np.stack(base_boards), axis=0)

            for tcp, corners in zip(robot_poses, corners_2d_list):
                T_cam_board_pred = invert_transform(X) @ invert_transform(tcp) @ T_base_board_ref
                R = T_cam_board_pred[:3, :3]
                t = T_cam_board_pred[:3, 3]
                
                pcam = (R @ objp.T + t.reshape(3, 1)).T
                valid = pcam[:, 2] > 1e-8
                if not np.any(valid):
                    continue
                    
                rvec, _ = cv2.Rodrigues(R)
                img_pts, _ = cv2.projectPoints(objp, rvec, t, self.intrinsics, self.dist_coeffs)
                img_pts = img_pts.reshape(-1, 2)
                
                u_proj = img_pts[valid, 0]
                v_proj = img_pts[valid, 1]
                
                det = corners[valid]
                err = np.sqrt((det[:, 0] - u_proj) ** 2 + (det[:, 1] - v_proj) ** 2)
                frame_errors.append(float(np.mean(err)))
        else:
            if board_to_tcp is not None:
                T_tcp_board_ref = pose_to_mat(board_to_tcp)
            else:
                base_to_tcp_board_list = [invert_transform(tcp) @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
                T_tcp_board_ref = np.mean(np.stack(base_to_tcp_board_list), axis=0)

            for tcp, corners in zip(robot_poses, corners_2d_list):
                T_cam_board_pred = invert_transform(X) @ tcp @ T_tcp_board_ref
                R = T_cam_board_pred[:3, :3]
                t = T_cam_board_pred[:3, 3]
                
                pcam = (R @ objp.T + t.reshape(3, 1)).T
                valid = pcam[:, 2] > 1e-8
                if not np.any(valid):
                    continue
                    
                rvec, _ = cv2.Rodrigues(R)
                img_pts, _ = cv2.projectPoints(objp, rvec, t, self.intrinsics, self.dist_coeffs)
                img_pts = img_pts.reshape(-1, 2)
                
                u_proj = img_pts[valid, 0]
                v_proj = img_pts[valid, 1]
                
                det = corners[valid]
                err = np.sqrt((det[:, 0] - u_proj) ** 2 + (det[:, 1] - v_proj) ** 2)
                frame_errors.append(float(np.mean(err)))

        return np.array(frame_errors, dtype=np.float64)

    def calculate_position_error(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        X: np.ndarray,
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
            board_to_base: 标定板相对于基座的位姿 (Eye-on-Hand用)
            board_to_tcp: 标定板相对于TCP的位姿 (Eye-to-Hand用)

        Returns:
            errors: 位置误差列表 (米)
        """
        errors: List[float] = []

        for tcp, cam_pose in zip(robot_poses, camera_poses):
            if self.mode == 'eye_on_hand':
                T_measured = tcp @ X @ cam_pose
                p_measured = T_measured[:3, 3]

                # 与粗略估计比较
                if board_to_base is not None:
                    T_expected = pose_to_mat(board_to_base)
                    p_expected = T_expected[:3, 3]
                    error = float(np.linalg.norm(p_measured - p_expected))
                    errors.append(error)
                else:
                    errors.append(float(0.0))
            else:
                T_measured = X @ cam_pose
                p_measured = T_measured[:3, 3]

                # 与粗略估计比较
                if board_to_tcp is not None:
                    T_board_tcp = pose_to_mat(board_to_tcp)
                    T_expected = tcp @ T_board_tcp
                    p_expected = T_expected[:3, 3]
                    error = float(np.linalg.norm(p_measured - p_expected))
                    errors.append(error)
                else:
                    errors.append(float(0.0))

        return np.array(errors)

    def calculate_rotation_error(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        X: np.ndarray,
        board_to_base: Optional[List[float]] = None,
        board_to_tcp: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        计算旋转误差（相对角，单位 rad）

        对比对象与位置误差保持一致：
        - Eye-on-Hand: 将观测到的标定板姿态(T_base_board)与 board_to_base 粗略姿态比较。
                - Eye-to-Hand: 将观测到的标定板姿态(T_base_board)与
                    由 board_to_tcp 转换得到的基座系参考姿态(T_base_tcp @ T_board_tcp)比较。

        若未提供粗略姿态，则以观测平均姿态作为参考。
        """
        errors_rad: List[float] = []

        if self.mode == 'eye_on_hand':
            measured_list = [tcp @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
            if board_to_base is not None:
                T_ref = pose_to_mat(board_to_base)
            else:
                T_ref = np.mean(np.stack(measured_list), axis=0)

            R_ref = T_ref[:3, :3]
            for T_meas in measured_list:
                angle_rad = self._rotation_angle_error_rad(T_meas[:3, :3], R_ref)
                errors_rad.append(float(angle_rad))
        else:
            measured_list = [X @ cam_pose for cam_pose in camera_poses]
            if board_to_tcp is not None:
                T_board_tcp = pose_to_mat(board_to_tcp)
                for tcp, T_meas in zip(robot_poses, measured_list):
                    T_ref = tcp @ T_board_tcp
                    angle_rad = self._rotation_angle_error_rad(T_meas[:3, :3], T_ref[:3, :3])
                    errors_rad.append(float(angle_rad))
            else:
                T_ref = np.mean(np.stack(measured_list), axis=0)
                R_ref = T_ref[:3, :3]
                for T_meas in measured_list:
                    angle_rad = self._rotation_angle_error_rad(T_meas[:3, :3], R_ref)
                    errors_rad.append(float(angle_rad))

        return np.array(errors_rad, dtype=np.float64)

    def calculate_pose_error_components(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        X: np.ndarray,
        board_to_base: Optional[List[float]] = None,
        board_to_tcp: Optional[List[float]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate signed pose error components in the reference target frame.

        Returns:
            position_components: shape (N, 3), unit meter, expressed in reference target frame.
            rotation_components: shape (N, 3), unit rad, rotation vector of reference-to-measured error.
        """
        position_components: List[np.ndarray] = []
        rotation_components: List[np.ndarray] = []

        if self.mode == 'eye_on_hand':
            measured_list = [tcp @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
            if board_to_base is not None:
                ref_list = [pose_to_mat(board_to_base) for _ in measured_list]
            else:
                T_ref = np.mean(np.stack(measured_list), axis=0)
                ref_list = [T_ref for _ in measured_list]
        else:
            measured_list = [X @ cam_pose for cam_pose in camera_poses]
            if board_to_tcp is not None:
                T_tcp_board = pose_to_mat(board_to_tcp)
                ref_list = [tcp @ T_tcp_board for tcp in robot_poses[:len(measured_list)]]
            else:
                T_ref = np.mean(np.stack(measured_list), axis=0)
                ref_list = [T_ref for _ in measured_list]

        for T_measured, T_ref in zip(measured_list, ref_list):
            T_error = invert_transform(T_ref) @ T_measured
            position_components.append(np.asarray(T_error[:3, 3], dtype=np.float64))
            rotation_components.append(matrix_to_rotvec(T_error[:3, :3]))

        return (
            np.asarray(position_components, dtype=np.float64),
            np.asarray(rotation_components, dtype=np.float64),
        )

    def calculate_spatial_consistency(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate target-pose consistency without using a manually measured reference pose.

        Eye-on-Hand: target should be fixed in base.
        Eye-to-Hand: target should be fixed in TCP.
        """
        if self.mode == 'eye_on_hand':
            target_poses = [tcp @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
        else:
            target_poses = [invert_transform(tcp) @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]

        T_ref = self._average_transforms(target_poses)
        pos_components: List[np.ndarray] = []
        rot_components: List[np.ndarray] = []
        pos_errors: List[float] = []
        rot_errors: List[float] = []

        for T_target in target_poses:
            T_error = invert_transform(T_ref) @ T_target
            pos_component = np.asarray(T_error[:3, 3], dtype=np.float64)
            rot_component = matrix_to_rotvec(T_error[:3, :3])
            pos_components.append(pos_component)
            rot_components.append(rot_component)
            pos_errors.append(float(np.linalg.norm(pos_component)))
            rot_errors.append(float(np.linalg.norm(rot_component)))

        return (
            np.asarray(pos_errors, dtype=np.float64),
            np.asarray(rot_errors, dtype=np.float64),
            np.asarray(pos_components, dtype=np.float64),
            np.asarray(rot_components, dtype=np.float64),
        )

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

        if self.mode == 'eye_on_hand':
            target_poses = [tcp @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
            T_ref = self._average_transforms(target_poses)
            predicted_camera_poses = [invert_transform(X) @ invert_transform(tcp) @ T_ref for tcp in robot_poses]
        else:
            target_poses = [invert_transform(tcp) @ X @ cam_pose for tcp, cam_pose in zip(robot_poses, camera_poses)]
            T_ref = self._average_transforms(target_poses)
            predicted_camera_poses = [invert_transform(X) @ tcp @ T_ref for tcp in robot_poses]

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
                T_base_board_ref = np.mean(np.stack(base_boards), axis=0)
        else:
            if board_to_tcp is not None:
                T_tcp_board_ref = pose_to_mat(board_to_tcp)
            else:
                tcp_board_list = [invert_transform(tcp) @ X @ cam_pose for tcp, cam_pose in zip(robot_poses[:n], camera_poses[:n])]
                T_tcp_board_ref = np.mean(np.stack(tcp_board_list), axis=0)

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
                pair_n = min(len(det_pts), len(reproj_pts))
                frame_err = np.linalg.norm(det_pts[:pair_n] - reproj_pts[:pair_n], axis=1)
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
            return {'mean': 0, 'max': 0, 'min': 0, 'std': 0, 'median': 0}

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
