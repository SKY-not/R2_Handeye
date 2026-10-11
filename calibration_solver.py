#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标定计算模块 - 负责 AX=XB 求解
"""

import numpy as np
import os
import sys
import cv2
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, cast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration.solver_axxb import HandEyeSolver
from calibration.optimizer import HandEyeOptimizer
from calibration.transforms import invert_transform, validate_transform
from calibration.targets import target_config, checkerboard_points, tag_correspondences
from calibration.session import new_run_directory, write_json
from config import CALIBRATION_CONFIG, get_results_path, REALSENSE_CONFIG


class DataCollectorProtocol(Protocol):
    """Minimal collector interface consumed by CalibrationSolver."""

    intrinsics: np.ndarray
    dist_coeffs: np.ndarray

    def get_saved_data(self) -> Sequence[Mapping[str, Any]]:
        ...


class CalibrationSolver:
    """手眼标定求解器"""

    def __init__(
        self,
        mode: str,
        intrinsics: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        backend: str = 'checkerboard'
    ) -> None:
        """
        初始化求解器

        Args:
            mode: 'eye_on_hand' 或 'eye_to_hand'
        """
        self.mode = mode
        self.backend = backend
        self.target_config = target_config(backend)
        self.session_metadata = None
        self.legacy_pose_only = False
        self.output_path = None
        self.source_session = None
        self.solver = HandEyeSolver(mode)
        self.optimizer = HandEyeOptimizer(mode)

        # 相机内参（优先使用外部传入的真实内参）
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
        min_pts = CALIBRATION_CONFIG.get('min_calibration_points', 6)
        self.min_points_required = max(6, int(cast(int, min_pts)))

        if dist_coeffs is not None:
            self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    def _build_checkerboard_object_points(self) -> np.ndarray:
        """Build checkerboard object points in board coordinate system."""
        return checkerboard_points(self.target_config)

    @staticmethod
    def _rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Convert OpenCV rvec/tvec to 4x4 homogeneous transform."""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.reshape(3)
        return T

    @staticmethod
    def _invert_transform(T: np.ndarray) -> np.ndarray:
        """Invert 4x4 homogeneous transform."""
        return invert_transform(T)

    def load_data(
        self,
        data_collector: DataCollectorProtocol
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        加载采集的数据

        Args:
            data_collector: CalibDataCollector 对象

        Returns:
            robot_poses: 机器人位姿列表 (T_base_tcp)
            camera_poses: 相机观测位姿列表 (T_cam_board)
            corners_2d_list: 每帧角点像素坐标
            images: 与有效帧对齐的 RGB 图像列表
            tag_corners_list: 与有效帧对齐的 AprilTag 角点列表
            tag_ids_list: 与有效帧对齐的 AprilTag id 列表
        """
        self.session_metadata = getattr(data_collector, 'session_metadata', None)
        self.legacy_pose_only = bool(getattr(data_collector, 'legacy_pose_only', False))
        self.source_session = getattr(data_collector, 'data_path', {}).get('root')
        if self.session_metadata is None and not self.legacy_pose_only:
            raise ValueError('Session metadata required; explicitly import legacy poses for consistency-only analysis')
        if self.session_metadata is not None and (
                self.session_metadata['mode'] != self.mode or self.session_metadata['backend'] != self.backend):
            raise ValueError('Solver mode/backend differs from the recorded session')
        self.target_config = getattr(data_collector, 'target_config', self.target_config)
        saved_data = data_collector.get_saved_data()

        # 使用采集会话的相机模型；离线复算不读取当前硬件或当前配置。
        collector_intrinsics = getattr(data_collector, 'intrinsics', None)
        if collector_intrinsics is not None:
            collector_intrinsics_np = np.asarray(collector_intrinsics, dtype=np.float64)
            if collector_intrinsics_np.shape == (3, 3):
                self.intrinsics = collector_intrinsics_np
        collector_dist = getattr(data_collector, 'dist_coeffs', None)
        if collector_dist is not None:
            collector_dist_np = np.asarray(collector_dist, dtype=np.float64).reshape(-1, 1)
            if collector_dist_np.shape[0] >= 4:
                self.dist_coeffs = collector_dist_np

        if len(saved_data) < self.min_points_required:
            raise ValueError(
                f"数据点不足: {len(saved_data)}, 需要至少{self.min_points_required}个"
            )

        print(f"加载了 {len(saved_data)} 个数据点")

        self.sample_indices = []
        robot_poses = []
        camera_poses = []
        corners_2d_list = []
        images = []
        tag_corners_list = []
        tag_ids_list = []
        objp = self._build_checkerboard_object_points() if self.backend == 'checkerboard' else None
        dist_coeffs = self.dist_coeffs

        for d_any in saved_data:
            d = cast(Dict[str, Any], d_any)
            tcp = validate_transform(d['tcp'], f"TCP frame {d['index']}")
            if self.backend in ('apriltag', 'apriltag_board'):
                tag_pose = d.get('tag_pose')
                if tag_pose is None:
                    print(f"  警告: 帧 {d['index']} 缺少 tag_pose，跳过")
                    continue
                camera_pose = np.asarray(tag_pose, dtype=np.float64)
                if camera_pose.shape != (4, 4):
                    print(f"  警告: 帧 {d['index']} tag_pose 维度异常，跳过")
                    continue
                validate_transform(camera_pose, f"camera frame {d['index']}")
                corners_2d = np.zeros((0, 2), dtype=np.float32)
                tag_corners = d.get('tag_corners')
                if tag_corners is None:
                    tag_corners_arr = np.zeros((0, 2), dtype=np.float32)
                else:
                    tag_corners_arr = np.asarray(tag_corners, dtype=np.float32).reshape(-1, 2)
                tag_ids = d.get('tag_ids')
                if tag_ids is None:
                    tag_ids_arr = np.zeros((0,), dtype=np.int32)
                else:
                    tag_ids_arr = np.asarray(tag_ids, dtype=np.int32).reshape(-1)
                if not self.legacy_pose_only:
                    tag_correspondences(self.target_config, tag_ids_arr, tag_corners_arr,
                                        self.backend == 'apriltag_board')
            else:
                corners = d['corners']

                if corners is None:
                    print(f"  警告: 帧 {d['index']} 数据不完整，跳过")
                    continue

                corners_2d = corners.reshape(-1, 2).astype(np.float32)
                if objp is None:
                    print(f"  警告: 帧 {d['index']} 缺少 checkerboard 点模型，跳过")
                    continue
                if len(corners_2d) != len(objp) or not np.isfinite(corners_2d).all():
                    raise ValueError(f"Invalid checkerboard corners in frame {d['index']}")
                ok, rvec, tvec = cv2.solvePnP(
                    objp,
                    corners_2d,
                    self.intrinsics,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if not ok:
                    print(f"  警告: 帧 {d['index']} PnP求解失败，跳过")
                    continue

                camera_pose = self._rvec_tvec_to_transform(rvec, tvec)
                validate_transform(camera_pose, f"PnP frame {d['index']}")
                tag_corners_arr = np.zeros((0, 2), dtype=np.float32)
                tag_ids_arr = np.zeros((0,), dtype=np.int32)

            self.sample_indices.append(str(d['index']))
            robot_poses.append(tcp)
            camera_poses.append(camera_pose)
            corners_2d_list.append(corners_2d)
            tag_corners_list.append(tag_corners_arr)
            tag_ids_list.append(tag_ids_arr)
            if d['rgb'] is not None:
                if self.session_metadata is not None and [d['rgb'].shape[1], d['rgb'].shape[0]] != self.session_metadata['image_size']:
                    raise ValueError(f"Image resolution mismatch in frame {d['index']}")
                images.append(d['rgb'])
            else:
                if not self.legacy_pose_only:
                    raise ValueError(f"Missing image in frame {d['index']}")
                images.append(np.zeros((1, 1, 3), dtype=np.uint8))

        if len(robot_poses) < self.min_points_required:
            raise ValueError(
                f"有效数据不足: {len(robot_poses)}, 需要至少{self.min_points_required}个"
            )

        return robot_poses, camera_poses, corners_2d_list, images, tag_corners_list, tag_ids_list

    def solve(
        self,
        robot_poses: List[np.ndarray],
        camera_poses: List[np.ndarray],
        corners_2d_list: List[np.ndarray],
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        执行标定求解

        Args:
            robot_poses: 机器人位姿列表
            camera_poses: 相机观测位姿列表
            corners_2d_list: 角点像素坐标列表

        Returns:
            result: {
                'X': 手眼变换矩阵,
                'z_scale': compatibility value fixed at 1.0,
                'optimization_result': 优化结果
            }
        """
        print("\n" + "=" * 50)
        print("标定计算")
        print("=" * 50)

        # 1. SVD 求解
        print("\n[1/2] SVD 求解...")
        X_svd = self.solver.solve_axxb_svd(robot_poses, camera_poses)
        print("SVD 求解完成")

        # 打印SVD结果
        pos = X_svd[:3, 3]
        print(f"  位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")

        # 2. 非线性优化
        print("\n[2/2] 非线性优化...")
        try:
            X_opt, z_scale, opt_result = self.optimizer.optimize(
                robot_poses=robot_poses,
                camera_data=camera_poses,
                intrinsics=self.intrinsics,
                initial_X=X_svd,
                position_weight=float(CALIBRATION_CONFIG.get('spatial_position_weight', 0.5)),
                rotation_weight=float(CALIBRATION_CONFIG.get('spatial_rotation_weight', 0.5)),
                position_scale_m=float(CALIBRATION_CONFIG['position_scale_m']),
                rotation_scale_rad=float(CALIBRATION_CONFIG['rotation_scale_rad']),
            )
            validate_transform(X_opt, 'optimized hand-eye transform')
            if (not opt_result.success or not np.isfinite(opt_result.fun)
                    or opt_result.fun > opt_result.initial_fun + max(1e-12, opt_result.initial_fun * 1e-9)):
                X_opt = X_svd.copy()
                opt_result.success = False
                print('优化未通过收敛/代价检查，保留 SVD 初值')
            print("非线性优化完成")
            print(f"  优化是否成功: {opt_result.success}")
            print(f"  最终目标函数值 (误差): {opt_result.fun:.6e}")
            print(
                "  spatial weights: "
                f"position={float(CALIBRATION_CONFIG.get('spatial_position_weight', 0.5)):.3f}, "
                f"rotation={float(CALIBRATION_CONFIG.get('spatial_rotation_weight', 0.5)):.3f}"
            )
        except Exception as e:
            print(f"优化过程出现异常: {e}")
            X_opt = X_svd
            z_scale = 1.0
            opt_result = None
            print("回退使用SVD结果作为最终解")

        print(f"  z_scale (fixed): {z_scale:.6f}")

        # 打印优化后结果
        pos = X_opt[:3, 3]
        print(f"  位置: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")

        # 保存结果
        if save:
            self._save_result(X_opt, z_scale, opt_result)

        return {
            'X': X_opt,
            'initial_X': X_svd,
            'output_path': self.output_path,
            'diagnostics': self.solver.diagnostics,
            'z_scale': z_scale,
            'optimization_result': opt_result,
            'robot_poses': robot_poses,
            'camera_poses': camera_poses,
            'corners_2d_list': corners_2d_list
        }

    def _save_result(self, X: np.ndarray, z_scale: float, optimization=None) -> None:
        """保存结果到文件"""
        results_path = str(new_run_directory(Path(get_results_path(self.mode)) / 'runs'))
        self.output_path = results_path
        write_json(Path(results_path) / 'result.json', {
            'mode': self.mode, 'backend': self.backend, 'source_session': self.source_session,
            'session': self.session_metadata, 'legacy_pose_only': self.legacy_pose_only,
            'target': self.target_config, 'diagnostics': self.solver.diagnostics,
            'transform': X, 'optimization_accepted': bool(optimization is not None and optimization.success),
            'position_scale_m': CALIBRATION_CONFIG['position_scale_m'],
            'rotation_scale_rad': CALIBRATION_CONFIG['rotation_scale_rad'],
        })

        # 保存手眼变换矩阵
        np.savetxt(os.path.join(results_path, 'handeye_transform.txt'), X, delimiter=' ')

        # Keep this compatibility file for existing result consumers.
        np.savetxt(os.path.join(results_path, 'depth_scale.txt'), np.array([z_scale]), delimiter=' ')

        # 保存详细信息
        with open(os.path.join(results_path, 'calibration_info.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Mode: {self.mode}\n")
            f.write(f"Z-scale: {z_scale:.6f}\n")
            f.write(f"\nTransform (4x4):\n")
            f.write(str(X))

        print(f"\n结果已保存至: {results_path}")
