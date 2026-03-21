#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
棋盘格角点检测与深度恢复模块
用于手眼标定的特征提取
"""

import numpy as np
import cv2
from typing import Any, Dict, Optional, Tuple, cast


class CheckerboardExtractor:
    """棋盘格角点提取器"""

    def __init__(self, checkerboard_size: Tuple[int, int] = (5, 5), square_size: float = 0.024) -> None:
        """
        初始化角点提取器

        Args:
            checkerboard_size: 棋盘格内角点数量 (cols, rows)
            square_size: 方格大小 (米)
        """
        self.checkerboard_size = checkerboard_size  # (cols, rows)
        self.square_size = square_size

        # 亚像素角点优化参数
        self.refine_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30, 0.001
        )

    def detect_corners(
        self,
        gray_image: np.ndarray,
        refine: bool = True
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        检测棋盘格角点

        Args:
            gray_image: 灰度图
            refine: 是否进行亚像素级优化

        Returns:
            success: 检测是否成功
            corners: 角点坐标 (N, 1, 2)
            corners_refined: 亚像素级角点 (如果refine=True)

        Note:
            棋盘格检测可直接在原始图像上进行，不需要先传入畸变参数做矫正。
            畸变参数主要在后续 solvePnP 估计位姿时使用。
        """
        # 检测棋盘格
        success, corners = cv2.findChessboardCorners(
            gray_image,
            self.checkerboard_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not success:
            return False, None, None

        # 亚像素级优化
        if refine:
            corners_refined = cv2.cornerSubPix(
                gray_image,
                corners,
                (5, 5),  # 搜索窗口大小
                (-1, -1),  # 死区大小
                self.refine_criteria
            )
            return True, corners, corners_refined

        return True, corners, corners

    def get_center_pixel(self, corners: np.ndarray) -> np.ndarray:
        """
        获取棋盘格中心像素坐标

        Args:
            corners: 角点坐标

        Returns:
            center_px: 中心像素坐标 [u, v]
        """
        # 棋盘格中心点索引
        center_idx = (self.checkerboard_size[0] * self.checkerboard_size[1]) // 2
        center_corner = corners[center_idx, 0]
        center_px = np.round(center_corner).astype(int)
        return np.asarray(center_px, dtype=np.int32)

    def get_corners_3d(self, corners: np.ndarray, depth_image: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """
        将角点像素坐标转换为3D相机坐标

        Args:
            corners: 角点像素坐标 (N, 1, 2)
            depth_image: 深度图 (H, W), 单位: 米
            intrinsics: 相机内参矩阵 (3, 3)

        Returns:
            corners_3d: 角点3D坐标 (N, 3)
        """
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        corners_3d = []

        for i in range(len(corners)):
            u = int(np.round(corners[i, 0, 0]))
            v = int(np.round(corners[i, 0, 1]))

            # 检查边界
            if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
                z = depth_image[v, u]
                if z > 0:  # 有效深度
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    corners_3d.append([x, y, z])
                else:
                    corners_3d.append([0, 0, 0])
            else:
                corners_3d.append([0, 0, 0])

        return np.array(corners_3d)

    def estimate_board_pose(self, corners_3d: np.ndarray) -> np.ndarray:
        """
        估计棋盘格板相对于相机的位姿

        Args:
            corners_3d: 棋盘格角点在相机坐标系下的3D坐标

        Returns:
            H_cb: 棋盘格相对于相机的齐次变换矩阵 (4, 4)
        """
        # 生成3D世界坐标 (假设棋盘格在XY平面上，Z=0)
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        # 使用RANSAC或最小二乘拟合平面
        # 简化: 使用PCA找主平面
        centered = corners_3d - corners_3d.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]

        # 确保法向量指向相机
        if normal[2] > 0:
            normal = -normal

        # 计算旋转矩阵 (Z轴指向棋盘格法向)
        z_axis = normal / np.linalg.norm(normal)
        x_axis = np.cross([0, 0, 1], z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            x_axis = np.array([1, 0, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])

        # 质心作为位置
        t = corners_3d.mean(axis=0)

        # 构建齐次变换矩阵
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = t

        return H

    def extract(self, color_image: np.ndarray, depth_image: np.ndarray, intrinsics: np.ndarray) -> Dict[str, Any]:
        """
        提取棋盘格特征 (完整流程)

        Args:
            color_image: 彩色图 (BGR)
            depth_image: 深度图 (米)
            intrinsics: 相机内参

        Returns:
            result: 包含以下键的字典:
                - success: 是否成功
                - corners_2d: 角点2D坐标
                - corners_3d: 角点3D坐标
                - center_px: 中心像素坐标
                - board_pose: 棋盘格位姿 (4x4)
                - visualized: 可视化图像
        """
        # 转为灰度图
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 检测角点
        success, corners, corners_refined = self.detect_corners(gray, refine=True)

        if not success:
            return {
                'success': False,
                'message': '未检测到棋盘格'
            }
        assert corners_refined is not None

        # 获取中心像素坐标
        center_px = self.get_center_pixel(corners_refined)

        # 获取深度
        u, v = center_px
        if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
            depth = depth_image[v, u]
        else:
            depth = 0

        if depth <= 0:
            return {
                'success': False,
                'message': '深度值无效'
            }

        # 获取所有角点3D坐标
        corners_3d = self.get_corners_3d(corners_refined, depth_image, intrinsics)

        # 估计棋盘格位姿
        board_pose = self.estimate_board_pose(corners_3d)

        # 可视化
        vis = color_image.copy()
        cv2.drawChessboardCorners(vis, self.checkerboard_size, corners_refined, success)
        start_pt = tuple(corners_refined[0, 0].astype(int))
        end_pt = tuple(corners_refined[-1, 0].astype(int))
        cv2.circle(vis, start_pt, 9, (0, 255, 255), -1)
        cv2.putText(vis, 'START', (start_pt[0] + 8, start_pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(vis, end_pt, 9, (255, 0, 255), -1)
        cv2.putText(vis, 'END', (end_pt[0] + 8, end_pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.circle(vis, tuple(center_px), 10, (0, 255, 0), -1)  # 标记中心

        return {
            'success': True,
            'corners_2d': corners_refined,
            'corners_3d': corners_3d,
            'center_px': center_px,
            'center_depth': depth,
            'board_pose': board_pose,
            'visualized': vis
        }


def visualize_checkerboard(
    color_image: np.ndarray,
    corners: np.ndarray,
    checkerboard_size: Tuple[int, int],
    center_idx: Optional[int] = None
) -> np.ndarray:
    """
    可视化棋盘格检测结果

    Args:
        color_image: 彩色图
        corners: 角点坐标
        checkerboard_size: 棋盘格大小
        center_idx: 中心点索引

    Returns:
        vis: 可视化图像
    """
    vis = color_image.copy()
    cv2.drawChessboardCorners(vis, checkerboard_size, corners, True)

    if corners is not None and len(corners) > 0:
        start_pt = tuple(corners[0, 0].astype(int))
        end_pt = tuple(corners[-1, 0].astype(int))
        cv2.circle(vis, start_pt, 9, (0, 255, 255), -1)
        cv2.putText(vis, 'START', (start_pt[0] + 8, start_pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(vis, end_pt, 9, (255, 0, 255), -1)
        cv2.putText(vis, 'END', (end_pt[0] + 8, end_pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    if center_idx is not None:
        center = corners[center_idx, 0]
        cv2.circle(vis, tuple(center.astype(int)), 15, (0, 255, 0), -1)

    return vis


def load_and_process(
    image_path: str,
    depth_path: str,
    intrinsics: np.ndarray,
    checkerboard_size: Tuple[int, int] = (5, 5)
) -> Dict[str, Any]:
    """
    加载并处理图像

    Args:
        image_path: 彩色图路径
        depth_path: 深度图路径 (.npy)
        intrinsics: 相机内参
        checkerboard_size: 棋盘格大小

    Returns:
        result: 处理结果字典
    """
    # 加载图像
    color_img = cv2.imread(image_path)
    if color_img is None:
        return {'success': False, 'message': f'无法读取图像: {image_path}'}
    depth_img = np.load(depth_path)

    # 提取特征
    extractor = CheckerboardExtractor(checkerboard_size)
    result = extractor.extract(color_img, depth_img, intrinsics)

    return result


if __name__ == "__main__":
    # 测试代码
    import os

    # 测试检测
    extractor = CheckerboardExtractor(checkerboard_size=(5, 5))

    # 创建测试图像
    print("棋盘格提取器已创建")
    print(f"  角点数量: {extractor.checkerboard_size}")
    print(f"  方格大小: {extractor.square_size}m")
