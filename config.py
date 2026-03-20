#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手眼标定系统配置文件
"""

import numpy as np
import os
from typing import Dict, List, Optional

# ============================================
# 项目根目录
# ============================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================
# UR3 机械臂配置
# ============================================
UR3_CONFIG = {
    'tcp_host_ip': '192.168.56.102',  # UR3 IP地址
    'tcp_port': 30003,
    'workspace_limits': None,  # 运行时从示教点动态计算
    'default_velocity': 1.05,
    'default_acceleration': 1.4,
}

# ============================================
# RealSense D405 相机配置
# ============================================
REALSENSE_CONFIG = {
    'device_id': None,  # 自动选择第一个设备
    'width': 1280,
    'height': 720,
    'fps': 30,
    # D405 内参
    'default_intrinsics': {
        'fx': 591.3669592841128, 'fy': 590.1281027314918 ,
        'cx': 643.8559356720739, 'cy': 370.3876182383728,
    }
}

# ============================================
# 标定板配置
# ============================================
# Eye-on-Hand: Camera on tool end, checkerboard fixed on base (stationary)
# Eye-to-Hand: Camera fixed in workspace, checkerboard on tool end
CHECKERBOARD_CONFIG = {
    'size': (11, 8),  # 内角点数量 (cols, rows)
    'square_size': 0.006,  # 棋盘格方格大小 (米)
    # 粗略位姿 (仅用于可视化/误差计算参考，不参与标定求解)
    # Eye-on-Hand: 标定板相对于机器人基座的粗略位姿
    # 格式: [x, y, z, rx, ry, rz] - 平移(米), 旋转向量(度)
    'board_to_base_rough': [0.0, 0.0, 0.0, 0, 0, 0],
    # Eye-to-Hand: 标定板相对于TCP(法兰盘中心)的粗略位姿
    'board_to_tcp_rough': [-0.095, 0, 0.0104, 180, 0, 0],
}

# ============================================
# 标定模式配置
# ============================================
CALIBRATION_MODES = ['eye_on_hand', 'eye_to_hand']

# ============================================
# 标定参数配置
# ============================================
CALIBRATION_CONFIG = {
    'z_scale_init': 1.0,  # 深度缩放因子初始值
    'z_scale_bounds': (0.95, 1.05),  # 深度缩放因子搜索范围
    'optimization_method': 'Nelder-Mead',
    'min_calibration_points': 6,  # 最少标定点数
}

# ============================================
# 采样配置
# ============================================
SAMPLING_CONFIG = {
    'grid_step': 0.05,  # 网格采样间距 (米)
    'num_samples': 30,  # 自动采样数量
    'sampling_method': 'grid',  # 'grid', 'random', 'adaptive'
    'min_distance_from_teach': 0.03,  # 采样点与示教点的最小距离
}

# ============================================
# 可视化配置
# ============================================
VISUALIZATION_CONFIG = {
    'coordinate_axis_length': 0.1,  # 坐标系轴长度 (米)
    'show_checkerboard': True,
    'show_robot': True,
    'figsize': (12, 10),
}

# ============================================
# 数据路径配置
# ============================================
def get_data_path(mode: str) -> Dict[str, str]:
    """获取指定模式的数据路径"""
    base_path = os.path.join(PROJECT_ROOT, 'data', mode)
    return {
        'root': base_path,
        'teach_poses': os.path.join(base_path, 'teach_poses'),
        'poses': os.path.join(base_path, 'poses'),
        'images': os.path.join(base_path, 'images'),
    }

def get_results_path(mode: str) -> str:
    """获取指定模式的结果路径"""
    return os.path.join(PROJECT_ROOT, 'results', mode)


# ============================================
# 辅助函数
# ============================================
def load_teach_poses(mode: str) -> List[np.ndarray]:
    """加载示教点"""
    poses_dir = get_data_path(mode)['teach_poses']
    poses = []
    for i in range(8):
        pose_file = os.path.join(poses_dir, f'teach_{i}.txt')
        if os.path.exists(pose_file):
            poses.append(np.loadtxt(pose_file))
    return poses


def compute_workspace_from_teach(teach_poses: List[np.ndarray]) -> Optional[np.ndarray]:
    """从示教点计算工作空间边界 (AABB)"""
    if len(teach_poses) == 0:
        return None

    poses_array = np.array([p[:3, 3] for p in teach_poses])  # 提取位置

    workspace = np.array([
        [poses_array[:, 0].min(), poses_array[:, 0].max()],
        [poses_array[:, 1].min(), poses_array[:, 1].max()],
        [poses_array[:, 2].min(), poses_array[:, 2].max()],
    ])
    return workspace
