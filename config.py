#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手眼标定系统配置文件
"""

import numpy as np
import os
from typing import Dict, List, Optional

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# UR3 机械臂配置
UR3_CONFIG = {
    'tcp_host_ip': '192.168.56.102',  # UR3 IP地址
    'tcp_port': 30003,
    'workspace_limits': None,  # 运行时从示教点动态计算
    'default_velocity': 1.05,
    'default_acceleration': 1.4,
}

# RealSense D405 相机配置
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

# 标定板配置
# Eye-on-Hand: Camera on tool end, checkerboard fixed on base (stationary)
# Eye-to-Hand: Camera fixed in workspace, checkerboard on tool end
CHECKERBOARD_CONFIG = {
    'size': (11, 8),  # 内角点数量 (cols, rows)
    'square_size': 0.006,  # 棋盘格方格大小 (米)
    # 粗略位姿 (仅用于可视化/误差计算参考，不参与标定求解)
    # Eye-on-Hand: 标定板相对于机器人基座的粗略位姿
    # 格式: [x, y, z, rx, ry, rz] - 平移(米), 旋转向量(角度)
    'board_to_base_rough': [-0.0553, -0.3491, 0.0437, -70.5, -163.36, 7.45],
    # Eye-to-Hand: 标定板相对于TCP(法兰盘中心)的粗略位姿
    # 'board_to_tcp_rough': [-0.095, 0, 0.006, 0, 180, 0],
    'board_to_tcp_rough': [0.005, 0, 0.075, 0, -90, 0],
}

# 标定模式配置
CALIBRATION_MODES = ['eye_on_hand', 'eye_to_hand']

# SVD 采集与特征分析配置
SVD_CONFIG = {
    'data_root': os.path.join(PROJECT_ROOT, 'data', 'svd'),
    'images_dirname': 'images',
    'features_filename': 'svd_features.csv',
}

# 标定参数配置
CALIBRATION_CONFIG = {
    'z_scale_init': 1.0,  # 深度缩放因子初始值
    'z_scale_bounds': (0.95, 1.05),  # 深度缩放因子搜索范围
    'optimization_method': 'Nelder-Mead',
    'min_calibration_points': 6,  # 最少标定点数
}

# AprilTag 标定配置
# 参考位姿见 CHECKERBOARD_CONFIG 中的 board_to_base_rough 和 board_to_tcp_rough
APRILTAG_CONFIG = {
    'family': 'tag36h11',
    'tag_size': 0.03,  # 米
    'target_tag_id': 1,
    'decision_margin_threshold': 20.0,
    'min_area_ratio': 0.0005,
}

# AprilTag 标定结果测试配置
APRILTAG_TEST_CONFIG = {
    'mode': 'eye_to_hand',
    'tag_family': 'tag36h11',
    'tag_size': 0.01,  # AprilTag边长 (米)
    'target_tag_id': 0,
    'decision_margin_threshold': 20.0,
    'axis_length': 0.030,  # 图像可视化坐标轴长度 (米)
    # 目标: tag坐标系下TCP位姿 [x, y, z, rx, ry, rz]
    # 平移单位米, 旋转为Rodrigues旋转向量(弧度)
    # 't_tag_tcp_target': [0.0, -0.20, 0.20, 0.0, np.pi, 0.0],  # 目标位姿 (TCP在tag坐标系下) eye_to_hand
    't_tag_tcp_target': [0.0, 0.0, -0.30, 0.0, 0.0, 0.0],  # 目标位姿 (TCP在tag坐标系下) eye_on_hand
    'rtde_velocity': 0.05,
    'rtde_acceleration': 0.05,
    'dry_run': False,
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'coordinate_axis_length': 0.1,  # 坐标系轴长度 (米)
    'show_checkerboard': True,
    'show_robot': True,
    'figsize': (12, 10),
}

# 数据路径配置
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


def get_svd_data_path() -> Dict[str, str]:
    """获取 SVD 采集与分析数据路径。"""
    data_root = str(SVD_CONFIG['data_root'])
    images_dir = os.path.join(data_root, str(SVD_CONFIG['images_dirname']))
    features_path = os.path.join(data_root, str(SVD_CONFIG['features_filename']))
    return {
        'root': data_root,
        'images': images_dir,
        'features': features_path,
    }
