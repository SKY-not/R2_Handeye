#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手眼标定系统主程序
整合设备连接、数据采集、标定计算、可视化、误差计算
"""

import sys
import os
import numpy as np
from typing import List, Sequence, cast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from device_manager import DeviceManager
from data_collector import CalibDataCollector
from calibration_solver import CalibrationSolver
from error_calculator import ErrorCalculator
from result_visualizer import ResultVisualizer
from config import CHECKERBOARD_CONFIG, CALIBRATION_CONFIG


def _rough_pose_deg_to_rad(pose: List[float]) -> List[float]:
    """Convert rough pose rotation vector from degree to radian."""
    pose_rad = [float(v) for v in pose]
    pose_rad[3] = float(np.deg2rad(pose_rad[3]))
    pose_rad[4] = float(np.deg2rad(pose_rad[4]))
    pose_rad[5] = float(np.deg2rad(pose_rad[5]))
    return pose_rad


def select_mode() -> str:
    """选择标定模式"""
    print("\n" + "=" * 50)
    print("请选择标定模式:")
    print("  1. Eye-on-Hand (眼在手上) - 相机安装在机械臂末端")
    print("  2. Eye-to-Hand (眼在手外) - 相机固定在工作空间")
    print("=" * 50)

    while True:
        choice = input("\n请输入选项 (1/2): ").strip()
        if choice == '1':
            return 'eye_on_hand'
        elif choice == '2':
            return 'eye_to_hand'
        else:
            print("无效选择，请重新输入")


def select_backend() -> str:
    """选择特征后端"""
    print("\n" + "=" * 50)
    print("请选择特征后端:")
    print("  1. Checkerboard (棋盘格)")
    print("  2. AprilTag")
    print("=" * 50)

    while True:
        choice = input("\n请输入选项 (1/2): ").strip()
        if choice == '1':
            return 'checkerboard'
        elif choice == '2':
            return 'apriltag'
        else:
            print("无效选择，请重新输入")


def load_rough_pose(mode: str) -> List[float]:
    """
    加载粗略位姿

    Args:
        mode: 标定模式

    Returns:
        board_to_base 或 board_to_tcp (取决于模式)
    """
    if mode == 'eye_on_hand':
        # Eye-on-Hand: 标定板相对于基座的位姿
        pose_deg_any = CHECKERBOARD_CONFIG.get('board_to_base_rough', [0, 0, 0, 0, 0, 0])
        pose_deg = [float(v) for v in cast(Sequence[float], pose_deg_any)]
        pose = _rough_pose_deg_to_rad(pose_deg)
        print(f"\n加载 Eye-on-Hand 粗略位姿(角度输入): {pose_deg}")
        print(f"转换后用于计算(弧度): {pose}")
        return pose
    else:
        # Eye-to-Hand: 标定板相对于TCP的位姿
        pose_deg_any = CHECKERBOARD_CONFIG.get('board_to_tcp_rough', [0, 0, 0, 0, 0, 0])
        pose_deg = [float(v) for v in cast(Sequence[float], pose_deg_any)]
        pose = _rough_pose_deg_to_rad(pose_deg)
        print(f"\n加载 Eye-to-Hand 粗略位姿(角度输入): {pose_deg}")
        print(f"转换后用于计算(弧度): {pose}")
        return pose


def main() -> None:
    """主函数"""
    print("=" * 50)
    print("手眼标定系统")
    print("=" * 50)

    # 1. 选择模式
    mode = select_mode()
    mode_name = "Eye-on-Hand" if mode == "eye_on_hand" else "Eye-to-Hand"
    print(f"\n已选择: {mode_name}")

    backend = select_backend()
    print(f"已选择后端: {backend}")

    # 加载粗略位姿 (仅用于可视化/误差计算参考)
    rough_pose = load_rough_pose(mode)

    # 2. 连接设备
    device_mgr = DeviceManager()
    success = device_mgr.connect()

    if not success:
        print("\n设备连接失败，请检查配置后重试")
        return

    robot = device_mgr.get_robot()
    camera = device_mgr.get_camera()
    if robot is None or camera is None:
        print("设备对象为空，请检查连接流程")
        device_mgr.disconnect()
        return

    # 3. 数据采集
    print("\n" + "=" * 50)
    print("数据采集")
    print("=" * 50)

    collector = CalibDataCollector(robot, camera, mode, backend=backend)

    # 询问是新建采集还是使用已有数据
    print("\n请选择:")
    print("  1. 新建数据采集")
    print("  2. 使用已有数据")

    choice = input("请输入选项 (1/2): ").strip()

    if choice == '1':
        collector.clear_old_data()
        collector.collect_loop()
    else:
        data = collector.get_saved_data()
        print(f"已加载 {len(data)} 个已有数据点")

    min_required_cfg = CALIBRATION_CONFIG.get('min_calibration_points', 6)
    min_required = max(6, int(cast(int, min_required_cfg)))
    current_count = len(collector.get_saved_data())
    if current_count < min_required:
        print(f"\n当前有效样本数: {current_count}，少于最小要求: {min_required}")
        print("请选择:")
        print("  1. 继续采集")
        print("  2. 退出流程")
        decision = input("请输入选项 (1/2): ").strip()
        if decision == '1':
            collector.collect_loop()
        else:
            print("用户选择退出流程")
            device_mgr.disconnect()
            return

    # 4. 标定计算
    solver = CalibrationSolver(
        mode,
        intrinsics=camera.intrinsics,
        dist_coeffs=getattr(camera, 'dist_coeffs', None),
        backend=backend
    )
    try:
        robot_poses, camera_poses, corners_2d_list, images = solver.load_data(collector)
    except ValueError as e:
        print(f"错误: {e}")
        print("请继续采集数据后重试，或退出流程。")
        device_mgr.disconnect()
        return

    result = solver.solve(robot_poses, camera_poses, corners_2d_list)

    # 5. 误差计算
    print("\n" + "=" * 50)
    print("误差计算")
    print("=" * 50)

    error_calc = ErrorCalculator(
        mode,
        intrinsics=camera.intrinsics,
        dist_coeffs=camera.dist_coeffs,
        backend=backend
    )

    # 根据模式设置参数
    if mode == 'eye_on_hand':
        position_errors = error_calc.calculate_position_error(
            robot_poses, camera_poses, result['X'], result['z_scale'],
            board_to_base=rough_pose
        )
        rotation_errors = error_calc.calculate_rotation_error(
            robot_poses,
            camera_poses,
            result['X'],
            board_to_base=rough_pose
        )
    else:
        position_errors = error_calc.calculate_position_error(
            robot_poses, camera_poses, result['X'], result['z_scale'],
            board_to_tcp=rough_pose
        )
        rotation_errors = error_calc.calculate_rotation_error(
            robot_poses,
            camera_poses,
            result['X'],
            board_to_tcp=rough_pose
        )

    # calculate_rotation_error returns radians; convert to degrees for display.
    rotation_errors_deg = np.degrees(rotation_errors)

    reproj_errors = np.array([], dtype=np.float64)
    if backend != 'apriltag':
        reproj_errors = error_calc.calculate_reprojection_error(
            robot_poses,
            camera_poses,
            corners_2d_list,
            result['X'],
            result['z_scale']
        )

    error_calc.print_error_report(position_errors, "位置误差报告", unit='m')
    error_calc.print_error_report(rotation_errors_deg, "旋转误差报告 (deg)", unit='deg')
    if backend != 'apriltag':
        error_calc.print_error_report(reproj_errors, "重投影误差报告 (px)", unit='px')
    else:
        print("\nAprilTag 后端已跳过重投影误差计算")

    if backend != 'apriltag':
        show_reproj_frames = input("是否逐帧查看重投影角点对比? (y/n): ").strip().lower()
        if show_reproj_frames == 'y':
            if mode == 'eye_on_hand':
                error_calc.visualize_reprojection_frames(
                    images,
                    robot_poses,
                    camera_poses,
                    corners_2d_list,
                    result['X'],
                    board_to_base=rough_pose
                )
            else:
                error_calc.visualize_reprojection_frames(
                    images,
                    robot_poses,
                    camera_poses,
                    corners_2d_list,
                    result['X'],
                    board_to_tcp=rough_pose
                )

    # 6. 可视化
    print("\n" + "=" * 50)
    print("可视化")
    print("=" * 50)

    visualizer = ResultVisualizer(mode)

    show_viz = input("是否显示3D可视化? (y/n): ").strip().lower()
    if show_viz == 'y':
        if mode == 'eye_on_hand':
            visualizer.visualize(
                robot_poses, result['X'], result['z_scale'],
                camera_poses=camera_poses,
                board_to_base=rough_pose
            )
        else:
            visualizer.visualize(
                robot_poses, result['X'], result['z_scale'],
                camera_poses=camera_poses,
                board_to_tcp=rough_pose
            )

    show_errors = input("是否显示误差分布图? (y/n): ").strip().lower()
    if show_errors == 'y':
        visualizer.visualize_position_rotation_errors(
            position_errors,
            rotation_errors_deg,
            pos_unit='mm',
            rot_unit='deg',
            pos_scale=1000.0,
            rot_scale=1.0
        )

    # 7. 清理
    device_mgr.disconnect()

    print("\n" + "=" * 50)
    print("标定完成!")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
