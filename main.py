#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手眼标定系统主程序
整合设备连接、数据采集、标定计算、可视化、误差计算
"""

import sys
import os
import numpy as np
from typing import List
from pathlib import Path
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_collector import CalibDataCollector
from calibration_solver import CalibrationSolver
from error_calculator import ErrorCalculator
from result_visualizer import ResultVisualizer
from config import get_data_path
from calibration.targets import target_config


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
    print("  3. Four-AprilTag Board")
    print("=" * 50)

    while True:
        choice = input("\n请输入选项 (1/2/3): ").strip()
        if choice == '1':
            return 'checkerboard'
        elif choice == '2':
            return 'apriltag'
        elif choice == '3':
            return 'apriltag_board'
        else:
            print("无效选择，请重新输入")


def load_rough_pose(mode: str, backend: str, target=None):
    config = target_config(backend) if target is None else target
    key = 'board_to_base_rough' if mode == 'eye_on_hand' else 'board_to_tcp_rough'
    value = config.get(key)
    return None if value is None else _rough_pose_deg_to_rad(value)


def select_session(mode: str) -> Path:
    base = Path(get_data_path(mode)['root'])
    sessions = sorted((base / 'sessions').glob('*/session.json'), reverse=True)
    choices = [p.parent for p in sessions]
    if (base / 'poses').exists():
        choices.append(base)
    if not choices:
        raise ValueError('No saved sessions found; collect a new session first')
    for i, path in enumerate(choices, 1):
        suffix = ' [legacy: pose consistency only]' if not (path / 'session.json').exists() else ''
        print(f'{i}. {path.name}{suffix}')
    while True:
        choice = input('Session number: ').strip()
        if choice.isdigit() and 1 <= int(choice) <= len(choices):
            return choices[int(choice) - 1]


def evaluate(collector, solver, result, robot, camera, corners, images, tag_corners, tag_ids):
    calculator = ErrorCalculator(collector.mode, solver.intrinsics, solver.dist_coeffs,
                                 collector.backend, target=collector.target_config)
    X = result['X']
    initial_pos, initial_rot, _, _ = calculator.calculate_spatial_consistency(robot, camera, result['initial_X'])
    positions, rotations, position_components, rotation_components = calculator.calculate_spatial_consistency(robot, camera, X)
    print(f'SVD -> final mean residual: {initial_pos.mean()*1000:.3f} -> {positions.mean()*1000:.3f} mm; '
          f'{np.degrees(initial_rot).mean():.3f} -> {np.degrees(rotations).mean():.3f} deg')
    calculator.print_error_report(positions, 'Training-sample spatial consistency', unit='m')
    calculator.print_error_report(np.degrees(rotations), 'Training-sample rotation consistency', unit='deg')
    print('These are fit residuals, not independent absolute-accuracy measurements.')
    rough = None if collector.legacy_pose_only else load_rough_pose(collector.mode, collector.backend, collector.target_config)
    reference = {'board_to_base' if collector.mode == 'eye_on_hand' else 'board_to_tcp': rough}
    if rough is not None:
        calculator.print_error_report(calculator.calculate_position_error(robot, camera, X, **reference),
                                      'Deviation from manually estimated reference (not ground truth)', unit='m')
        calculator.print_error_report(np.degrees(calculator.calculate_rotation_error(robot, camera, X, **reference)),
                                      'Rotation deviation from manually estimated reference', unit='deg')
    projection = None
    if not collector.legacy_pose_only:
        if collector.backend == 'checkerboard':
            projection = calculator.calculate_reprojection_error(robot, camera, corners, X)
        else:
            observed = calculator.calculate_apriltag_observed_reprojection_error(camera, tag_corners, tag_ids, images)
            calculator.print_error_report(observed, 'Observed-tag reprojection', unit='px')
            projection = calculator.calculate_apriltag_chain_reprojection_error(robot, camera, tag_corners, tag_ids, X, images)
        calculator.print_error_report(projection, 'Full-chain reprojection', unit='px')
    if result['output_path']:
        path = Path(result['output_path']) / 'residuals.csv'
        with path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['frame', 'translation_mm', 'rotation_deg', 'chain_reprojection_px'])
            for i, (pos, rot) in enumerate(zip(positions, rotations)):
                pixel = '' if projection is None else projection[i]
                writer.writerow([solver.sample_indices[i], pos * 1000, np.degrees(rot), pixel])
    if collector.backend == 'checkerboard' and input('View corner reprojections? (y/n): ').lower() == 'y':
        calculator.visualize_reprojection_frames(images, robot, camera, corners, X)
    visualizer = ResultVisualizer(collector.mode)
    if input('Show 3D visualization? (y/n): ').lower() == 'y':
        visualizer.visualize(robot, X, camera_poses=camera, **reference)
    if input('Show consistency plots? (y/n): ').lower() == 'y':
        visualizer.visualize_position_rotation_errors(positions, np.degrees(rotations),
                                                       pos_unit='mm', rot_unit='deg', pos_scale=1000)
        visualizer.visualize_pose_error_components(position_components, rotation_components,
                                                   pos_unit='mm', rot_unit='deg', pos_scale=1000,
                                                   rot_scale=180 / np.pi,
                                                   title='Consistency in the mean target frame')


def main() -> None:
    mode, backend = select_mode(), select_backend()
    choice = input('1: 新建采集，2: 离线复算已有会话：').strip()
    manager = None
    try:
        if choice == '1':
            from device_manager import DeviceManager
            manager = DeviceManager()
            if not manager.connect():
                return
            collector = CalibDataCollector(manager.get_robot(), manager.get_camera(), mode, backend)
            collector.start_new_session()
            collector.collect_loop()
        elif choice == '2':
            root = select_session(mode)
            legacy = not (root / 'session.json').exists()
            if legacy:
                print('旧数据缺少相机与标定板元信息，只能评价历史位姿一致性。')
                if input('以历史位姿模式继续（不计算重投影）？(y/n): ').strip().lower() != 'y':
                    return
            collector = CalibDataCollector.from_session(str(root), mode, backend, legacy_pose_only=legacy)
        else:
            raise ValueError('Choose 1 or 2')
        solver = CalibrationSolver(mode, collector.intrinsics, collector.dist_coeffs, backend)
        robot, camera, corners, images, tag_corners, tag_ids = solver.load_data(collector)
        result = solver.solve(robot, camera, corners)
        evaluate(collector, solver, result, robot, camera, corners, images, tag_corners, tag_ids)
        print(f"Result saved: {result['output_path']}")
    finally:
        if manager is not None:
            manager.disconnect()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Calibration failed: {exc}')
        sys.exit(1)
