#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Result visualization module - Unified coordinate system for TCP, camera, calibration board display
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
from typing import Any, List, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CHECKERBOARD_CONFIG


def pose_to_mat(pose: Union[np.ndarray, List[float]]) -> np.ndarray:
    """Convert pose array to 4x4 transformation matrix"""
    if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
        return pose

    x, y, z, rx, ry, rz = pose

    # Rotation vector to rotation matrix (Rodrigues)
    theta = np.linalg.norm([rx, ry, rz])
    if theta > 1e-6:
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


class ResultVisualizer:
    """Calibration result visualizer - Unified to UR3 base coordinate system"""

    def __init__(self, mode: str) -> None:
        """
        Initialize visualizer

        Args:
            mode: 'eye_on_hand' or 'eye_to_hand'
        """
        self.mode = mode

    def visualize(
        self,
        robot_poses: List[np.ndarray],
        X: np.ndarray,
        z_scale: float = 1.0,
        camera_poses: Optional[List[np.ndarray]] = None,
        board_to_base: Optional[List[float]] = None,
        board_to_tcp: Optional[List[float]] = None
    ) -> None:
        """
        Visualize calibration result

        Args:
            robot_poses: TCP pose list (in base coordinate system)
            X: hand-eye transformation matrix (camera relative to TCP or base)
            z_scale: depth scale factor
            camera_poses: board pose list in camera frame (T_cam_board)
            board_to_base: rough pose of calibration board relative to base (for Eye-on-Hand)
            board_to_tcp: rough pose of calibration board relative to TCP (for Eye-to-Hand)
        """
        print("\nGenerating visualization...")

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 1. Draw all TCP positions (blue dots)
        tcp_positions = np.array([p[:3, 3] for p in robot_poses])
        ax.scatter(tcp_positions[:, 0], tcp_positions[:, 1], tcp_positions[:, 2],
                  c='blue', marker='o', s=80, label='TCP (flange)', alpha=0.7)

        # 2. Calculate and draw camera positions based on mode
        if self.mode == 'eye_on_hand':
            # Eye-on-Hand: Camera moves with TCP
            # Camera in base coordinate = TCP @ X
            camera_positions = []
            for tcp in robot_poses:
                T_cam_in_base = tcp @ X
                camera_positions.append(T_cam_in_base[:3, 3])
            camera_positions = np.array(camera_positions)
            ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                      c='red', marker='^', s=60, label='Camera (Eye-on-Hand)', alpha=0.7)

            # Draw camera coordinate frame (using first position orientation)
            self._draw_coordinate_frame(ax, robot_poses[0] @ X, 'Camera', 'red', 0.05)

        else:
            # Eye-to-Hand: Camera is fixed
            # Camera in base coordinate = X (directly from calibration result)
            cam_pos = X[:3, 3]
            ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]],
                      c='red', marker='^', s=150, label='Camera (Eye-to-Hand)', alpha=0.9)

            # Draw camera coordinate frame
            self._draw_coordinate_frame(ax, X, 'Camera', 'red', 0.1)

        # 3. Draw calibration board positions (using rough pose as reference)
        if self.mode == 'eye_on_hand' and board_to_base is not None:
            # Calibration board fixed on base
            T_board = pose_to_mat(board_to_base)
            self._draw_coordinate_frame(ax, T_board, 'Board (ref)', 'green', 0.1)

            # Draw measured board positions from observations
            if camera_poses is not None:
                for tcp, cam_pose in zip(robot_poses, camera_poses):
                    T_board_measured = tcp @ X @ cam_pose
                    p = T_board_measured[:3, 3]
                    ax.scatter([p[0]], [p[1]], [p[2]],
                              c='green', marker='s', s=30, alpha=0.4)

        elif self.mode == 'eye_to_hand' and board_to_tcp is not None:
            # Calibration board moves with TCP (reference trajectory)
            T_board_rough = pose_to_mat(board_to_tcp)
            for tcp in robot_poses:
                T_board_ref = tcp @ T_board_rough
                ax.scatter([T_board_ref[0, 3]], [T_board_ref[1, 3]], [T_board_ref[2, 3]],
                          c='green', marker='s', s=30, alpha=0.4)

            # Draw a sample board coordinate frame
            if robot_poses:
                self._draw_coordinate_frame(ax, robot_poses[0] @ T_board_rough,
                                          'Board (ref)', 'green', 0.1)

            # Draw measured board positions from observations
            if camera_poses is not None:
                for cam_pose in camera_poses:
                    T_board_measured = X @ cam_pose
                    p = T_board_measured[:3, 3]
                    ax.scatter([p[0]], [p[1]], [p[2]],
                              c='lime', marker='x', s=25, alpha=0.5)

        # 4. Draw base coordinate frame
        self._draw_coordinate_frame(ax, np.eye(4), 'Base', 'black', 0.15)

        # Set figure properties
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)

        mode_name = "Eye-on-Hand" if self.mode == "eye_on_hand" else "Eye-to-Hand"
        ax.set_title(f'{mode_name} Calibration Result\n(Unified coordinate: UR3 Base)', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)

        # Set equal axis scale
        self._set_equal_axis(ax, tcp_positions, robot_poses)

        plt.tight_layout()
        plt.show()

    def visualize_errors(self, errors: np.ndarray) -> None:
        """
        Visualize error distribution

        Args:
            errors: error array
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Error bar chart
        ax1 = axes[0]
        ax1.bar(range(1, len(errors)+1), errors * 1000, color='steelblue', edgecolor='black')
        ax1.axhline(y=np.mean(errors) * 1000, color='r', linestyle='--',
                   label=f'Mean: {np.mean(errors)*1000:.3f}mm')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Error (mm)')
        ax1.set_title('Position Error per Frame')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Error histogram
        ax2 = axes[1]
        ax2.hist(errors * 1000, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.mean(errors) * 1000, color='r', linestyle='--',
                   label=f'Mean: {np.mean(errors)*1000:.3f}mm')
        ax2.set_xlabel('Error (mm)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Reprojection Error Analysis', fontsize=14)
        plt.tight_layout()
        plt.show()

    def _draw_coordinate_frame(
        self,
        ax: Any,
        T: np.ndarray,
        label: str,
        color: str,
        length: float = 0.1
    ) -> None:
        """
        Draw coordinate frame

        Args:
            ax: matplotlib 3D axis
            T: 4x4 transformation matrix
            label: label
            color: color ('red', 'green', 'blue', 'black')
            length: axis length
        """
        pos = T[:3, 3]
        R = T[:3, :3]

        # Calculate each axis endpoint
        x_axis = pos + R[:, 0] * length
        y_axis = pos + R[:, 1] * length
        z_axis = pos + R[:, 2] * length

        # Draw origin point
        ax.scatter([pos[0]], [pos[1]], [pos[2]], color=color, s=100)

        # Draw axes
        ax.quiver(pos[0], pos[1], pos[2],
                 x_axis[0]-pos[0], x_axis[1]-pos[1], x_axis[2]-pos[2],
                 color='r', arrow_length_ratio=0.1, linewidth=2)
        ax.quiver(pos[0], pos[1], pos[2],
                 y_axis[0]-pos[0], y_axis[1]-pos[1], y_axis[2]-pos[2],
                 color='g', arrow_length_ratio=0.1, linewidth=2)
        ax.quiver(pos[0], pos[1], pos[2],
                 z_axis[0]-pos[0], z_axis[1]-pos[1], z_axis[2]-pos[2],
                 color='b', arrow_length_ratio=0.1, linewidth=2)

        # Label
        ax.text(pos[0], pos[1], pos[2], label, color=color, fontsize=10,
               weight='bold')

    def _set_equal_axis(self, ax: Any, *positions: np.ndarray) -> None:
        """Set equal axis scale"""
        all_points = []
        for pos_array in positions:
            if pos_array is not None and len(pos_array) > 0:
                all_points.append(pos_array)

        if not all_points:
            return

        all_points = np.vstack(all_points)
        max_range = np.abs(all_points).max() * 1.2

        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
