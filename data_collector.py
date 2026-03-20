#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data collection module - responsible for collecting calibration data.
Supports teach-by-demo with keyboard workflow:
    - Space: detect checkerboard corners for current frame
    - Enter: save last successful detection
    - Esc: exit collection loop
"""

import os
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import cv2
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration.feature_extractor import CheckerboardExtractor
from config import CHECKERBOARD_CONFIG, CALIBRATION_CONFIG, get_data_path


class CalibDataCollector:
    """Calibration data collector"""

    def __init__(self, robot: Any, camera: Any, mode: str) -> None:
        """
        Initialize data collector

        Args:
            robot: robot object
            camera: camera object
            mode: 'eye_on_hand' or 'eye_to_hand'
        """
        self.robot = robot
        self.camera = camera
        self.mode = mode

        # Checkerboard config
        cb_size = CHECKERBOARD_CONFIG['size']
        cb_square = CHECKERBOARD_CONFIG['square_size']
        self.extractor = CheckerboardExtractor(cb_size, cb_square)

        # Data save path
        self.data_path = get_data_path(mode)
        self._ensure_dirs()

        # Collection counter
        self.frame_count = 0

        # Camera intrinsics
        self.intrinsics = camera.intrinsics

        # Real-time preview control
        self._preview_active = False
        self._preview_thread = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()

        # Latest validated detection from Space key, consumed by Enter key
        self._pending_detection = None
        self.min_frames_required = max(6, int(CALIBRATION_CONFIG.get('min_calibration_points', 6)))

    def _ensure_dirs(self) -> None:
        """Ensure data directories exist"""
        os.makedirs(self.data_path['poses'], exist_ok=True)
        os.makedirs(self.data_path['images'], exist_ok=True)

    def get_frame_index(self) -> int:
        """Get next frame index"""
        # Find existing max index
        existing = []
        poses_dir = self.data_path['poses']
        if os.path.exists(poses_dir):
            for f in os.listdir(poses_dir):
                if f.startswith('tcp_') and f.endswith('.txt'):
                    idx = int(f.split('_')[1].split('.')[0])
                    existing.append(idx)

        if existing:
            return max(existing) + 1
        return 1

    def capture_and_detect(self) -> Dict[str, Any]:
        """
        Capture one frame and detect corners

        Returns:
            dict: {
                'success': bool,
                'rgb': RGB image,
                'depth': depth map,
                'corners': corner coordinates,
                'corners_refined': sub-pixel corners,
                'rgb_image_path': RGB image save path,
                'depth_image_path': depth image save path
            }
        """
        # Get image
        rgb, depth = self.camera.get_data()
        if rgb is None or depth is None:
            return {'success': False}

        # Convert to grayscale
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        # Detect corners
        success, corners, corners_refined = self.extractor.detect_corners(gray, refine=True)

        if not success:
            return {
                'success': False,
                'rgb': rgb,
                'depth': depth,
                'corners': None,
                'corners_refined': None
            }

        return {
            'success': True,
            'rgb': rgb,
            'depth': depth,
            'corners': corners,
            'corners_refined': corners_refined
        }

    def save_frame(self, frame_data: Dict[str, Any]) -> bool:
        """
        Save one frame of data

        Args:
            frame_data: data returned by capture_and_detect

        Returns:
            bool: whether save is successful
        """
        if not frame_data['success']:
            print("  [X] Corner detection failed, cannot save")
            return False

        # Get frame index
        idx = self.get_frame_index()

        # 1. Save TCP pose
        tcp_pose = self.robot.get_transform_matrix()
        tcp_path = os.path.join(self.data_path['poses'], f'tcp_{idx:03d}.txt')
        np.savetxt(tcp_path, tcp_pose, delimiter=' ')
        print(f"  TCP pose saved: {tcp_path}")

        # 2. Save corner coordinates
        corners = frame_data['corners_refined']
        corners_path = os.path.join(self.data_path['poses'], f'corners_{idx:03d}.txt')
        # Save as (N, 2) format
        corners_reshaped = corners.reshape(-1, 2)
        np.savetxt(corners_path, corners_reshaped, delimiter=' ')
        print(f"  Corner coordinates saved: {corners_path}")

        # 3. Save RGB image
        rgb_path = os.path.join(self.data_path['images'], f'rgb_{idx:03d}.png')
        cv2.imwrite(rgb_path, frame_data['rgb'])
        print(f"  RGB image saved: {rgb_path}")

        # 4. Save depth image
        depth_path = os.path.join(self.data_path['images'], f'depth_{idx:03d}.npy')
        np.save(depth_path, frame_data['depth'])
        print(f"  Depth image saved: {depth_path}")

        self.frame_count += 1
        print(f"  [OK] Frame {idx} saved (total {self.frame_count} frames)")

        return True

    def collect_single(self) -> bool:
        """
        Legacy single-frame collection helper.
        Prefer using collect_loop keyboard workflow (Space/Enter/Esc).

        Returns:
            bool: whether save is successful
        """
        print("\n" + "=" * 50)
        print("Single Frame Data Collection")
        print("=" * 50)
        print("Step 1: Press Enter to run one-shot corner detection")
        print("Step 2: Press Enter again to confirm save")
        print("        Press 'q' to cancel")
        print("        Preview window will continuously show camera feed")
        print("=" * 50)

        # Wait for user to press Enter to show detection result
        input()

        # Capture and detect
        frame_data = self.capture_and_detect()

        if not frame_data['success']:
            print("\nCorner detection failed, please adjust camera position")
            return False

        # Pause real-time preview, show corner detection result
        self.stop_preview()
        self._show_detection_result(frame_data)

        # Wait for user confirmation
        print("\nPress Enter to confirm save, or 'q' to cancel...")
        key = input()
        if key.lower() == 'q':
            print("Cancelled")
            # Resume real-time preview
            self.start_preview()
            return False

        # Save data
        result = self.save_frame(frame_data)

        # Resume real-time preview display
        cv2.destroyWindow('Corner Detection Result')
        self.start_preview()

        return result

    def detect_current_frame(self) -> bool:
        """Capture one frame, detect corners, and show the result for user confirmation."""
        frame_data = self.capture_and_detect()
        if not frame_data['success']:
            self._pending_detection = None
            print("[X] Corner detection failed. Please adjust board pose/light and try Space again.")
            return False

        self._pending_detection = frame_data
        self._show_detection_result(frame_data)
        print("[OK] Corner detection success. Press Enter to save this frame.")
        return True

    def _show_detection_result(self, frame_data: Dict[str, Any]) -> None:
        """
        Show corner detection result

        Args:
            frame_data: frame data
        """
        rgb = frame_data['rgb'].copy()
        corners = frame_data['corners_refined']

        # Draw corners
        cv2.drawChessboardCorners(rgb, CHECKERBOARD_CONFIG['size'], corners, True)

        # Display
        cv2.imshow('Corner Detection Result', rgb)
        cv2.waitKey(100)

    def _preview_loop(self) -> None:
        """Background thread: real-time camera image display"""
        # Create resizable window
        cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)

        while self._preview_active:
            try:
                rgb, depth = self.camera.get_data()

                # Check image validity (fix black screen issue)
                if rgb is None or rgb.size == 0 or rgb.mean() < 1.0:
                    time.sleep(0.05)
                    continue

                # Save latest frame for main thread
                with self._frame_lock:
                    self._latest_frame = rgb.copy()

                # Display real-time preview
                preview = rgb.copy()
                cv2.putText(preview, "Space: detect  Enter: save  Esc: exit", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(preview, f"Collected: {self.frame_count} / Min: {self.min_frames_required}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                cv2.imshow('Camera Preview', preview)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Preview error: {e}")
                time.sleep(0.1)

        cv2.destroyWindow('Camera Preview')

    def start_preview(self) -> None:
        """Start real-time preview thread"""
        if self._preview_active:
            return

        self._preview_active = True
        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._preview_thread.start()
        print("Real-time preview started")

    def stop_preview(self) -> None:
        """Stop real-time preview thread"""
        self._preview_active = False
        if self._preview_thread:
            self._preview_thread.join(timeout=1.0)
        cv2.destroyAllWindows()
        print("Real-time preview stopped")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get latest frame image"""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def collect_loop(self) -> int:
        """
        Loop collection mode:
        Repeatedly execute single frame collection until user chooses to exit

        Returns:
            int: total frames collected
        """
        print("\n" + "=" * 50)
        print("Start Loop Collection")
        print("=" * 50)
        print("At each position:")
        print("  1. Move robot to new position (teach by demo)")
        print("  2. Press Space to detect checkerboard corners")
        print("  3. Press Enter to save current valid detection")
        print("  4. Press Esc to exit collection")
        print(f"  Minimum recommended frames: {self.min_frames_required}")
        print("  Preview window continuously shows camera image")
        print("=" * 50)

        # Start real-time preview
        self.start_preview()

        while True:
            key = cv2.waitKey(50) & 0xFF

            # Esc exits collection mode
            if key == 27:
                print("\\nESC pressed. Exiting collection mode.")
                break

            # Space triggers detection preview
            if key == ord(' '):
                self.detect_current_frame()

            # Enter saves latest valid detection
            elif key in (13, 10):
                if self._pending_detection is None:
                    print("[!] No valid detection to save. Press Space first.")
                    continue

                saved = self.save_frame(self._pending_detection)
                if saved:
                    self._pending_detection = None

            time.sleep(0.02)

        # Stop real-time preview
        self.stop_preview()

        print(f"\nCollection complete, total {self.frame_count} frames")
        if self.frame_count < self.min_frames_required:
            print(f"[!] Collected frames < recommended minimum ({self.min_frames_required}).")
        return self.frame_count

    def get_saved_data(self) -> List[Dict[str, Any]]:
        """
        Get all saved data

        Returns:
            list: [{'tcp': 4x4 matrix, 'corners': corners, 'rgb': RGB image, 'depth': depth map}, ...]
        """
        data = []
        poses_dir = self.data_path['poses']
        images_dir = self.data_path['images']

        if not os.path.exists(poses_dir):
            return data

        # Get all TCP files
        tcp_files = sorted([f for f in os.listdir(poses_dir) if f.startswith('tcp_')])

        for tcp_file in tcp_files:
            idx = tcp_file.split('_')[1].split('.')[0]

            # Load TCP pose
            tcp_path = os.path.join(poses_dir, tcp_file)
            tcp = np.loadtxt(tcp_path)

            # Load corners
            corners_path = os.path.join(poses_dir, f'corners_{idx}.txt')
            if os.path.exists(corners_path):
                corners = np.loadtxt(corners_path).reshape(-1, 1, 2)
            else:
                corners = None

            # Load images
            rgb_path = os.path.join(images_dir, f'rgb_{idx}.png')
            rgb = cv2.imread(rgb_path) if os.path.exists(rgb_path) else None

            depth_path = os.path.join(images_dir, f'depth_{idx}.npy')
            depth = np.load(depth_path) if os.path.exists(depth_path) else None

            data.append({
                'tcp': tcp,
                'corners': corners,
                'rgb': rgb,
                'depth': depth,
                'index': idx
            })

        return data


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """Convert pose array to 4x4 transformation matrix"""
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
