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
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
import numpy as np
import cv2
import threading
import time

# Set Qt font directory before importing cv2 to reduce runtime font warnings.
_qt_font_dir = "/usr/share/fonts/truetype/dejavu"
if os.path.isdir(_qt_font_dir):
    os.environ.setdefault("QT_QPA_FONTDIR", _qt_font_dir)
    os.environ.setdefault("OPENCV_QT_FONTDIR", _qt_font_dir)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration.feature_extractor import CheckerboardExtractor
from config import APRILTAG_CONFIG, CHECKERBOARD_CONFIG, CALIBRATION_CONFIG, get_data_path

AprilTagDetector: Any = None
try:
    from pyapriltags import Detector as _AprilTagDetector
    AprilTagDetector = _AprilTagDetector
except ImportError:
    AprilTagDetector = None


class CalibDataCollector:
    """Calibration data collector"""

    def __init__(self, robot: Any, camera: Any, mode: str, backend: str = 'checkerboard') -> None:
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
        self.backend = backend

        # Checkerboard config
        cb_size_any = CHECKERBOARD_CONFIG['size']
        cb_size_seq = cast(Sequence[int], cb_size_any)
        cb_cols, cb_rows = int(cb_size_seq[0]), int(cb_size_seq[1])
        cb_square = float(cast(float, CHECKERBOARD_CONFIG['square_size']))
        self.extractor = CheckerboardExtractor((cb_cols, cb_rows), cb_square)
        self.cb_size: Tuple[int, int] = (cb_cols, cb_rows)

        # Data save path
        self.data_path = get_data_path(mode)
        self._ensure_dirs()

        # Collection counter
        self.frame_count = 0

        # Camera intrinsics
        self.intrinsics = camera.intrinsics
        self.dist_coeffs = np.asarray(getattr(camera, 'dist_coeffs', np.zeros((5, 1))), dtype=np.float64).reshape(-1, 1)

        # AprilTag backend config
        self.apriltag_detector: Optional[Any] = None
        self.apriltag_family = str(APRILTAG_CONFIG['family'])
        self.apriltag_size = float(cast(float, APRILTAG_CONFIG['tag_size']))
        self.apriltag_id = int(cast(int, APRILTAG_CONFIG['target_tag_id']))
        self.apriltag_margin_th = float(cast(float, APRILTAG_CONFIG['decision_margin_threshold']))
        self.apriltag_min_area_ratio = float(cast(float, APRILTAG_CONFIG['min_area_ratio']))

        if self.backend == 'apriltag':
            if AprilTagDetector is None:
                raise ImportError("未安装 pyapriltags，无法使用 AprilTag 后端")
            self.apriltag_detector = AprilTagDetector(families=self.apriltag_family)

        # Real-time preview control
        self._preview_active = False
        self._preview_thread = None
        self._latest_frame = None
        self._frame_lock = threading.Lock()

        # Latest validated detection from Space key, consumed by Enter key
        self._pending_detection: Optional[Dict[str, Any]] = None
        min_pts = CALIBRATION_CONFIG.get('min_calibration_points', 6)
        self.min_frames_required = max(6, int(cast(int, min_pts)))

        

    def _ensure_dirs(self) -> None:
        """Ensure data directories exist"""
        os.makedirs(self.data_path['poses'], exist_ok=True)
        os.makedirs(self.data_path['images'], exist_ok=True)

    def clear_old_data(self) -> None:
        """Clear old calibration data before starting a new collection"""
        import shutil
        for dir_key in ['poses', 'images']:
            dir_path = self.data_path.get(dir_key)
            if dir_path and os.path.exists(dir_path):
                # We could delete the entire directory and recreate it
                # or just delete all files and subdirectories in it
                for filename in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"警告：无法删除旧文件 {file_path}: {e}")
        print("已清理历史采集数据。")

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

        if self.backend == 'apriltag':
            return self._capture_and_detect_apriltag(rgb, depth, gray)

        # Detect checkerboard corners
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

    def _capture_and_detect_apriltag(self, rgb: np.ndarray, depth: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Detect AprilTag on undistorted image and use pyapriltags direct pose output."""
        if self.apriltag_detector is None:
            return {'success': False, 'rgb': rgb, 'depth': depth}

        h, w = gray.shape
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        new_k, _ = cv2.getOptimalNewCameraMatrix(
            intrinsics,
            self.dist_coeffs,
            (w, h),
            alpha=0,
            newImgSize=(w, h)
        )

        undistorted_gray = cv2.undistort(gray, intrinsics, self.dist_coeffs, None, new_k)
        undistorted_rgb = cv2.undistort(rgb, intrinsics, self.dist_coeffs, None, new_k)

        camera_params = [
            float(new_k[0, 0]),
            float(new_k[1, 1]),
            float(new_k[0, 2]),
            float(new_k[1, 2]),
        ]

        detections = self.apriltag_detector.detect(
            undistorted_gray,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=self.apriltag_size
        )

        target = None
        for det in detections:
            if int(det.tag_id) != self.apriltag_id:
                continue
            margin = float(getattr(det, 'decision_margin', 0.0))
            corners_2d = np.asarray(det.corners, dtype=np.float64).reshape(-1, 2)
            if corners_2d.shape[0] != 4:
                continue
            poly_area = float(abs(cv2.contourArea(corners_2d.astype(np.float32))))
            area_ratio = poly_area / float(w * h)
            if margin < self.apriltag_margin_th:
                continue
            if area_ratio < self.apriltag_min_area_ratio:
                continue
            target = det
            break

        if target is None:
            return {
                'success': False,
                'rgb': rgb,
                'depth': depth,
                'corners': None,
                'corners_refined': None
            }

        tag_pose = np.eye(4, dtype=np.float64)
        tag_pose[:3, :3] = np.asarray(target.pose_R, dtype=np.float64).reshape(3, 3)
        tag_pose[:3, 3] = np.asarray(target.pose_t, dtype=np.float64).reshape(3)

        return {
            'success': True,
            'rgb': rgb,
            'display_rgb': undistorted_rgb,
            'depth': depth,
            'tag_pose': tag_pose,
            'tag_id': int(target.tag_id),
            'tag_decision_margin': float(getattr(target, 'decision_margin', 0.0)),
            'tag_corners': np.asarray(target.corners, dtype=np.float32).reshape(-1, 2),
            'corners': None,
            'corners_refined': None
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

        # 2. Save detection payload
        if self.backend == 'apriltag':
            tag_pose = np.asarray(frame_data.get('tag_pose'), dtype=np.float64)
            if tag_pose.shape != (4, 4):
                print("  [X] AprilTag pose 无效，不能保存")
                return False
            tag_pose_path = os.path.join(self.data_path['poses'], f'tag_pose_{idx:03d}.txt')
            np.savetxt(tag_pose_path, tag_pose, delimiter=' ')
            print(f"  Tag pose saved: {tag_pose_path}")

            tag_corners = frame_data.get('tag_corners')
            if tag_corners is not None:
                tag_corners_path = os.path.join(self.data_path['poses'], f'tag_corners_{idx:03d}.txt')
                np.savetxt(tag_corners_path, np.asarray(tag_corners).reshape(-1, 2), delimiter=' ')
        else:
            corners = frame_data['corners_refined']
            corners_path = os.path.join(self.data_path['poses'], f'corners_{idx:03d}.txt')
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
        rgb = frame_data.get('display_rgb', frame_data['rgb']).copy()

        if self.backend == 'apriltag':
            tag_corners = frame_data.get('tag_corners')
            tag_id = frame_data.get('tag_id', -1)
            tag_margin = frame_data.get('tag_decision_margin', 0.0)
            if tag_corners is None:
                return
            pts = np.asarray(tag_corners, dtype=np.int32).reshape(-1, 2)
            cv2.polylines(rgb, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            center = np.mean(pts, axis=0).astype(int)
            cv2.circle(rgb, tuple(center), 6, (0, 0, 255), -1)
            cv2.putText(
                rgb,
                f'ID={tag_id} margin={float(tag_margin):.1f}',
                (int(center[0]) + 10, int(center[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            cv2.imshow('Corner Detection Result', rgb)
            cv2.waitKey(100)
            return

        corners = frame_data['corners_refined']
        if corners is None:
            return

        # Draw corners
        cv2.drawChessboardCorners(rgb, self.cb_size, corners, True)

        # Mark start/end corner to make corner ordering explicit.
        if corners is not None and len(corners) > 0:
            start_pt = tuple(corners[0, 0].astype(int))
            end_pt = tuple(corners[-1, 0].astype(int))

            cv2.circle(rgb, start_pt, 9, (0, 255, 255), -1)
            cv2.putText(
                rgb,
                'START',
                (start_pt[0] + 8, start_pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.circle(rgb, end_pt, 9, (255, 0, 255), -1)
            cv2.putText(
                rgb,
                'END',
                (end_pt[0] + 8, end_pt[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )

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
                target_hint = "Space: detect target  Enter: save  Esc: exit"
                cv2.putText(preview, target_hint, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(preview, f"Collected: {self.frame_count} / Min: {self.min_frames_required}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                cv2.imshow('Camera Preview', preview)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Preview error: {e}")
                time.sleep(0.1)

        cv2.destroyWindow('Camera Preview')

    def _update_preview_once(self) -> int:
        """Render one preview frame in main thread and return keyboard key code."""
        try:
            rgb, _ = self.camera.get_data()
            if rgb is None or rgb.size == 0 or rgb.mean() < 1.0:
                return cv2.waitKey(30) & 0xFF

            with self._frame_lock:
                self._latest_frame = rgb.copy()

            preview = rgb.copy()
            target_hint = "Space: detect target  Enter: save  Esc: exit"
            cv2.putText(preview, target_hint, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(preview, f"Collected: {self.frame_count} / Min: {self.min_frames_required}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            cv2.imshow('Camera Preview', preview)
        except Exception as e:
            print(f"Preview error: {e}")

        return cv2.waitKey(30) & 0xFF

    def start_preview(self) -> None:
        """Prepare preview window (UI is updated in main thread)."""
        if self._preview_active:
            return

        self._preview_active = True
        cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
        print("Real-time preview started")

    def stop_preview(self) -> None:
        """Stop real-time preview thread"""
        self._preview_active = False
        if self._preview_thread:
            self._preview_thread.join(timeout=1.0)
            self._preview_thread = None
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
        target_name = "AprilTag" if self.backend == 'apriltag' else "checkerboard corners"
        print("At each position:")
        print("  1. Move robot to new position (teach by demo)")
        print(f"  2. Press Space to detect {target_name}")
        print("  3. Press Enter to save current valid detection")
        print("  4. Press Esc to exit collection")
        print(f"  Minimum recommended frames: {self.min_frames_required}")
        print("  Preview window continuously shows camera image")
        print("=" * 50)

        # Start preview window in main thread
        self.start_preview()

        while True:
            key = self._update_preview_once()

            # Esc exits collection mode
            if key == 27:
                print("\nESC pressed. Exiting collection mode.")
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
                    cv2.destroyWindow('Corner Detection Result')

            time.sleep(0.01)

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
        data: List[Dict[str, Any]] = []
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

            tag_pose_path = os.path.join(poses_dir, f'tag_pose_{idx}.txt')
            if os.path.exists(tag_pose_path):
                tag_pose = np.loadtxt(tag_pose_path)
            else:
                tag_pose = None

            # Load images
            rgb_path = os.path.join(images_dir, f'rgb_{idx}.png')
            rgb = cv2.imread(rgb_path) if os.path.exists(rgb_path) else None

            depth_path = os.path.join(images_dir, f'depth_{idx}.npy')
            depth = np.load(depth_path) if os.path.exists(depth_path) else None

            data.append({
                'tcp': tcp,
                'corners': corners,
                'tag_pose': tag_pose,
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
