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
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, cast
import numpy as np
import cv2
import threading
import time
from scipy.spatial.transform import Rotation

# Set Qt font directory before importing cv2 to reduce runtime font warnings.
_qt_font_dir = "/usr/share/fonts/truetype/dejavu"
if os.path.isdir(_qt_font_dir):
    os.environ.setdefault("QT_QPA_FONTDIR", _qt_font_dir)
    os.environ.setdefault("OPENCV_QT_FONTDIR", _qt_font_dir)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from calibration.feature_extractor import CheckerboardExtractor
from calibration.transforms import invert_transform
from config import APRILTAG_BOARD_CONFIG, APRILTAG_CONFIG, CHECKERBOARD_CONFIG, CALIBRATION_CONFIG, get_data_path

AprilTagDetector: Any = None
try:
    from pyapriltags import Detector as _AprilTagDetector
    AprilTagDetector = _AprilTagDetector
except ImportError:
    AprilTagDetector = None


class CaptureFrameData(TypedDict, total=False):
    """One captured frame and its detection payload."""

    success: bool
    rgb: np.ndarray
    display_rgb: np.ndarray
    depth: np.ndarray
    corners: Optional[np.ndarray]
    corners_refined: Optional[np.ndarray]
    tag_pose: np.ndarray
    tag_id: int
    tag_ids: List[int]
    tag_decision_margin: float
    tag_decision_margins: List[float]
    tag_corners: np.ndarray


class SavedFrameData(TypedDict):
    """One persisted calibration sample loaded from disk."""

    tcp: np.ndarray
    corners: Optional[np.ndarray]
    tag_pose: Optional[np.ndarray]
    rgb: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    index: str


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
        if self.backend == 'apriltag_board':
            self.apriltag_family = str(APRILTAG_BOARD_CONFIG['family'])
            self.apriltag_size = float(cast(float, APRILTAG_BOARD_CONFIG['tag_size']))
            board_ids = cast(Sequence[int], APRILTAG_BOARD_CONFIG['tag_ids'])
            self.apriltag_board_ids = [int(tag_id) for tag_id in board_ids]
            raw_centers = cast(Dict[int, Sequence[float]], APRILTAG_BOARD_CONFIG['tag_centers'])
            self.apriltag_board_centers = {
                int(tag_id): np.asarray(center, dtype=np.float64).reshape(3)
                for tag_id, center in raw_centers.items()
            }
            self.apriltag_id = self.apriltag_board_ids[0]
            self.apriltag_margin_th = float(cast(float, APRILTAG_BOARD_CONFIG['decision_margin_threshold']))
            self.apriltag_min_area_ratio = float(cast(float, APRILTAG_BOARD_CONFIG['min_area_ratio']))
        else:
            self.apriltag_family = str(APRILTAG_CONFIG['family'])
            self.apriltag_size = float(cast(float, APRILTAG_CONFIG['tag_size']))
            self.apriltag_id = int(cast(int, APRILTAG_CONFIG['target_tag_id']))
            self.apriltag_board_ids = [self.apriltag_id]
            self.apriltag_board_centers: Dict[int, np.ndarray] = {}
            self.apriltag_margin_th = float(cast(float, APRILTAG_CONFIG['decision_margin_threshold']))
            self.apriltag_min_area_ratio = float(cast(float, APRILTAG_CONFIG['min_area_ratio']))

        if self.backend in ('apriltag', 'apriltag_board'):
            if AprilTagDetector is None:
                raise ImportError("未安装 pyapriltags，无法使用 AprilTag 后端")
            self.apriltag_detector = AprilTagDetector(families=self.apriltag_family)

        # Real-time preview control
        self._preview_active = False
        self._latest_frame = None
        self._frame_lock = threading.Lock()

        # Latest validated detection from Space key, consumed by Enter key
        self._pending_detection: Optional[CaptureFrameData] = None
        min_pts = CALIBRATION_CONFIG.get('min_calibration_points', 6)
        self.min_frames_required = max(6, int(cast(int, min_pts)))

        

    def _ensure_dirs(self) -> None:
        """Ensure data directories exist"""
        os.makedirs(self.data_path['poses'], exist_ok=True)
        os.makedirs(self.data_path['images'], exist_ok=True)

    def clear_old_data(self) -> None:
        """Clear old calibration data before starting a new collection"""
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

    def capture_and_detect(self) -> 'CaptureFrameData':
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
        if self.backend == 'apriltag_board':
            return self._capture_and_detect_apriltag_board(rgb, depth, gray)

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

    @staticmethod
    def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Build a 4x4 transform from a rotation matrix and translation vector."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        return T

    def _detect_apriltags_with_pose(
        self,
        rgb: np.ndarray,
        gray: np.ndarray
    ) -> Tuple[np.ndarray, List[Any], int, int]:
        """Detect AprilTags on an undistorted image using pyapriltags pose output."""
        if self.apriltag_detector is None:
            return rgb, [], gray.shape[1], gray.shape[0]

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
        return undistorted_rgb, list(detections), w, h

    def _capture_and_detect_apriltag(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        gray: np.ndarray
    ) -> 'CaptureFrameData':
        """Detect AprilTag on undistorted image and use pyapriltags direct pose output."""
        if self.apriltag_detector is None:
            return {'success': False, 'rgb': rgb, 'depth': depth}

        undistorted_rgb, detections, w, h = self._detect_apriltags_with_pose(rgb, gray)

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

        tag_pose = self._make_transform(target.pose_R, target.pose_t)

        return {
            'success': True,
            'rgb': rgb,
            'display_rgb': undistorted_rgb,
            'depth': depth,
            'tag_pose': tag_pose,
            'tag_id': int(target.tag_id),
            'tag_decision_margin': float(getattr(target, 'decision_margin', 0.0)),
            'tag_ids': [int(target.tag_id)],
            'tag_decision_margins': [float(getattr(target, 'decision_margin', 0.0))],
            'tag_corners': np.asarray(target.corners, dtype=np.float32).reshape(-1, 2),
            'corners': None,
            'corners_refined': None
        }

    def _board_to_tag_transform(self, tag_id: int) -> np.ndarray:
        """Return T_board_tag for a configured tag on the rigid AprilTag board."""
        center = self.apriltag_board_centers[tag_id]
        return self._make_transform(np.eye(3, dtype=np.float64), center)

    def _average_board_poses(
        self,
        board_poses: List[np.ndarray],
        margins: List[float]
    ) -> np.ndarray:
        """Fuse per-tag T_camera_board estimates in one frame."""
        translations = np.asarray([pose[:3, 3] for pose in board_poses], dtype=np.float64)
        rotations = Rotation.from_matrix([pose[:3, :3] for pose in board_poses])

        weights = np.asarray(margins, dtype=np.float64)
        if weights.shape[0] != len(board_poses) or float(np.sum(weights)) <= 1e-9:
            weights = np.ones(len(board_poses), dtype=np.float64)
        weights = weights / float(np.sum(weights))

        fused = np.eye(4, dtype=np.float64)
        fused[:3, 3] = np.average(translations, axis=0, weights=weights)
        fused[:3, :3] = rotations.mean(weights=weights).as_matrix()
        return fused

    def _capture_and_detect_apriltag_board(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        gray: np.ndarray
    ) -> 'CaptureFrameData':
        """Fuse visible AprilTags into one board-center pose using pyapriltags poses."""
        if self.apriltag_detector is None:
            return {'success': False, 'rgb': rgb, 'depth': depth}

        undistorted_rgb, detections, w, h = self._detect_apriltags_with_pose(rgb, gray)
        board_poses: List[np.ndarray] = []
        margins: List[float] = []
        visible_ids: List[int] = []
        visible_corners: List[np.ndarray] = []

        for det in detections:
            tag_id = int(det.tag_id)
            if tag_id not in self.apriltag_board_ids or tag_id not in self.apriltag_board_centers:
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

            T_camera_tag = self._make_transform(det.pose_R, det.pose_t)
            T_board_tag = self._board_to_tag_transform(tag_id)
            T_camera_board = T_camera_tag @ invert_transform(T_board_tag)
            board_poses.append(T_camera_board)
            margins.append(margin)
            visible_ids.append(tag_id)
            visible_corners.append(np.asarray(det.corners, dtype=np.float32).reshape(4, 2))

        if not board_poses:
            return {
                'success': False,
                'rgb': rgb,
                'depth': depth,
                'corners': None,
                'corners_refined': None
            }

        fused_pose = self._average_board_poses(board_poses, margins)

        return {
            'success': True,
            'rgb': rgb,
            'display_rgb': undistorted_rgb,
            'depth': depth,
            'tag_pose': fused_pose,
            'tag_id': int(visible_ids[0]),
            'tag_ids': visible_ids,
            'tag_decision_margin': float(np.mean(margins)),
            'tag_decision_margins': margins,
            'tag_corners': np.vstack(visible_corners).astype(np.float32),
            'corners': None,
            'corners_refined': None
        }

    def save_frame(self, frame_data: 'CaptureFrameData') -> bool:
        """
        Save one frame of data

        Args:
            frame_data: data returned by capture_and_detect

        Returns:
            bool: whether save is successful
        """
        if not frame_data.get('success', False):
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
        if self.backend in ('apriltag', 'apriltag_board'):
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

            tag_ids = frame_data.get('tag_ids')
            if tag_ids is not None:
                tag_ids_path = os.path.join(self.data_path['poses'], f'tag_ids_{idx:03d}.txt')
                np.savetxt(tag_ids_path, np.asarray(tag_ids, dtype=np.int32).reshape(-1, 1), fmt='%d')
        else:
            corners = frame_data['corners_refined']
            if corners is None:
                print("  [X] Corner detection payload 无效，不能保存")
                return False
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

    def _show_detection_result(self, frame_data: 'CaptureFrameData') -> None:
        """
        Show corner detection result

        Args:
            frame_data: frame data
        """
        rgb = frame_data.get('display_rgb', frame_data['rgb']).copy()

        if self.backend in ('apriltag', 'apriltag_board'):
            tag_corners = frame_data.get('tag_corners')
            tag_id = frame_data.get('tag_id', -1)
            tag_ids = frame_data.get('tag_ids', [tag_id])
            tag_margin = frame_data.get('tag_decision_margin', 0.0)
            if tag_corners is None:
                return
            pts_all = np.asarray(tag_corners, dtype=np.int32).reshape(-1, 4, 2)
            for pts in pts_all:
                cv2.polylines(rgb, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                center_i = np.mean(pts, axis=0).astype(int)
                cv2.circle(rgb, tuple(center_i), 4, (0, 0, 255), -1)
            center = np.mean(pts_all.reshape(-1, 2), axis=0).astype(int)
            cv2.putText(
                rgb,
                f'ID={tag_ids} margin={float(tag_margin):.1f}',
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
        if self.backend == 'apriltag_board':
            target_name = "four-AprilTag board"
        elif self.backend == 'apriltag':
            target_name = "AprilTag"
        else:
            target_name = "checkerboard corners"
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

    def get_saved_data(self) -> List['SavedFrameData']:
        """
        Get all saved data

        Returns:
            list: [{'tcp': 4x4 matrix, 'corners': corners, 'rgb': RGB image, 'depth': depth map}, ...]
        """
        data: List[SavedFrameData] = []
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
