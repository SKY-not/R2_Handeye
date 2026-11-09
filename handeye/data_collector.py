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
from typing import Any, List, Optional, TypedDict, cast
import numpy as np
import cv2
import threading
import time
import json
from pathlib import Path

# Set Qt font directory before importing cv2 to reduce runtime font warnings.
_qt_font_dir = "/usr/share/fonts/truetype/dejavu"
if os.path.isdir(_qt_font_dir):
    os.environ.setdefault("QT_QPA_FONTDIR", _qt_font_dir)
    os.environ.setdefault("OPENCV_QT_FONTDIR", _qt_font_dir)


from handeye.calibration.transforms import invert_transform, matrix_to_rotvec, validate_transform
from handeye.session import data_paths, new_run_directory, read_manifest, write_json
from handeye.camera.model import Rectifier, validate_camera_matrix
from handeye.robot.interface import validate_zero_tcp_offset
from handeye.targets.detector import TargetDetector
from copy import deepcopy
from handeye.config import get_data_path, target_definition, load_config

class CaptureFrameData(TypedDict, total=False):
    """One captured frame and its detection payload."""

    tcp: np.ndarray
    captured_at_ns: int
    capture_duration_ms: float
    success: bool
    rgb: np.ndarray
    display_rgb: np.ndarray
    depth: Optional[np.ndarray]
    corners: Optional[np.ndarray]
    corners_refined: Optional[np.ndarray]
    tag_pose: np.ndarray
    charuco_ids: List[int]
    board_pose: np.ndarray
    tag_id: int
    tag_ids: List[int]
    tag_decision_margin: float
    tag_decision_margins: List[float]
    tag_corners: np.ndarray
    tcp_offset: np.ndarray
    tcp_state_source: Optional[str]
    robot_timestamp_us: Optional[int]


class SavedFrameData(TypedDict):
    """One persisted calibration sample loaded from disk."""

    tcp: np.ndarray
    corners: Optional[np.ndarray]
    tag_pose: Optional[np.ndarray]
    tag_corners: Optional[np.ndarray]
    tag_ids: Optional[np.ndarray]
    rgb: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    index: str


class CalibDataCollector(TargetDetector):
    """Calibration data collector"""

    def __init__(self, robot: Any, camera: Any, mode: str, backend: str = 'checkerboard', *, capture_config=None,
                 solver_config=None, target=None, reference=None, data_root=None) -> None:
        """
        Initialize data collector

        Args:
            robot: robot object
            camera: camera object
            mode: 'eye_on_hand' or 'eye_to_hand'
        """
        self.capture_config = deepcopy(load_config('configs/capture.json') if capture_config is None else capture_config)
        self.solver_config = deepcopy({} if solver_config is None else solver_config)
        self.reference = deepcopy({} if reference is None else reference)
        self.data_root = data_root
        self.robot = robot
        self.camera = camera
        self.mode = mode
        self.backend = backend
        self.target_config = target_definition(backend) if target is None else deepcopy(target)
        self.session_metadata = None
        self.legacy_pose_only = False

        # Data save path
        self.data_path = get_data_path(mode) if data_root is None else get_data_path(mode, data_root)


        # Collection counter
        self.frame_count = 0

        # Camera intrinsics
        self.intrinsics = np.eye(3) if camera is None else validate_camera_matrix(camera.intrinsics).copy()
        self.dist_coeffs = np.asarray(getattr(camera, 'dist_coeffs', np.zeros((5, 1))), dtype=np.float64).reshape(-1, 1)

        self.apriltag_detector: Optional[Any] = None
        self._configure_target()
        self.rectifier = Rectifier(self.intrinsics, self.dist_coeffs)

        # Real-time preview control
        self._preview_active = False
        self._latest_frame = None
        self._frame_lock = threading.Lock()

        # Latest validated detection from Space key, consumed by Enter key
        self._pending_detection: Optional[CaptureFrameData] = None
        min_pts = self.solver_config.get('min_calibration_points', 6)
        self.min_frames_required = max(6, int(cast(int, min_pts)))

        

    def _ensure_dirs(self) -> None:
        """Ensure data directories exist"""
        Path(self.data_path['frames']).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_session(cls, root: str, mode: str, backend: str,
                     legacy_pose_only: bool = False) -> 'CalibDataCollector':
        """Load recorded observations without importing or connecting camera hardware."""
        metadata = read_manifest(Path(root)) if (Path(root) / 'session.json').is_file() else None
        if metadata is not None and (metadata.get('mode') != mode or metadata.get('backend') != backend):
            raise ValueError('Data mode/backend differs from requested mode/backend')
        collector = cls(None, None, mode, backend, capture_config={}, solver_config={},
                        target=None if metadata is None else metadata['target'], reference={})
        collector.use_session(root, legacy_pose_only)
        return collector

    def start_new_session(self) -> None:
        if self.camera is None or self.robot is None:
            raise RuntimeError('Collection requires a connected camera and robot')
        offset = validate_zero_tcp_offset(self.robot.get_tcp_offset())
        root = new_run_directory(Path((get_data_path(self.mode) if self.data_root is None else get_data_path(self.mode, self.data_root))['root']))
        self.data_path = data_paths(root)
        self._ensure_dirs()
        self.frame_count = 0
        self._pending_detection = None
        self.legacy_pose_only = False
        self.intrinsics = validate_camera_matrix(self.camera.intrinsics).copy()
        self.dist_coeffs = np.asarray(self.camera.dist_coeffs, dtype=float).reshape(-1, 1)
        self.rectifier = Rectifier(self.intrinsics, self.dist_coeffs)
        self.session_metadata = {
            'schema_version': 2, 'kind': 'calibration', 'status': 'collecting', 'mode': self.mode, 'backend': self.backend,
            'target': self.target_config,
            'capture_config': self.capture_config,
            'tcp_definition': {'frame': 'flange_zero_offset', 'offset_m_rotvec_rad': offset,
                               'verified': True, 'source': 'robot state interface',
                               'verified_at_ns': time.time_ns()},
            'camera_matrix': self.intrinsics, 'dist_coeffs': self.dist_coeffs.ravel(),
            'camera_id': getattr(self.camera, 'device_id', None),
            'distortion_model': getattr(self.camera, 'distortion_model', 'opencv'),
            'source_camera_model': getattr(self.camera, 'source_camera_model', None),
            'image_size': [self.camera.im_width, self.camera.im_height],
            'corner_space': 'raw' if self.backend in ('checkerboard', 'charuco') else 'rectified_alpha0',
            'translation_unit': 'm', 'rotation_representation': 'rotation_vector', 'rotation_unit': 'rad',
            'depth_unit': 'm', 'depth_alignment': 'image',
        }
        write_json(root / 'session.json', self.session_metadata)
        print(f'新采集会话: {root}（历史数据保留）')

    def use_session(self, root: str, legacy_pose_only: bool = False) -> None:
        path = Path(root)
        if not (path / 'session.json').exists():
            if not legacy_pose_only or self.backend in ('checkerboard', 'charuco'):
                raise ValueError('旧数据缺少 session.json；AprilTag 可显式选择仅分析历史位姿，棋盘格请重新采集')
            self.legacy_pose_only = True
            self.session_metadata = None
            print('旧数据模式：仅使用历史位姿；不报告重投影误差，不允许追加采集。')
        else:
            metadata = read_manifest(path)
            definition = metadata.get('tcp_definition')
            if definition is not None and ('offset_m_rotvec_rad' in definition or definition.get('verified')):
                validate_zero_tcp_offset(definition.get('offset_m_rotvec_rad'))
            if metadata['mode'] != self.mode or metadata['backend'] != self.backend:
                raise ValueError('会话的标定模式/后端与当前选择不一致')
            expected_space = 'raw' if self.backend in ('checkerboard', 'charuco') else 'rectified_alpha0'
            if metadata['corner_space'] != expected_space:
                raise ValueError('会话角点坐标系与检测后端不一致')
            self.session_metadata = metadata
            self.legacy_pose_only = False
            self.target_config = metadata['target']
            self.intrinsics = validate_camera_matrix(np.asarray(metadata['camera_matrix']))
            self.dist_coeffs = np.asarray(metadata['dist_coeffs'], dtype=float).reshape(-1, 1)
            self.rectifier = Rectifier(self.intrinsics, self.dist_coeffs)
            self._configure_target()
        folder = 'frames' if self.session_metadata and self.session_metadata['schema_version'] == 2 else 'poses'
        if not (path / folder).is_dir():
            raise ValueError(f'No pose directory in session: {path}')
        self.data_path = data_paths(path)
        pattern = 'frame_*.json' if self.session_metadata is not None else 'tcp_*.txt'
        self.frame_count = sum(1 for _ in (path / folder).glob('*/frame.json' if folder == 'frames' else pattern))
        self._pending_detection = None

    def _check_capture_camera(self) -> None:
        if self.camera is None or self.robot is None:
            raise RuntimeError('This is an offline session; no capture hardware is attached')
        if self.legacy_pose_only:
            raise ValueError('不能向缺少元信息的旧数据追加采集，请新建会话')
        if self.session_metadata is None:
            self.start_new_session()
        metadata = self.session_metadata
        if (not np.allclose(self.intrinsics, self.camera.intrinsics)
                or not np.allclose(self.dist_coeffs.ravel(), np.asarray(self.camera.dist_coeffs).ravel())
                or metadata['image_size'] != [self.camera.im_width, self.camera.im_height]
                or metadata['camera_id'] != getattr(self.camera, 'device_id', None)):
            raise ValueError('当前相机与会话相机配置不同，请新建采集会话')

    def get_frame_index(self) -> int:
        """Get next frame index"""
        folders = Path(self.data_path['frames'])
        existing = [int(p.name) for p in folders.iterdir() if p.is_dir() and p.name.isdigit()] if folders.exists() else []
        return max(existing, default=0) + 1

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
        self._check_capture_camera()
        before_sample = self.robot.get_calibration_sample()
        validate_zero_tcp_offset(before_sample['tcp_offset'])
        before = validate_transform(before_sample['transform'], 'zero TCP before capture').copy()
        started = time.monotonic_ns()
        captured_at = time.time_ns()
        rgb, depth = self.camera.get_data()
        rgb = None if rgb is None else rgb.copy()
        depth = None if depth is None else depth.copy()
        after_sample = self.robot.get_calibration_sample()
        offset = validate_zero_tcp_offset(after_sample['tcp_offset'])
        after = validate_transform(after_sample['transform'], 'zero TCP after capture').copy()
        if rgb is None:
            return {'success': False}
        delta = invert_transform(before) @ after
        if (np.linalg.norm(delta[:3, 3]) > float(self.capture_config['capture_max_translation_m'])
                or np.linalg.norm(matrix_to_rotvec(delta[:3, :3])) > float(self.capture_config['capture_max_rotation_rad'])):
            print('采集期间机器人移动，已拒绝此帧；请静止后重试。')
            return {'success': False}
        snapshot = {'tcp': after, 'captured_at_ns': captured_at,
                    'capture_duration_ms': (time.monotonic_ns() - started) / 1e6,
                    'tcp_offset': offset, 'tcp_state_source': after_sample.get('source'),
                    'robot_timestamp_us': after_sample.get('robot_timestamp_us'),
                    'robot_before': before, 'robot_after': after,
                    'camera_timing': getattr(self.camera, 'last_frame_metadata', {})}
        expected = self.session_metadata['image_size']
        if [rgb.shape[1], rgb.shape[0]] != expected:
            raise ValueError('实际图像尺寸与会话配置不同')

        return {**self.detect_image(rgb, depth), **snapshot}

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

        if 'tcp' not in frame_data or 'captured_at_ns' not in frame_data:
            raise ValueError('Missing capture-time TCP snapshot; capture again before saving')
        validate_zero_tcp_offset(frame_data.get('tcp_offset'))
        self._check_capture_camera()
        idx = self.get_frame_index()

        from handeye.session import save_frame_data
        root = Path(self.data_path['frames']) / f'{idx:06d}'
        save_frame_data(root, frame_data, self.target_config, self.intrinsics, self.dist_coeffs,
                        self.session_metadata['corner_space'])
        self.frame_count += 1
        print(f'Saved frame {idx}: {root}')
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

        if self.backend == 'charuco':
            cv2.aruco.drawDetectedCornersCharuco(rgb, frame_data['corners_refined'],
                                               np.asarray(frame_data['charuco_ids'], dtype=np.int32).reshape(-1, 1))
            cv2.imshow('Corner Detection Result', rgb)
            cv2.waitKey(100)
            return
        if self.backend == 'apriltag_board':
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
        else:
            target_name = "ChArUco corners" if self.backend == "charuco" else "checkerboard corners"
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
        if self.session_metadata and self.session_metadata['schema_version'] == 2:
            from handeye.session import load_frame_data
            return load_frame_data(Path(self.data_path['root']), require_robot=True)
        data: List[SavedFrameData] = []
        poses_dir = self.data_path['poses']
        images_dir = self.data_path['images']

        if not os.path.exists(poses_dir):
            return data

        # Get all TCP files
        tcp_files = sorted([f for f in os.listdir(poses_dir) if f.startswith('tcp_')])

        for tcp_file in tcp_files:
            idx = tcp_file.split('_')[1].split('.')[0]
            if self.session_metadata is not None and not (Path(poses_dir) / f'frame_{idx}.json').exists():
                continue
            marker = Path(poses_dir) / f'frame_{idx}.json'
            if marker.is_file():
                frame_metadata = json.loads(marker.read_text(encoding='utf-8'))
                if 'tcp_offset_m_rotvec_rad' in frame_metadata:
                    validate_zero_tcp_offset(frame_metadata['tcp_offset_m_rotvec_rad'])

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

            tag_corners_path = os.path.join(poses_dir, f'tag_corners_{idx}.txt')
            if os.path.exists(tag_corners_path):
                tag_corners = np.loadtxt(tag_corners_path).reshape(-1, 2)
            else:
                tag_corners = None

            tag_ids_path = os.path.join(poses_dir, f'tag_ids_{idx}.txt')
            if os.path.exists(tag_ids_path):
                tag_ids = np.loadtxt(tag_ids_path, dtype=np.int32).reshape(-1)
            else:
                tag_ids = None

            # Load images
            rgb_path = os.path.join(images_dir, f'rgb_{idx}.png')
            rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED) if os.path.exists(rgb_path) else None

            depth = None  # Depth is archived, but unused by pose-only hand-eye calibration.

            if self.session_metadata is not None and rgb is None:
                raise ValueError(f'Missing image for committed frame {idx}')
            data.append({
                'tcp': tcp,
                'corners': corners,
                'tag_pose': tag_pose,
                'tag_corners': tag_corners,
                'tag_ids': tag_ids,
                'rgb': rgb,
                'depth': depth,
                'index': idx
            })

        return data
