"""Shared pyapriltags detection and direct pose estimation."""
from __future__ import annotations
from typing import Any, List, Tuple
import cv2
import numpy as np
from handeye.targets.feature_extractor import CheckerboardExtractor
from handeye.targets.geometry import tag_points, solve_board_pose, reprojection_errors, checkerboard_points
from handeye.camera.model import Rectifier


class TargetDetector:
    def __init__(self, target, camera):
        self.target_config = target
        self.camera = camera
        self.backend = 'checkerboard' if target['type'] == 'checkerboard' else 'apriltag_board'
        self.intrinsics = np.asarray(camera.intrinsics)
        self.dist_coeffs = np.asarray(camera.dist_coeffs)
        self.rectifier = Rectifier(self.intrinsics, self.dist_coeffs)
        self._configure_target()

    def detect_image(self, image, depth=None):
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.target_config.get('type') == 'apriltag':
            size = (image.shape[1], image.shape[0])
            K, _ = cv2.getOptimalNewCameraMatrix(self.intrinsics, self.dist_coeffs, size, 1.0, size)
            display = cv2.undistort(image, self.intrinsics, self.dist_coeffs, None, K)
            mono = display if display.ndim == 2 else cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
            detections = self.apriltag_detector.detect(mono, estimate_tag_pose=True,
                camera_params=(K[0,0], K[1,1], K[0,2], K[1,2]), tag_size=self.target_config['tag_size'])
            for det in detections:
                if int(det.tag_id) not in self.apriltag_board_ids or det.decision_margin < self.apriltag_margin_th:
                    continue
                if getattr(det, 'pose_R', None) is None or getattr(det, 'pose_t', None) is None:
                    continue
                corners = np.asarray(det.corners).reshape(4,2)
                if abs(cv2.contourArea(corners.astype(np.float32))) / (size[0]*size[1]) < self.apriltag_min_area_ratio:
                    continue
                T = self._make_transform(det.pose_R, det.pose_t)
                error = float(reprojection_errors(T, tag_points(self.target_config, int(det.tag_id)), corners, K, np.zeros(5)).mean())
                limit = self.target_config.get('max_reprojection_error_px')
                if limit is not None and error > limit:
                    continue
                return {'success':True, 'rgb':image, 'depth':depth, 'display_rgb':display,
                        'tag_pose':T, 'tag_ids':[int(det.tag_id)], 'tag_corners':corners,
                        'tag_decision_margin':float(det.decision_margin),
                        'reprojection_error_px':error, 'projection_matrix':K, 'projection_distortion':np.zeros(5)}
            return {'success':False}
        if self.backend == 'apriltag_board':
            return self._capture_and_detect_apriltag_board(image, depth, gray)
        success, corners, refined = self.extractor.detect_corners(gray, refine=True)
        return {'success':success, 'rgb':image, 'depth':depth, 'corners':corners, 'corners_refined':refined}

    def _configure_target(self) -> None:
        cfg = self.target_config
        if self.backend == 'checkerboard':
            self.cb_size = (int(cfg['size'][0]), int(cfg['size'][1]))
            self.extractor = CheckerboardExtractor(self.cb_size, float(cfg['square_size']))
            return
        self.apriltag_family = str(cfg['family'])
        self.apriltag_board_ids = [int(i) for i in cfg['tag_ids']]
        if len(set(self.apriltag_board_ids)) != len(self.apriltag_board_ids):
            raise ValueError('Duplicate tag IDs in board configuration')
        for tag_id in self.apriltag_board_ids:
            tag_points(cfg, tag_id)
        self.apriltag_margin_th = float(cfg['decision_margin_threshold'])
        self.apriltag_min_area_ratio = float(cfg['min_area_ratio'])
        if self.camera is None:
            return  # Offline replay only needs the saved geometry and observations.
        from pyapriltags import Detector as AprilTagDetector
        options = {'nthreads':2, 'quad_sigma':0.0, 'decode_sharpening':0.25, 'debug':0} if cfg.get('type') == 'apriltag' else {}
        self.apriltag_detector = AprilTagDetector(families=self.apriltag_family,
                                                 quad_decimate=1.0, refine_edges=1, **options)

    @staticmethod
    def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Build a 4x4 transform from a rotation matrix and translation vector."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        return T

    def _detect_board_tags(
        self,
        rgb: np.ndarray,
        gray: np.ndarray
    ) -> Tuple[np.ndarray, List[Any], int, int]:
        """Estimate each visible tag pose with pyapriltags on the rectified image."""
        if self.apriltag_detector is None:
            return rgb, [], gray.shape[1], gray.shape[0]

        h, w = gray.shape
        undistorted_gray, _ = self.rectifier.apply(gray)
        undistorted_rgb, _ = self.rectifier.apply(rgb)

        detections = self.apriltag_detector.detect(
            undistorted_gray,
            estimate_tag_pose=True,
            camera_params=(self.rectifier.new_K[0, 0], self.rectifier.new_K[1, 1],
                           self.rectifier.new_K[0, 2], self.rectifier.new_K[1, 2]),
            tag_size=self.target_config['tag_size'],
        )
        return undistorted_rgb, list(detections), w, h

    def _capture_and_detect_apriltag_board(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        gray: np.ndarray
    ) -> 'CaptureFrameData':
        """Fuse direct tag poses into the configured board frame."""
        if self.apriltag_detector is None:
            return {'success': False, 'rgb': rgb, 'depth': depth}

        undistorted_rgb, detections, w, h = self._detect_board_tags(rgb, gray)
        margins: List[float] = []
        visible_ids: List[int] = []
        visible_corners: List[np.ndarray] = []
        visible_poses: List[np.ndarray] = []

        for det in detections:
            tag_id = int(det.tag_id)
            if tag_id not in self.apriltag_board_ids:
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

            if getattr(det, 'pose_R', None) is None or getattr(det, 'pose_t', None) is None:
                continue
            margins.append(margin)
            visible_poses.append(self._make_transform(det.pose_R, det.pose_t))
            visible_ids.append(tag_id)
            visible_corners.append(np.asarray(det.corners, dtype=np.float32).reshape(4, 2))

        if not visible_ids:
            return {
                'success': False,
                'rgb': rgb,
                'depth': depth,
                'corners': None,
                'corners_refined': None
            }

        try:
            fused_pose, _ = solve_board_pose(self.target_config, np.asarray(visible_ids),
                                             np.vstack(visible_corners), self.rectifier.new_K,
                                             np.asarray(visible_poses), np.asarray(margins))
        except (ValueError, cv2.error) as exc:
            print(f'AprilTag board rejected: {exc}')
            return {'success': False}

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

