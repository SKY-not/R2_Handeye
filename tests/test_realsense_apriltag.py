"""SDK-oracle camera checks and direct AprilTag board pose regressions."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from handeye.camera.model import InverseBrownRectifier
from handeye.camera.realsense import RealSenseCamera
from handeye.calibration.transforms import pose_to_mat
from handeye.targets.detector import TargetDetector
from handeye.targets.geometry import tag_points, tag_transform, solve_board_pose, target_config

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

K = np.array([[653.669, 0., 639.931], [0., 652.61, 360.909], [0., 0., 1.]])
D = np.array([-.052754826843738556, .055751752108335495,
              -.0003578081668820232, .000154135370394215, -.01853051595389843])


def intrinsics():
    i = rs.intrinsics()
    i.width, i.height = 1280, 720
    i.fx, i.fy, i.ppx, i.ppy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    i.model, i.coeffs = rs.distortion.inverse_brown_conrady, D.tolist()
    return i


@unittest.skipIf(rs is None, 'RealSense SDK is optional for offline calculations')
class InverseBrownTests(unittest.TestCase):
    def test_map_matches_sdk_projection_and_deprojection_across_image(self):
        i = intrinsics()
        rectifier = InverseBrownRectifier(K, D)
        rectifier.prepare((i.width, i.height))
        maps = rectifier.maps
        for v in np.linspace(0, i.height-1, 7).astype(int):
            for u in np.linspace(0, i.width-1, 9).astype(int):
                ray = [(u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], 1.]
                expected = rs.rs2_project_point_to_pixel(i, ray)
                source = [float(maps[0][v, u]), float(maps[1][v, u])]
                np.testing.assert_allclose(source, expected, atol=2e-4, rtol=0)
                actual = rs.rs2_deproject_pixel_to_point(i, source, 1.)
                np.testing.assert_allclose(actual, ray, atol=5e-7, rtol=0)
        rectifier.prepare((i.width, i.height))
        self.assertIs(rectifier.maps, maps)

    def test_color_and_metric_depth_share_sdk_pixel_mapping(self):
        i = intrinsics()
        row, col = np.indices((i.height, i.width), dtype=np.float32)
        image = np.dstack((col, row, np.ones_like(row)))
        depth = (row*i.width+col).astype(np.float32) * .001
        depth[340:380, 620:660] = 0
        rectifier = InverseBrownRectifier(K, D)
        color, metric = rectifier.apply(image, depth)
        for u, v in [(100, 100), (1000, 100), (200, 600), (1100, 600), (640, 360)]:
            source = rs.rs2_project_point_to_pixel(i, [(u-i.ppx)/i.fx, (v-i.ppy)/i.fy, 1.])
            np.testing.assert_allclose(color[v, u, :2], source, atol=.04, rtol=0)
            x, y = np.rint(source).astype(int)
            self.assertEqual(metric[v, u], depth[y, x])
        self.assertEqual(metric[360, 640], 0)
        with self.assertRaisesRegex(ValueError, 'aligned'):
            rectifier.apply(image, depth[:-1])

    def test_all_camera_getters_expose_matching_rectified_images(self):
        i = intrinsics()
        profile = Mock()
        profile.as_video_stream_profile.return_value.get_intrinsics.return_value = i
        camera = RealSenseCamera()
        camera.intrinsics = K.copy()
        camera.dist_coeffs = camera._get_dist_coeffs(profile).reshape(-1, 1)
        self.assertEqual(camera.distortion_model, 'none')
        np.testing.assert_array_equal(camera.dist_coeffs, np.zeros((5, 1)))
        self.assertEqual(camera.source_camera_model['distortion_model'], 'inverse_brown_conrady')
        np.testing.assert_allclose(camera.source_camera_model['dist_coeffs'], D)
        row, col = np.indices((i.height, i.width))
        raw_image = np.dstack((col % 256, row % 256, col*0)).astype(np.uint8)
        raw_depth = ((row+col) % 5000).astype(np.uint16)
        color, depth_frame = Mock(), Mock()
        color.get_data.return_value = raw_image
        depth_frame.get_data.return_value = raw_depth
        frames = Mock()
        frames.get_color_frame.return_value = color
        frames.get_depth_frame.return_value = depth_frame
        camera.pipeline = Mock()
        camera.pipeline.wait_for_frames.return_value = frames
        camera.align = Mock()
        camera.align.process.return_value = frames
        camera.connected, camera.depth_scale = True, .001
        expected_image, expected_depth = camera._color_rectifier.apply(
            raw_image, raw_depth.astype(np.float32)*camera.depth_scale)
        image, depth = camera.get_data()
        np.testing.assert_array_equal(image, expected_image)
        np.testing.assert_array_equal(depth, expected_depth)
        np.testing.assert_array_equal(camera.get_color_image(), expected_image)
        np.testing.assert_array_equal(camera.get_depth_image(), expected_depth)
        np.testing.assert_allclose(camera.project_to_3d(800, 400, .2),
            [(800-K[0, 2])*.2/K[0, 0], (400-K[1, 2])*.2/K[1, 1], .2])
        camera.enable_depth = False
        camera.align = None
        image, depth = camera.get_data()
        self.assertIsNone(depth)
        np.testing.assert_array_equal(image, expected_image)

    def test_standard_brown_output_is_unchanged(self):
        i = intrinsics()
        i.model = rs.distortion.brown_conrady
        profile = Mock()
        profile.as_video_stream_profile.return_value.get_intrinsics.return_value = i
        camera = RealSenseCamera()
        camera.intrinsics = K.copy()
        coefficients = camera._get_dist_coeffs(profile)
        self.assertIsNone(camera._color_rectifier)
        np.testing.assert_allclose(coefficients, D)


class DirectBoardPoseTests(unittest.TestCase):
    def test_margin_weighted_translation_and_rotation(self):
        cfg = target_config('apriltag_board')
        ids = np.array([1, 2, 3, 4])
        margins = np.array([20., 40., 60., 80.])
        angles = np.array([-.01, .01, .02, -.02])
        translations = np.array([[.001, 0, .4], [-.001, 0, .4], [0, .001, .4], [0, -.001, .4]])
        expected_angle = np.arctan2(np.sum(margins*np.sin(angles)), np.sum(margins*np.cos(angles)))
        expected = pose_to_mat([*np.average(translations, axis=0, weights=margins), 0, 0, expected_angle])
        candidates = [pose_to_mat([*t, 0, 0, a]) for t, a in zip(translations, angles)]
        poses = np.array([p @ tag_transform(cfg, int(i)) for p, i in zip(candidates, ids)])
        points = np.vstack([tag_points(cfg, int(i)) for i in ids])
        camera_K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
        corners, _ = cv2.projectPoints(points, np.array([0, 0, expected_angle]), expected[:3, 3], camera_K, np.zeros(5))
        with patch('cv2.solvePnP', side_effect=AssertionError('OpenCV PnP forbidden')), \
             patch('cv2.solvePnPGeneric', side_effect=AssertionError('OpenCV PnP forbidden')), \
             patch('cv2.solvePnPRefineLM', side_effect=AssertionError('OpenCV PnP forbidden')):
            result, error = solve_board_pose(cfg, ids, corners, camera_K, poses, margins)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        self.assertLess(error, 1e-9)

    @unittest.skipUnless(hasattr(cv2, 'aruco'), 'OpenCV marker generation is needed only for this synthetic test')
    def test_real_pyapriltags_detection_and_board_fusion_without_opencv_pnp(self):
        try:
            import pyapriltags
        except ImportError:
            self.skipTest('pyapriltags is a hardware capture dependency')
        cfg = target_config('apriltag_board')
        cfg.update(decision_margin_threshold=20, min_area_ratio=.0005,
                   min_visible_tags=2, max_reprojection_error_px=2.)
        camera_K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])
        camera = SimpleNamespace(intrinsics=camera_K, dist_coeffs=np.zeros(5))
        image = np.full((480, 640), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        for tag_id in cfg['tag_ids']:
            center = np.asarray(cfg['tag_centers'][str(tag_id)])
            u, v = np.rint(center[:2]*600/.15 + [320, 240]).astype(int)
            # OpenCV's generated orientation is rotated pi relative to the
            # canonical axes returned by pyapriltags for this dictionary.
            marker = np.rot90(cv2.aruco.generateImageMarker(dictionary, tag_id, 80), 2)
            image[v-40:v+40, u-40:u+40] = marker
        detector = TargetDetector(cfg, camera)
        with patch('cv2.solvePnP', side_effect=AssertionError('OpenCV PnP forbidden')), \
             patch('cv2.solvePnPGeneric', side_effect=AssertionError('OpenCV PnP forbidden')), \
             patch('cv2.solvePnPRefineLM', side_effect=AssertionError('OpenCV PnP forbidden')):
            result = detector.detect_image(image)
        self.assertTrue(result['success'])
        self.assertEqual(set(result['tag_ids']), set(cfg['tag_ids']))
        np.testing.assert_allclose(result['tag_pose'][:3, 3], [0, 0, .15], atol=.002)
        np.testing.assert_allclose(result['tag_pose'][:3, :3], np.eye(3), atol=.02)


if __name__ == '__main__':
    unittest.main()
