"""Hardware-free regression tests: python -m unittest discover -s tests -p test_calibration_core.py."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handeye.calibration.transforms import (average_transforms, invert_transform, matrix_to_rotvec,
                                    pose_to_mat, rotvec_to_matrix, validate_transform)
from handeye.calibration.solver_axxb import HandEyeSolver
from handeye.calibration.optimizer import HandEyeOptimizer
from handeye.calibration.targets import (checkerboard_points, tag_points, tag_correspondences,
                                 solve_board_pose, target_config)
from handeye.session import data_paths, read_manifest, write_json
from handeye.calibration_solver import CalibrationSolver
from handeye.camera.model import Rectifier, opencv_distortion
from handeye.targets.geometry import tag_transform
from handeye.data_collector import CalibDataCollector
from handeye.error_calculator import ErrorCalculator
from handeye.robot.ur_robot import URRobot


def synthetic(mode, count=12):
    rng = np.random.default_rng(42)
    def transform():
        T = np.eye(4)
        T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        T[:3, 3] = rng.uniform(-0.4, 0.4, 3)
        return T
    X, Y = transform(), transform()
    robot = [transform() for _ in range(count)]
    camera = [invert_transform(X) @ (invert_transform(g) if mode == 'eye_on_hand' else g) @ Y
              for g in robot]
    return X, robot, camera


class TransformTests(unittest.TestCase):
    def test_pi_and_small_angle_roundtrip(self):
        for axis in ([1, 0, 0], [1, 2, 3]):
            axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
            for angle in (0, 1e-10, np.pi - 1e-9, np.pi):
                R = Rotation.from_rotvec(axis * angle).as_matrix()
                np.testing.assert_allclose(rotvec_to_matrix(matrix_to_rotvec(R)), R, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(matrix_to_rotvec(np.diag([1., -1., -1.]))), np.pi)

    def test_mean_stays_rigid(self):
        mean = average_transforms([pose_to_mat([0, 0, 0, 0, 0, a]) for a in (-0.5, 0.5)])
        validate_transform(mean)
        np.testing.assert_allclose(mean, np.eye(4), atol=1e-14)

    def test_invalid_transform_rejected(self):
        bad = np.eye(4); bad[0, 0] = 2
        with self.assertRaises(ValueError):
            validate_transform(bad)


class SolverTests(unittest.TestCase):
    def test_both_modes_recover_transform(self):
        for mode in ('eye_on_hand', 'eye_to_hand'):
            X, robot, camera = synthetic(mode)
            initial = HandEyeSolver(mode).solve_axxb_svd(robot, camera)
            np.testing.assert_allclose(initial, X, atol=1e-12)
            fitted, _, result = HandEyeOptimizer(mode).optimize(robot, camera, initial_X=initial)
            self.assertTrue(result.success)
            np.testing.assert_allclose(fitted, X, atol=1e-9)

    def test_translation_only_and_single_axis_rejected(self):
        for rotations in (False, True):
            robot = [pose_to_mat([i * .01, 0, 0, 0, 0, i * .1 if rotations else 0]) for i in range(6)]
            with self.assertRaisesRegex(ValueError, 'independent axes'):
                HandEyeSolver('eye_to_hand').solve_axxb_svd(robot, robot)

    def test_mismatched_lengths_rejected(self):
        _, robot, camera = synthetic('eye_to_hand')
        with self.assertRaises(ValueError):
            HandEyeSolver('eye_to_hand').solve_axxb_svd(robot, camera[:-1])

    def test_noisy_refinement_reduces_same_dimensionless_objective(self):
        X, robot, camera = synthetic('eye_to_hand')
        rng = np.random.default_rng(3)
        camera = [c @ pose_to_mat(np.r_[rng.normal(0, .0003, 3), rng.normal(0, .002, 3)]) for c in camera]
        initial = HandEyeSolver('eye_to_hand').solve_axxb_svd(robot, camera)
        fitted, _, result = HandEyeOptimizer('eye_to_hand').optimize(robot, camera, initial_X=initial)
        self.assertTrue(result.success)
        self.assertLessEqual(result.fun, result.initial_fun + 1e-9)
        validate_transform(fitted)

    def test_failed_or_worse_optimizer_never_saved_as_final(self):
        X, robot, camera = synthetic('eye_to_hand')
        bad = X.copy(); bad[0, 3] += 123
        for success, cost in [(False, 1.), (True, 100.)]:
            solver = CalibrationSolver('eye_to_hand', backend='apriltag_board')
            result = OptimizeResult(success=success, fun=cost, initial_fun=2.)
            with patch.object(solver.optimizer, 'optimize', return_value=(bad, 1., result)), \
                    patch.object(solver, '_save_result') as save, contextlib.redirect_stdout(io.StringIO()):
                solved = solver.solve(robot, camera, [], save=True)
            np.testing.assert_allclose(solved['X'], X, atol=1e-12)
            np.testing.assert_allclose(save.call_args.args[0], X, atol=1e-12)


class FragmentedSocket:
    def __init__(self, data, fragment=7):
        self.data, self.fragment = data, fragment
    def recv(self, size):
        count = min(size, self.fragment)
        chunk, self.data = self.data[:count], self.data[count:]
        return chunk
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def packet(timestamp, pose):
    data = bytearray(1108)
    struct.pack_into('!i', data, 0, len(data))
    struct.pack_into('!d', data, 4, timestamp)
    struct.pack_into('!6d', data, 444, *pose)
    struct.pack_into('!6d', data, 252, *range(6))
    return bytes(data)


class RobotStreamTests(unittest.TestCase):
    def test_fragmented_and_coalesced_packets(self):
        expected = [.1, .2, .3, .4, .5, .6]
        for fragment in (1, 7, 1500):
            sock = FragmentedSocket(packet(1., [0]*6) + packet(2., expected), fragment)
            with patch('handeye.robot.ur_robot.socket.create_connection', return_value=sock):
                np.testing.assert_allclose(URRobot().get_tool_pose(), expected)

    def test_closed_truncated_and_stale_stream_rejected(self):
        with self.assertRaises(ConnectionError):
            URRobot._read_packet(FragmentedSocket(packet(1., [0]*6)[:30]))
        with self.assertRaises(ValueError):
            URRobot._read_packet(FragmentedSocket(struct.pack('!i', -1)))
        sock = FragmentedSocket(packet(1., [0]*6) * 2)
        with patch('handeye.robot.ur_robot.socket.create_connection', return_value=sock), self.assertRaises(ValueError):
            URRobot().get_tool_pose()


K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])


class TargetTests(unittest.TestCase):
    def test_fused_board_pose_with_rotated_tags_and_partial_visibility(self):
        config = target_config('apriltag_board')
        config['tag_rotvecs'][2] = [0, 0, np.pi / 2]
        expected = pose_to_mat([.01, .02, .4, .3, -.2, .1])
        for ids in (np.array([1, 2, 3, 4]), np.array([2, 4])):
            points = np.vstack([tag_points(config, int(i)) for i in ids])
            corners, _ = cv2.projectPoints(points, matrix_to_rotvec(expected[:3, :3]), expected[:3, 3], K, np.zeros(5))
            tag_poses = np.array([expected @ tag_transform(config, int(i)) for i in ids])
            fitted, error = solve_board_pose(config, ids, corners, K, tag_poses, np.ones(len(ids)))
            np.testing.assert_allclose(fitted, expected, atol=1e-7)
            self.assertLess(error, 1e-7)

    def test_wrong_duplicate_and_missing_ids_rejected(self):
        config = target_config('apriltag_board')
        for ids, count in [([1, 1], 8), ([1, 99], 8), ([1, 2], 7), ([], 0)]:
            with self.assertRaises(ValueError):
                tag_correspondences(config, ids, np.zeros((count, 2)))

    def test_corner_relabel_is_not_hidden(self):
        config = target_config('apriltag_board')
        points = tag_points(config, config['tag_ids'][0])
        T = pose_to_mat([0, 0, .4, 0, 0, 0])
        pixels, _ = cv2.projectPoints(points, np.zeros(3), T[:3, 3], K, np.zeros(5))
        calc = ErrorCalculator('eye_to_hand', K, np.zeros(5), 'apriltag_board')
        error = calc._apriltag_frame_reprojection_error(T, np.roll(pixels, 1, axis=0),
                                                       np.array([config['tag_ids'][0]]), K, np.zeros(5))
        self.assertGreater(error, 10)

    def test_inconsistent_board_rejected(self):
        config = target_config('apriltag_board')
        ids = np.array([1, 2, 3, 4])
        points = np.vstack([tag_points(config, int(i)) for i in ids])
        pixels, _ = cv2.projectPoints(points, np.array([.3, -.2, .1]), np.array([0., 0., .4]), K, np.zeros(5))
        pixels[4:8, 0, 0] += 40
        with self.assertRaises(ValueError):
            expected = pose_to_mat([0., 0., .4, .3, -.2, .1])
            tag_poses = np.array([expected @ tag_transform(config, int(i)) for i in ids])
            solve_board_pose(config, ids, pixels, K, tag_poses, np.ones(len(ids)))


class CameraModelTests(unittest.TestCase):
    def test_distortion_models_not_silently_reinterpreted(self):
        for model in ('inverse_brown_conrady', 'modified_brown_conrady', 'kannala_brandt4'):
            with self.assertRaises(ValueError):
                opencv_distortion(model, [.1, 0, 0, 0, 0])
        np.testing.assert_array_equal(opencv_distortion('none', np.zeros(5)), np.zeros(5))

    def test_rectification_maps_cached(self):
        rectifier = Rectifier(K, np.zeros(5))
        with patch('handeye.camera.model.cv2.initUndistortRectifyMap', wraps=cv2.initUndistortRectifyMap) as build:
            rectifier.apply(np.zeros((480, 640), dtype=np.uint8))
            rectifier.apply(np.zeros((480, 640, 3), dtype=np.uint8))
            self.assertEqual(build.call_count, 1)


class FakeCamera:
    intrinsics = K.copy()
    dist_coeffs = np.zeros(5)
    im_width, im_height = 640, 480
    device_id = 'fake-camera'
    distortion_model = 'none'
    def get_data(self):
        return np.zeros((480, 640, 3), dtype=np.uint8), np.ones((480, 640))


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path_patch = patch('handeye.data_collector.get_data_path', return_value=data_paths(self.root))
        self.path_patch.start(); self.addCleanup(self.path_patch.stop)
        self.robot = unittest.mock.Mock()
        self.robot.get_transform_matrix.return_value = np.eye(4)
        self.robot.get_tcp_offset.return_value = np.zeros(6)
        self.robot.get_calibration_sample.side_effect = lambda: {
            'transform': self.robot.get_transform_matrix(), 'tcp_offset': np.zeros(6),
            'source': 'synthetic zero-TCP fixture', 'robot_timestamp_us': None}
        self.collector = CalibDataCollector(self.robot, FakeCamera(), 'eye_to_hand', 'checkerboard')
        points = checkerboard_points(self.collector.target_config)
        corners, _ = cv2.projectPoints(points, np.array([.1,.2,.05]), np.array([0.,0.,.5]), K, np.zeros(5))
        self.collector.extractor.detect_corners = lambda *args, **kwargs: (True, corners, corners)

    def test_save_uses_capture_snapshot_and_preserves_previous_sessions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            frame = self.collector.capture_and_detect()
            old_root = Path(self.collector.data_path['root'])
            self.robot.get_transform_matrix.return_value = pose_to_mat([.1, 0, 0, 0, 0, 0])
            self.collector.save_frame(frame)
            saved = np.asarray(json.loads((old_root / 'frames/000001/robot.json').read_text())['transform'])
            np.testing.assert_allclose(saved, np.eye(4))
            self.assertEqual(self.robot.get_transform_matrix.call_count, 2)
            self.collector.start_new_session()
        self.assertTrue((old_root / 'frames/000001/robot.json').exists())
        self.assertNotEqual(old_root, Path(self.collector.data_path['root']))
        metadata = read_manifest(old_root)
        self.assertEqual(metadata['camera_id'], 'fake-camera')
        self.assertTrue(metadata['tcp_definition']['verified'])
        self.assertEqual(metadata['tcp_definition']['offset_m_rotvec_rad'], [0] * 6)
        loaded = CalibDataCollector.from_session(str(old_root), 'eye_to_hand', 'checkerboard')
        self.assertEqual(len(loaded.get_saved_data()), 1)

    def test_capture_accepts_gray_camera_without_depth(self):
        self.collector.camera.get_data = lambda: (np.zeros((480,640),dtype=np.uint8), None)
        with contextlib.redirect_stdout(io.StringIO()):
            sample=self.collector.capture_and_detect()
            self.assertTrue(sample['success'])
            self.collector.save_frame(sample)
        root=Path(self.collector.data_path['root'])
        self.assertFalse((root/'frames/000001/depth.npy').exists())
        self.assertEqual(self.collector.get_saved_data()[0]['rgb'].ndim,2)

    def test_motion_during_capture_rejected(self):
        self.robot.get_transform_matrix.side_effect = [np.eye(4), pose_to_mat([.01, 0, 0, 0, 0, 0])]
        with contextlib.redirect_stdout(io.StringIO()):
            frame = self.collector.capture_and_detect()
        self.assertFalse(frame['success'])

    def test_nonzero_tcp_cannot_start_collection(self):
        self.robot.get_tcp_offset.return_value = np.array([0, 0, .1, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'Calibration requires TCP offset'):
            self.collector.start_new_session()
        self.assertFalse(list(self.root.rglob('session.json')))

    def test_tcp_offset_change_during_capture_is_rejected(self):
        self.robot.get_calibration_sample.side_effect = [
            {'transform': np.eye(4), 'tcp_offset': np.zeros(6)},
            {'transform': np.eye(4), 'tcp_offset': np.array([0, 0, 0, .1, 0, 0])}]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            self.collector.capture_and_detect()
        self.assertEqual(self.collector.get_saved_data(), [])

    def test_failed_image_write_never_commits_frame(self):
        with contextlib.redirect_stdout(io.StringIO()):
            frame = self.collector.capture_and_detect()
            with patch('handeye.data_collector.cv2.imwrite', return_value=False), self.assertRaises(OSError):
                self.collector.save_frame(frame)
        self.assertEqual(self.collector.get_saved_data(), [])

    def test_wrong_backend_and_camera_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.collector.start_new_session()
        with self.assertRaises(ValueError):
            CalibDataCollector.from_session(self.collector.data_path['root'], 'eye_to_hand', 'apriltag_board')
        self.collector.camera = FakeCamera()
        self.collector.camera.intrinsics = K * 2
        with self.assertRaises(ValueError):
            self.collector.capture_and_detect()

    def test_legacy_requires_explicit_opt_in(self):
        with self.assertRaises(ValueError):
            CalibDataCollector.from_session(str(self.root), 'eye_to_hand', 'apriltag_board')


class EvaluationTests(unittest.TestCase):
    def test_recorded_board_session_offline_pipeline(self):
        """Reload JSON geometry, solve, save an isolated run, and render its CSV report."""
        from handeye import pipeline as main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from tests.fixtures import write_session
            X = write_session(root)
            collector = CalibDataCollector.from_session(str(root), 'eye_to_hand', 'apriltag_board')
            solver = CalibrationSolver('eye_to_hand', K, np.zeros(5), 'apriltag_board')
            results = root / 'results'; results.mkdir()
            archived = results / 'handeye_transform.txt'; archived.write_text('preserve me')
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch('handeye.calibration_solver.get_results_path', return_value=str(results)), \
                    patch('builtins.input', return_value='n'):
                robot, camera, corners, images, tc, ti = solver.load_data(collector)
                result = solver.solve(robot, camera, corners)
                main.evaluate(collector, solver, result, robot, camera, corners, images, tc, ti)
            np.testing.assert_allclose(result['X'], X, atol=1e-7)
            self.assertEqual(archived.read_text(), 'preserve me')
            self.assertTrue((Path(result['output_path']) / 'result.json').exists())
            lines = (Path(result['output_path']) / 'residuals.csv').read_text().splitlines()
            self.assertEqual(len(lines), 9)

    def test_eye_to_hand_reference_free_errors_use_tcp_frame(self):
        X, robot, camera = synthetic('eye_to_hand')
        calc = ErrorCalculator('eye_to_hand', K, np.zeros(5), 'apriltag_board')
        np.testing.assert_allclose(calc.calculate_position_error(robot, camera, X), 0, atol=1e-12)
        np.testing.assert_allclose(calc.calculate_rotation_error(robot, camera, X), 0, atol=1e-12)

    def test_static_relative_residuals_cannot_validate_handeye(self):
        X, _, poses = synthetic('eye_to_hand')
        def residuals(x):
            values = x @ np.asarray(poses)
            return invert_transform(average_transforms(values)) @ values
        np.testing.assert_allclose(residuals(X), residuals(np.eye(4)), atol=1e-12)

    def test_observation_z_analysis_uses_camera_axes(self):
        from handeye.analysis import observation as module
        poses = np.stack([pose_to_mat([0, 0, z, 0, np.pi / 2, 0]) for z in (.1, .102)])
        _, translation, _ = module.observation_residuals(poses)
        np.testing.assert_allclose(translation[:, 2], [-1, 1], atol=1e-10)
        np.testing.assert_allclose(translation[:, 0], [0, 0], atol=1e-10)

    def test_empty_evaluation_is_not_zero_error(self):
        stats = ErrorCalculator('eye_to_hand', K, np.zeros(5)).compute_statistics(np.array([]))
        self.assertTrue(np.isnan(stats['mean']))


if __name__ == '__main__':
    unittest.main()
