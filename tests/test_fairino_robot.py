"""FAIRINO state conversion, stale feedback and offline integration; no network."""
import contextlib
import ctypes
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from handeye.config import load_config, load_workflow, resolve_path
from handeye.device_manager import create_robot, create_motion_interfaces
from handeye.robot.fairino_robot import FairinoRobot, _load_sdk
from handeye.session import write_json


class State(ctypes.Structure):
    _fields_ = [('frame_cnt', ctypes.c_uint8),
                ('millisecond', ctypes.c_uint16),
                ('flange_cur_pos', ctypes.c_double * 6),
                ('toolCoord', ctypes.c_double * 6)]


def state(frame=1, pose=None, offset=None, millisecond=1):
    packet = State()
    packet.frame_cnt = frame
    packet.millisecond = millisecond
    packet.flange_cur_pos[:] = [100, 200, 300, 90, 90, 0] if pose is None else pose
    packet.toolCoord[:] = [0.] * 6 if offset is None else offset
    return packet


class FakeRPC:
    initial = None
    instances = []

    def __init__(self, ip):
        self.ip = ip
        self.robot_state_pkg = self.initial
        self.is_conect = self.SDK_state = self.sock_cli_state_state = True
        self.reconnect_flag = self.closeRPC_state = False
        self.stop_event = threading.Event()
        self.socket_closed = False
        self.sock_cli_state = SimpleNamespace(close=self._close_socket)
        self.robot = object()
        self.close_calls = 0
        self.instances.append(self)

    def _close_socket(self):
        self.socket_closed = True

    def CloseRPC(self):
        self.close_calls += 1


class Clock:
    def __init__(self):
        self.now = 0.
        self.pending = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        if self.pending:
            FakeRPC.instances[-1].robot_state_pkg = self.pending.pop(0)


class FairinoStateTests(unittest.TestCase):
    def setUp(self):
        FakeRPC.initial = state(1)
        FakeRPC.instances = []
        self.clock = Clock()
        self.clock.pending = [state(2)]
        sdk = SimpleNamespace(RPC=FakeRPC, RobotStatePkg=State)
        for item in (patch('handeye.robot.fairino_robot._load_sdk', return_value=sdk),
                     patch('handeye.robot.fairino_robot.time.monotonic', self.clock.monotonic),
                     patch('handeye.robot.fairino_robot.time.sleep', self.clock.sleep)):
            item.start()
            self.addCleanup(item.stop)
        self.robot = FairinoRobot('192.0.2.1', 'unused-sdk', state_timeout_s=.02)
        self.addCleanup(self.robot.disconnect)

    def test_millimetres_and_mixed_axis_rpy_become_base_transform(self):
        sample = self.robot.get_calibration_sample()
        np.testing.assert_allclose(sample['transform'][:3, 3], [.1, .2, .3])
        # Fixed-axis X=90 then Y=90; treating RPY as a rotation vector fails.
        np.testing.assert_allclose(sample['transform'][:3, :3],
                                   [[0, 1, 0], [0, 0, -1], [-1, 0, 0]], atol=1e-14)
        np.testing.assert_array_equal(sample['tcp_offset'], np.zeros(6))
        self.assertIsNone(sample['robot_timestamp_us'])
        self.assertEqual(FakeRPC.instances[0].ip, '192.0.2.1')
        packet = FakeRPC.instances[0].robot_state_pkg
        packet.flange_cur_pos[0] = 9999
        packet.toolCoord[0] = 5
        self.assertEqual(sample['transform'][0, 3], .1)
        np.testing.assert_array_equal(sample['tcp_offset'], np.zeros(6))

    def test_nonzero_and_nonfinite_raw_tool_offsets_are_rejected(self):
        for offset in ([1e-15, 0, 0, 0, 0, 0], [0, 0, 0, 0, 360, 0],
                       [0, 0, .1, 0, 0, 0], [float('nan')] * 6):
            with self.subTest(offset=offset):
                self.clock.pending = [state(2, offset=offset)]
                self.robot.disconnect()
                with self.assertRaisesRegex(ValueError, 'Calibration requires TCP offset'):
                    self.robot.get_calibration_sample()

    def test_nonfinite_pose_is_rejected(self):
        self.clock.pending = [state(2, pose=[0, 0, float('inf'), 0, 0, 0])]
        with self.assertRaisesRegex(ValueError, 'six finite values'):
            self.robot.get_calibration_sample()

    def test_initial_class_placeholder_and_cached_or_duplicate_packets_time_out(self):
        for initial, pending in ((State, []), (state(1), []), (state(1), [state(1)])):
            with self.subTest(initial=initial, duplicate=bool(pending)):
                self.robot.disconnect()
                FakeRPC.initial = initial
                self.clock.pending = pending
                with self.assertRaisesRegex(TimeoutError, 'feedback did not advance'):
                    self.robot.get_calibration_sample()

    def test_first_real_packet_is_accepted_after_sdk_placeholder(self):
        FakeRPC.initial = State
        np.testing.assert_array_equal(self.robot.get_tcp_offset(), np.zeros(6))

    def test_frame_counter_wrap_and_full_wrap_with_new_clock_are_accepted(self):
        for before, after in ((255, 0), (1, 1)):
            with self.subTest(before=before, after=after):
                self.robot.disconnect()
                FakeRPC.initial = state(before, millisecond=100)
                self.clock.pending = [state(after, millisecond=101)]
                sample = self.robot.get_calibration_sample()
                self.assertIn(f'frame {after}', sample['source'])

    def test_each_sample_requires_feedback_newer_than_the_call(self):
        self.robot.get_calibration_sample()
        # Reusing a previously accepted cached state must not pass capture again.
        with self.assertRaises(TimeoutError):
            self.robot.get_calibration_sample()

    def test_disconnect_flags_are_rejected_even_if_a_packet_is_cached(self):
        for name, value in (('is_conect', False), ('SDK_state', False),
                            ('sock_cli_state_state', False), ('reconnect_flag', True)):
            with self.subTest(flag=name):
                self.robot.disconnect()
                self.robot._connect()
                setattr(FakeRPC.instances[-1], name, value)
                with self.assertRaises(ConnectionError):
                    self.robot.get_calibration_sample()

    def test_feedback_loss_disables_vendor_retry_and_disconnect_is_idempotent(self):
        self.robot.get_calibration_sample()
        client = FakeRPC.instances[-1]
        self.assertFalse(client.reconnect())
        with self.assertRaises(ConnectionError):
            self.robot.get_calibration_sample()
        self.robot.disconnect()
        self.robot.disconnect()
        self.assertTrue(client.stop_event.is_set())
        self.assertTrue(client.closeRPC_state)
        self.assertTrue(client.socket_closed)
        self.assertIsNone(client.robot_state_pkg)
        self.assertEqual(client.close_calls, 1)

    def test_converted_fr3_frames_use_existing_offline_solver_in_both_modes(self):
        from handeye.pipeline import calibrate_session
        from tests.fixtures import write_session
        for mode in ('eye_to_hand', 'eye_on_hand'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'data'
                expected = write_session(source, mode=mode, schema=2)
                manifest = json.loads((source / 'session.json').read_text(encoding='utf-8'))
                manifest['tcp_definition'] = {'verified': True, 'frame': 'flange_zero_offset',
                                              'offset_m_rotvec_rad': [0.] * 6}
                manifest['capture_config'] = {'robot': load_config('configs/robots/fr3_lab.json')}
                write_json(source / 'session.json', manifest)
                for i, path in enumerate(sorted((source / 'frames').glob('*/robot.json'))):
                    recorded = json.loads(path.read_text(encoding='utf-8'))
                    transform = np.asarray(recorded['transform'])
                    pose = np.r_[transform[:3, 3] * 1000,
                                 Rotation.from_matrix(transform[:3, :3]).as_euler('xyz', degrees=True)]
                    self.clock.pending = [state(i + 20, pose=pose)]
                    sample = self.robot.get_calibration_sample()
                    recorded.update(transform=sample['transform'], source=sample['source'])
                    write_json(path, recorded)
                with patch('handeye.robot.fairino_robot._load_sdk',
                           side_effect=AssertionError('Offline solve must not load the SDK')), \
                        contextlib.redirect_stdout(io.StringIO()):
                    result = calibrate_session(source, output_root=root / 'results')
                np.testing.assert_allclose(result['X'], expected, atol=1e-8)


class FairinoSetupTests(unittest.TestCase):
    def test_factory_and_shared_capture_configuration_select_fr3_without_loading_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = load_config('configs/capture.json')
            capture['robot_config'] = 'configs/robots/fr3_lab.json'
            path = Path(directory) / 'capture.json'
            write_json(path, capture)
            settings = load_workflow(path, robot=True)
            with patch('handeye.robot.fairino_robot._load_sdk',
                       side_effect=AssertionError('Factory must not load SDK or connect')):
                robot = create_robot(settings['robot'])
            self.assertIsInstance(robot, FairinoRobot)
            self.assertEqual(robot.sdk_path, resolve_path(settings['robot']['sdk_path']))
            with self.assertRaisesRegex(ValueError, 'UR only'):
                create_motion_interfaces(settings['robot'])

    def test_timeout_must_be_positive_and_finite(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FairinoRobot('192.0.2.1', 'missing-sdk', state_timeout_s=value)

    def test_missing_sdk_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'sdk_path must contain'):
                _load_sdk(directory)

    def test_sdk_without_tool_offset_feedback_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'fairino'
            package.mkdir()
            (package / 'Robot.py').write_text(
                'class RPC: pass\nclass RobotStatePkg: pass\n', encoding='utf-8')
            with self.assertRaisesRegex(ImportError, 'toolCoord'):
                _load_sdk(directory)


if __name__ == '__main__':
    unittest.main()
