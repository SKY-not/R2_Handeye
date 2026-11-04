"""Zero-offset TCP contract and read-only UR state parsing; no robot connections."""
import struct
import unittest
from unittest.mock import patch
import numpy as np
from handeye.robot.ur_robot import URRobot
from tests.test_calibration_core import FragmentedSocket


def state(timestamp=1, offset=None, pose=None, include_offset=True):
    pose = [.1, .2, .3, .01, .02, .03] if pose is None else pose
    offset = [0.] * 6 if offset is None else offset
    values = list(pose) + (list(offset) if include_offset else [])
    cartesian = struct.pack('!IB', 5 + len(values) * 8, 4) + struct.pack('!' + str(len(values)) + 'd', *values)
    mode = struct.pack('!IBQ', 13, 0, timestamp)
    payload = mode + cartesian
    return struct.pack('!IB', 5 + len(payload), 16) + payload


class ZeroTCPTests(unittest.TestCase):
    def test_fragmented_state_pairs_pose_with_zero_offset(self):
        for fragment in (1, 7, 1500):
            stream = FragmentedSocket(state(1) + state(2), fragment)
            with patch('handeye.robot.ur_robot.socket.create_connection', return_value=stream) as connect:
                sample = URRobot().get_calibration_sample()
            np.testing.assert_allclose(sample['transform'][:3, 3], [.1, .2, .3])
            np.testing.assert_array_equal(sample['tcp_offset'], np.zeros(6))
            self.assertEqual(sample['robot_timestamp_us'], 2)
            self.assertEqual(connect.call_args.args[0][1], 30012)

    def test_translation_rotation_and_small_nonzero_offsets_are_rejected(self):
        for offset in ([.001, 0, 0, 0, 0, 0], [0, 0, 0, 0, .01, 0], [1e-15, 0, 0, 0, 0, 0]):
            stream = FragmentedSocket(state(1) + state(2, offset))
            with patch('handeye.robot.ur_robot.socket.create_connection', return_value=stream), \
                    self.assertRaisesRegex(ValueError, 'Calibration requires TCP offset'):
                URRobot().get_calibration_sample()

    def test_missing_nonfinite_or_truncated_offset_is_rejected(self):
        with self.assertRaises(ValueError):
            URRobot._read_cartesian_state(FragmentedSocket(state(include_offset=False)))
        with self.assertRaises(ValueError):
            URRobot._read_cartesian_state(FragmentedSocket(state(offset=[float('nan')] * 6)))
        with self.assertRaises(ConnectionError):
            URRobot._read_cartesian_state(FragmentedSocket(state()[:-3]))
        for offset in (None, [], [0.] * 5):
            with self.assertRaises(ValueError):
                URRobot.validate_zero_tcp_offset(offset)

    def test_stale_secondary_states_are_rejected(self):
        stream = FragmentedSocket(state(5) + state(5))
        with patch('handeye.robot.ur_robot.socket.create_connection', return_value=stream), \
                self.assertRaisesRegex(ValueError, 'timestamps'):
            URRobot().get_calibration_sample()


if __name__ == '__main__':
    unittest.main()
