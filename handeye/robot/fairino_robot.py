"""Read-only FAIRINO state adapter for SDK v2.1.6 / controller v3.8.6."""
import importlib.util
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from handeye.config import resolve_path
from handeye.robot.interface import validate_zero_tcp_offset


def _load_sdk(path):
    # Load the configured Python source explicitly, independently of PYTHONPATH
    # and any installed fairino package or Python-version-specific binary.
    source = resolve_path(path) / 'fairino' / 'Robot.py'
    if not source.is_file():
        raise FileNotFoundError(
            f'FAIRINO SDK source not found: {source}; '
            'sdk_path must contain the fairino/Robot.py source file')
    spec = importlib.util.spec_from_file_location('_handeye_fairino_sdk', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (not hasattr(module, 'RPC') or not hasattr(module, 'RobotStatePkg')
            or not all(hasattr(module.RobotStatePkg, name) for name in
                       ('frame_cnt', 'flange_cur_pos', 'toolCoord'))):
        raise ImportError('FAIRINO SDK must provide v2.1.6 state feedback including toolCoord')
    return module


def _pose_to_transform(pose):
    """FAIRINO fixed-axis XYZ RPY [mm, degrees] -> T_base_flange [metres]."""
    values = np.asarray(pose, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError('FAIRINO pose must contain six finite values')
    transform = np.eye(4)
    transform[:3, 3] = values[:3] / 1000.0
    transform[:3, :3] = Rotation.from_euler('xyz', values[3:], degrees=True).as_matrix()
    return transform


class FairinoRobot:
    """Expose the existing zero-TCP calibration contract without motion commands."""

    def __init__(self, tcp_host_ip, sdk_path, state_timeout_s=2.0):
        self.tcp_host_ip = str(tcp_host_ip).strip()
        self.sdk_path = resolve_path(sdk_path)
        self.state_timeout_s = float(state_timeout_s)
        if not self.tcp_host_ip:
            raise ValueError('FAIRINO controller IP must not be empty')
        if not math.isfinite(self.state_timeout_s) or self.state_timeout_s <= 0:
            raise ValueError('FAIRINO state_timeout_s must be positive and finite')
        self._client = None
        self._state_type = None

    def _connect(self):
        sdk = _load_sdk(self.sdk_path)

        class StateClient(sdk.RPC):
            def reconnect(self):
                # The vendor retry loop ignores CloseRPC's stop event. End this
                # read-only session on loss of feedback instead of leaving that
                # loop running after collection has failed or been closed.
                self.SDK_state = False
                self.sock_cli_state_state = False
                return False

        self._state_type = sdk.RobotStatePkg
        self._client = StateClient(self.tcp_host_ip)

    @staticmethod
    def _state_key(state):
        # frame_cnt wraps at 255; compare for change, not numeric increase.
        # Calendar fields also distinguish a complete counter wrap.
        return tuple(getattr(state, name, 0) for name in
                     ('frame_cnt', 'year', 'mouth', 'day', 'hour', 'minute',
                      'second', 'millisecond'))

    def _fresh_state(self):
        if self._client is None:
            self._connect()
        client = self._client
        initial = client.robot_state_pkg
        initial_key = (self._state_key(initial)
                       if isinstance(initial, self._state_type) else None)
        deadline = time.monotonic() + self.state_timeout_s
        while True:
            if (not client.is_conect or not client.SDK_state
                    or not client.sock_cli_state_state or client.reconnect_flag
                    or client.closeRPC_state):
                raise ConnectionError('FAIRINO state connection lost; reconnect before collecting')
            state = client.robot_state_pkg
            if isinstance(state, self._state_type) and state is not initial:
                # The SDK replaces this ctypes object for each checked packet.
                # Copy one packet so pose and active offset cannot cross frames.
                snapshot = self._state_type.from_buffer_copy(state)
                if self._state_key(snapshot) != initial_key:
                    return snapshot
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('FAIRINO state feedback did not advance before state_timeout_s')
            time.sleep(min(0.005, remaining))

    def get_calibration_sample(self):
        state = self._fresh_state()
        # Verify raw values before conversion: even a tiny nonzero RPY offset
        # must fail the same exact-zero contract as the UR adapter.
        offset = validate_zero_tcp_offset(state.toolCoord)
        return {
            'transform': _pose_to_transform(state.flange_cur_pos),
            'tcp_offset': offset,
            'source': f'FAIRINO SDK flange/zero-TCP state (frame {state.frame_cnt})',
            # This SDK exposes a calendar clock, not UR's controller timestamp.
            'robot_timestamp_us': None,
        }

    def get_tcp_offset(self):
        return self.get_calibration_sample()['tcp_offset']

    def disconnect(self):
        client, self._client = self._client, None
        if client is None:
            return
        # Set these before closing the socket so the receiver does not retry.
        client.closeRPC_state = True
        client.stop_event.set()
        try:
            client.CloseRPC()
        finally:
            # CloseRPC is skipped by the SDK's decorator after an RPC failure.
            sock = client.sock_cli_state
            if sock is not None:
                sock.close()
            client.sock_cli_state = None
            client.robot_state_pkg = None
            client.robot = None
