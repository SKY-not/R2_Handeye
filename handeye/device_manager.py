"""Small driver factories; hardware libraries load only when needed."""
from handeye.config import load_workflow
import math


def create_camera(settings):
    if settings['driver'] != 'realsense':
        raise ValueError(f'Unsupported camera driver: {settings["driver"]}')
    if settings.get('intrinsics_source', 'device') != 'device':
        raise ValueError('RealSense currently uses device intrinsics')
    from handeye.camera.realsense import RealSenseCamera
    return RealSenseCamera(**{k: settings[k] for k in
        ('width', 'height', 'fps', 'device_id', 'enable_depth') if k in settings})


def create_robot(settings):
    if settings['driver'] == 'fairino':
        from handeye.robot.fairino_robot import FairinoRobot
        return FairinoRobot(settings['tcp_host_ip'], settings['sdk_path'],
                            state_timeout_s=settings.get('state_timeout_s', 2.0))
    if settings['driver'] != 'ur':
        raise ValueError(f'Unsupported robot driver: {settings["driver"]}')
    from handeye.robot.ur_robot import URRobot
    robot = URRobot(settings['tcp_host_ip'], settings['tcp_port'],
                    tcp_state_port=settings['state_port'], workspace_limits=settings.get('workspace_limits'))
    robot.joint_vel = float(settings['default_velocity'])
    robot.joint_acc = float(settings['default_acceleration'])
    if not all(math.isfinite(x) and x > 0 for x in (robot.joint_vel,robot.joint_acc)):
        raise ValueError('Robot velocity and acceleration must be positive')
    return robot


def create_motion_interfaces(settings, *, receive_only=False):
    if settings['driver'] != 'ur':
        raise ValueError('Motion validation currently supports UR only')
    from handeye.robot.motion import connect_rtde
    return connect_rtde(settings['tcp_host_ip'], receive_only=receive_only)


class DeviceManager:
    def __init__(self, capture_config=None):
        self.settings = load_workflow('configs/capture.json', robot=True) if capture_config is None else capture_config
        self.robot = self.camera = None
        self.connected = False

    def connect(self):
        try:
            self.robot = create_robot(self.settings['robot'])
            self.robot.get_calibration_sample()
            self.camera = create_camera(self.settings['camera'])
            self.camera.connect()
            self.connected = True
            return True
        except Exception:
            self.disconnect()
            raise

    def disconnect(self):
        try:
            if self.camera is not None:
                self.camera.disconnect()
        finally:
            if self.robot is not None:
                self.robot.disconnect()
            self.connected = False

    def get_robot(self):
        return self.robot

    def get_camera(self):
        return self.camera
