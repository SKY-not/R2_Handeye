#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UR3 机械臂通信接口
支持获取末端位姿、关节角度，以及关节空间/任务空间运动控制
"""

import socket
import struct
import time
import numpy as np
from typing import Any, Optional, cast
from calibration.transforms import matrix_to_rotvec, rotvec_to_matrix


class URRobot:
    """UR3 机械臂控制类"""

    def __init__(
        self,
        tcp_host_ip: str = "192.168.50.100",
        tcp_port: int = 30003,
        workspace_limits: Optional[np.ndarray] = None
    ) -> None:
        """
        初始化UR3机械臂连接

        Args:
            tcp_host_ip: UR3 IP地址
            tcp_port: TCP端口号 (默认30003)
            workspace_limits: 工作空间限制 [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        """
        self.tcp_host_ip = tcp_host_ip
        self.tcp_port = tcp_port
        self.tcp_socket: Optional[socket.socket] = None

        # 工作空间限制
        if workspace_limits is None:
            workspace_limits_arr = np.array([[-0.7, 0.7], [-0.7, 0.7], [0.00, 0.6]], dtype=np.float64)
        else:
            workspace_limits_arr = np.asarray(workspace_limits, dtype=np.float64)
        self.workspace_limits: np.ndarray = workspace_limits_arr

        # 速度与加速度参数
        self.joint_acc = 1.4  # 关节加速度
        self.joint_vel = 1.05  # 关节速度
        self.tool_acc = 0.5   # 工具加速度
        self.tool_vel = 0.2   # 工具速度

        # 容差设置
        self.joint_tolerance = 0.01
        self.tool_pose_tolerance: list[float] = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]

    def connect(self) -> None:
        """建立TCP连接"""
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        assert self.tcp_socket is not None
        self.tcp_socket.connect((self.tcp_host_ip, self.tcp_port))
        print(f"已连接到UR3: {self.tcp_host_ip}:{self.tcp_port}")

    def disconnect(self) -> None:
        """断开TCP连接"""
        if self.tcp_socket:
            self.tcp_socket.close()
            self.tcp_socket = None

    def is_connected(self) -> bool:
        """检查连接状态"""
        return self.tcp_socket is not None

    # ==================== 位姿获取 ====================

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = sock.recv(size - len(chunks))
            if not chunk:
                raise ConnectionError('UR state stream closed before the frame was complete')
            chunks.extend(chunk)
        return bytes(chunks)

    @classmethod
    def _read_packet(cls, sock: socket.socket) -> bytes:
        header = cls._recv_exact(sock, 4)
        size = struct.unpack('!i', header)[0]
        if not 492 <= size <= 4096:
            raise ValueError(f'Invalid UR realtime message size: {size}')
        return header + cls._recv_exact(sock, size - 4)

    def get_state(self) -> bytes:
        """Read complete length-prefixed messages from the UR realtime stream."""
        with socket.create_connection((self.tcp_host_ip, self.tcp_port), timeout=5.0) as sock:
            first = self._read_packet(sock)
            current = self._read_packet(sock)
        before = struct.unpack_from('!d', first, 4)[0]
        after = struct.unpack_from('!d', current, 4)[0]
        if not np.isfinite([before, after]).all() or after <= before:
            raise ValueError('UR state timestamps did not advance')
        return current

    def get_joint_positions(self) -> np.ndarray:
        return self._parse_state_data(self.get_state(), 'joint_data')

    def get_tool_pose(self) -> np.ndarray:
        """Return actual TCP [m, m, m, rad, rad, rad]."""
        return self._parse_state_data(self.get_state(), 'cartesian_info')

    def get_transform_matrix(self) -> np.ndarray:
        """
        获取末端齐次变换矩阵 (4x4)

        Returns:
            np.array: 4x4 齐次变换矩阵
        """
        tool_pose = self.get_tool_pose()
        return self.pose_to_transform(tool_pose)

    # ==================== 运动控制 ====================

    def move_j(self, joint_config: np.ndarray, k_acc: float = 1.0, k_vel: float = 1.0, wait: bool = True) -> None:
        """
        关节空间运动 (movej)

        Args:
            joint_config: 6个关节角度 (弧度)
            k_acc: 加速度缩放系数
            k_vel: 速度缩放系数
            wait: 是否阻塞等待运动完成
        """
        if not self.is_connected():
            self.connect()

        tcp_command = "movej([%f" % joint_config[0]
        for i in range(1, 6):
            tcp_command += ",%f" % joint_config[i]
        tcp_command += "],a=%f,v=%f)\n" % (k_acc * self.joint_acc, k_vel * self.joint_vel)

        sock = self.tcp_socket
        assert sock is not None
        sock.send(str.encode(tcp_command))

        if wait:
            self._wait_for_joint(joint_config)
        else:
            time.sleep(0.5)  # 等待命令发送完成
            if self.tcp_socket is not None:
                self.tcp_socket.close()
                self.tcp_socket = None

    def move_j_p(self, tool_config: np.ndarray, k_acc: float = 1.0, k_vel: float = 1.0, wait: bool = True) -> None:
        """
        任务空间运动 (movej with pose)

        Args:
            tool_config: [x, y, z, rx, ry, rz] (位置: 米, 旋转向量: 弧度)
            k_acc: 加速度缩放系数
            k_vel: 速度缩放系数
            wait: 是否阻塞等待运动完成
        """
        if not self.is_connected():
            self.connect()

        # RPY转旋转向量
        rpy = tool_config[3:6]
        rot_vec = self.rpy_to_rotvec(rpy)

        tcp_command = "def process():\n"
        tcp_command += " movej(get_inverse_kin(p[%f,%f,%f,%f,%f,%f]),a=%f,v=%f)\n" % (
            tool_config[0], tool_config[1], tool_config[2],
            rot_vec[0], rot_vec[1], rot_vec[2],
            k_acc * self.joint_acc, k_vel * self.joint_vel)
        tcp_command += "end\n"

        sock = self.tcp_socket
        assert sock is not None
        sock.send(str.encode(tcp_command))

        if wait:
            self._wait_for_pose(tool_config)
        else:
            time.sleep(0.5)
            if self.tcp_socket is not None:
                self.tcp_socket.close()
                self.tcp_socket = None

    def move_l(self, tool_config: np.ndarray, k_acc: float = 1.0, k_vel: float = 1.0, wait: bool = True) -> None:
        """
        直线运动 (movel)

        Args:
            tool_config: [x, y, z, rx, ry, rz]
            k_acc: 加速度缩放系数
            k_vel: 速度缩放系数
            wait: 是否阻塞等待运动完成
        """
        if not self.is_connected():
            self.connect()

        rpy = tool_config[3:6]
        rot_vec = self.rpy_to_rotvec(rpy)

        tcp_command = "def process():\n"
        tcp_command += " movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f)\n" % (
            tool_config[0], tool_config[1], tool_config[2],
            rot_vec[0], rot_vec[1], rot_vec[2],
            k_acc * self.tool_acc, k_vel * self.tool_vel)
        tcp_command += "end\n"

        sock = self.tcp_socket
        assert sock is not None
        sock.send(str.encode(tcp_command))

        if wait:
            self._wait_for_pose(tool_config)
        else:
            time.sleep(0.5)
            if self.tcp_socket is not None:
                self.tcp_socket.close()
                self.tcp_socket = None

    def go_home(self, home_config: Optional[np.ndarray] = None) -> None:
        """
        机械臂归位

        Args:
            home_config: 归位关节角度，默认使用标准归位姿态
        """
        if home_config is None:
            home_config = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, 0, 0], dtype=np.float64)
        self.move_j(home_config)

    # ==================== 辅助函数 ====================

    def _parse_state_data(self, data: bytes, subpackage: str) -> np.ndarray:
        """
        解析TCP状态数据

        Args:
            data: 原始状态数据
            subpackage: 'joint_data' 或 'cartesian_info'

        Returns:
            np.array: 解析后的数据
        """
        offsets = {'joint_data': 252, 'cartesian_info': 444}
        if subpackage not in offsets:
            raise ValueError(f'Unknown subpackage: {subpackage}')
        if len(data) < 492 or struct.unpack_from('!i', data)[0] != len(data):
            raise ValueError('Incomplete or misaligned UR realtime packet')
        values = np.asarray(struct.unpack_from('!6d', data, offsets[subpackage]))
        if not np.isfinite(values).all():
            raise ValueError(f'Non-finite UR state: {subpackage}')
        return values

    def _wait_for_joint(self, target_joints: np.ndarray, timeout: float = 30) -> None:
        """等待关节运动到目标位置"""
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                print("警告: 等待关节运动超时")
                break
            current_joints = self.get_joint_positions()
            if all(np.abs(current_joints[i] - target_joints[i]) < self.joint_tolerance for i in range(6)):
                break
            time.sleep(0.01)
        if self.tcp_socket is not None:
            self.tcp_socket.close()
            self.tcp_socket = None
        time.sleep(0.5)  # 等待机械臂稳定

    def _wait_for_pose(self, target_pose: np.ndarray, timeout: float = 30) -> None:
        """等待末端运动到目标位姿"""
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                print("警告: 等待末端运动超时")
                break
            current_pose = self.get_tool_pose()
            if all(np.abs(current_pose[i] - target_pose[i]) < self.tool_pose_tolerance[i] for i in range(3)):
                break
            time.sleep(0.01)
        if self.tcp_socket is not None:
            self.tcp_socket.close()
            self.tcp_socket = None
        time.sleep(0.5)  # 等待机械臂稳定

    # ==================== 坐标变换 ====================

    @staticmethod
    def rpy_to_rotvec(rpy: np.ndarray) -> np.ndarray:
        """RPY转旋转向量"""
        R = URRobot.rpy_to_R(rpy)
        return URRobot.R_to_rotvec(R)

    @staticmethod
    def rotvec_to_rpy(rotvec: np.ndarray) -> np.ndarray:
        """旋转向量转RPY"""
        R = URRobot.rotvec_to_R(rotvec)
        return URRobot.R_to_rpy(R)

    @staticmethod
    def rpy_to_R(rpy: np.ndarray) -> np.ndarray:
        """RPY转旋转矩阵"""
        r, p, y = rpy
        rot_x = np.array([[1, 0, 0],
                          [0, np.cos(r), -np.sin(r)],
                          [0, np.sin(r), np.cos(r)]])
        rot_y = np.array([[np.cos(p), 0, np.sin(p)],
                          [0, 1, 0],
                          [-np.sin(p), 0, np.cos(p)]])
        rot_z = np.array([[np.cos(y), -np.sin(y), 0],
                          [np.sin(y), np.cos(y), 0],
                          [0, 0, 1]])
        return np.asarray(rot_z @ rot_y @ rot_x, dtype=np.float64)

    @staticmethod
    def R_to_rpy(R: np.ndarray) -> np.ndarray:
        """旋转矩阵转RPY"""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            r = np.arctan2(R[2, 1], R[2, 2])
            p = np.arctan2(-R[2, 0], sy)
            y = np.arctan2(R[1, 0], R[0, 0])
        else:
            r = np.arctan2(-R[1, 2], R[1, 1])
            p = np.arctan2(-R[2, 0], sy)
            y = 0
        return np.array([r, p, y], dtype=np.float64)

    @staticmethod
    def R_to_rotvec(R: np.ndarray) -> np.ndarray:
        """旋转矩阵转旋转向量"""
        return matrix_to_rotvec(R)

    @staticmethod
    def rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
        """旋转向量转旋转矩阵 (Rodrigues formula)"""
        return rotvec_to_matrix(rotvec)

    @staticmethod
    def pose_to_transform(pose: np.ndarray) -> np.ndarray:
        """
        位姿数组转齐次变换矩阵

        Args:
            pose: [x, y, z, rx, ry, rz]

        Returns:
            np.array: 4x4 齐次变换矩阵
        """
        x, y, z, rx, ry, rz = pose
        R = URRobot.rotvec_to_R(np.array([rx, ry, rz]))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def transform_to_pose(T: np.ndarray) -> np.ndarray:
        """
        齐次变换矩阵转位姿数组

        Args:
            T: 4x4 齐次变换矩阵

        Returns:
            np.array: [x, y, z, rx, ry, rz]
        """
        pos = T[:3, 3]
        rotvec = URRobot.R_to_rotvec(T[:3, :3])
        return np.concatenate([pos, rotvec])

    # ==================== 工作空间检查 ====================

    def is_in_workspace(self, pose: np.ndarray) -> bool:
        """检查位姿是否在工作空间内"""
        pos = pose[:3]
        return bool(
            self.workspace_limits[0, 0] <= pos[0] <= self.workspace_limits[0, 1] and
            self.workspace_limits[1, 0] <= pos[1] <= self.workspace_limits[1, 1] and
            self.workspace_limits[2, 0] <= pos[2] <= self.workspace_limits[2, 1]
        )


if __name__ == "__main__":
    # 测试代码
    robot = URRobot(tcp_host_ip="192.168.50.100")

    # 获取当前位姿
    print("获取当前位姿...")
    pose = robot.get_tool_pose()
    print(f"当前末端位姿: {pose}")

    # 获取关节角度
    joints = robot.get_joint_positions()
    print(f"当前关节角度: {joints}")

    # 获取变换矩阵
    T = robot.get_transform_matrix()
    print("末端变换矩阵:")
    print(T)
