#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robotiq 夹爪控制接口
"""

import socket
from typing import Optional


class RobotiqGripper:
    """Robotiq 85 夹爪控制类"""

    def __init__(self) -> None:
        self.socket = None
        self.host = None
        self.port = None

    def connect(self, host: str, port: int = 63352) -> None:
        """连接到夹爪"""
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        print(f"已连接到Robotiq夹爪: {host}:{port}")

    def disconnect(self) -> None:
        """断开连接"""
        if self.socket:
            self.socket.close()
            self.socket = None

    def _send_command(self, command: str) -> bytes:
        """发送命令"""
        if not self.socket:
            raise RuntimeError("夹爪未连接")
        self.socket.send(command.encode())
        return self.socket.recv(1024)

    def _reset(self) -> None:
        """重置夹爪"""
        command = "SET RG 0 0\r\n"
        self._send_command(command)

    def activate(self) -> None:
        """激活夹爪"""
        command = "SET RG 1 1\r\n"
        self._send_command(command)

    def get_current_position(self) -> int:
        """获取当前夹爪位置 (0-255)"""
        command = "GET RG\r\n"
        response = self._send_command(command)
        # 解析响应获取位置
        try:
            parts = response.decode().split()
            if len(parts) >= 3:
                return int(parts[2])
        except:
            pass
        return 0

    def move_and_wait_for_pos(self, position: int, speed: int = 255, force: int = 255) -> None:
        """
        移动到指定位置并等待

        Args:
            position: 目标位置 (0-255, 0=打开, 255=关闭)
            speed: 速度 (0-255)
            force: 力 (0-255)
        """
        command = f"SET RG {position} {speed} {force}\r\n"
        self._send_command(command)
        # 等待到达目标位置
        current_pos = self.get_current_position()
        while abs(current_pos - position) > 5:
            current_pos = self.get_current_position()

    def open(self, speed: int = 255, force: int = 255) -> None:
        """打开夹爪"""
        self.move_and_wait_for_pos(0, speed, force)

    def close(self, speed: int = 255, force: int = 255) -> None:
        """关闭夹爪"""
        self.move_and_wait_for_pos(255, speed, force)
