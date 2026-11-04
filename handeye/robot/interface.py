"""Robot state contract: T_base_tcp in metres, rotation vectors in radians."""
from typing import Protocol
import numpy as np


def validate_zero_tcp_offset(offset):
    value = np.asarray(offset, dtype=float)
    if value.shape != (6,) or not np.isfinite(value).all() or np.any(value != 0):
        raise ValueError('Calibration requires TCP offset [0, 0, 0, 0, 0, 0]')
    return value.copy()


class RobotState(Protocol):
    def get_tcp_offset(self) -> np.ndarray: ...
    def get_calibration_sample(self) -> dict: ...
    def disconnect(self) -> None: ...
