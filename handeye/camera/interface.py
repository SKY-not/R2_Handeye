"""Camera contract: BGR or gray image, optional aligned metric depth."""
from typing import Protocol
import numpy as np


class Camera(Protocol):
    intrinsics: np.ndarray
    dist_coeffs: np.ndarray
    im_width: int
    im_height: int
    device_id: str | None
    def connect(self) -> None: ...
    def get_data(self) -> tuple[np.ndarray, np.ndarray | None]: ...
    def disconnect(self) -> None: ...
