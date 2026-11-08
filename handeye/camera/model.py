"""OpenCV camera-model validation and cached rectification."""
import cv2
import numpy as np


def validate_camera_matrix(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError('Invalid camera matrix')
    return K


def opencv_distortion(model: str, coefficients: np.ndarray) -> np.ndarray:
    """Never interpret an inverse/modified model as standard OpenCV distortion."""
    model = model.lower().split('.')[-1]
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if not np.isfinite(coefficients).all():
        raise ValueError('Non-finite camera distortion')
    if model == 'none' or (coefficients.size == 5 and np.all(coefficients == 0)):
        return np.zeros(5)
    if model not in ('brown_conrady', 'opencv') or coefficients.size not in (4, 5, 8, 12, 14):
        raise ValueError(f'Unsupported distortion model: {model}; provide calibrated OpenCV intrinsics')
    return coefficients


class Rectifier:
    """Build maps once per image size; keep pixels and camera matrix together."""
    def __init__(self, K: np.ndarray, distortion: np.ndarray) -> None:
        self.K = validate_camera_matrix(K).copy()
        self.distortion = opencv_distortion('opencv', distortion)
        self.size = None
        self.new_K = None
        self.maps = None

    def prepare(self, size: tuple[int, int]) -> np.ndarray:
        if size != self.size:
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.distortion, size, 0, size)
            self.maps = cv2.initUndistortRectifyMap(self.K, self.distortion, None,
                                                 self.new_K, size, cv2.CV_16SC2)
            self.size = size
        return self.new_K

    def apply(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K = self.prepare((image.shape[1], image.shape[0]))
        return cv2.remap(image, *self.maps, interpolation=cv2.INTER_LINEAR), K


class InverseBrownRectifier:
    """Rectify RealSense inverse Brown images onto the same pinhole K.

    The map follows librealsense 2.56.5's rs2_project_point_to_pixel,
    including the radially scaled tangential terms:
    https://github.com/realsenseai/librealsense/blob/v2.56.5/src/rs.cpp
    """
    def __init__(self, K: np.ndarray, coefficients: np.ndarray) -> None:
        self.K = validate_camera_matrix(K).copy()
        self.coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1).copy()
        if self.coefficients.size != 5 or not np.isfinite(self.coefficients).all():
            raise ValueError('Inverse Brown distortion requires five finite coefficients')
        self.size = None
        self.maps = None

    def prepare(self, size: tuple[int, int]) -> np.ndarray:
        if size != self.size:
            width, height = size
            if width <= 0 or height <= 0:
                raise ValueError('Invalid rectification image size')
            row, col = np.indices((height, width), dtype=np.float64)
            x = (col - self.K[0, 2]) / self.K[0, 0]
            y = (row - self.K[1, 2]) / self.K[1, 1]
            k1, k2, p1, p2, k3 = self.coefficients
            radius2 = x*x + y*y
            scale = 1 + radius2 * (k1 + radius2 * (k2 + radius2 * k3))
            rx, ry = x*scale, y*scale
            source_x = rx + 2*p1*rx*ry + p2*(radius2 + 2*rx*rx)
            source_y = ry + 2*p2*rx*ry + p1*(radius2 + 2*ry*ry)
            self.maps = ((source_x*self.K[0, 0] + self.K[0, 2]).astype(np.float32),
                         (source_y*self.K[1, 1] + self.K[1, 2]).astype(np.float32))
            if not all(np.isfinite(m).all() for m in self.maps):
                raise ValueError('Non-finite inverse Brown rectification map')
            self.size = size
        return self.K

    def apply(self, image: np.ndarray, depth: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray | None]:
        if depth is not None and depth.shape != image.shape[:2]:
            raise ValueError('Depth must be aligned to the color image before rectification')
        self.prepare((image.shape[1], image.shape[0]))
        color = cv2.remap(image, *self.maps, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Nearest sampling preserves metric Z without blending invalid zero depth.
        metric_depth = None if depth is None else cv2.remap(
            depth, *self.maps, interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return color, metric_depth
