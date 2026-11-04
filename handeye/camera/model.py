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
