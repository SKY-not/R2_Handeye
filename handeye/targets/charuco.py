"""ChArUco observations in raw pixels, using the MultiCamCalib board convention."""
import cv2
import numpy as np


def make_board(config):
    if not hasattr(cv2, 'aruco') or not hasattr(cv2.aruco, 'CharucoDetector'):
        raise RuntimeError('ChArUco requires OpenCV with aruco.CharucoDetector support')
    dims = config['squares']
    square, marker = float(config['square_length_m']), float(config['marker_length_m'])
    if len(dims) != 2 or any(type(i) is not int or i < 3 for i in dims):
        raise ValueError('ChArUco requires at least 3 x 3 squares')
    if not np.isfinite([square, marker]).all() or not 0 < marker < square:
        raise ValueError('Require 0 < marker length < square length')
    name = config['dictionary']
    code = getattr(cv2.aruco, name, None) if isinstance(name, str) and name.startswith('DICT_') else None
    if type(code) is not int:
        raise ValueError('Unknown ChArUco dictionary')
    dictionary = cv2.aruco.getPredefinedDictionary(code)
    if dims[0] * dims[1] // 2 > len(dictionary.bytesList):
        raise ValueError('ChArUco dictionary has too few markers')
    board = cv2.aruco.CharucoBoard(tuple(dims), square, marker, dictionary)
    board.setLegacyPattern(False)
    return board


def correspondences(config, ids, pixels):
    ids = np.asarray(ids)
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    points = make_board(config).getChessboardCorners().astype(float)
    minimum = max(6, int(config.get('min_visible_corners', 6)))
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(set(ids.tolist())) != len(ids):
        raise ValueError('Invalid or duplicate ChArUco corner IDs')
    if len(ids) < minimum or np.any(ids < 0) or np.any(ids >= len(points)):
        raise ValueError('Insufficient or out-of-range ChArUco corner IDs')
    if pixels.shape != (len(ids), 2) or not np.isfinite(pixels).all():
        raise ValueError('Invalid ChArUco image points')
    points = points[ids]
    if np.linalg.matrix_rank(points[:, :2] - points[:, :2].mean(axis=0)) < 2:
        raise ValueError('ChArUco corners are collinear')
    return points, pixels


class CharucoTarget:
    def __init__(self, config):
        self.config = config
        self.board = make_board(config)

    def detect(self, image, K, distortion, depth=None):
        from handeye.targets.geometry import reprojection_errors
        from handeye.calibration.transforms import pose_to_mat
        params = cv2.aruco.CharucoParameters()
        params.cameraMatrix = np.asarray(K, dtype=float)
        params.distCoeffs = np.asarray(distortion, dtype=float)
        detector = cv2.aruco.CharucoDetector(self.board, params)
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)
        if ids is None:
            return {'success': False}
        order = np.argsort(ids.ravel())
        ids = ids.ravel()[order]
        corners = corners.reshape(-1, 2)[order]
        try:
            points, corners = correspondences(self.config, ids, corners)
            ok, rvec, tvec = cv2.solvePnP(points, corners, K, distortion)
            if not ok:
                return {'success': False}
            pose = pose_to_mat(np.r_[tvec.ravel(), rvec.ravel()])
            error = float(reprojection_errors(pose, points, corners, K, distortion).mean())
            if error > float(self.config.get('max_reprojection_error_px', 2.)):
                return {'success': False}
        except (ValueError, cv2.error):
            return {'success': False}
        return {'success': True, 'rgb': image, 'depth': depth,
                'corners': corners.reshape(-1, 1, 2).astype(np.float32),
                'corners_refined': corners.reshape(-1, 1, 2).astype(np.float32),
                'charuco_ids': ids.tolist(), 'board_pose': pose,
                'reprojection_error_px': error}
