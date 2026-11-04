"""Synthetic eight-frame session reused by hardware-free regression tests."""
from pathlib import Path
import cv2
import numpy as np
from handeye.calibration.transforms import pose_to_mat, invert_transform, matrix_to_rotvec
from handeye.calibration.targets import target_config, tag_points
from handeye.camera.model import Rectifier
from handeye.session import write_json

K = np.array([[600., 0, 320], [0, 600, 240], [0, 0, 1]])


def write_session(root, mode='eye_to_hand', schema=1):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if schema == 1:
        (root / 'poses').mkdir(); (root / 'images').mkdir()
    else:
        (root / 'frames').mkdir()
    config = target_config('apriltag_board')
    config['tag_rotvecs'][2] = [0, 0, np.pi / 2]
    metadata = {'schema_version': schema, 'kind':'calibration', 'mode': mode, 'backend': 'apriltag_board',
                'corner_space': 'rectified_alpha0', 'camera_matrix': K, 'dist_coeffs': np.zeros(5),
                'camera_id': 'synthetic', 'image_size': [640, 480], 'target': config}
    write_json(root / 'session.json', metadata)
    ids = np.array([1, 2, 3, 4])
    points = np.vstack([tag_points(config, int(i)) for i in ids])
    rectified_K = Rectifier(K, np.zeros(5)).prepare((640, 480))
    X = pose_to_mat([.2, -.1, .3, .2, -.4, .1])
    Y = pose_to_mat([.01, .02, .03, .1, 0, 0])
    rng = np.random.default_rng(7)
    for i in range(8):
        C = pose_to_mat(np.r_[rng.uniform(-.02, .02, 2), .4 + rng.uniform(-.03, .03),
                              rng.uniform(-.4, .4, 3)])
        G = X @ C @ invert_transform(Y) if mode == 'eye_to_hand' else Y @ invert_transform(C) @ invert_transform(X)
        pixels, _ = cv2.projectPoints(points, matrix_to_rotvec(C[:3, :3]), C[:3, 3], rectified_K, np.zeros(5))
        index = f'{i + 1:03d}'
        if schema == 2:
            from handeye.session import save_frame_data
            sample = {'success':True,'rgb':np.zeros((480,640),dtype=np.uint8),'depth':None,
                'tag_pose':C,'tag_ids':ids,'tag_corners':pixels.reshape(-1,2),'tcp':G,
                'tcp_offset':np.zeros(6),'captured_at_ns':i}
            save_frame_data(root / 'frames' / f'{i+1:06d}',sample,config,K,np.zeros(5),'rectified_alpha0')
            continue
        for prefix, data in [('tcp', G), ('tag_pose', C), ('tag_corners', pixels.reshape(-1, 2)), ('tag_ids', ids)]:
            np.savetxt(root / 'poses' / f'{prefix}_{index}.txt', data,
                       fmt='%d' if prefix == 'tag_ids' else '%.18e')
        cv2.imwrite(str(root / 'images' / f'rgb_{index}.png'), np.zeros((480, 640, 3), dtype=np.uint8))
        write_json(root / 'poses' / f'frame_{index}.json', {'captured_at_ns': i})
    return X
