"""Session manifests and atomic JSON writes, independent of hardware and UI."""
from datetime import datetime
import json
from pathlib import Path

import numpy as np


def _json_text(value, level=0):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        if not value:
            return '{}'
        items = [json.dumps(str(k), ensure_ascii=False) + ': ' + _json_text(v, level + 1)
                 for k, v in value.items()]
        return '{\n' + ',\n'.join('  ' * (level + 1) + x for x in items) + '\n' + '  ' * level + '}'
    if isinstance(value, (list, tuple)):
        if not any(isinstance(x, (list, tuple, dict, np.ndarray)) for x in value):
            return json.dumps(value, ensure_ascii=False, allow_nan=False)
        return '[\n' + ',\n'.join('  ' * (level + 1) + _json_text(x, level + 1) for x in value) + '\n' + '  ' * level + ']'
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(_json_text(value) + '\n', encoding='utf-8')
    temporary.replace(path)


def new_run_directory(parent):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime('%Y%m%d_%H%M%S')
    for sequence in range(100000):
        root = parent / (name if sequence == 0 else f'{name}_{sequence:03d}')
        try:
            root.mkdir()
            return root
        except FileExistsError:
            continue
    raise RuntimeError('Cannot allocate a new data/result directory')


def data_paths(root: Path) -> dict[str, str]:
    return {key: str(root / folder) for key, folder in
            [('root', ''), ('frames', 'frames'), ('poses', 'poses'), ('images', 'images'), ('teach_poses', 'teach_poses')]}


def read_manifest(root: Path) -> dict:
    metadata = json.loads((root / 'session.json').read_text(encoding='utf-8'))
    if metadata.get('schema_version') not in (1, 2):
        raise ValueError('Unsupported calibration session schema')
    if metadata.get('corner_space') not in ('raw', 'rectified_alpha0', 'rectified_alpha1'):
        raise ValueError('Unknown corner coordinate system')
    return metadata

def load_handeye(source, mode=None):
    """Explicit result directory or matrix file; no fallback to historical paths."""
    from handeye.config import resolve_path
    from handeye.calibration.transforms import validate_transform
    path = resolve_path(source)
    if path.is_dir():
        path = path / 'handeye_transform.txt'
    if not path.is_file():
        raise FileNotFoundError(f'Hand-eye matrix not found: {path}; supply --result')
    metadata_path = path.parent / 'result.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else None
    if metadata is not None and mode is not None and metadata['mode'] != mode:
        raise ValueError(f'Result mode {metadata["mode"]} does not match requested {mode}')
    matrix = validate_transform(np.loadtxt(path), 'selected hand-eye matrix')
    if metadata is not None and not np.allclose(matrix, metadata['transform'], rtol=0, atol=1e-10):
        raise ValueError('Matrix file differs from its result.json')
    return matrix, {'matrix_file': str(path), 'mode': None if metadata is None else metadata['mode'],
                    'source_session': None if metadata is None else metadata.get('source_session')}



def begin_run(kind, *, sources, configuration, output_root=None):
    from handeye.config import PROJECT_ROOT, resolve_path
    root = new_run_directory(PROJECT_ROOT / 'outputs' / kind
                             if output_root is None else resolve_path(output_root))
    write_json(root / 'run.json', {'kind': kind, 'status': 'running', 'sources': sources,
                                  'configuration': configuration})
    return root


def finish_run(root, status='complete', **fields):
    root = Path(root)
    metadata = json.loads((root / 'run.json').read_text(encoding='utf-8'))
    metadata.update(status=status, **fields)
    metadata['artifacts'] = [p.name for p in sorted(root.iterdir()) if p.is_file()]
    write_json(root / 'run.json', metadata)


def save_frame_data(root, sample, target, K, distortion, corner_space):
    import cv2
    from handeye.targets.geometry import checkerboard_points, tag_correspondences, reprojection_errors
    from handeye.camera.model import Rectifier
    from handeye.calibration.transforms import pose_to_mat, validate_transform
    root = Path(root)
    image = sample['rgb']
    if target['type'] == 'charuco':
        from handeye.targets.charuco import correspondences
        points, corners = correspondences(target, sample['charuco_ids'], sample['corners_refined'])
        pose = validate_transform(sample['board_pose'])
        projection_K, projection_dist = K, distortion
        ids = []
    elif target['type'] == 'checkerboard':
        points = checkerboard_points(target)
        corners = np.asarray(sample['corners_refined']).reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(points, corners, K, distortion)
        if not ok:
            raise ValueError('Checkerboard PnP failed')
        pose = pose_to_mat(np.r_[tvec.ravel(), rvec.ravel()])
        projection_K, projection_dist = K, distortion
        ids = []
    else:
        ids = sample['tag_ids']
        points, corners = tag_correspondences(target, ids, sample['tag_corners'])
        pose = validate_transform(sample['tag_pose'])
        projection_K = sample.get('projection_matrix')
        if projection_K is None:
            projection_K = Rectifier(K, distortion).prepare((image.shape[1], image.shape[0]))
        projection_dist = np.zeros(5)
    error = float(reprojection_errors(pose, points, corners, projection_K, projection_dist).mean())
    root.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(root / 'image.png'), image):
        raise OSError(f'Failed to save image: {root}')
    depth = sample.get('depth')
    if depth is not None:
        np.save(root / 'depth.npy', depth)
    write_json(root / 'observation.json', {
        'transform':pose, 'image_points':corners, 'object_points':points, 'tag_ids':ids,
        'charuco_ids':sample.get('charuco_ids', []),
        'corner_space':corner_space, 'projection_matrix':projection_K,
        'projection_distortion':np.asarray(projection_dist).ravel(),
        'reprojection_error_px':error, 'decision_margin':sample.get('tag_decision_margin')})
    if 'tcp' in sample:
        write_json(root / 'robot.json', {'transform':sample['tcp'],
            'before_transform':sample.get('robot_before', sample['tcp']),
            'after_transform':sample.get('robot_after', sample['tcp']),
            'tcp_offset_m_rotvec_rad':sample['tcp_offset'],
            'source':sample.get('tcp_state_source'), 'robot_timestamp_us':sample.get('robot_timestamp_us')})
    write_json(root / 'frame.json', {'captured_at_ns':sample['captured_at_ns'],
        'capture_duration_ms':sample.get('capture_duration_ms'), 'camera_timing':sample.get('camera_timing', {}),
        'image':'image.png', 'depth':None if depth is None else 'depth.npy'})


def load_frame_data(root, *, require_robot=False):
    import cv2
    from handeye.robot.interface import validate_zero_tcp_offset
    from handeye.camera.model import Rectifier
    from handeye.targets.geometry import checkerboard_points, tag_correspondences
    from handeye.calibration.transforms import validate_transform
    metadata = read_manifest(Path(root))
    result = []
    for marker in sorted((Path(root) / 'frames').glob('*/frame.json')):
        folder = marker.parent
        observation = json.loads((folder / 'observation.json').read_text(encoding='utf-8'))
        image = cv2.imread(str(folder / 'image.png'), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'Missing image for committed frame {folder.name}')
        if [image.shape[1],image.shape[0]] != metadata['image_size']:
            raise ValueError('Saved image resolution differs from session')
        if observation['corner_space'] != metadata['corner_space']:
            raise ValueError('Saved corner coordinate system differs from session')
        K = np.asarray(metadata['camera_matrix'])
        distortion = np.asarray(metadata['dist_coeffs'])
        space = metadata['corner_space']
        if space == 'rectified_alpha0':
            K = Rectifier(K,distortion).prepare(tuple(metadata['image_size']))
        elif space == 'rectified_alpha1':
            size = tuple(metadata['image_size'])
            K, _ = cv2.getOptimalNewCameraMatrix(K,distortion,size,1.,size)
        projection_dist = distortion.ravel() if space == 'raw' else np.zeros(5)
        if (not np.allclose(observation['projection_matrix'],K,rtol=0,atol=1e-8)
                or not np.allclose(observation['projection_distortion'],projection_dist,rtol=0,atol=1e-10)):
            raise ValueError('Saved projection model differs from session')
        target = metadata['target']
        if target.get('type') == 'charuco':
            from handeye.targets.charuco import correspondences
            expected_points, _ = correspondences(target, observation['charuco_ids'], observation['image_points'])
        elif target.get('type',metadata.get('backend')) == 'checkerboard':
            expected_points = checkerboard_points(target)
        else:
            expected_points, _ = tag_correspondences(target,observation['tag_ids'],observation['image_points'])
        if not np.allclose(observation['object_points'],expected_points,rtol=0,atol=1e-10):
            raise ValueError('Saved object points differ from target geometry')
        validate_transform(observation['transform'],'saved target pose')
        robot = None
        if require_robot:
            robot = json.loads((folder / 'robot.json').read_text(encoding='utf-8'))
            validate_zero_tcp_offset(robot['tcp_offset_m_rotvec_rad'])
        ids = np.asarray(observation['tag_ids'], dtype=int)
        pixels = np.asarray(observation['image_points'], dtype=float)
        result.append({'tcp': None if robot is None else np.asarray(robot['transform']),
            'corners': pixels.reshape(-1,1,2) if not len(ids) else None,
            'tag_corners':pixels if len(ids) else None, 'tag_ids':ids,
            'tag_pose':np.asarray(observation['transform']), 'rgb':image, 'depth':None,
            'charuco_ids':observation.get('charuco_ids', []),
            'index':folder.name, 'observation':observation,
            'frame':json.loads(marker.read_text(encoding='utf-8'))})
    return result
