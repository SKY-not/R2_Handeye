"""Load only selected configurations; relative paths use the project root."""
from copy import deepcopy
import json
import math
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / 'configs'
CALIBRATION_MODES = ('eye_on_hand', 'eye_to_hand')


def resolve_path(path):
    source = Path(path).expanduser()
    return source.resolve() if source.is_absolute() else (PROJECT_ROOT / source).resolve()


def load_config(path):
    value = json.loads(resolve_path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Configuration must be a JSON object: {path}')
    return value


def load_workflow(path, *, robot=False):
    settings = load_config(path)
    for name in (('robot', 'camera', 'target') if robot else ('camera', 'target')):
        settings[name] = load_config(settings[name + '_config'])
    profiles = load_config(settings['detection_config'])
    profile = settings['detection_profile']
    if profile not in profiles:
        raise ValueError(f'Unknown detection profile: {profile}')
    settings['detection'] = deepcopy(profiles[profile])
    target = settings['target']
    if target.get('type') not in ('checkerboard', 'charuco', 'apriltag_board', 'apriltag'):
        raise ValueError(f'Unsupported target type: {target.get("type")}')
    if target['type'] == 'charuco':
        from handeye.targets.charuco import make_board
        board = make_board(target)
        minimum = settings['detection'].get('min_visible_corners', 6)
        limit = settings['detection'].get('max_reprojection_error_px', 2.)
        if type(minimum) is not int or not 6 <= minimum <= len(board.getChessboardCorners()):
            raise ValueError('Invalid ChArUco minimum corner count')
        if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
            raise ValueError('Invalid ChArUco reprojection limit')
    if target['type'] in ('apriltag_board', 'apriltag'):
        if not 1 <= int(settings['detection']['min_visible_tags']) <= len(target['tag_ids']):
            raise ValueError('Detection min_visible_tags is incompatible with target geometry')
    if target['type'] == 'apriltag':
        if len(target['tag_ids']) != 1:
            raise ValueError('A single-tag target must declare exactly one ID')
        tag = str(target['tag_ids'][0])
        if any(target['tag_centers'][tag]) or any(target.get('tag_rotvecs', {}).get(tag,[0,0,0])):
            raise ValueError('Single-tag coordinates must use the tag center and native axes')
    for key in ('capture_max_translation_m','capture_max_rotation_rad'):
        if key in settings and (not math.isfinite(settings[key]) or settings[key] < 0):
            raise ValueError(f'{key} must be finite and nonnegative')
    return settings


def target_definition(backend, solver_config=None):
    """Defaults for direct library callers; saved sessions supply their own geometry."""
    if backend not in ('checkerboard', 'charuco', 'apriltag_board'):
        raise ValueError(f'Unknown target backend: {backend}')
    filename = 'four_apriltag' if backend == 'apriltag_board' else backend
    value = load_config(f'configs/targets/{filename}.json')
    value.update(load_config('configs/detection.json')[backend])
    return value


def effective_target(settings):
    target = deepcopy(settings['target'])
    target.update(settings['detection'])
    return target


def get_data_path(mode, root=None):
    base = (resolve_path(root) if root is not None else PROJECT_ROOT / 'data') / mode
    return {key: str(base / folder) for key, folder in
            [('root', ''), ('frames', 'frames'), ('poses', 'poses'), ('images', 'images')]}


def get_results_path(mode):
    return str(PROJECT_ROOT / 'outputs' / mode)
