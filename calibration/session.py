"""Session manifests and atomic JSON writes, independent of hardware and UI."""
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np


def write_json(path: Path, value: dict) -> None:
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f'Not JSON serializable: {type(item)}')
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=convert, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def new_run_directory(parent: Path) -> Path:
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid4().hex[:6]
    root = parent / name
    root.mkdir(parents=True, exist_ok=False)
    return root


def data_paths(root: Path) -> dict[str, str]:
    return {key: str(root / folder) for key, folder in
            [('root', ''), ('poses', 'poses'), ('images', 'images'), ('teach_poses', 'teach_poses')]}


def read_manifest(root: Path) -> dict:
    metadata = json.loads((root / 'session.json').read_text(encoding='utf-8'))
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported calibration session schema')
    if metadata.get('corner_space') not in ('raw', 'rectified_alpha0'):
        raise ValueError('Unknown corner coordinate system')
    return metadata
