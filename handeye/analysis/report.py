"""Shared report serialization and figures for both analysis workflows."""
import csv
from pathlib import Path
import numpy as np
from handeye.session import write_json
from handeye.config import load_config


def statistics(values):
    array = np.asarray(values, dtype=float)
    valid = array[np.isfinite(array)]
    return {'count': int(len(array)), 'invalid_count':int(len(array)-len(valid)),
            **{key: None if not len(valid) else float(function(valid)) for key, function in
               [('mean',np.mean),('max',np.max),('min',np.min),('std',np.std),('median',np.median)]}}


def save_report(root, report, rows, *, visualization_config='configs/visualization.json', show=False):
    root = Path(root)
    settings = load_config(visualization_config)
    write_json(root / 'evaluation.json', report)
    with (root / 'residuals.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib.pyplot as plt
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1,2,figsize=settings['error_figsize'])
    for axis, key, label in zip(axes, ('translation_mm','rotation_deg'), ('Position residual (mm)','Rotation residual (deg)')):
        values = [row[key] for row in rows]
        axis.plot(range(1,len(rows)+1), values, 'o-')
        axis.set(xlabel='Sample', ylabel=label)
        axis.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(figures / 'pose_errors.png', dpi=150)
    components, axes = plt.subplots(2,1,figsize=settings['components_figsize'])
    for axis, prefix, unit in zip(axes, ('d','r'), ('mm','deg')):
        for component in ('x','y','z'):
            key = prefix + component + '_' + unit
            axis.plot(range(1,len(rows)+1), [row[key] for row in rows], label=component)
        axis.set(xlabel='Sample', ylabel=unit)
        axis.legend(); axis.grid(alpha=.3)
    components.tight_layout(); components.savefig(figures / 'components.png', dpi=150)
    projection, ax = plt.subplots(figsize=settings['reprojection_figsize'])
    for key in ('observed_reprojection_px','chain_reprojection_px'):
        if key in rows[0] and any(row[key] is not None for row in rows):
            ax.plot(range(1,len(rows)+1), [np.nan if r[key] is None else r[key] for r in rows], 'o-', label=key)
    ax.set(xlabel='Sample', ylabel='Reprojection error (px)')
    if ax.lines:
        ax.legend()
    projection.tight_layout(); projection.savefig(figures / 'reprojection.png', dpi=150)
    for key, value in report.items():
        if isinstance(value, dict) and 'mean' in value:
            print(f'{key}: mean={value["mean"]}, std={value["std"]}, max={value["max"]}')
    if show:
        plt.show()
    for item in (fig,components,projection):
        plt.close(item)
