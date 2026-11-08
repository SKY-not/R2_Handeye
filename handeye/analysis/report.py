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


def save_report(root, report, rows, *, visualization_config='configs/visualization.json', show=False, context=None):
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
    if 'reference' in report:
        fig.suptitle('Consistency residuals relative to mean board pose')
    fig.tight_layout()
    fig.savefig(figures / 'pose_errors.png', dpi=150)
    components, axes = plt.subplots(2,1,figsize=settings['components_figsize'])
    for axis, prefix, unit in zip(axes, ('d','r'), ('mm','deg')):
        for component in ('x','y','z'):
            key = prefix + component + '_' + unit
            axis.plot(range(1,len(rows)+1), [row[key] for row in rows], label=component)
        axis.set(xlabel='Sample', ylabel=unit)
        axis.legend(); axis.grid(alpha=.3)
    if 'reference' in report:
        components.suptitle('Consistency components in mean board frame')
    components.tight_layout(); components.savefig(figures / 'components.png', dpi=150)
    projection, ax = plt.subplots(figsize=settings['reprojection_figsize'])
    for key in ('observed_reprojection_px','chain_reprojection_px'):
        if key in rows[0] and any(row[key] is not None for row in rows):
            ax.plot(range(1,len(rows)+1), [np.nan if r[key] is None else r[key] for r in rows], 'o-', label=key)
    ax.set(xlabel='Sample', ylabel='Reprojection error (px)')
    if ax.lines:
        ax.legend()
    projection.tight_layout(); projection.savefig(figures / 'reprojection.png', dpi=150)
    charts = [fig, components, projection]
    if 'reference' in report:
        reference_fig, axes = plt.subplots(1,2,figsize=settings['error_figsize'])
        reference_fig.suptitle('Deviation from configured reference board pose')
        for axis,key,label in zip(axes, ('reference_position_mm','reference_rotation_deg'),
                                  ('Position deviation (mm)','Rotation deviation (deg)')):
            values = [row.get(key) for row in rows]
            axis.set(xlabel='Sample', ylabel=label)
            axis.grid(alpha=.3)
            if any(value is not None for value in values):
                axis.plot(range(1,len(rows)+1), values, 'o-')
            else:
                axis.text(.5,.5,'Reference pose unavailable',ha='center',va='center',transform=axis.transAxes)
        reference_fig.tight_layout()
        reference_fig.savefig(figures / 'reference_pose_errors.png', dpi=150)
        reference_components, axes = plt.subplots(2,1,figsize=settings['components_figsize'])
        reference_components.suptitle('Reference deviation components in reference board frame')
        for axis,prefix,unit in zip(axes, ('reference_d','reference_r'), ('mm','deg')):
            axis.set(xlabel='Sample', ylabel=unit)
            axis.grid(alpha=.3)
            for component in 'xyz':
                values = [row.get(prefix+component+'_'+unit) for row in rows]
                if any(value is not None for value in values):
                    axis.plot(range(1,len(rows)+1), values, label=component)
            if axis.lines:
                axis.legend()
            else:
                axis.text(.5,.5,'Reference pose unavailable',ha='center',va='center',transform=axis.transAxes)
        reference_components.tight_layout()
        reference_components.savefig(figures / 'reference_components.png', dpi=150)
        charts.extend([reference_fig, reference_components])
        reference = report['reference']
        if reference is not None:
            for key in ('position_mm','rotation_deg'):
                value = reference[key]
                print(f'reference_{key}: mean={value["mean"]}, std={value["std"]}, max={value["max"]}')
        else:
            print('Reference pose unavailable; reference plots contain no error values.')
    for key, value in report.items():
        if isinstance(value, dict) and 'mean' in value:
            print(f'{key}: mean={value["mean"]}, std={value["std"]}, max={value["max"]}')
    from handeye.analysis.html_report import write_html_report
    write_html_report(root, report, rows, context=context)
    if show:
        plt.show()
    for item in charts:
        plt.close(item)
