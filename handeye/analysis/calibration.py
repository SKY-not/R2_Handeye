"""One evaluator shared by calibrate and analyze calibration; never solves hand-eye."""
import json
from pathlib import Path
import numpy as np
from handeye.config import load_config, resolve_path
from handeye.error_calculator import ErrorCalculator
from handeye.targets.geometry import checkerboard_points, reprojection_errors
from handeye.session import read_manifest, load_handeye, begin_run, finish_run, write_json
from handeye.analysis.report import statistics, save_report

DEFAULT_REFERENCE_CONFIG = 'configs/references/lab_board_mount.json'


def reference_pose(path, collector):
    if path is None:
        return None, None
    value = load_config(path)
    reference = value.get(collector.backend, value)
    if reference.get('target_id') not in (None, collector.backend, collector.target_config.get('target_id')):
        raise ValueError('Reference target differs from selected data')
    key = 'board_to_base_rough' if collector.mode == 'eye_on_hand' else 'board_to_tcp_rough'
    pose = reference.get(key)
    if pose is None:
        return None, reference
    pose = np.asarray(pose, dtype=float).copy()
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError('Reference pose must contain six finite values')
    if reference['rotation_representation'] != 'rotation_vector':
        raise ValueError('Reference must declare rotation_vector representation')
    factors = {'m':1., 'mm':.001}
    angles = {'rad':1., 'deg':np.pi/180}
    if reference['translation_unit'] not in factors or reference['rotation_unit'] not in angles:
        raise ValueError('Unsupported reference units')
    pose[:3] *= factors[reference['translation_unit']]
    pose[3:] *= angles[reference['rotation_unit']]
    tcp = reference.get('tcp_definition')
    if tcp is not None and np.any(np.asarray(tcp, dtype=float) != 0):
        raise ValueError('Reference TCP differs from the required zero-offset calibration TCP')
    return pose, reference


def evaluate(collector, solver, result, robot, camera, corners, images, tag_corners, tag_ids,
             visualize=False, *, reference_config=None, scope='training_sample_fit',
             visualization_config='configs/visualization.json'):
    calculator = ErrorCalculator(collector.mode, solver.intrinsics, solver.dist_coeffs,
                                 collector.backend, target=collector.target_config)
    X = result['X']
    positions, rotations, pc, rc = calculator.calculate_spatial_consistency(robot, camera, X)
    report = {'sample_count':len(robot), 'scope':scope, 'residual_frame':'mean_target',
              'spatial_position_mm':statistics(positions*1000),
              'spatial_rotation_deg':statistics(np.degrees(rotations)),
              'reference':None, 'observed_reprojection_px':None, 'chain_reprojection_px':None}
    reference_config = DEFAULT_REFERENCE_CONFIG if reference_config is None else reference_config
    ref, reference = reference_pose(reference_config, collector)
    rp = rr = rpc = rrc = None
    if ref is not None and not collector.legacy_pose_only:
        kwargs = {'board_to_base' if collector.mode == 'eye_on_hand' else 'board_to_tcp':ref}
        rpc, rrc = calculator.calculate_pose_error_components(robot,camera,X,**kwargs)
        rp = np.linalg.norm(rpc, axis=1)*1000
        rr = np.degrees(np.linalg.norm(rrc, axis=1))
        report['reference'] = {
            'definition':reference, 'source':str(resolve_path(reference_config)),
            'residual_frame':'reference_target',
            'position_mm':statistics(rp), 'rotation_deg':statistics(rr),
            'position_components_mm':{axis:statistics(rpc[:,j]*1000) for j,axis in enumerate('xyz')},
            'rotation_components_deg':{axis:statistics(np.degrees(rrc[:,j])) for j,axis in enumerate('xyz')},
        }
    observed = chain = None
    if not collector.legacy_pose_only:
        if collector.backend == 'charuco':
            from handeye.calibration.transforms import target_poses, average_transforms, invert_transform
            reference = average_transforms(target_poses(collector.mode, robot, camera, X))
            left = [invert_transform(t) for t in robot] if collector.mode == 'eye_on_hand' else robot
            predictions = invert_transform(X) @ np.asarray(left) @ reference
            def errors(poses):
                values = []
                for pose, points, pixels in zip(poses, solver.object_points_list, corners):
                    try:
                        values.append(float(reprojection_errors(pose, points, pixels, solver.intrinsics, solver.dist_coeffs).mean()))
                    except ValueError:
                        values.append(float('nan'))
                return np.asarray(values)
            observed, chain = errors(camera), errors(predictions)
        elif collector.backend == 'checkerboard':
            points = checkerboard_points(collector.target_config)
            observed = np.asarray([reprojection_errors(pose,points,pixels,solver.intrinsics,solver.dist_coeffs).mean()
                                   for pose,pixels in zip(camera,corners)])
            chain = calculator.calculate_reprojection_error(robot,camera,corners,X)
        else:
            observed = calculator.calculate_apriltag_observed_reprojection_error(camera,tag_corners,tag_ids,images)
            chain = calculator.calculate_apriltag_chain_reprojection_error(robot,camera,tag_corners,tag_ids,X,images)
        report['observed_reprojection_px'] = statistics(observed)
        report['chain_reprojection_px'] = statistics(chain)
    rows = []
    def finite(value):
        return None if value is None or not np.isfinite(value) else float(value)
    for i,index in enumerate(solver.sample_indices):
        rows.append({'frame':index, 'translation_mm':positions[i]*1000, 'rotation_deg':np.degrees(rotations[i]),
                     **dict(zip(('dx_mm','dy_mm','dz_mm'),pc[i]*1000)),
                     **dict(zip(('rx_deg','ry_deg','rz_deg'),np.degrees(rc[i]))),
                     'observed_reprojection_px':None if observed is None else finite(observed[i]),
                     'chain_reprojection_px':None if chain is None else finite(chain[i]),
                     'reference_position_mm':None if rp is None else float(rp[i]),
                     'reference_rotation_deg':None if rr is None else float(rr[i]),
                     **dict(zip(('reference_dx_mm','reference_dy_mm','reference_dz_mm'),
                                [None]*3 if rpc is None else rpc[i]*1000)),
                     **dict(zip(('reference_rx_deg','reference_ry_deg','reference_rz_deg'),
                                [None]*3 if rrc is None else np.degrees(rrc[i])))})
    print(f'Evaluation scope: {scope}; reference deviations are not independent absolute accuracy.')
    if result.get('output_path'):
        context = {'mode':collector.mode, 'backend':collector.backend,
                   'source_session':solver.source_session, 'transform':np.asarray(X).tolist()}
        save_report(result['output_path'], report, rows, visualization_config=visualization_config,
                    show=visualize, context=context)
    return report


def analyze_calibration(result, *, data=None, mode=None, reference_config=None,
                        output_root=None, visualize=False, visualization_config='configs/visualization.json'):
    from handeye.pipeline import load_calibration_data
    reference_config = DEFAULT_REFERENCE_CONFIG if reference_config is None else reference_config
    matrix, source = load_handeye(result, mode=mode)
    path = data or source['source_session']
    if path is None:
        raise ValueError('A standalone matrix requires --data')
    root_data = resolve_path(path)
    metadata = read_manifest(root_data)
    expected_mode = source['mode'] or mode
    if expected_mode is None:
        raise ValueError('A standalone matrix requires explicit --mode')
    if metadata['mode'] != expected_mode:
        raise ValueError('Data mode differs from selected hand-eye result')
    original = source['source_session']
    same_data = original is not None and resolve_path(original) == root_data
    if original and not same_data and (resolve_path(original) / 'session.json').exists():
        previous = read_manifest(resolve_path(original))
        if previous.get('camera_id') and metadata.get('camera_id') and previous['camera_id'] != metadata['camera_id']:
            raise ValueError('Analysis data was captured by a different camera')
    collector, loader, frames = load_calibration_data(root_data, mode=expected_mode, for_analysis=True)
    scope = 'training_sample_fit' if same_data else 'separate_dataset_consistency'
    root = begin_run('analysis', sources={**source,'data':str(root_data)},
                     configuration={'kind':'calibration', 'reference_config':None if reference_config is None else str(resolve_path(reference_config)),
                                    'visualization':load_config(visualization_config)}, output_root=output_root)
    try:
        write_json(root / 'analysis.json', {'kind':'calibration','data_id':root_data.name,
            'data':str(root_data),'matrix_file':source['matrix_file'],'scope':scope})
        evaluate(collector, loader, {'X':matrix,'output_path':str(root)}, *frames,
                 reference_config=reference_config, scope=scope, visualize=visualize, visualization_config=visualization_config)
    except Exception as error:
        finish_run(root,'failed',error=str(error)); raise
    finish_run(root)
    return root
