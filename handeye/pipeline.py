"""Collection and offline solve orchestration. Evaluation is shared with analyze."""
from pathlib import Path
import json
from handeye.config import load_config, load_workflow, effective_target, resolve_path
from handeye.data_collector import CalibDataCollector
from handeye.calibration_solver import CalibrationSolver
from handeye.session import read_manifest, write_json
from handeye.analysis.calibration import evaluate


def load_calibration_data(session, *, mode=None, backend=None, solver_config=None,
                          output_root=None, legacy_poses=False, for_analysis=False):
    backend = 'apriltag_board' if backend == 'four_apriltag' else backend
    root = resolve_path(session)
    if (root / 'session.json').is_file():
        metadata = read_manifest(root)
        if metadata.get('kind', 'calibration') != 'calibration':
            raise ValueError('Hand-eye analysis requires calibration data with robot poses')
        mode = metadata['mode'] if mode is None else mode
        backend = metadata['backend'] if backend is None else backend
    elif not (legacy_poses and mode and backend):
        raise ValueError('Missing session.json: legacy poses require --legacy-poses, --mode and --backend')
    # Analysis needs only the existing pose loader, no optimizer configuration.
    settings = {'min_calibration_points':6} if for_analysis else load_config(solver_config or 'configs/solver.json')
    collector = CalibDataCollector.from_session(str(root), mode, backend, legacy_pose_only=legacy_poses)
    solver = CalibrationSolver(mode, collector.intrinsics, collector.dist_coeffs, backend,
        solver_config=settings, target=collector.target_config,
        output_root=None if output_root is None else resolve_path(output_root))
    if for_analysis:
        solver.min_points_required = 2
    solver.config_source = None if for_analysis else str(resolve_path(solver_config or 'configs/solver.json'))
    return collector, solver, solver.load_data(collector)


def calibrate_session(session, *, mode=None, backend=None, solver_config=None, output_root=None,
                      legacy_poses=False, visualize=False, reference_config=None,
                      visualization_config='configs/visualization.json'):
    collector, solver, data = load_calibration_data(session, mode=mode, backend=backend,
        solver_config=solver_config, output_root=output_root, legacy_poses=legacy_poses)
    definition = (collector.session_metadata or {}).get('tcp_definition')
    if not definition or not definition.get('verified'):
        print('Recorded TCP offset is unverified; offline analysis does not certify zero-offset acquisition.')
    result = solver.solve(*data[:3])
    path = Path(result['output_path']) / 'result.json'
    metadata = json.loads(path.read_text(encoding='utf-8'))
    try:
        result['evaluation'] = evaluate(collector, solver, result, *data, visualize=visualize,
            reference_config=reference_config, visualization_config=visualization_config)
        metadata.update(status='complete', evaluation_file='evaluation.json')
        metadata['configuration']['visualization'] = load_config(visualization_config)
        metadata['configuration']['reference_source'] = None if reference_config is None else str(resolve_path(reference_config))
    except Exception as error:
        metadata.update(status='evaluation_failed', error=str(error))
        write_json(path, metadata)
        raise
    write_json(path, metadata)
    return result


def collect_session(mode, *, config='configs/capture.json', data_root=None):
    from handeye.device_manager import DeviceManager
    settings = load_workflow(config, robot=True)
    target = effective_target(settings)
    if target['type'] not in ('checkerboard','apriltag_board'):
        raise ValueError('Hand-eye collection requires checkerboard or apriltag_board, not a single-tag observation target')
    manager = DeviceManager(settings)
    try:
        manager.connect()
        collector = CalibDataCollector(manager.get_robot(),manager.get_camera(),mode,target['type'],
            capture_config=settings, target=target, reference={}, data_root=data_root)
        collector.start_new_session()
        collector.session_metadata['configuration_source'] = str(resolve_path(config))
        write_json(Path(collector.data_path['root']) / 'session.json',collector.session_metadata)
        try:
            collector.collect_loop()
            collector.session_metadata['status'] = 'complete'
        except Exception as error:
            collector.session_metadata.update(status='failed', error=str(error))
            raise
        finally:
            collector.session_metadata['sample_count'] = collector.frame_count
            write_json(Path(collector.data_path['root']) / 'session.json',collector.session_metadata)
            collector.stop_preview()
        print('Data saved:',collector.data_path['root'])
        return collector.data_path['root']
    finally:
        manager.disconnect()
