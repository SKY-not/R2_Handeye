"""Static visual repeatability: camera-frame observations, without robot or hand-eye."""
import csv
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from handeye.config import load_config, load_workflow, effective_target, resolve_path, PROJECT_ROOT
from handeye.session import (new_run_directory, write_json, read_manifest, save_frame_data,
                             load_frame_data, begin_run, finish_run)
from handeye.calibration.transforms import average_transforms, validate_transform
from handeye.targets.geometry import reprojection_errors
from handeye.analysis.report import statistics, save_report


def capture_observations(config='configs/observation.json', *, data_root=None):
    import cv2
    from handeye.device_manager import create_camera
    from handeye.targets.detector import TargetDetector
    settings = load_workflow(config)
    target = effective_target(settings)
    interval = float(settings['sample_interval_s'])
    if not np.isfinite(interval) or interval < 0:
        raise ValueError('sample_interval_s must be finite and nonnegative')
    camera = create_camera(settings['camera'])
    root = None
    count = 0
    try:
        camera.connect()
        detector = TargetDetector(target,camera)
        root = new_run_directory((PROJECT_ROOT / 'data' if data_root is None else resolve_path(data_root)) / 'observation')
        (root / 'frames').mkdir()
        space = {'checkerboard':'raw','apriltag_board':'rectified_alpha0','apriltag':'rectified_alpha1'}[target['type']]
        metadata = {'schema_version':2,'kind':'observation','corner_space':space,
            'target':target,'configuration_source':str(resolve_path(config)), 'capture_config':settings,
            'camera_matrix':camera.intrinsics,'dist_coeffs':np.asarray(camera.dist_coeffs).ravel(),
            'image_size':[camera.im_width,camera.im_height],'camera_id':camera.device_id,
            'distortion_model':getattr(camera,'distortion_model','opencv'),
            'source_camera_model':getattr(camera,'source_camera_model',None),
            'translation_unit':'m','rotation_unit':'rad','rotation_representation':'rotation_vector',
            'depth_unit':'m','depth_alignment':'image','status':'collecting'}
        write_json(root / 'session.json',metadata)
        last = float('-inf')
        print('Keep camera and target fixed. Enter=save observation, Space/Esc=finish and analyze.')
        while True:
            image,depth = camera.get_data()
            captured_at = time.time_ns()
            if image is None:
                continue
            if [image.shape[1],image.shape[0]] != metadata['image_size']:
                raise ValueError('Camera resolution changed during capture')
            sample = detector.detect_image(image,depth)
            display = sample.get('display_rgb',image).copy()
            if display.ndim == 2:
                display = cv2.cvtColor(display,cv2.COLOR_GRAY2BGR)
            cv2.putText(display,f'Samples: {count} | Enter=save Space/Esc=finish | detected={sample["success"]}',
                        (10,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),1)
            cv2.imshow('Observation consistency',display)
            key = cv2.waitKey(1) & 0xff
            if key in (27,32):
                break
            now = time.monotonic()
            if key in (10,13) and sample['success'] and now-last >= interval:
                sample.update(captured_at_ns=captured_at,camera_timing=getattr(camera,'last_frame_metadata',{}))
                save_frame_data(root / 'frames' / f'{count+1:06d}',sample,target,camera.intrinsics,camera.dist_coeffs,space)
                count += 1
                last = now
        metadata.update(status='complete',sample_count=count)
        write_json(root / 'session.json',metadata)
    except Exception as error:
        if root is not None and (root / 'session.json').exists():
            metadata.update(status='failed',error=str(error),sample_count=count)
            write_json(root / 'session.json',metadata)
        raise
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()
    print('Observation data saved:',root)
    return root


def _load_observations(path):
    if path.is_file():
        # Historical validate CSV: camera poses are available, raw corners are not.
        with path.open(newline='',encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        poses = [np.array([[float(row[f'camera_T_{i}{j}']) for j in range(4)] for i in range(4)]) for row in rows]
        return poses, [row['sample_index'] for row in rows], [float(row['reprojection_error_px']) for row in rows], [float(row['decision_margin']) for row in rows], 'recorded_csv_values'
    metadata = read_manifest(path)
    if metadata.get('kind') != 'observation':
        raise ValueError('Observation analysis requires static observation data (not moving calibration samples)')
    frames = load_frame_data(path)
    poses,indices,errors,margins = [],[],[],[]
    for frame in frames:
        observation = frame['observation']
        if observation['corner_space'] != metadata['corner_space']:
            raise ValueError('Observation corner space differs from session')
        pose = np.asarray(observation['transform'])
        errors.append(float(reprojection_errors(pose,np.asarray(observation['object_points']),
            np.asarray(observation['image_points']),np.asarray(observation['projection_matrix']),
            np.asarray(observation['projection_distortion'])).mean()))
        poses.append(pose); indices.append(frame['index'])
        margins.append(observation.get('decision_margin'))
    return poses,indices,errors,margins,'recomputed_from_saved_correspondences'


def observation_residuals(poses):
    poses = np.asarray([validate_transform(pose) for pose in poses])
    mean = average_transforms(poses)
    position = (poses[:,:3,3]-mean[:3,3])*1000
    rotation = np.degrees(Rotation.from_matrix(mean[:3,:3].T @ poses[:,:3,:3]).as_rotvec())
    return mean,position,rotation


def analyze_observation(data, *, output_root=None, visualize=False,
                        visualization_config='configs/visualization.json'):
    path = resolve_path(data)
    poses,indices,projection,margins,projection_source = _load_observations(path)
    if len(poses) < 2:
        raise ValueError('At least two saved observations are required; captured data remains available')
    mean,position,rotation = observation_residuals(poses)
    pos_norm,rot_norm = np.linalg.norm(position,axis=1),np.linalg.norm(rotation,axis=1)
    dz_std = float(np.std(position[:,2]))
    outliers = [] if dz_std == 0 else [indices[i] for i in np.flatnonzero(np.abs(position[:,2])/dz_std > 3)]
    report = {'sample_count':len(poses),'scope':'static_visual_repeatability','position_frame':'camera',
        'rotation_residual_frame':'mean_target','mean_transform':mean,
        'position_mm':statistics(pos_norm),'rotation_deg':statistics(rot_norm),
        'position_components_mm':{axis:statistics(position[:,i]) for i,axis in enumerate('xyz')},
        'rotation_components_deg':{axis:statistics(rotation[:,i]) for i,axis in enumerate('xyz')},
        'observed_reprojection_px':statistics(projection),'reprojection_source':projection_source,
        'decision_margin':statistics([np.nan if x is None else x for x in margins]),'z_outlier_frames':outliers}
    rows = [{'frame':index,'translation_mm':pos_norm[i],'rotation_deg':rot_norm[i],
        **dict(zip(('dx_mm','dy_mm','dz_mm'),position[i])),
        **dict(zip(('rx_deg','ry_deg','rz_deg'),rotation[i])),
        'observed_reprojection_px':projection[i],'decision_margin':margins[i]} for i,index in enumerate(indices)]
    root = begin_run('analysis',sources={'data':str(path)},configuration={
        'kind':'observation','visualization':load_config(visualization_config)},output_root=output_root)
    try:
        write_json(root / 'analysis.json',{'kind':'observation','data':str(path),'data_id':path.name})
        save_report(root,report,rows,visualization_config=visualization_config,show=visualize,
                    context={'source_session':str(path)})
        print('Static visual repeatability does not establish hand-eye accuracy.')
    except Exception as error:
        finish_run(root,'failed',error=str(error)); raise
    finish_run(root)
    return root
