"""Hardware-free workflow, configuration and non-overwriting persistence regression tests."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
import cv2
import numpy as np
from handeye.config import load_config, load_workflow, PROJECT_ROOT
from handeye.pipeline import calibrate_session
from handeye.session import load_handeye, write_json, read_manifest, load_frame_data, new_run_directory
from handeye.analysis.calibration import analyze_calibration
from handeye.analysis.observation import analyze_observation
from tests.fixtures import write_session


def content_snapshot(root):
    return {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class StructureTests(unittest.TestCase):
    def test_calibrate_and_analyze_share_metrics_in_both_modes(self):
        for mode in ('eye_to_hand','eye_on_hand'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'input'
                expected = write_session(source,mode=mode,schema=2)
                before = content_snapshot(source)
                with contextlib.redirect_stdout(io.StringIO()):
                    first = calibrate_session(source,output_root=root)
                    with patch('handeye.calibration_solver.CalibrationSolver.solve',side_effect=AssertionError('Analysis must not solve')):
                        analysis = analyze_calibration(first['output_path'],output_root=root)
                np.testing.assert_allclose(first['X'],expected,atol=1e-10)
                a = read_json(Path(first['output_path']) / 'evaluation.json')
                b = read_json(analysis / 'evaluation.json')
                self.assertEqual(a,b)
                self.assertLess(a['chain_reprojection_px']['mean'],1e-4)
                self.assertEqual(before,content_snapshot(source))
                self.assertNotEqual(first['output_path'],str(analysis))
                expected_figures = {'pose_errors.png','components.png','reprojection.png',
                                    'reference_pose_errors.png','reference_components.png'}
                for output in (Path(first['output_path']), analysis):
                    self.assertEqual({p.name for p in (output/'figures').glob('*.png')}, expected_figures)
                    html = (output/'report.html').read_text()
                    self.assertEqual(html.count('class="plot-button"'),5)
                    self.assertIn('T_base_camera' if mode=='eye_to_hand' else 'T_tcp_camera', html)
                    self.assertIn('id="frame-details"', html)
                saved = Path(first['output_path'])
                for number in (saved/'handeye_transform.txt').read_text().split():
                    self.assertRegex(number, r'^-?[0-9]+[.][0-9]{8}$')
                matrix, _ = load_handeye(saved)
                np.testing.assert_array_equal(matrix, read_json(saved/'result.json')['transform'])
                np.testing.assert_allclose(matrix, first['X'], atol=5.1e-9, rtol=0)
                from decimal import Decimal
                serialized = json.loads((saved/'result.json').read_text(), parse_float=Decimal)['transform']
                self.assertTrue(all(value.as_tuple().exponent >= -8 for row in serialized for value in row))
                if mode == 'eye_on_hand':
                    self.assertIsNone(a['reference'])  # No configured base-to-board reference.
                else:
                    self.assertIsNotNone(a['reference'])
                self.assertEqual(read_json(Path(first['output_path']) / 'result.json')['status'],'complete')
                self.assertNotIn('sha256',json.dumps(read_json(Path(first['output_path']) / 'result.json')))

    def test_reference_plots_use_configured_pose_and_matching_components(self):
        from handeye.calibration.transforms import pose_to_mat, invert_transform, matrix_to_rotvec
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'input'
            write_session(source, schema=2)
            actual_mount = pose_to_mat([.01,.02,.03,.1,0,0])
            expected_error = pose_to_mat([.01,0,0,0,0,.1])
            reference = actual_mount @ invert_transform(expected_error)
            reference_pose = np.r_[reference[:3,3], np.degrees(matrix_to_rotvec(reference[:3,:3]))]
            path = root/'reference.json'
            write_json(path, {'target_id':'apriltag_board','translation_unit':'m',
                'rotation_unit':'deg','rotation_representation':'rotation_vector',
                'tcp_definition':[0]*6, 'board_to_tcp_rough':reference_pose})
            with contextlib.redirect_stdout(io.StringIO()):
                result = calibrate_session(source, output_root=root, reference_config=path)
                analysis = analyze_calibration(result['output_path'], output_root=root, reference_config=path)
            report = read_json(analysis/'evaluation.json')
            self.assertAlmostEqual(report['reference']['position_mm']['mean'], 10., places=4)
            self.assertAlmostEqual(report['reference']['rotation_deg']['mean'], np.degrees(.1), places=4)
            self.assertLess(report['spatial_position_mm']['mean'], 1e-4)
            self.assertLess(report['chain_reprojection_px']['mean'], 1e-4)
            with (analysis/'residuals.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                position = [float(row['reference_d'+axis+'_mm']) for axis in 'xyz']
                rotation = [float(row['reference_r'+axis+'_deg']) for axis in 'xyz']
                np.testing.assert_allclose(position, [10.,0,0], atol=2e-5, rtol=0)
                np.testing.assert_allclose(rotation, [0,0,np.degrees(.1)], atol=2e-5, rtol=0)
                self.assertAlmostEqual(np.linalg.norm(position), float(row['reference_position_mm']), places=10)
                self.assertAlmostEqual(np.linalg.norm(rotation), float(row['reference_rotation_deg']), places=10)

    def test_v1_legacy_session_still_loads_without_live_target_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'input'
            expected = write_session(source)
            with patch('handeye.data_collector.target_definition',side_effect=AssertionError('live target read')), \
                 patch('handeye.calibration_solver.target_config',side_effect=AssertionError('live target read')), \
                 contextlib.redirect_stdout(io.StringIO()):
                result = calibrate_session(source,output_root=root)
            np.testing.assert_allclose(result['X'],expected,atol=1e-10)

    def test_repeated_runs_preserve_data_and_effective_solver_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'input';write_session(source)
            before=content_snapshot(source)
            config=load_config('configs/solver.json');config.update(spatial_position_weight=.4,spatial_rotation_weight=.6)
            write_json(root/'solver.json',config)
            with contextlib.redirect_stdout(io.StringIO()):
                first=calibrate_session(source,output_root=root,solver_config=root/'solver.json')
                second=calibrate_session(source,output_root=root)
            self.assertNotEqual(first['output_path'],second['output_path'])
            self.assertEqual(read_json(Path(first['output_path'])/'result.json')['configuration']['solver'],config)
            self.assertEqual(before,content_snapshot(source))

    def test_different_dataset_and_mismatched_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_session(root/'training');write_session(root/'heldout',schema=2)
            with contextlib.redirect_stdout(io.StringIO()):
                result=calibrate_session(root/'training',output_root=root)
                out=analyze_calibration(result['output_path'],data=root/'heldout',output_root=root)
            self.assertEqual(read_json(out/'evaluation.json')['scope'],'separate_dataset_consistency')
            with self.assertRaises(ValueError):
                analyze_calibration(result['output_path'],mode='eye_on_hand',output_root=root)

    def test_reference_is_explicit_and_units_are_respected(self):
        from handeye.analysis.calibration import reference_pose
        from types import SimpleNamespace
        collector=SimpleNamespace(mode='eye_to_hand',backend='apriltag_board',target_config={})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.json'
            write_json(path,{'translation_unit':'mm','rotation_unit':'deg','rotation_representation':'rotation_vector',
                'board_to_tcp_rough':[0,0,70,0,-90,0],'tcp_definition':[0]*6})
            pose,_=reference_pose(path,collector)
            np.testing.assert_allclose(pose,[0,0,.07,0,-np.pi/2,0])
            self.assertEqual(reference_pose(None,collector),(None,None))

    def test_known_nonzero_tcp_rejected_in_both_schemas(self):
        from handeye.data_collector import CalibDataCollector
        for schema in (1,2):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);write_session(root,schema=schema)
                metadata=read_json(root/'session.json')
                metadata['tcp_definition']={'offset_m_rotvec_rad':[0,0,.1,0,0,0],'verified':False}
                write_json(root/'session.json',metadata)
                with self.assertRaisesRegex(ValueError,'Calibration requires TCP offset'):
                    CalibDataCollector.from_session(str(root),'eye_to_hand','apriltag_board')
                del metadata['tcp_definition'];write_json(root/'session.json',metadata)
                frame=root/('poses/frame_001.json' if schema==1 else 'frames/000001/robot.json')
                value=read_json(frame);value['tcp_offset_m_rotvec_rad']=[0,0,0,.1,0,0];write_json(frame,value)
                collector=CalibDataCollector.from_session(str(root),'eye_to_hand','apriltag_board')
                with self.assertRaisesRegex(ValueError,'Calibration requires TCP offset'):
                    collector.get_saved_data()

    def test_selected_result_integrity_and_missing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(FileNotFoundError):load_handeye(root)
            np.savetxt(root/'handeye_transform.txt',np.eye(4))
            write_json(root/'result.json',{'mode':'eye_to_hand','transform':np.eye(4)})
            with self.assertRaises(ValueError):load_handeye(root,mode='eye_on_hand')
            matrix=np.eye(4);matrix[0,3]=.1;np.savetxt(root/'handeye_transform.txt',matrix)
            with self.assertRaisesRegex(ValueError,'differs'):load_handeye(root)

    def test_new_gray_data_without_depth_and_projection_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_session(root,schema=2)
            frames=load_frame_data(root,require_robot=True)
            self.assertEqual(frames[0]['rgb'].ndim,2)
            self.assertFalse(list(root.rglob('depth.npy')))
            path=root/'frames/000001/observation.json';value=read_json(path)
            value['projection_matrix'][0][0] *= 2;write_json(path,value)
            with self.assertRaisesRegex(ValueError,'projection model'):load_frame_data(root,require_robot=True)

    def test_observation_analysis_recomputes_projection_without_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'data';write_session(data,schema=2)
            metadata=read_json(data/'session.json');metadata['kind']='observation';metadata.pop('mode')
            write_json(data/'session.json',metadata)
            for path in data.glob('frames/*/robot.json'):path.unlink()
            # Corrupt the cached scalar: reprojecting saved correspondences must ignore it.
            path=data/'frames/000001/observation.json';value=read_json(path);value['reprojection_error_px']=999
            write_json(path,value)
            before=content_snapshot(data)
            with patch('handeye.device_manager.create_camera',side_effect=AssertionError('hardware')),contextlib.redirect_stdout(io.StringIO()):
                output=analyze_observation(data,output_root=root)
            report=read_json(output/'evaluation.json')
            self.assertLess(report['observed_reprojection_px']['mean'],1e-4)
            self.assertEqual(report['position_frame'],'camera')
            self.assertEqual(before,content_snapshot(data))

    def test_observation_rejects_moving_calibration_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write_session(root,schema=2)
            with self.assertRaisesRegex(ValueError,'static observation'):analyze_observation(root)

    def test_historical_validation_csv_remains_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'old.csv'
            fields=['sample_index','reprojection_error_px','decision_margin']+[f'camera_T_{i}{j}' for i in range(4) for j in range(4)]
            with path.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                for k in range(3):
                    T=np.eye(4);T[2,3]=.4+k*.001
                    writer.writerow({'sample_index':k,'reprojection_error_px':.2,'decision_margin':50,
                        **{f'camera_T_{i}{j}':T[i,j] for i in range(4) for j in range(4)}})
            with contextlib.redirect_stdout(io.StringIO()):output=analyze_observation(path,output_root=root)
            self.assertEqual(read_json(output/'evaluation.json')['reprojection_source'],'recorded_csv_values')

    def test_device_factory_applies_supported_robot_parameters(self):
        from handeye.device_manager import create_robot
        settings=load_config('configs/robots/ur3_lab.json')
        settings.update(default_velocity=.4,default_acceleration=.7,workspace_limits=[[-1,1],[-2,2],[0,1]])
        robot=create_robot(settings)
        self.assertEqual(robot.joint_vel,.4);self.assertEqual(robot.joint_acc,.7)
        np.testing.assert_equal(robot.workspace_limits,settings['workspace_limits'])

    def test_motion_uses_selected_configuration_and_result(self):
        from handeye.move_validation import run_move_validation
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);np.savetxt(root/'matrix.txt',np.eye(4))
            settings=load_config('configs/motion.json');settings['rtde_velocity']=.012
            write_json(root/'motion.json',settings)
            args=SimpleNamespace(config=str(root/'motion.json'),result=str(root/'matrix.txt'),mode='eye_to_hand',
                visualization_config='configs/visualization.json',robot_ip=None,target_tag_id=None,dry_run=True,output_root=root)
            with patch('handeye.move_validation._run_move') as run:
                output=run_move_validation(args)
            self.assertEqual(run.call_args.args[0].motion_settings['rtde_velocity'],.012)
            record=read_json(output/'run.json')
            self.assertEqual(record['sources']['matrix_file'],str(root/'matrix.txt'))
            self.assertTrue(record['configuration']['motion']['dry_run'])

    def test_configuration_does_not_import_or_load_devices_for_help(self):
        for arguments in (['collect'],['calibrate'],['analyze'],['analyze','calibration'],['move_tcp']):
            run=subprocess.run([sys.executable,'-B','-m','scripts.'+arguments[0],*arguments[1:]],cwd=PROJECT_ROOT,capture_output=True,text=True)
            self.assertEqual(run.returncode,2,run.stderr)
            self.assertNotIn('pyrealsense2',run.stderr)
        self.assertFalse((PROJECT_ROOT/'scripts/validate.py').exists())

    def test_capture_and_observation_have_separate_references(self):
        capture=load_workflow('configs/capture.json',robot=True)
        observation=load_workflow('configs/observation.json')
        self.assertIn('robot',capture);self.assertNotIn('robot',observation)
        self.assertEqual(capture['target']['type'],'charuco')
        self.assertEqual(observation['target']['type'],'apriltag')
        self.assertNotIn('detection',load_config('configs/solver.json'))

    def test_directory_collision_and_compact_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('handeye.session.datetime') as clock:
                clock.now.return_value.strftime.return_value='fixed'
                first=new_run_directory(root);second=new_run_directory(root)
            self.assertEqual(first.name,'fixed');self.assertEqual(second.name,'fixed_001')
            write_json(first/'test.json',{'pose':[0,0,.07,0,-90,0],'matrix':np.eye(4)})
            text=(first/'test.json').read_text()
            self.assertIn('"pose": [0, 0, 0.07, 0, -90, 0]',text)
            self.assertIn('[1.0, 0.0, 0.0, 0.0]',text)


    def test_static_capture_saves_gray_frames_without_robot_or_depth(self):
        from handeye.analysis.observation import capture_observations
        from handeye.targets.geometry import checkerboard_points
        from tests.test_calibration_core import FakeCamera
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            settings=load_config('configs/observation.json')
            settings.update(target_config='configs/targets/checkerboard.json',detection_profile='checkerboard',sample_interval_s=0)
            write_json(root/'observation.json',settings)
            camera=FakeCamera();camera.connect=Mock();camera.disconnect=Mock()
            camera.get_data=lambda: (np.zeros((480,640),dtype=np.uint8),None)
            geometry=load_config('configs/targets/checkerboard.json')
            corners,_=cv2.projectPoints(checkerboard_points(geometry),np.array([.1,.2,.05]),np.array([0.,0.,.5]),camera.intrinsics,camera.dist_coeffs)
            with patch('handeye.device_manager.create_camera',return_value=camera), \
                 patch('handeye.device_manager.create_robot',side_effect=AssertionError('robot connection')), \
                 patch('handeye.targets.feature_extractor.CheckerboardExtractor.detect_corners',return_value=(True,corners,corners)), \
                 patch('cv2.imshow'),patch('cv2.destroyAllWindows'),patch('cv2.waitKey',side_effect=[13,13,32]), \
                 contextlib.redirect_stdout(io.StringIO()):
                data=capture_observations(root/'observation.json',data_root=root)
                output=analyze_observation(data,output_root=root)
            self.assertEqual(read_json(data/'session.json')['sample_count'],2)
            self.assertFalse(list(data.rglob('robot.json')))
            self.assertFalse(list(data.rglob('depth.npy')))
            self.assertEqual(read_json(output/'evaluation.json')['sample_count'],2)
            camera.disconnect.assert_called_once()

    def test_motion_dry_run_never_opens_control_or_sends_motion(self):
        from handeye.move_validation import run_move_validation
        from tests.test_calibration_core import FakeCamera
        from types import SimpleNamespace
        for mode in ('eye_to_hand','eye_on_hand'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);np.savetxt(root/'matrix.txt',np.eye(4))
                args=SimpleNamespace(config='configs/motion.json',result=str(root/'matrix.txt'),mode=mode,
                    visualization_config='configs/visualization.json',robot_ip=None,target_tag_id=None,dry_run=True,output_root=root)
                camera=FakeCamera();camera.connect=Mock();camera.disconnect=Mock()
                receiver=Mock();receiver.getActualTCPPose.return_value=[0.]*6
                detection=SimpleNamespace(pose_R=np.eye(3),pose_t=np.array([0.,0.,.4]))
                with patch('handeye.move_validation.create_camera',return_value=camera), \
                     patch('handeye.move_validation.create_motion_interfaces',return_value=(None,receiver)) as interfaces, \
                     patch('handeye.move_validation._create_apriltag_detector'), \
                     patch('handeye.move_validation._detect_target_tag',return_value=([],detection)), \
                     patch('cv2.imshow'),patch('cv2.destroyAllWindows'),patch('cv2.waitKey',side_effect=[13,27]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    output=run_move_validation(args)
                if mode=='eye_to_hand':
                    interfaces.assert_not_called()
                else:
                    interfaces.assert_called_once_with(args.motion_settings['robot'],receive_only=True)
                    receiver.disconnect.assert_called_once()
                events=read_json(output/'events.json')['events']
                self.assertEqual(len(events),1);self.assertTrue(events[0]['dry_run'])
                self.assertNotIn('moveJ_return',events[0])
                camera.disconnect.assert_called_once()

    def test_realsense_color_only_does_not_access_depth(self):
        from handeye.camera.realsense import RealSenseCamera
        camera=RealSenseCamera(enable_depth=False)
        camera.connected=True
        frame=Mock();frame.get_data.return_value=np.zeros((2,3,3),dtype=np.uint8)
        frame.get_timestamp.return_value=1.;frame.get_frame_timestamp_domain.return_value='hardware_clock'
        frames=Mock();frames.get_color_frame.return_value=frame
        camera.pipeline=Mock();camera.pipeline.wait_for_frames.return_value=frames
        image,depth=camera.get_data()
        self.assertEqual(image.shape,(2,3,3));self.assertIsNone(depth)
        frames.get_depth_frame.assert_not_called()
        with self.assertRaisesRegex(ValueError,'disabled'):camera.get_depth_image()


if __name__ == '__main__':
    unittest.main()
