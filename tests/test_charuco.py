"""ChArUco geometry, detection and offline pipeline regression."""
import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from handeye.config import target_definition, load_workflow
from handeye.targets.charuco import make_board, correspondences, CharucoTarget
from handeye.session import save_frame_data, write_json, load_frame_data
from handeye.pipeline import load_calibration_data
from handeye.analysis.calibration import evaluate
from handeye.calibration.transforms import pose_to_mat, invert_transform


class CharucoTests(unittest.TestCase):
    def setUp(self):
        self.config = target_definition('charuco')
        self.K = np.array([[1200., 0, 680], [0, 1200., 530], [0, 0, 1]])

    def test_default_and_rotated_partial_detection(self):
        self.assertEqual(load_workflow('configs/capture.json', robot=True)['target']['type'], 'charuco')
        board = make_board(self.config)
        self.assertEqual(len(board.getChessboardCorners()), 88)
        image = cv2.copyMakeBorder(board.generateImage((1200, 900)), 80, 80, 80, 80, cv2.BORDER_CONSTANT, value=255)
        detector = CharucoTarget(self.config)
        for image in (image, cv2.rotate(image, cv2.ROTATE_180)):
            sample = detector.detect(image, self.K, np.zeros(5))
            self.assertTrue(sample['success'])
            self.assertEqual(len(sample['charuco_ids']), 88)
        image[:, :600] = 255
        sample = detector.detect(image, self.K, np.zeros(5))
        self.assertTrue(sample['success'])
        self.assertLess(len(sample['charuco_ids']), 88)
        self.assertFalse(detector.detect(np.full_like(image, 255), self.K, np.zeros(5))['success'])

    def test_invalid_ids(self):
        for ids in ([0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 88], [0, 1, 2, 3, 4, 5], [0., 1., 2., 3., 4., 5.]):
            with self.assertRaises(ValueError):
                correspondences(self.config, ids, np.zeros((6, 2)))

    def test_session_loader_and_both_mode_evaluations(self):
        distortion = np.array([.08, -.03, .001, -.002, 0.])
        X = pose_to_mat([.02, -.01, .04, .02, -.03, .01])
        reference = pose_to_mat([.03, .02, .8, .05, -.08, .03])
        for mode in ('eye_on_hand', 'eye_to_hand'):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'frames').mkdir()
                write_json(root / 'session.json', {'schema_version':2, 'kind':'calibration', 'mode':mode,
                    'backend':'charuco', 'target':self.config, 'corner_space':'raw',
                    'camera_matrix':self.K, 'dist_coeffs':distortion, 'image_size':[1360,1060]})
                for i in range(6):
                    tcp = pose_to_mat([.01*i, -.005*i, .003*i, .01*i, -.015*i, .008*i])
                    pose = invert_transform(X) @ (invert_transform(tcp) if mode == 'eye_on_hand' else tcp) @ reference
                    ids = np.arange(i, 88, 2)
                    points = make_board(self.config).getChessboardCorners()[ids].astype(float)
                    pixels, _ = cv2.projectPoints(points, cv2.Rodrigues(pose[:3,:3])[0], pose[:3,3], self.K, distortion)
                    sample = {'success':True, 'rgb':np.full((1060,1360),255,np.uint8),
                        'charuco_ids':ids.tolist(), 'corners_refined':pixels, 'board_pose':pose,
                        'captured_at_ns':i, 'tcp':tcp, 'tcp_offset':np.zeros(6)}
                    save_frame_data(root / 'frames' / f'{i:06d}',sample,self.config,self.K,distortion,'raw')
                collector, solver, frames = load_calibration_data(root, for_analysis=True)
                report = evaluate(collector,solver,{'X':X},*frames)
                self.assertLess(report['observed_reprojection_px']['mean'], 1e-6)
                self.assertLess(report['chain_reprojection_px']['mean'], 1e-6)
                self.assertIsNone(report['reference'])
                observation = root / 'frames/000000/observation.json'
                value = json.loads(observation.read_text())
                value['charuco_ids'][0] = 87
                write_json(observation,value)
                with self.assertRaises(ValueError):
                    load_frame_data(root, require_robot=True)


if __name__ == '__main__':
    unittest.main()
