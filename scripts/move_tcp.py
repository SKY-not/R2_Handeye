"""AprilTag target motion check. --dry-run still uses the camera, but sends no motion."""
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, help='Calibration run directory or matrix file')
    parser.add_argument('--mode', choices=['eye_to_hand', 'eye_on_hand'])
    parser.add_argument('--robot-ip')
    parser.add_argument('--target-tag-id', type=int)
    parser.add_argument('--config', default='configs/motion.json')
    parser.add_argument('--output-root')
    parser.add_argument('--visualization-config', default='configs/visualization.json')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    from handeye.move_validation import run_move_validation
    try:
        print(run_move_validation(args))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'Motion validation failed: {error}\n')

if __name__ == '__main__':
    main()
