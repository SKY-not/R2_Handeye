"""Solve recorded hand-eye data, then evaluate with the shared analysis functions."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,help='Recorded data directory')
    parser.add_argument('--solver-config',default='configs/solver.json')
    parser.add_argument('--reference-config')
    parser.add_argument('--visualization-config',default='configs/visualization.json')
    parser.add_argument('--output-root')
    parser.add_argument('--visualize',action='store_true')
    parser.add_argument('--legacy-poses',action='store_true')
    parser.add_argument('--mode',choices=['eye_to_hand','eye_on_hand'],help='For importing legacy poses')
    parser.add_argument('--backend',choices=['checkerboard','four_apriltag'],help='For importing legacy poses')
    args = vars(parser.parse_args())
    args['session'] = args.pop('data')
    from handeye.pipeline import calibrate_session
    try:
        print(calibrate_session(**args)['output_path'])
    except (ValueError,OSError,RuntimeError) as error:
        parser.exit(1,f'Calibration failed: {error}\n')


if __name__ == '__main__':
    main()
