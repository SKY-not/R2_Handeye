"""Analyze static visual observations or an explicitly selected hand-eye result."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='kind',required=True)
    observation = sub.add_parser('observation',help='Static camera/target repeatability')
    inputs = observation.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--capture',action='store_true')
    inputs.add_argument('--data',help='Observation directory or historical validation CSV')
    observation.add_argument('--config',help='Capture only; defaults to configs/observation.json')
    observation.add_argument('--data-root',help='Capture only; parent of observation/<data_id>')
    calibration = sub.add_parser('calibration',help='Offline hand-eye quality analysis')
    calibration.add_argument('--result',required=True)
    calibration.add_argument('--data',help='Defaults to selected result source data')
    calibration.add_argument('--mode',choices=['eye_to_hand','eye_on_hand'],help='Required for a standalone matrix')
    calibration.add_argument('--reference-config')
    for command in (observation,calibration):
        command.add_argument('--output-root')
        command.add_argument('--visualize',action='store_true')
        command.add_argument('--visualization-config',default='configs/visualization.json')
    args = parser.parse_args()
    try:
        if args.kind == 'observation':
            from handeye.analysis.observation import capture_observations, analyze_observation
            if not args.capture and (args.config or args.data_root):
                parser.error('--config and --data-root apply only to --capture')
            data = capture_observations(args.config or 'configs/observation.json',data_root=args.data_root) if args.capture else args.data
            output = analyze_observation(data,output_root=args.output_root,visualize=args.visualize,
                                         visualization_config=args.visualization_config)
        else:
            from handeye.analysis.calibration import analyze_calibration
            options = vars(args).copy(); options.pop('kind')
            output = analyze_calibration(**options)
        print('Analysis saved:',output)
    except (ValueError,OSError,RuntimeError,ImportError,KeyError) as error:
        parser.exit(1,f'Analysis failed: {error}\n')


if __name__ == '__main__':
    main()
