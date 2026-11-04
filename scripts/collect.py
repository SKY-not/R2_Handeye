"""Collect hand-eye data using explicit workflow and device configurations."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['eye_to_hand','eye_on_hand'])
    parser.add_argument('--config',default='configs/capture.json')
    parser.add_argument('--data-root')
    args = parser.parse_args()
    from handeye.pipeline import collect_session
    try:
        print(collect_session(**vars(args)))
    except (ValueError,OSError,RuntimeError,ImportError) as error:
        parser.exit(1,f'Collection failed: {error}\n')


if __name__ == '__main__':
    main()
