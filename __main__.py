import argparse
import os
import os.path

import library
import library.runner_regression


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--runs_count', required=True)

    assert 'CUDA_VISIBLE_DEVICES' in os.environ, \
        'CUDA_VISIBLE_DEVICES should be specified'

    return parser.parse_args()


if __name__ == '__main__':
    parsed_args = parse_args()
    if parsed_args.mode == "regression":
        library.runner_regression.run_regression(
            parsed_args.model,
            int(parsed_args.runs_count),
        )
 