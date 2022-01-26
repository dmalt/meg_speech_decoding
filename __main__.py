import argparse
import os
import os.path
import json

import library
import library.runner_regression
import library.runner_classification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--patient', required=True)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--runs_count', required=False) # for regression

    assert 'CUDA_VISIBLE_DEVICES' in os.environ, \
        'CUDA_VISIBLE_DEVICES should be specified'

    return parser.parse_args()


if __name__ == '__main__':
    parsed_args = parse_args()
    
    for dir_name in ["results", "model_dumps"]:
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
            print(f"{dir_name} dir created")
    
    patients_dict = {}
    with open("library/patients.json", "r") as patients_file:
        for patient in json.load(patients_file):
            patients_dict[patient["name"]] = patient

    assert parsed_args.patient in patients_dict
    patient = patients_dict[parsed_args.patient]

    if parsed_args.mode == "regression":
        library.runner_regression.run_regression(
            parsed_args.model,
            patient,
            int(parsed_args.runs_count),
            parsed_args.debug
        )
    elif parsed_args.mode == "classification":
        library.runner_classification.run_classification(
            parsed_args.model,
            patient,
            parsed_args.debug
        )      
 