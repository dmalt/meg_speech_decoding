import copy
import datetime
import json
import os

import h5py
import numpy as np
import sklearn
import sklearn.preprocessing
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from . import bench_models_classification, bench_models_regression
from .runner_common import (
    CLASSIFICATION_MODE,
    MODEL_DUMPS_DIR,
    REGRESSION_MODE,
    RESULTS_DIR,
    WORDS_REMAP,
    data_generator,
)

# TODO: REMOVE IT
CLASSIFICATION_MODEL_CLASS = bench_models_classification.Mel2WordSimple

MAX_ITERATIONS_COUNT = 10_000
METRIC_ITERATIONS = 1_000
EARLY_STOP_STEPS = 5_000


# DEBUG

# DEBUG_DATA_LIMIT = SAMPLING_RATE * 180
# MAX_ITERATIONS_COUNT = 500
# METRIC_ITERATIONS = 100


def process_batch(bench_model, generator, is_train, iteration, max_words_length):
    loss_function = nn.CrossEntropyLoss()

    if is_train:
        bench_model.model.train()
    else:
        bench_model.model.eval()

    x_batch, y_batch = next(generator)
    x_batch = prepare_x_batch_for_net(x_batch, max_words_length)
    non_silent_indexes = np.where(y_batch != 0)[0]

    assert x_batch.shape[0] == y_batch.shape[0]

    x_batch = torch.FloatTensor(x_batch)
    y_batch = torch.LongTensor(y_batch)

    if torch.cuda.is_available():
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()

    if is_train:
        bench_model.optimizer.zero_grad()

    y_predicted = bench_model.model(x_batch)
    assert not torch.any(torch.isnan(y_predicted))

    loss = loss_function(y_predicted, y_batch)

    if is_train:
        loss.backward()
        bench_model.optimizer.step()

    assert (
        y_predicted.shape[0] == y_batch.shape[0]
    ), f"{y_predicted.shape[0]} != {y_batch.shape[0]}"

    metrics = {}

    y_predicted_numpy = y_predicted.cpu().detach().numpy().argmax(axis=1)
    y_batch_numpy = y_batch.cpu().detach().numpy()

    metrics["loss"] = float(loss.cpu().detach().numpy())

    metrics["accuracy"] = float(np.mean(y_predicted_numpy == y_batch_numpy))
    if len(non_silent_indexes) > 0:
        metrics["accuracy (without silent class)"] = float(
            np.mean(y_predicted_numpy[non_silent_indexes] == y_batch_numpy[non_silent_indexes])
        )

    for key, value in metrics.items():
        bench_model.logger.add_value(key, is_train, value, iteration)

    return metrics


def load_words_info(filepath):
    phrases_info = []
    with open(filepath) as phrases_file:
        for row in phrases_file:
            row = row.strip()
            if len(row) == 0:
                continue
            splitted_row = row.split("\t")
            splitted_row[0] = int(splitted_row[0])
            splitted_row[1] = int(splitted_row[1])
            assert splitted_row[2] in WORDS_REMAP, f"phrase {splitted_row[2]} not allowed"
            phrases_info.append(splitted_row)
    return phrases_info


def get_random_predictions(bench_model, generator, iterations, max_words_length):
    Y_batch = []
    Y_predicted = []
    for index, (x_batch, y_batch) in enumerate(generator):
        x_batch = prepare_x_batch_for_net(x_batch, max_words_length)
        x_batch = torch.FloatTensor(x_batch)
        if torch.cuda.is_available():
            x_batch = x_batch.cuda()
        y_predicted = bench_model.model(x_batch).cpu().detach().numpy().argmax(axis=1)
        assert x_batch.shape[0] == y_predicted.shape[0]
        Y_predicted.append(y_predicted)
        Y_batch.append(y_batch)
        if index > iterations:
            break

    Y_predicted = np.concatenate(Y_predicted, axis=0)
    Y_batch = np.concatenate(Y_batch, axis=0)
    return Y_batch, Y_predicted


def predict_regression(bench_model_regression, ecog):
    X_predicted = []
    all_data_generator = data_generator(
        ecog,
        [],
        bench_model_regression.BATCH_SIZE,
        bench_model_regression.LAG_BACKWARD,
        bench_model_regression.LAG_FORWARD,
        shuffle=False,
        infinite=False,
    )
    bench_model_regression.model.eval()
    for ecog_batch in all_data_generator:
        ecog_batch = torch.FloatTensor(ecog_batch)
        if torch.cuda.is_available():
            ecog_batch = ecog_batch.cuda()
        x_predicted = bench_model_regression.model(ecog_batch).cpu().data.numpy()
        assert ecog_batch.shape[0] == x_predicted.shape[0]
        X_predicted.append(x_predicted)

    X_predicted = np.concatenate(X_predicted, axis=0)
    X_predicted = np.pad(
        X_predicted,
        [
            (
                bench_model_regression.LAG_BACKWARD,
                bench_model_regression.LAG_FORWARD,
            ),
            (0, 0),
        ],
    )
    assert X_predicted.shape[0] == ecog.shape[0]
    return X_predicted


def prepare_frames(x, words_info, downsampling_coef):
    x_frames = []
    classes = []

    last_phrase_end = 0
    for phrase_start, phrase_end, phrase in words_info:
        if last_phrase_end != 0 and last_phrase_end != phrase_start:
            assert last_phrase_end < phrase_start
            x_frames.append(
                x[int(last_phrase_end / downsampling_coef) : int(phrase_start / downsampling_coef)]
            )
            classes.append(WORDS_REMAP["silent"])

        x_frames.append(
            x[int(phrase_start / downsampling_coef) : int(phrase_end / downsampling_coef)]
        )
        classes.append(WORDS_REMAP[phrase])
        last_phrase_end = phrase_end

    return x_frames, classes


def fix_class_imbalance(X, Y):
    non_silent_indexes = np.where(Y != 0)[0]
    possible_classes = len(set(Y))
    silent_indexes = np.where(Y == 0)[0]

    selected_silent_indexes = np.random.choice(
        silent_indexes, int(len(non_silent_indexes) * 1.0 / possible_classes)
    )
    selected_indexes = np.sort(np.concatenate([selected_silent_indexes, non_silent_indexes]))

    X = X[selected_indexes]
    Y = Y[selected_indexes]

    assert len(X) == len(Y)
    return X, Y


def batch_iterator(X, Y, batch_size=1):
    length = len(X)
    for index in range(0, length, batch_size):
        current_slice = slice(index, min(index + batch_size, length))
        yield copy.deepcopy(X[current_slice]), copy.deepcopy(Y[current_slice])


def random_iterator(X, Y, batch_size=1):
    random_core = np.arange(0, len(X))
    while True:
        current_indexes = np.random.choice(random_core, batch_size)
        yield copy.deepcopy(X[current_indexes]), copy.deepcopy(Y[current_indexes])


def prepare_x_batch_for_net(x_batch, max_words_length):
    x_batch = [x.transpose() for x in x_batch]
    for i in range(len(x_batch)):
        if x_batch[i].shape[1] >= max_words_length:
            x_batch[i] = x_batch[i][:, :max_words_length]
        else:
            x_batch[i] = np.pad(
                x_batch[i],
                pad_width=[
                    (0, 0),
                    (0, max_words_length - x_batch[i].shape[1]),
                ],
            )
    return np.array(x_batch)


def split_name(filename_):
    filename = copy.deepcopy(filename_).split("/")[-1]
    filename = ".".join(filename.split(".")[:-1])
    mode, patient, model_name, date = filename.split("___")
    return mode, patient, model_name, date


def get_words_filepath(data_filepath):
    return ".".join(data_filepath.split(".")[:-1]) + "_words.txt"


def calc_max_words_length(X):
    lenth_list = []
    for file_x in X:
        for x in file_x:
            lenth_list.append(x.shape[0])
    return int(np.percentile(lenth_list, 95))


def run_classification(regression_bench_model_name, patient, is_debug=False):
    assert hasattr(bench_models_regression, regression_bench_model_name)
    bench_regression_model = getattr(bench_models_regression, regression_bench_model_name)(patient)

    ecog_preprocessed_cache = []
    for filepath in patient["files_list"]:
        with h5py.File(filepath, "r+") as input_file:
            data = input_file["RawData"]["Samples"][()]

        ecog = data[:, patient["ecog_channels"]].astype("double")
        #         if is_debug:
        #             ecog = ecog[:DEBUG_DATA_LIMIT]
        x = bench_regression_model.preprocess_ecog(ecog, patient["sampling_rate"]).astype(
            "float32"
        )
        ecog_preprocessed_cache.append(x)
        if is_debug and len(ecog_preprocessed_cache) >= 2:
            break

    all_models_files = []

    for filename in os.listdir(MODEL_DUMPS_DIR):
        if not filename.endswith(".pth"):
            continue
        mode, patient_name, model_name, date = split_name(filename)
        if (
            mode == REGRESSION_MODE
            and model_name == regression_bench_model_name
            and patient_name == patient["name"]
        ):
            all_models_files.append(filename)

    for regression_model_filename in all_models_files:
        bench_regression_model = getattr(bench_models_regression, regression_bench_model_name)(
            patient
        )
        assert regression_bench_model_name in regression_model_filename
        print("Start File:", regression_model_filename)

        regression_model_file_path = f"{MODEL_DUMPS_DIR}/{regression_model_filename}"
        _, patient_name, _, date = split_name(regression_model_filename)
        assert patient_name == patient["name"]

        bench_regression_model.model.load_state_dict(torch.load(regression_model_file_path))

        X = []
        Y = []

        for index, filepath in enumerate(patient["files_list"]):
            with h5py.File(filepath, "r+") as input_file:
                data = input_file["RawData"]["Samples"][()]
            words_info = load_words_info(get_words_filepath(filepath))

            #             if is_debug:
            #                 debug_limit = 0
            #                 for start, end, word in words_info:
            #                     if start > DEBUG_DATA_LIMIT:
            #                         break
            #                     debug_limit += 1
            #                 words_info = words_info[:debug_limit]

            ecog = ecog_preprocessed_cache[index]

            x = predict_regression(bench_regression_model, ecog).astype("float32")
            x = sklearn.preprocessing.scale(x, copy=False)
            x_frames, classes = prepare_frames(
                x, words_info, bench_regression_model.downsampling_coef
            )
            x_frames = np.array(x_frames)
            classes = np.array(classes)

            assert x_frames.shape[0] == classes.shape[0]

            x_frames, classes = fix_class_imbalance(x_frames, classes)

            assert x_frames.shape[0] == classes.shape[0]

            X.append(x_frames)
            Y.append(classes)

            if is_debug and len(X) >= 2:
                break

        max_words_length = calc_max_words_length(X)
        bench_model = CLASSIFICATION_MODEL_CLASS(
            bench_regression_model.OUTPUT_SIZE,
            patient,
            regression_bench_model_name,
        )

        test_start_file_index = bench_model.TEST_START_FILE_INDEX if not is_debug else 1
        X_train = np.concatenate(X[:test_start_file_index], axis=0)
        Y_train = np.concatenate(Y[:test_start_file_index], axis=0)

        X_val = np.concatenate(X[test_start_file_index:], axis=0)
        Y_val = np.concatenate(Y[test_start_file_index:], axis=0)

        X_test = np.concatenate(X[test_start_file_index:], axis=0)
        Y_test = np.concatenate(Y[test_start_file_index:], axis=0)

        assert X_train.shape[0] == Y_train.shape[0]
        assert X_val.shape[0] == Y_val.shape[0]
        assert X_test.shape[0] == Y_test.shape[0]

        batch_size = bench_model.BATCH_SIZE

        train_generator = random_iterator(X_train, Y_train, batch_size)
        val_generator = random_iterator(X_val, Y_val, batch_size)
        test_generator = random_iterator(X_test, Y_test, batch_size)

        max_metric = -float("inf")
        model_filename = (
            f"{CLASSIFICATION_MODE}___{patient['name']}___{regression_bench_model_name}___{date}"
        )
        model_path = f"{MODEL_DUMPS_DIR}/{model_filename}.pth"
        max_iterations_count = MAX_ITERATIONS_COUNT if not is_debug else 1_000
        best_iteration = 0

        for iteration in range(max_iterations_count):
            process_batch(bench_model, train_generator, True, iteration, max_words_length)
            with torch.no_grad():
                metrics = process_batch(
                    bench_model,
                    val_generator,
                    False,
                    iteration,
                    max_words_length,
                )
                is_last_iteration = iteration == (max_iterations_count - 1)
                if iteration % 1000 == 0 or is_last_iteration:
                    smoothed_metric = bench_model.logger.get_smoothed_value("accuracy")
                    if smoothed_metric >= max_metric:
                        max_metric = smoothed_metric
                        best_iteration = iteration
                        torch.save(bench_model.model.state_dict(), model_path)
                    else:
                        assert iteration >= best_iteration
                        if (iteration - best_iteration) > EARLY_STOP_STEPS:
                            print(
                                f"Stopping model. Iteration {iteration} {round(smoothed_metric, 2)}. Best iteration {best_iteration} {round(max_metric, 2)}."
                            )
                            break

        bench_model.model.load_state_dict(torch.load(model_path))
        bench_model.model.eval()

        result = {}
        result["train_accuracy"] = float(
            accuracy_score(
                *get_random_predictions(
                    bench_model,
                    train_generator,
                    METRIC_ITERATIONS,
                    max_words_length,
                )
            )
        )
        result["val_accuracy"] = float(
            accuracy_score(
                *get_random_predictions(
                    bench_model,
                    val_generator,
                    METRIC_ITERATIONS,
                    max_words_length,
                )
            )
        )
        result["test_accuracy"] = float(
            accuracy_score(
                *get_random_predictions(
                    bench_model,
                    test_generator,
                    METRIC_ITERATIONS,
                    max_words_length,
                )
            )
        )
        result["train_logs"] = bench_model.logger.train_logs
        result["val_logs"] = bench_model.logger.test_logs
        result["iterations"] = iteration

        with open(f"{RESULTS_DIR}/{model_filename}.json", "w") as result_file:
            json.dump(result, result_file)
