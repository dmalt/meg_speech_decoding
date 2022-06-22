from typing import Generator, Iterable, TypeVar

import numpy as np  # type: ignore
import torch  # type: ignore

MODEL_DUMPS_DIR = "model_dumps"
RESULTS_DIR = "results"


REGRESSION_MODE = "regression"
CLASSIFICATION_MODE = "classification"

WORDS_REMAP = {
    "silent": 0,
    "женя": 1,
    "широко": 2,
    "шагает": 3,
    "желтых": 4,
    "штанах": 5,
    "шуру": 6,
    "ужалил": 7,
    "шершень": 8,
    "лара": 9,
    "ловко": 10,
    "крутит": 11,
    "руль": 12,
    "левой": 13,
    "рукой": 14,
    "лирику": 15,
    "любит": 16,
    "лиля": 17,
    "бабушка": 18,
    "боится": 19,
    "барабанов": 20,
    "белого": 21,
    "барана": 22,
    "больно": 23,
    "бодает": 24,
    "бешеный": 25,
    "бык": 26,
}


def data_generator(*pargs, **kwargs):
    # keep this temporarily till further refactoring
    pass


def get_random_predictions(model, generator, iterations):
    Y_batch = []
    Y_predicted = []
    for index, (x_batch, y_batch) in enumerate(generator):
        x_batch = torch.FloatTensor(x_batch)
        if torch.cuda.is_available():
            x_batch = x_batch.cuda()
        y_predicted = model(x_batch).cpu().detach().numpy()
        assert x_batch.shape[0] == y_predicted.shape[0]
        Y_predicted.append(y_predicted)
        Y_batch.append(y_batch)
        if index > iterations:
            break

    Y_predicted = np.concatenate(Y_predicted, axis=0)
    Y_batch = np.concatenate(Y_batch, axis=0)
    return Y_batch, Y_predicted


T = TypeVar("T", covariant=True)


def infinite(x: Iterable[T]) -> Generator[T, None, None]:
    while True:
        iterator = iter(x)
        for item in iterator:
            yield item
