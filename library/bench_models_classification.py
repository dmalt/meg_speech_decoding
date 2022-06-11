import torch
from torch.utils.tensorboard import SummaryWriter  # pyright: ignore

from . import models_classification
from .loggers import LearningLogStorerClasification
from .runner_common import WORDS_REMAP


class BenchModelClassification:
    def __init__(self, input_shape, output_shape, frequency):
        raise NotImplementedError

    def fit(self, X_train, Y_train, X_test, Y_test):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def slice_target(self, Y):
        return Y


class Mel2WordSimple(BenchModelClassification):
    OUTPUT_CLASSES = len(WORDS_REMAP)
    LEARNING_RATE = 0.0003
    BATCH_SIZE = 50

    def __init__(self, input_size, subject, regression_bench_model_name):
        self.input_size = input_size
        self.subject = subject
        self.TEST_START_FILE_INDEX = self.subject[
            "test_start_file_classification_index"
        ]
        self.regression_bench_model_name = regression_bench_model_name
        self.model = models_classification.Mel2WordSimple(
            self.input_size, self.OUTPUT_CLASSES
        )
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.LEARNING_RATE
        )
        self.logger = LearningLogStorerClasification(
            SummaryWriter(
                comment=f"___classification___{self.subject['name']}___{regression_bench_model_name}"
            )
        )
