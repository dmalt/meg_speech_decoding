from . import models_classification
from .runner_common import WORDS_REMAP
from .loggers import LearningLogStorerClasification

from torch.utils.tensorboard import SummaryWriter

import torch

class BenchModelClaffification:
    def __init__(self, input_shape, output_shape, frequency):
        raise NotImplementedError

    def fit(self, X_train, Y_train, X_test, Y_test):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def slice_target(self, Y):
        return Y
    
class Mel2WordSimple__80MELS(BenchModelClaffification):
    INPUT_SIZE = 80
    OUTPUT_CLASSES = len(WORDS_REMAP)
    LEARNING_RATE = 0.0003
    BATCH_SIZE = 50

    def __init__(self):
        self.model = models_classification.Mel2WordSimple(self.INPUT_SIZE, self.OUTPUT_CLASSES).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.logger = LearningLogStorerClasification(SummaryWriter(comment=f"_{str(self.__class__.__name__)}"))
