import torch

import numpy as np

from .models_regression import SimpleNet
from . import common_preprocessing


class LearningLogStorer:
    def __init__(self, tensorboard_writer=None):
        self.tensorboard_writer = tensorboard_writer
        self.train_logs = {"loss": [], "correlation": [], "correlation_speech": []}
        self.test_logs = {"loss": [], "correlation": [], "correlation_speech": []}

    def add_value(self, name, is_train, value, iteration):
        train_or_test = "train" if is_train else "test"
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(f'{train_or_test}/{name}', value, iteration)
        if is_train:
            self.train_logs[name].append((iteration, value))
        else:
            self.test_logs[name].append((iteration, value))
        return

        
class BenchModelRegressionBase:
    #LAG_BACKWARD
    #LAG_FORWARD
    #LEARNING_RATE
    #DOWNSAMPLING_COEF
    #BATCH_SIZE
    #SELECTED_CHANNELS

    def __init__(self):
        # model
        # optimizer
        # logger
        raise NotImplementedError
        
    def preprocess_ecog(self, ecog):
        raise NotImplementedError

    def preprocess_sound(self, sound):
        raise NotImplementedError
        
    def detect_voice(self, y_batch):
        raise NotImplementedError


class SimpleNetBase(BenchModelRegressionBase):
    def __init__(self):
        self.model = SimpleNet(self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAG_BACKWARD, self.LAG_FORWARD).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        #SummaryWriter(comment=f"{str(self.__class__.__name__)}")
        self.logger = LearningLogStorer()

    def preprocess_ecog(self, ecog, sampling_rate):
        return common_preprocessing.classic_ecog_pipeline(ecog, sampling_rate, self.DOWNSAMPLING_COEF, self.LOW_PASS_HZ, self.HIGH_PASS_HZ)[:, self.SELECTED_CHANNELS]


class SimpleNetBase_WithLSTM__CNANNELS_6_11__LAG_1500_0__80MELS(SimpleNetBase):
    LAG_BACKWARD = 1500
    LAG_FORWARD = 0
    LEARNING_RATE = 0.0003
    DOWNSAMPLING_COEF = 10
    BATCH_SIZE = 100
    SELECTED_CHANNELS = [6, 7, 8, 9, 10, 11]
    INPUT_SIZE = len(SELECTED_CHANNELS)
    
    # ecog params
    HIGH_PASS_HZ = 20
    LOW_PASS_HZ = 200

    # mels settings
    N_MELS = 80
    F_MAX = 8000
    OUTPUT_SIZE = N_MELS
    
    def __init__(self,):
        super(self.__class__, self).__init__()

    def preprocess_sound(self, sound, sampling_rate, ecog_size):
        return common_preprocessing.classic_melspectrogram_pipeline(sound, sampling_rate, self.DOWNSAMPLING_COEF, ecog_size, self.N_MELS, self.F_MAX)
    
    def detect_voice(self, y_batch):
        return np.sum(y_batch > 1, axis=1) > int(self.N_MELS * 0.25)

    
    
    
    
