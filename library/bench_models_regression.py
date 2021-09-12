import torch

import numpy as np

from .models_regression import SimpleNet, Encoder
from . import common_preprocessing
from torch.utils.tensorboard import SummaryWriter


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
    BATCH_SIZE = 100
    DOWNSAMPLING_COEF = 10
    LEARNING_RATE = 0.0003

    # ecog params
    HIGH_PASS_HZ = 10
    LOW_PASS_HZ = 200
    
    def __init__(self):
        self.model = SimpleNet(self.INPUT_SIZE, self.OUTPUT_SIZE, self.LAG_BACKWARD, self.LAG_FORWARD).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.logger = LearningLogStorer(SummaryWriter(comment=f"_{str(self.__class__.__name__)}"))

    def preprocess_ecog(self, ecog, sampling_rate):
        return common_preprocessing.classic_ecog_pipeline(ecog, sampling_rate, self.DOWNSAMPLING_COEF, self.LOW_PASS_HZ, self.HIGH_PASS_HZ)[:, self.SELECTED_CHANNELS]
    
    
class SimpleNetMelsBase(SimpleNetBase):
    # mels settings
    N_MELS = 80
    F_MAX = 8000
    OUTPUT_SIZE = N_MELS

    def preprocess_sound(self, sound, sampling_rate, ecog_size):
        return common_preprocessing.classic_melspectrogram_pipeline(sound, sampling_rate, self.DOWNSAMPLING_COEF, ecog_size, self.N_MELS, self.F_MAX)
    
    def detect_voice(self, y_batch):
        return np.sum(y_batch > 1, axis=1) > int(self.N_MELS * 0.25)


class SimpleNetBase_WithLSTM__CNANNELS_6_11__LAG_1500_0__80MELS(SimpleNetMelsBase):
    LAG_BACKWARD = 1500
    LAG_FORWARD = 0

    SELECTED_CHANNELS = [6, 7, 8, 9, 10, 11]
    INPUT_SIZE = len(SELECTED_CHANNELS)
    
    
class SimpleNetBase_WithLSTM__CNANNELS_6_10__LAG_1500_0__80MELS(SimpleNetMelsBase):
    LAG_BACKWARD = 1500
    LAG_FORWARD = 0

    SELECTED_CHANNELS = [6, 7, 8, 9, 10]
    INPUT_SIZE = len(SELECTED_CHANNELS)


class SimpleNetBase_WithLSTM__CNANNELS_6_11__LAG_1500_1500__80MELS(SimpleNetMelsBase):
    LAG_BACKWARD = 1500
    LAG_FORWARD = 1500

    SELECTED_CHANNELS = [6, 7, 8, 9, 10, 11]
    INPUT_SIZE = len(SELECTED_CHANNELS)


class SimpleNetBase_WithLSTM__CNANNELS_6_11__LAG_0_1500__80MELS(SimpleNetMelsBase):
    LAG_BACKWARD = 0
    LAG_FORWARD = 1500

    SELECTED_CHANNELS = [6, 7, 8, 9, 10, 11]
    INPUT_SIZE = len(SELECTED_CHANNELS)
    
    
class SimpleNetBase_WithLSTM__CNANNELS_0_5__LAG_1500_0__80MELS(SimpleNetMelsBase):
    LAG_BACKWARD = 1500
    LAG_FORWARD = 0

    SELECTED_CHANNELS = [0, 1, 2, 3, 4, 5]
    INPUT_SIZE = len(SELECTED_CHANNELS)
    
    
class SimpleNetBase_WithLSTM__CNANNELS_12_17__LAG_1500_0__80MELS(SimpleNetMelsBase):
    LAG_BACKWARD = 1500
    LAG_FORWARD = 0

    SELECTED_CHANNELS = [12, 13, 14, 15, 16, 17]
    INPUT_SIZE = len(SELECTED_CHANNELS)


class SimpleNetBase_WithLSTM__CNANNELS_18_23__LAG_1500_0__80MELS(SimpleNetMelsBase):
    LAG_BACKWARD = 1500
    LAG_FORWARD = 0

    SELECTED_CHANNELS = [18, 19, 20, 21, 22, 23]
    INPUT_SIZE = len(SELECTED_CHANNELS)
        

class SimpleNetBase_WithLSTM__CNANNELS_24_29__LAG_1500_0__80MELS(SimpleNetMelsBase):
    LAG_BACKWARD = 1500
    LAG_FORWARD = 0

    SELECTED_CHANNELS = [24, 25, 26, 27, 28, 29]
    INPUT_SIZE = len(SELECTED_CHANNELS)

    
############################################


class LargeNetBase(BenchModelRegressionBase):
    BATCH_SIZE = 100
    DOWNSAMPLING_COEF = 10
    LEARNING_RATE = 0.0003

    # ecog params
    HIGH_PASS_HZ = 10
    LOW_PASS_HZ = 200
    
    def __init__(self):
        ecog_frame = 96
        ecog_stride = 24
        self.model = Encoder(ecog_frame + self.LAG_BACKWARD + self.LAG_FORWARD, ecog_stride).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)
        self.logger = LearningLogStorer(SummaryWriter(comment=f"_{str(self.__class__.__name__)}"))

    def preprocess_ecog(self, ecog, sampling_rate):
        return common_preprocessing.classic_ecog_pipeline(ecog, sampling_rate, self.DOWNSAMPLING_COEF, self.LOW_PASS_HZ, self.HIGH_PASS_HZ)[:, self.SELECTED_CHANNELS]
    
    
class LargeNetMelsBase(LargeNetBase):
    # mels settings
    N_MELS = 80
    F_MAX = 8000
    OUTPUT_SIZE = N_MELS

    def preprocess_sound(self, sound, sampling_rate, ecog_size):
        return common_preprocessing.classic_melspectrogram_pipeline(sound, sampling_rate, self.DOWNSAMPLING_COEF, ecog_size, self.N_MELS, self.F_MAX)
    
    def detect_voice(self, y_batch):
        return np.sum(y_batch > 1, axis=1) > int(self.N_MELS * 0.25)
    
    
class LargeNet__CNANNELS_6_11__LAG_1500_1500__80MELS(LargeNetMelsBase):
    LAG_BACKWARD = 1500
    LAG_FORWARD = 1500

    SELECTED_CHANNELS = [6, 7, 8, 9, 10, 11]
    INPUT_SIZE = len(SELECTED_CHANNELS)
    