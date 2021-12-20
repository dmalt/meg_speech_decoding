import numpy as np

class LearningLogStorerBase:
    def add_value(self, name, is_train, value, iteration):
        train_or_test = "train" if is_train else "test"
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(f'{train_or_test}/{name}', value, iteration)
        if is_train:
            self.train_logs[name].append((iteration, value))
        else:
            self.test_logs[name].append((iteration, value))
        return
    
    def get_smoothed_value(self, name, window=100):
        return np.mean([value for iteration, value in self.test_logs[name][-window:]])
            


class LearningLogStorer(LearningLogStorerBase):
    def __init__(self, tensorboard_writer=None):
        self.tensorboard_writer = tensorboard_writer
        self.train_logs = {"loss": [], "correlation": [], "correlation_speech": []}
        self.test_logs = {"loss": [], "correlation": [], "correlation_speech": []}
        
        
class LearningLogStorerClasification(LearningLogStorerBase):
    def __init__(self, tensorboard_writer=None):
        self.tensorboard_writer = tensorboard_writer
        self.train_logs = {"loss": [], "accuracy": [], "accuracy (without silent class)": []}
        self.test_logs = {"loss": [], "accuracy": [], "accuracy (without silent class)": []}