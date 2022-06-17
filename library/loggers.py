from __future__ import annotations

from typing import Any

import numpy as np


class LearningLogStorerBase:
    train_logs: dict
    test_logs: dict
    tensorboard_writer: Any

    def add_value(self, name: str, is_train: bool, value: np.floating, iteration: int) -> None:
        train_or_test = "train" if is_train else "test"
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(f"{train_or_test}/{name}", value, iteration)
        if is_train:
            self.train_logs[name].append((iteration, value))
        else:
            self.test_logs[name].append((iteration, value))
        return

    def get_smoothed_value(self, name: str, window: int = 100) -> np.floating:
        return np.mean([value for _, value in self.test_logs[name][-window:]])


class LearningLogStorer(LearningLogStorerBase):
    def __init__(self, tensorboard_writer: Any | None = None):
        self.tensorboard_writer = tensorboard_writer
        self.train_logs = {
            "loss": [],
            "correlation": [],
            "correlation_speech": [],
        }
        self.test_logs = {
            "loss": [],
            "correlation": [],
            "correlation_speech": [],
        }


class LearningLogStorerClasification(LearningLogStorerBase):
    def __init__(self, tensorboard_writer: Any | None = None):
        self.tensorboard_writer = tensorboard_writer
        self.train_logs = {
            "loss": [],
            "accuracy": [],
            "accuracy (without silent class)": [],
        }
        self.test_logs = {
            "loss": [],
            "accuracy": [],
            "accuracy (without silent class)": [],
        }
