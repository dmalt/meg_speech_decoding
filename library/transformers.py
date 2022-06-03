from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from functools import wraps
from typing import Callable, Optional, Protocol

import numpy as np
import numpy.typing as npt
import sklearn.preprocessing as skp  # type: ignore
from joblib import Memory  # type: ignore

from . import signal_processing as sp
from .type_aliases import Array, Array32, Signal32, as_32

memory = Memory("/home/altukhov/Data/speech/cachedir", verbose=0)
log = logging.getLogger(__name__)


class Transformer(Protocol):
    def __call__(self, sound: Array, sr: float) -> tuple[Array32, float]:
        ...


class TargetTransformer(Transformer, Protocol):
    def detect_voice(self, y_batch: Array) -> Optional[npt.NDArray[np.bool_]]:
        ...


@dataclass
class AlignedData:
    """Timeseries data with X and Y of equal length and sampling rate"""
    X: Signal32
    Y: Signal32

    def __post_init__(self):
        assert len(self.X.data) == len(self.Y.data)
        assert self.X.sr == self.Y.sr


@dataclass
class DataTransformer:
    transform_x: Callable[[Signal32], Signal32]
    transform_y: Callable[[Signal32], Signal32]

    def transform(self, X: Signal32, Y: Signal32) -> AlignedData:
        X = self.transform_x(X)
        Y = self.transform_y(Y)
        Y = as_32(sp.align_samples(Y, X))
        return AlignedData(X, Y)


@dataclass
class MegPipeline:
    highpass: float
    lowpass: float
    notch_freqs: list[float]
    selected_channels: Optional[list[int]] = None

    def __post_init__(self):
        time_decorator = Timer(type(self).__name__, "Transforming data", log)
        self.transform = memory.cache(time_decorator(sp.preprocess_meg))

        self.lowpass = None if self.lowpass == "None" else self.lowpass
        self.highpass = None if self.highpass == "None" else self.highpass

    def __call__(self, signal: Array, sr: float) -> tuple[Array32, float]:
        return self.transform(signal, sr, **asdict(self))


class Scaler:
    def __call__(self, signal: Array, sr: float) -> tuple[Array32, float]:
        return skp.scale(signal.astype("float64")).astype("float32"), sr

    def detect_voice(self, y_batch: Array) -> None:
        return None


@dataclass
class EcogPipeline:
    dsamp_coef: int
    highpass: Optional[float]
    lowpass: Optional[float]
    notch_narrow_freqs: list[float]
    notch_wide_freqs: list[float]
    selected_channels: Optional[list[int]] = None

    def __post_init__(self):
        time_decorator = Timer(type(self).__name__, "Transforming data", log)
        self.transform = memory.cache(time_decorator(sp.preprocess_ecog))

        self.lowpass = None if self.lowpass == "None" else self.lowpass
        self.highpass = None if self.highpass == "None" else self.highpass

    def __call__(self, ecog: Array, sr: float) -> tuple[Array32, float]:
        return self.transform(ecog, sr, **asdict(self))


class Timer:
    def __init__(self, caller, message, logger):
        self.caller = caller
        self.message = message
        self.logger = logger

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.logger.info(self.caller + ": " + self.message)
            t1 = time.perf_counter()
            res = func(*args, **kwargs)
            self.logger.info(f"{self.caller}: Done in {time.perf_counter() - t1:.2f} sec")
            return res

        return wrapper


@dataclass
class MelspectrogramPipeline:
    dsamp_coef: int
    n_mels: int
    f_max: float

    def __post_init__(self):
        time_decorator = Timer(type(self).__name__, "Transforming data", log)
        self.transform = memory.cache(time_decorator(sp.melspectrogram_pipeline))

    def __call__(self, signal: Array, sr: float) -> tuple[Array32, float]:
        return self.transform(signal, sr, self.n_mels, self.f_max, self.dsamp_coef)

    def detect_voice(self, y_batch: Array) -> npt.NDArray[np.bool_]:
        res: npt.NDArray[np.bool_] = np.sum(y_batch > 1, axis=1) > int(self.n_mels * 0.25)
        return res
