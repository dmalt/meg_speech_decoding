from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional

from joblib import Memory  # type: ignore
from ndp.signal import Signal, T
from ndp.signal import pipelines as spp

from library.func_utils import log_execution_time

memory = Memory("/home/altukhov/Data/speech/cachedir", verbose=0)
log = logging.getLogger(__name__)


@dataclass
class MegPipeline:
    l_freq: Optional[float]
    h_freq: Optional[float]
    notch_freqs: list[float]
    selected_channels: Optional[list[int]]

    def __post_init__(self) -> None:
        time_decorator = log_execution_time(desc="transforming MEG data")
        self.transform = memory.cache(time_decorator(spp.preprocess_meg))
        # self.transform = (time_decorator(spp.preprocess_meg))

    def __call__(self, signal: Signal[T]) -> Signal[T]:
        return self.transform(signal, **asdict(self))


@dataclass
class EcogPipeline:
    dsamp_coef: int
    highpass: Optional[float]
    lowpass: Optional[float]
    notch_narrow_freqs: list[float]
    notch_wide_freqs: list[float]
    selected_channels: Optional[list[int]] = None

    def __post_init__(self) -> None:
        time_decorator = log_execution_time(desc="transforming ECoG data")
        self.transform = memory.cache(time_decorator(spp.preprocess_ecog))

    def __call__(self, ecog: Signal[T]) -> Signal[T]:
        return self.transform(ecog, **asdict(self))


@dataclass
class MelspectrogramPipeline:
    dsamp_coef: int
    n_mels: int
    f_max: float

    def __post_init__(self) -> None:
        time_decorator = log_execution_time(desc="melspectrogram computation")
        self.transform = memory.cache(time_decorator(spp.melspectrogram_pipeline))

    def __call__(self, signal: Signal[T]) -> Signal[T]:
        return self.transform(signal, **asdict(self))
