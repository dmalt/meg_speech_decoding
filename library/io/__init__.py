from dataclasses import dataclass

from ..transformers import TargetTransformer
from ..type_aliases import Array, Array32, Info, Transformer, VoiceDetector


@dataclass
class Lags:
    backward: int
    forward: int


@dataclass
class Transformers:
    transform_x: Transformer
    transform_y: TargetTransformer
