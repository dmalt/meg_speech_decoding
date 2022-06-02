from __future__ import annotations

from dataclasses import dataclass

from ..transformers import TargetTransformer
from ..type_aliases import Transformer


@dataclass
class Lags:
    backward: int
    forward: int


@dataclass
class Transformers:
    transform_x: Transformer
    transform_y: TargetTransformer
