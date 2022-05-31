from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EcogPatientConfig:
    sampling_rate: float
    files_list: list[str]
    ecog_channels: list[int]
    sound_channel: int


@dataclass
class MegPatientConfig:
    raw_path: str
    audio_path: str
    annotations_path: Optional[str]
