from __future__ import annotations

import logging

import h5py  # type: ignore
# import hydra  # type: ignore  # noqa
import librosa as lb  # type: ignore
import mne  # type: ignore

from .type_aliases import Array

log = logging.getLogger(__name__)


def read_ecog(h5_path: str, ecog_chs: list[int], sound_ch: int) -> tuple[Array, Array]:
    """Read ecog and audio signal from h5 file"""
    with h5py.File(h5_path, "r+") as input_file:
        data = input_file["RawData"]["Samples"][()]
    ecog = data[:, ecog_chs].astype("double")
    sound = data[:, sound_ch].astype("double")
    return ecog, sound


def read_meg(fif_path: str) -> tuple[Array, mne.Info]:
    """Read raw data in .fif format"""
    raw = mne.io.read_raw_fif(fif_path, verbose="ERROR", preload=True)
    raw.pick_types(meg=True)
    X = raw.get_data(reject_by_annotation="omit").T
    log.debug(raw.info)
    log.debug(f"Data length={len(X) / raw.info['sfreq']} sec")
    return X, raw.info


def read_audio(audio_path: str) -> tuple[Array, float]:
    """Read audio in .wav format"""
    sound, sound_sr = lb.load(audio_path, sr=None)
    log.debug(f"Sound length={len(sound) / sound_sr:.2f} sec, {sound_sr=}")
    return sound, sound_sr
