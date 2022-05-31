from __future__ import annotations

import logging

import h5py  # type: ignore
import librosa as lb  # type: ignore
import mne  # type: ignore
import numpy as np

from .config_schema import MegPatientConfig
from .type_aliases import Array, Array32

log = logging.getLogger(__name__)


def read_ecog(h5_path: str, ecog_chs: list[int], sound_ch: int) -> tuple[Array, Array]:
    """Read ecog and audio signal from h5 file"""
    with h5py.File(h5_path, "r+") as input_file:
        data = input_file["RawData"]["Samples"][()]
    ecog = data[:, ecog_chs].astype("double")
    sound = data[:, sound_ch].astype("double")
    return ecog, sound


def read_meg(patient: MegPatientConfig) -> tuple[Array32, float, Array32, float, mne.Info]:
    raw = mne.io.read_raw_fif(patient.raw_path, verbose="ERROR", preload=True)
    audio, audio_sr = lb.load(patient.audio_path, sr=None)
    assert abs(len(raw.times) / raw.info["sfreq"] - len(audio) / audio_sr) < 1
    info_audio = mne.create_info(["audio"], sfreq=audio_sr)
    audio_raw = mne.io.RawArray(audio[np.newaxis, :], info_audio)
    if patient.annotations_path is not None:
        log.debug(f"{patient.annotations_path=}")
        annots = mne.read_annotations(patient.annotations_path)
        raw.set_annotations(annots)
    _copy_annotations(raw, audio_raw)
    X = raw.get_data(picks="meg", reject_by_annotation="omit").T
    Y = np.squeeze(audio_raw.get_data(picks="audio", reject_by_annotation="omit"))
    log.debug(f"{len(X) / raw.info['sfreq']=:.3f}, {len(Y) / audio_sr=:.3f}")
    # Conversion to float32 for audio is crucical since float64 requires
    # too much memory when processing
    return X.astype("float32"), raw.info["sfreq"], Y.astype("float32"), audio_sr, raw.info


def read_meg_old(patient: MegPatientConfig):
    raw = mne.io.read_raw_fif(patient.raw_path, verbose="ERROR", preload=True)
    audio, audio_sr = lb.load(patient.audio_path, sr=None)
    # raw.filter(l_freq=10, h_freq=120)
    # raw.notch_filter([50, 100, 150])
    X = raw.get_data(picks="meg", reject_by_annotation="omit").T
    Y = audio
    log.info(f"{len(X) / raw.info['sfreq']=:.3f}, {len(Y) / audio_sr=:.3f}")
    # Conversion to float32 for audio is crucical since float64 requires
    # too much memory when processing
    return X.astype("float32"), raw.info["sfreq"], Y.astype("float32"), audio_sr, raw.info


def _copy_annotations(raw_from: mne.io.BaseRaw, raw_to: mne.io.BaseRaw) -> None:
    sr_from, sr_to = raw_from.info["sfreq"], raw_to.info["sfreq"]
    raw_to.set_meas_date(raw_from.info["meas_date"])
    # hack to set annotations when orig times of annotations differ;
    # see mne.Annotations.orig_time for more details. Note that data lengths (in seconds)
    # in raw_from and raw_to must be the same, otherwise raw_to.get_data will crash
    raw_to._first_samps = (raw_from._first_samps / sr_from * sr_to).astype(int)
    raw_to._last_samps = (raw_from._last_samps / sr_from * sr_to).astype(int)
    raw_to.set_annotations(raw_from.annotations)


def read_fif(fif_path: str) -> tuple[Array, mne.Info]:
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


def read_meg_chunks(patient):
    raw = mne.io.read_raw_fif(patient.raw_path, verbose="ERROR", preload=True)
    if patient.annotations_path is not None:
        annots = mne.read_annotations(patient.annotations_path)
        raw.set_annotations(annots)
    audio, audio_sr = lb.load(patient.audio_path, sr=None)
    X = raw.get_data(picks="meg").astype("float32").T
    onsets = map(lambda x: x - raw.first_samp / raw.info["sfreq"], raw.annotations.onset)
    annotations = sorted(zip(onsets, raw.annotations.duration))
    return (X, raw.info["sfreq"], audio, audio_sr, raw.info, annotations)
