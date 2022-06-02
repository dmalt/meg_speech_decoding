from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import librosa as lb  # type: ignore
import mne  # type: ignore
import numpy as np

from ..datasets import Composite, Continuous
from ..signal_processing import align_samples
from ..type_aliases import Array32, SignalArray32, VoiceDetector
from . import Lags, Transformers

log = logging.getLogger(__name__)


@dataclass
class PatientConfig:
    raw_path: str
    audio_path: str
    annotations_path: Optional[str]


@dataclass
class Info:
    mne_info: mne.Info


ContinuousDatasetPackage = Tuple[Continuous, VoiceDetector, Info]
CompositeDatasetPackage = Tuple[Composite, VoiceDetector, Info]
Annotations = List[Tuple[float, float]]


def read(patient: PatientConfig, lags: Lags, t: Transformers) -> ContinuousDatasetPackage:
    """Generate Continuous instance for meg"""
    X, Y, Y_sr, mne_info = _read(patient)
    X, new_sr = t.transform_x(X, mne_info["sfreq"])
    info = Info(mne_info)
    log.debug(f"{Y.dtype=}")
    Y, new_sound_sr = t.transform_y(Y, Y_sr)  # , len(X))
    log.info("Finished transforming target")
    Y = align_samples(Y, new_sound_sr, X, new_sr)
    assert len(X) == len(Y), f"{len(X)=} != {len(Y)=}"
    detect_voice = t.transform_y.detect_voice

    return Continuous(X, Y, lags.backward, lags.forward, new_sr), detect_voice, info


def read_chunks(patient: PatientConfig, lags: Lags, t: Transformers) -> CompositeDatasetPackage:
    X, Y, Y_sr, mne_info, annotations = _read_chunks(patient)
    log.debug(f"{len(X) / mne_info['sfreq']=:.2f}, {len(Y) / Y_sr=:.2f}")
    X, new_sr = t.transform_x(X, mne_info["sfreq"])
    info = Info(mne_info)
    log.debug(f"{Y.dtype=}")
    Y, new_sound_sr = t.transform_y(Y, Y_sr)  # , len(X))
    dv = t.transform_y.detect_voice
    log.debug("Finished transforming target")
    Y = align_samples(Y, new_sound_sr, X, new_sr)
    datasets: list[Continuous] = []
    for s in _get_good_slices(new_sr, annotations):
        log.debug(f"{s=}")
        X_slice, Y_slice = X[s, :], Y[s, :]
        if len(X_slice) < lags.backward + lags.forward + 1:
            continue
        datasets.append(Continuous(X_slice, Y_slice, lags.backward, lags.forward, new_sr))
    res = Composite(new_sr, datasets)
    log.debug(f"{len(res)=}")
    return res, dv, info


def _read(patient: PatientConfig) -> tuple[SignalArray32, SignalArray32, float, mne.Info]:
    raw = mne.io.read_raw_fif(patient.raw_path, verbose="ERROR", preload=True)
    audio, audio_sr = lb.load(patient.audio_path, sr=None)  # type: ignore
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
    return X.astype("float32"), Y.astype("float32"), audio_sr, raw.info


def _copy_annotations(raw_from: mne.io.BaseRaw, raw_to: mne.io.BaseRaw) -> None:
    sr_from, sr_to = raw_from.info["sfreq"], raw_to.info["sfreq"]
    raw_to.set_meas_date(raw_from.info["meas_date"])
    # hack to set annotations when orig times of annotations differ;
    # see mne.Annotations.orig_time for more details. Note that data lengths (in seconds)
    # in raw_from and raw_to must be the same, otherwise raw_to.get_data will crash
    raw_to._first_samps = (raw_from._first_samps / sr_from * sr_to).astype(int)
    raw_to._last_samps = (raw_from._last_samps / sr_from * sr_to).astype(int)
    raw_to.set_annotations(raw_from.annotations)


def _read_chunks(
    patient: PatientConfig,
) -> tuple[SignalArray32, Array32, float, mne.Info, list[tuple[Any, Any]]]:
    raw = mne.io.read_raw_fif(patient.raw_path, verbose="ERROR", preload=True)
    if patient.annotations_path is not None:
        annots = mne.read_annotations(patient.annotations_path)
        raw.set_annotations(annots)
    X: SignalArray32 = raw.get_data(picks="meg").astype("float32").T
    audio, audio_sr = lb.load(patient.audio_path, sr=None)  # type: ignore
    onsets = map(lambda x: x - raw.first_samp / raw.info["sfreq"], raw.annotations.onset)
    annotations = sorted(zip(onsets, raw.annotations.duration))
    return X, audio, audio_sr, cast(mne.Info, raw.info), annotations


def _get_good_slices(sr: float, annotations: Annotations) -> list[slice]:
    """Get slices for periods not annotated as bads"""
    events = []
    # when start and end occur simoultaneosly, first add new bad segment, then remove old
    START = 0
    END = 1
    for onset, duration in annotations:
        events.append((int(onset * sr), START))
        events.append((int((onset + duration) * sr), END))
    events.sort()
    bad_overlap_cnt = 0

    res = []
    last_start = 0
    for ev in events:
        if ev[1] == START:
            if bad_overlap_cnt == 0:
                res.append(slice(last_start, ev[0]))
            bad_overlap_cnt += 1
        elif ev[1] == END:
            bad_overlap_cnt -= 1
            if bad_overlap_cnt == 0:
                last_start = ev[0]
    # process last event
    if bad_overlap_cnt == 0:
        res.append(slice(last_start, None))
    return res
