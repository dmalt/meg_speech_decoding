from __future__ import annotations

from typing import Callable, Optional

import matplotlib  # type: ignore
import matplotlib.figure as mpl_fig  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import mne  # type: ignore
import numpy as np  # type: ignore
import numpy.typing as npt
import sklearn.preprocessing as skp  # type: ignore
from ndp.signal import Signal  # type: ignore

from library.func_utils import log_execution_time
from library.interpreter import ModelInterpreter
from library.models import SimpleNet  # type: ignore

matplotlib.use("TkAgg")

PLOT_CFG = dict(
    weights=dict(color="k", marker="."),
    patterns=dict(color="#1f77b4", marker="v"),
    true=dict(color="#ff7f0e", marker="o"),
    naive=dict(color="#d62728", marker="P"),
)


def plot_temporal_as_line(ax, f, y, style):
    ax.plot(f, y, markevery=10, label=style, **PLOT_CFG[style])


def plot_spatial_as_line(ax, t, style):
    tt = skp.minmax_scale(np.abs(t))
    ax.plot(tt, label=style, **PLOT_CFG[style])
    ax.set_xticks(range(5))


class MneInfoWithLayout(mne.Info):
    pass


class TopoVisualizer:
    def __init__(self, info: MneInfoWithLayout):
        self.info = info

    def __call__(self, ax, t, style: str | None = None):
        tt = skp.minmax_scale(np.abs(t))

        if len(tt) == 306:
            meg_idx = mne.pick_types(self.info, meg="mag")  # pyright: ignore
            info = mne.pick_info(self.info, sel=meg_idx)
            tt = (tt[::3] + tt[1::3] + tt[2::3]) / 3
            mne.viz.plot_topomap(tt, info, axes=ax, show=False)
        elif len(tt) == 102:
            meg_idx = mne.pick_types(self.info, meg="mag")  # pyright: ignore
            info = mne.pick_info(self.info, sel=meg_idx)
            mne.viz.plot_topomap(tt, info, axes=ax, show=False)


class InterpretPlotLayout:
    FREQ_XLIM = 150
    LEGEND_CFG = dict(
        bbox_to_anchor=(0, -4.1, 1, -4.1),
        loc="lower left",
        ncol=20,
        mode="expand",
        borderaxespad=0.0,
    )
    XLABEL = "Frequency, Hz"
    YLABEL = "Branch {i_branch}"
    FIGWIDTH = 12
    FIGHEIGHT = 6
    TEMPORAL_TITLE = "Temporal Patterns"
    SPATIAL_TITLE = "Spatial Patterns"

    def __init__(self, n_branches: int, plot_spatial: Callable, plot_temporal: Callable):
        assert n_branches > 0
        self.n_branches = n_branches
        self.fig, self.ax = plt.subplots(n_branches, 2)
        self.plot_spatial_single = plot_spatial
        self.plot_temporal_single = plot_temporal
        self.set_figure()

    def set_figure(self):
        self.fig.set_figwidth(self.FIGWIDTH)
        self.fig.set_figheight(self.FIGHEIGHT)
        plt.rc("font", family="serif", size=12)
        for i in range(self.n_branches):
            plt.setp(self.ax[i, 0], ylabel=self.YLABEL.format(i_branch=i + 1))
        plt.setp(self.ax[self.n_branches - 1, 0], xlabel=self.XLABEL)
        self.ax[0, 0].set_title(self.TEMPORAL_TITLE)
        self.ax[0, 1].set_title(self.SPATIAL_TITLE)
        plt.rc("font", family="serif", size=10)

    def add_temporal(self, f: np.ndarray, signals: list[np.ndarray], style: str) -> None:
        for i, y in enumerate(signals):
            self.plot_temporal_single(
                self.ax[i, 0], f[: self.FREQ_XLIM], y[: self.FREQ_XLIM], style
            )

    def add_spatial(self, data: list[np.ndarray], style: str) -> None:
        """
        Parameters
        ----------
        data: array of shape()
        """
        for i, tt in enumerate(data):
            self.plot_spatial_single(self.ax[i, 1], tt, style)

    def finalize(self):
        for i in range(self.n_branches):
            self.ax[i, 0].grid()
        self.ax[0, 0].legend(**self.LEGEND_CFG)
        # self.ax[0, 1].legend(**self.LEGEND_CFG)


class ContinuousDatasetPlotter:
    def __init__(self, X: Signal, Y: Signal):
        assert len(X) == len(Y) and X.sr == Y.sr
        n_channels, n_features = X.n_channels, Y.n_channels
        self.ch_inds = list(range(n_channels))
        self.feat_inds = list(range(n_channels, n_channels + n_features))
        ch_names_data = [f"channel {i + 1}" for i in range(n_channels)]
        ch_names_features = [f"feature {j + 1}" for j in range(n_features)]
        ch_names = ch_names_data + ch_names_features
        info = mne.create_info(sfreq=X.sr, ch_names=ch_names)
        data = np.concatenate((np.asarray(X).T, np.asarray(Y).T), axis=0)
        self.raw = mne.io.RawArray(data, info)

        onset, duration, description = [], [], []
        for a in X.annotations:
            onset.append(a.onset)
            duration.append(a.duration)
            description.append(a.type)
        self.raw.set_annotations(mne.Annotations(onset, duration, description))

    def plot(self, highpass: Optional[float] = None, lowpass: Optional[float] = None) -> None:
        if highpass or lowpass:
            raw = self.raw.copy().filter(l_freq=highpass, h_freq=lowpass, picks=self.ch_inds)
        else:
            raw = self.raw
        raw.plot(block=True, highpass=highpass, lowpass=lowpass)


@log_execution_time()
def get_model_weights_figure(
    model: SimpleNet, X: Signal[npt._32Bit], mne_info: MneInfoWithLayout, n_branches: int
) -> mpl_fig.Figure:
    mi = ModelInterpreter(model, X)
    freqs, weights, patterns = mi.get_temporal(nperseg=1000)
    sp = mi.get_spatial_patterns()
    sp_naive = mi.get_naive()
    plot_topo = TopoVisualizer(mne_info)
    pp = InterpretPlotLayout(n_branches, plot_topo, plot_temporal_as_line)

    pp.FREQ_XLIM = 150
    pp.add_temporal(freqs, weights, "weights")
    pp.add_temporal(freqs, patterns, "patterns")
    pp.add_spatial(sp, "patterns")
    pp.add_spatial(sp_naive, "naive")
    pp.finalize()
    plt.switch_backend("agg")
    return pp.fig
