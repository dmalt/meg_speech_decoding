from __future__ import annotations

import matplotlib.pyplot as plt  # type: ignore
import mne  # type: ignore
import numpy as np  # type: ignore
import sklearn.preprocessing as skp  # type: ignore

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


class TopoVisualizer:
    def __init__(self, info):
        self.info = info

    def __call__(self, ax, t, style=None):
        tt = skp.minmax_scale(np.abs(t))
        print(type(ax))

        if len(tt) == 306:
            meg_idx = mne.pick_types(self.info, meg="mag")
            info = mne.pick_info(self.info, sel=meg_idx)
            tt = (tt[::3] + tt[1::3] + tt[2::3]) / 3
            mne.viz.plot_topomap(tt, info, axes=ax, show=False)
        elif len(tt) == 102:
            meg_idx = mne.pick_types(self.info, meg="mag")
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

    def __init__(self, n_branches, plot_spatial, plot_temporal):
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
        plt.setp(self.ax[i, 0], xlabel=self.XLABEL)
        self.ax[0, 0].set_title(self.TEMPORAL_TITLE)
        self.ax[0, 1].set_title(self.SPATIAL_TITLE)
        plt.rc("font", family="serif", size=10)

    def add_temporal(
        self, f: np.ndarray, signals: list[np.ndarray], style: str
    ) -> None:
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
