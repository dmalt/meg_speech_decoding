from itertools import product

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import scipy.signal as scs  # type: ignore
import sklearn.preprocessing as skp  # type: ignore
import torch  # type: ignore
import tqdm  # type: ignore
from torch.autograd import Variable  # type: ignore

KOSTYL_CHOOSE = [0, 3, 1, 2]

NPERSEQ = 1000

COMPARISON_TOLERANCE = 1e-10
FREQ_XLIM = 250
N_BRANCHES = 4


WEIGHTS_COLOR = "k"
PATTENS_COLOR = "#1f77b4"
TRUE_COLOR = "#ff7f0e"
PATTERNS_NAIVE_COLOR = "#d62728"

WEIGHTS_MARKER = "."
PATTENS_MARKER = "v"
TRUE_MARKER = "o"
PATTERNS_NAIVE_MARKER = "P"

base = dict(linewidth=1.5, markevery=10)
params = dict(
    true=dict(label="True", marker="o", color="#ff7f0e").update(base),
    patterns=dict(label="Patterns", marker="v", color="#1f77b4").update(base),
    weights=dict(label="Weights", marker=".", color="k").update(base),
)

SR = 1000
# if kostyl_choose is not None:
#     # learned patterns can be permutated. here we manually need to fix it
#     self.convs_weights = self.convs_weights[kostyl_choose,:,:]
#     self.ica_weights = self.ica_weights[:,kostyl_choose]
#     self.ica_weights_scaled = self.ica_weights_scaled[:,kostyl_choose]
#     self.X_unmixed = self.X_unmixed[:, kostyl_choose]




def plot_temporal(model_interpreter, dataset):
    FINAL_FUGURE, FINAL_AXIS = plt.subplots(N_BRANCHES, 2)
    FINAL_FUGURE.set_figwidth(12)
    FINAL_FUGURE.set_figheight(6)
    plt.rc("font", family="serif", size=12)
    FINAL_FUGURE.tight_layout()
    for i in range(N_BRANCHES):
        plt.setp(FINAL_AXIS[i, 0], ylabel=f"Branch {i + 1}")
    plt.setp(FINAL_AXIS[3, 0], xlabel="Frequency, Hz")
    FINAL_AXIS[0, 0].set_title("Temporal Patterns")
    FINAL_AXIS[0, 1].set_title("Spatial Patterns")
    plt.rc("font", family="serif", size=10)

    freqs, ampletude, recovered = model_interpreter.get_temporal(
        dataset, sr, nperseg
    )
    for i, (unmixed_ch_id, convs_weights) in enumerate(anal_data):

        for y, p in zip((ampletude_true, recovered, ampletude), params):
            FINAL_AXIS[i, 0].plot(frequencies[:FREQ_XLIM], y[:FREQ_XLIM], **p)

        FINAL_AXIS[i, 0].grid()
    FINAL_AXIS[0, 0].legend(bbox_to_anchor=(1, -5.4), ncol=4)


interpret = np.cov(X_original.T) @ ica_weights_scaled
interpret_ossadtchi = get_ossagtchi_spatial_patterns(
    X_original, np.squeeze(convs_weights), ica_weights_scaled
)

print("Patters performance")
compare_weights(interpret)
print("Ossadtchi Patters performance")
compare_weights(interpret_ossadtchi)

METODS = ["True", "Patterns", "Patterns naive", "Weights"]
LS = [TRUE_MARKER, PATTENS_MARKER, PATTERNS_NAIVE_MARKER, WEIGHTS_MARKER]
LC = [TRUE_COLOR, PATTENS_COLOR, PATTERNS_NAIVE_COLOR, WEIGHTS_COLOR]
plt.rc("font", family="serif", size=12)
for i in range(0, sources_dimension):
    matrix = np.array(
        [
            mixing_matrix.T[:, i],
            interpret_ossadtchi[:, i],
            interpret[:, i],
            ica_weights[:, i],
        ]
    )
    lines = sklearn.preprocessing.minmax_scale(np.abs(matrix), axis=1)
    for line, label, line_style, line_color in list(
        zip(lines, METODS, LS, LC)
    ):
        if label == "Joint":
            continue
        FINAL_AXIS[i, 1].plot(
            line,
            label=label,
            linewidth=1 if label != "True" else 2,
            marker=line_style,
            color=line_color,
            markevery=1,
            markersize=None if label != "True" else 10,
        )
        FINAL_AXIS[i, 1].set_xticks(range(5))
plt.rc("font", family="serif", size=10)
FINAL_AXIS[0, 1].legend(bbox_to_anchor=(1, -5.4), ncol=4)
