from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import torch  # type: ignore
from hydra import compose, initialize  # type: ignore
from hydra.utils import instantiate  # type: ignore

import setup_utils  # type: ignore # noqa
from library.interpreter import ModelInterpreter
from library.visualize import (
    ContinuousDatasetPlotter,
    InterpretPlotLayout,
    TopoVisualizer,
    plot_spatial_as_line,
    plot_temporal_as_line,
)

setup_utils.setup_hydra()

# MODEL_DATE = "2022-05-09"
# MODEL_TIME = "13-47-04"
MODEL_DATE = "2022-05-19"
# MODEL_TIME = "18-42-04"
# MODEL_TIME = "23-17-30"
MODEL_TIME = "23-21-29"
OUTPUTS_DIR = Path(f"./outputs/{MODEL_DATE}/{MODEL_TIME}/")
CONFIG_PATH = OUTPUTS_DIR / ".hydra"
MODEL_PATH = OUTPUTS_DIR / "model_dumps/BenchModelRegressionBase.pth"

initialize(config_path=str(CONFIG_PATH), job_name="interpreter")
cfg = compose(config_name="config")

model = instantiate(cfg.model.regression)
model.load_state_dict(torch.load(MODEL_PATH))
dataset = instantiate(cfg.dataset)

plotter = ContinuousDatasetPlotter(dataset)
plotter.plot(highpass=50)
train, test = dataset.train_test_split(cfg.runner.train_test_ratio)
mi = ModelInterpreter(model, train, cfg.runner.batch_size)
f, a, p = mi.get_temporal(nperseg=1000)
sp = mi.get_spatial_patterns()
sp_naive = mi.get_naive()


# mm = train.info["mixing_matrix"].T
# mm = mm[:, [0, 3, 1, 2]]  # dirty fix to align true order with computed

plot_topo = TopoVisualizer(dataset.info["mne_info"])
# pp = InterpretPlotLayout(4, plot_spatial_as_line, plot_temporal_as_line)
pp = InterpretPlotLayout(4, plot_topo, plot_temporal_as_line)
pp.FREQ_XLIM = 150
pp.add_temporal(f, a, "weights")
pp.add_temporal(f, p, "patterns")
pp.add_spatial(sp, "patterns")
pp.add_spatial(sp_naive, "naive")
# pp.add_spatial(mm, "true")
pp.finalize()
plt.show()
