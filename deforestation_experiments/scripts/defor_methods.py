"""The deployed deforestation risk model's settings, the windows, and Verra's classes for alternative models.

The deployed functions are imported unchanged from landuse_tool.deforestation_tool (drivers: the random
forest, its training sample and importances); nothing in landuse_tool is changed here.

A split is (T1, T2, T3): the model is calibrated on the loss of T1+1..T2 with predictors at T1, and
tested on the loss of T2+1..T3 over the forest standing at T2 (VT0007's calibration and confirmation
periods).
"""

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import defor_data as DD                                                   # noqa: E402

if str(DD.TOOL) not in sys.path:
    sys.path.insert(0, str(DD.TOOL))
logging.getLogger("landuse_tool").setLevel(logging.ERROR)
from landuse_tool.deforestation_tool import drivers as DR                 # noqa: E402

THRESHOLD = 30                                   # % canopy, deployed default and the team's choice
SAMPLES = 5000                                   # per class, deployed default
# Hansen from 2013 only (team decision 2026-10-22), three-year windows
SPLITS = [(2013, 2016, 2019), (2016, 2019, 2022), (2018, 2021, 2024)]
RUNS = DD.ROOT / "outputs" / "runs"


def split_label(split):
    t1, t2, t3 = split
    return f"{t1 + 1}–{t2} → {t2 + 1}–{t3}"


# The deployed model: its predictors, and the year its hold-out reads each yearly layer at
DEPLOYED_PREDICTORS = ["Aspect", "Elevation", "Points of interest", "Population density", "Precipitation",
                       "Distance to railways", "Distance to roads", "Slope", "Soil moisture",
                       "Land-surface temperature", "Distance to waterways"]
LATEST = {"Land-surface temperature": 2024, "Precipitation": 2024, "Soil moisture": 2024,
          "Population density": 2020, "Night lights": 2024}   # the deployed run's years (2020 = last WorldPop)


# --------------------------------------------------------------------------
# Verra VT0007 (v1.0, 2024)

N_CLASSES = 30
COARSE_CELL_HA = 100_000          # VT0007 5.5: median REDD project size, 100,000 ha where unknown


def alternative_classes(risk, forest, method="geometric"):
    """VT0007 5.4.1: continuous risk rescaled to 1-2, then 30 geometric classes, narrow at high risk.
    method="equal_area": 30 classes of equal forest area instead (VT0007 allows any 1-30 scheme;
    riskmapjnr's best slicing)."""
    valid = forest & np.isfinite(risk)
    if method == "equal_area":
        from scipy.stats import rankdata
        classes = np.zeros(risk.shape, dtype="uint8")
        classes[valid] = np.minimum((rankdata(risk[valid]) - 1) / valid.sum() * N_CLASSES, N_CLASSES - 1).astype("uint8") + 1
        return classes
    lo, hi = np.nanmin(risk[valid]), np.nanmax(risk[valid])
    y = 3 - (1 + (risk - lo) / max(hi - lo, 1e-12))             # 1 = highest risk, 2 = lowest
    r = 0.5 ** (1 / N_CLASSES)
    limits = 2 * r ** np.arange(1, N_CLASSES + 1)                # 30 lower limits from ~1.98 down to 1
    classes = np.zeros(risk.shape, dtype="uint8")
    yy = np.where(valid, y, np.inf)
    classes[valid] = 1
    for k in range(2, N_CLASSES + 1):
        classes[valid & (yy < limits[k - 2])] = k
    return classes
