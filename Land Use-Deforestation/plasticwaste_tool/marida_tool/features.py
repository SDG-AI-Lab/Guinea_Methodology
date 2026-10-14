"""
Spectral features, computed identically to the deployed Guinea pipeline.

THE CONSTRAINT THAT SHAPES THIS MODULE

The point of the MARIDA work is to ask whether a learned classifier beats the
FDI threshold rule that Guinea currently ships. That comparison is only
meaningful if the FDI computed here is the SAME FDI the pipeline computes.

So `fdi()` below does not reimplement Biermann et al.'s equation from the
paper -- it calls `fdi_tool.spectral.compute_fdi`, the deployed function, with
MARIDA's bands mapped into the dictionary that function expects. If the
deployed formula is ever changed, this comparison changes with it, and no
silent divergence can open up between what is evaluated and what is run.

WHY THESE FEATURES AND NOT A DEEP MODEL

MARIDA's debris class is a few thousand pixels. A convolutional model over
patches has orders of magnitude more parameters than that supports, and the
architecture ablation already run for the dumpsite component (script 28)
established that in this module's data regime the binding constraint is the
volume of task-specific supervision, not model capacity. The same reasoning
applies here, so the model is a per-pixel classifier over interpretable
spectral features, and the burden of proof sits with anything more elaborate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_TOOL = Path(__file__).resolve().parent.parent
if str(_TOOL.parent) not in sys.path:
    sys.path.insert(0, str(_TOOL.parent))

from plasticwaste_tool.fdi_tool import spectral as SP   # noqa: E402

from .data import BANDS                                  # noqa: E402


def _as_dict(X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Columns of an [n, 11] reflectance array as the band dict SP expects.

    MARIDA spells the bands zero-padded ("B08"); the deployed pipeline spells
    them unpadded ("B8"). BOTH spellings are emitted rather than renaming one
    to match the other, so `fdi_tool.spectral` keeps receiving exactly the
    keys it was written against and nothing here depends on which convention
    a future caller happens to use.
    """
    d = {b: X[:, i].astype("float32") for i, b in enumerate(BANDS)}
    for b in list(d):
        if len(b) == 3 and b[1] == "0" and b[2].isdigit():
            d[f"B{b[2]}"] = d[b]          # B08 -> B8, B04 -> B4, ...
    return d


def fdi(X: np.ndarray) -> np.ndarray:
    """The deployed Floating Debris Index, not a local reimplementation."""
    return SP.compute_fdi(_as_dict(X))


def ndvi(X: np.ndarray) -> np.ndarray:
    return SP.compute_ndvi(_as_dict(X))


def ndwi(X: np.ndarray) -> np.ndarray:
    return SP.compute_ndwi(_as_dict(X))


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def build(X: np.ndarray) -> "object":
    """
    Feature table for one set of pixels.

    Eleven reflectances plus six indices. The indices are not redundant with
    the bands for a tree ensemble -- a tree can only split on one feature at a
    time, so a ratio it would otherwise need many splits to approximate is
    worth providing directly.

    FAI and NDVI are included because they are the standard floating-vegetation
    indices (Hu, C. (2009), A novel ocean color index to detect floating algae
    in the global ocean, Remote Sensing of Environment 113(10), 2118-2129) and
    the confuser this module most needs to separate from debris is Sargassum.
    Giving the model the index designed to FIND Sargassum is what lets the
    comparison in `evaluate.py` be about separability rather than about
    whether anyone remembered to compute the right band maths.
    """
    import pandas as pd

    X = np.asarray(X, dtype="float32")
    d = _as_dict(X)
    feats = {b: X[:, i] for i, b in enumerate(BANDS)}

    feats.update(_indices(d))

    # Red-edge slope: Sargassum has a vegetation red edge, plastic does not.
    feats["RE_slope"] = _ratio(d["B06"], d["B05"])

    return pd.DataFrame(feats)


def _indices(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """FDI, NDVI, NDWI, FAI and PI from a band dict carrying the zero-padded
    MARIDA names (B04, B08, B11) and the unpadded pipeline names (B4, B6, B8,
    B11) -- the dict `_as_dict` builds. One implementation, used by `build`
    (training) and by `from_composite` (deployment), so the two cannot drift.
    """
    out = {"FDI": SP.compute_fdi(d), "NDVI": SP.compute_ndvi(d),
           "NDWI": SP.compute_ndwi(d)}

    # Floating Algae Index (Hu 2009): same baseline-subtraction idea as FDI but
    # anchored on red and SWIR, so it responds to chlorophyll-bearing floating
    # matter. Where FDI and FAI disagree is where debris and Sargassum differ.
    lam = {"B4": 664.6, "B8": 832.8, "B11": 1613.7}
    f = (lam["B8"] - lam["B4"]) / (lam["B11"] - lam["B4"])
    out["FAI"] = (d["B08"] - (d["B04"] + (d["B11"] - d["B04"]) * f)).astype("float32")

    # Plastic Index (Themistocleous et al. 2020) -- NIR over NIR+Red.
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = d["B08"] / (d["B08"] + d["B04"])
    out["PI"] = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return out


# Composite (pipeline) band name -> MARIDA band name.
_COMPOSITE_TO_MARIDA = {"B3": "B03", "B4": "B04", "B6": "B06", "B8": "B08",
                        "B8A": "B8A", "B11": "B11", "B12": "B12"}


def from_composite(bands: Dict[str, np.ndarray], features: List[str]) -> np.ndarray:
    """
    The classifier's feature matrix for pixels of a Guinea composite.

    `bands` maps pipeline band names (B3, B4, B6, B8, B8A, B11, B12) to 1-D
    arrays of the pixels to score. Each band is given under its MARIDA name as
    well, and the indices come from `_indices`, so every column is computed by
    the same code `build` used on the training pixels. Returns an
    [n_pixels, len(features)] float32 array in the order of `features`.
    """
    d: Dict[str, np.ndarray] = {}
    for comp, mar in _COMPOSITE_TO_MARIDA.items():
        if comp in bands:
            a = np.asarray(bands[comp], dtype="float32")
            d[comp] = a
            d[mar] = a
    cols = {**{mar: d[mar] for mar in _COMPOSITE_TO_MARIDA.values() if mar in d},
            **_indices(d)}
    missing = [c for c in features if c not in cols]
    if missing:
        raise KeyError(f"composite lacks what the classifier needs: {missing}")
    return np.column_stack([cols[c] for c in features]).astype("float32")


FEATURE_NAMES: List[str] = BANDS + ["FDI", "NDVI", "NDWI", "FAI", "PI", "RE_slope"]
