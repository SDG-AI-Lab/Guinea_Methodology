"""
Spectral index computation — offline numpy port of the GEE server-side ops in
`api_connector.py`.

Why this exists as a separate, array-based module (mirrors landuse_tool's
design where `pipeline.run_from_arrays()` is independent of the fetch layer):
  1. Testability: unit tests / the synthetic self-test in `pipeline.py` can
     exercise the full FDI -> Otsu -> vectorize chain without a network call
     or GEE credentials.
  2. QA: `download_composite_bands()` lets you pull raw bands alongside the
     GEE-computed FDI/NDVI/NDWI, then recompute them here and diff the two —
     the same "don't just trust the graph" check used for landuse_tool's
     driver stack.

References
----------
Biermann, L., Clewley, D., Martinez-Vicente, V. & Topouzelis, K. (2020).
"Finding Plastic Patches in Coastal Waters using Optical Satellite Data."
Scientific Reports 10, 5364. https://doi.org/10.1038/s41598-020-62298-z
"""
# ---------------------------------------------------------------------------
# MOVED HERE from plasticwaste_tool/spectral.py by 24_build_component_packages.py.
# This file is now CANONICAL and is hand-maintained: the old top-level name is
# a re-export shim pointing back at it, not a second copy. Edit this file.
# ---------------------------------------------------------------------------

import logging
from typing import Dict

import numpy as np

from .. import config as pwcfg

logger = logging.getLogger(__name__)


def compute_fdi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Floating Debris Index (Biermann et al. 2020, Sci. Rep. 10, 5364, eq. 1-2):

        FDI    = R_NIR - R'_NIR
        R'_NIR = R_RE2 + (R_SWIR1 - R_RE2) * (lambda_NIR - lambda_RED)
                                            / (lambda_SWIR1 - lambda_RED) * 10

    with R_NIR = B8, R_RE2 = B6, R_SWIR1 = B11 and lambda_RED the B4 central
    wavelength, exactly as published. The factor of 10 is part of the
    published formula and is kept as given.

    Earlier versions used B8A as R_RE2 and lambda_B8A in the ratio; see the
    note on S2_CONFIG["lambda_nm"] in config.py.
    """
    lam = pwcfg.S2_CONFIG["lambda_nm"]
    b8, b6, b11 = bands["B8"], bands["B6"], bands["B11"]
    wl_factor = (lam["B8"] - lam["B4"]) / (lam["B11"] - lam["B4"])
    r_prime_nir = b6 + (b11 - b6) * wl_factor * 10
    return (b8 - r_prime_nir).astype("float32")


def compute_ndvi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    b8, b4 = bands["B8"], bands["B4"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (b8 - b4) / (b8 + b4)
    return ndvi.astype("float32")


def compute_ndwi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Modified NDWI (McFeeters 1996 form): (Green - NIR) / (Green + NIR).
    High values = open water."""
    b3, b8 = bands["B3"], bands["B8"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (b3 - b8) / (b3 + b8)
    return ndwi.astype("float32")


def apply_spectral_masks(fdi: np.ndarray, ndvi: np.ndarray, ndwi: np.ndarray,
                         ndvi_thresh: float = None, ndwi_thresh: float = None,
                         water: np.ndarray = None) -> np.ndarray:
    """
    Exclude vegetation (NDVI > thresh) and open water (NDWI > thresh) so only
    the residual floating-debris signature remains. Returns a copy of `fdi`
    with excluded pixels set to NaN.

    `water`: the composite's WATER band (JRC Global Surface Water occurrence,
    thresholded and buffered server-side — see `api_connector._surface_water_mask`).
    Required to keep coastal infrastructure out: the NDVI/NDWI pair only
    excludes vegetation and open water, so bare soil, sand, concrete and port
    rooftops all survive it, and their strong NIR/SWIR gives them a large FDI
    excess. Passing None reproduces the old, leakier behaviour.
    """
    ndvi_thresh = pwcfg.DETECTION_CONFIG["ndvi_threshold"] if ndvi_thresh is None else ndvi_thresh
    ndwi_thresh = pwcfg.DETECTION_CONFIG["ndwi_threshold"] if ndwi_thresh is None else ndwi_thresh
    keep = (ndvi < ndvi_thresh) & (ndwi < ndwi_thresh)
    if water is not None:
        keep &= np.nan_to_num(water, nan=0.0) > 0
    out = fdi.copy()
    out[~keep] = np.nan
    return out


def compare_with_gee(local_fdi: np.ndarray, gee_fdi: np.ndarray,
                     rtol: float = 1e-3) -> Dict:
    """
    QA helper: compare a locally-recomputed FDI array against the array
    downloaded from `api_connector` (which ran the same formula server-side
    in Earth Engine). Small floating-point differences are expected from GEE's
    internal reduction order; large ones indicate a band-selection or
    band-order mismatch.
    """
    valid = ~(np.isnan(local_fdi) | np.isnan(gee_fdi))
    if not np.any(valid):
        return {"n_compared": 0, "max_abs_diff": np.nan, "within_tolerance": False}
    diff = np.abs(local_fdi[valid] - gee_fdi[valid])
    return {
        "n_compared": int(valid.sum()),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "within_tolerance": bool(diff.max() < rtol),
    }
