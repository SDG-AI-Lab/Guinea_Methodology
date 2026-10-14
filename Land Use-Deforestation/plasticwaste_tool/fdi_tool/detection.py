"""
Hotspot classification: Otsu thresholding + vectorization (offline / numpy
port of `_otsu_threshold` / `_vectorize_hotspots` in `api_connector.py`), plus
a placeholder interface for the macro-scale dumpsite detector.

Otsu (1979): "A threshold selection method from gray-level histograms."
IEEE Trans. Systems, Man, and Cybernetics, 9(1), 62-66. Chosen because it is
unsupervised (no labelled plastic training data required at this stage) and
is the method used in the reference mesoscale plastic-detection literature
(e.g. Topouzelis et al. 2019/2020) that this pipeline follows.
"""
# ---------------------------------------------------------------------------
# MOVED HERE from plasticwaste_tool/detection.py (FDI half) by 24_build_component_packages.py.
# This file is now CANONICAL and is hand-maintained: the old top-level name is
# a re-export shim pointing back at it, not a second copy. Edit this file.
# ---------------------------------------------------------------------------

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import geopandas as gpd
from shapely.geometry import shape as shapely_shape
from rasterio.features import shapes as rasterio_shapes

from .. import config as pwcfg

logger = logging.getLogger(__name__)


# ============================================================================
# OTSU THRESHOLDING
# ============================================================================
def otsu_threshold(array: np.ndarray, bins: Optional[int] = None) -> Optional[float]:
    """
    Otsu's method on a single-band array, ignoring NaN. Falls back to
    scikit-image if available (numerically identical, faster); otherwise uses
    a pure-numpy histogram search equivalent to the GEE-side iteration in
    `api_connector._otsu_threshold`.

    Returns None if there are too few valid pixels to form a histogram (the
    caller should fall back to `DETECTION_CONFIG['fdi_manual_threshold']`,
    exactly as the GEE App prototype does via `ee.Algorithms.If`).
    """
    bins = bins or pwcfg.DETECTION_CONFIG["otsu_histogram_bins"]
    valid = array[~np.isnan(array)]
    if valid.size < 2 or np.ptp(valid) == 0:
        return None
    try:
        from skimage.filters import threshold_otsu
        return float(threshold_otsu(valid, nbins=bins))
    except Exception:
        logger.info("scikit-image unavailable; using manual Otsu histogram search.")
        counts, edges = np.histogram(valid, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        total = counts.sum()
        if total == 0:
            return None
        best_thresh, best_var = None, -1.0
        cum_counts = np.cumsum(counts)
        cum_means = np.cumsum(counts * centers)
        global_mean = cum_means[-1] / total
        for i in range(1, len(counts)):
            w0 = cum_counts[i - 1] / total
            w1 = 1.0 - w0
            if w0 == 0 or w1 == 0:
                continue
            m0 = cum_means[i - 1] / cum_counts[i - 1]
            m1 = (cum_means[-1] - cum_means[i - 1]) / (total - cum_counts[i - 1])
            inter_class_var = w0 * w1 * (m0 - m1) ** 2
            if inter_class_var > best_var:
                best_var, best_thresh = inter_class_var, centers[i]
        return float(best_thresh) if best_thresh is not None else None


def sigma_threshold(array: np.ndarray, n_sigma: Optional[float] = None) -> Optional[float]:
    """
    Anomaly threshold at mean + n*sigma of the (already masked) FDI distribution.

    This is the rule used by Digital Earth Africa's operational floating-marine-
    debris workflow, which computes FDI over water-masked pixels and flags
    mean + 3*sigma as debris. It is an anomaly-detection framing, and it suits
    FDI's actual distribution: floating debris is rare and elevated ABOVE the
    water baseline, so the signal lives in the upper tail rather than in a
    second histogram mode.

    Chosen over Otsu for whole-AOI detection on evidence, not preference --
    see `threshold_with_fallback` for why Otsu is not applicable at this scope.

    Reference: Digital Earth Africa, "Detection of Floating Marine Debris",
    https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/
    Real_world_examples/Floating_marine_debris.html
    Note their own caveat: the best threshold "will depend on the
    characteristics of your data and the specific application", so the
    multiplier is configurable and its sensitivity should be reported.

    LIMITATION -- the anomaly must be rare relative to the background. The
    flagged pixels are themselves part of the mean and sigma they are compared
    against, so if debris covers a large fraction of the masked population it
    inflates sigma and pulls the threshold up past itself, and the rule returns
    nothing. Measured on the synthetic fixture in `pipeline._synthetic_test`,
    where the target is ~1/3 of the surviving pixels, strengthening the target
    raised the threshold in step and never crossed it. On the real Conakry
    composite the regime is the intended one: 175 of 12,972 masked pixels
    (1.3%) sit beyond 3 sigma, and the peak reaches 6.7 sigma.

    So check the flagged fraction before trusting a run. A few percent is the
    working regime; a large fraction, or a suspiciously empty result over a
    small AOI, means this rule is the wrong instrument for that scene.
    """
    n_sigma = n_sigma if n_sigma is not None else pwcfg.DETECTION_CONFIG["threshold_n_sigma"]
    valid = array[np.isfinite(array)]
    if valid.size < 2:
        return None
    return float(valid.mean() + n_sigma * valid.std())


def threshold_with_fallback(array: np.ndarray) -> Tuple[float, str]:
    """
    Return (threshold, method) for the masked FDI array, dispatching on
    `DETECTION_CONFIG['threshold_method']`.

    Why the default is 'sigma' and not 'otsu'
    -----------------------------------------
    Otsu finds the split that best separates two modes, so it needs the scene to
    BE bimodal. Measured on the Conakry 2024-01 composite, the apparent
    bimodality came from bright coastal land, not from debris: once a positive
    water constraint was added, the Otsu threshold fell from -0.0077 to -0.0160,
    flagged 45% of all remaining water pixels, and its +/-10% area sensitivity
    degraded from -4.4/+5.3% to -17.5/+15.3%. It had been separating water from
    land all along.

    Both of those Otsu thresholds are also NEGATIVE, i.e. they select pixels
    merely less negative than typical water, whereas the published FDI signal
    for floating debris is an elevation ABOVE the water baseline.

    The literature supports this scoping. Kikaki et al. (2023), iScience 26,
    108402 -- "Large-scale detection of marine debris in coastal areas with
    Sentinel-2" -- do use FDI with an Otsu threshold, but only INSIDE a buffered
    hand-annotated region of suspected debris, as a label-refinement step; their
    deployed detector is a trained model. Within a small pre-localised region
    the two-class assumption holds. Across a whole coastal AOI it does not.

    'manual' is retained for reproducing the original GEE prototype, but note
    its 0.005 has no cited source -- it is an engineering fallback ("fallback if
    Otsu fails on sparse scenes"), and on this scene it happens to land at
    mean + 2*sigma, which is coincidence rather than derivation.
    """
    method = pwcfg.DETECTION_CONFIG.get("threshold_method", "sigma")

    if method == "sigma":
        t = sigma_threshold(array)
        if t is None:
            return pwcfg.DETECTION_CONFIG["fdi_manual_threshold"], "manual_fallback"
        return t, "sigma"

    if method == "manual":
        return pwcfg.DETECTION_CONFIG["fdi_manual_threshold"], "manual"

    t = otsu_threshold(array)
    if t is None:
        return pwcfg.DETECTION_CONFIG["fdi_manual_threshold"], "manual_fallback"
    return t, "otsu"


# ============================================================================
# HOTSPOT BINARY MASK + VECTORIZATION
# ============================================================================
def hotspot_binary_mask(masked_fdi: np.ndarray, threshold: Optional[float] = None) -> Tuple[np.ndarray, float, str]:
    """Threshold the (already NDVI/NDWI-masked) FDI array into a binary
    hotspot mask. Returns (mask, threshold_used, method)."""
    if threshold is None:
        threshold, method = threshold_with_fallback(masked_fdi)
    else:
        method = "manual_override"
    mask = np.where(np.isnan(masked_fdi), False, masked_fdi > threshold)
    return mask, threshold, method

def debris_binary_mask_de_africa(fdi: np.ndarray, ndvi: np.ndarray,
                                 water: np.ndarray,
                                 n_sigma: Optional[float] = None) -> Tuple[np.ndarray, Dict, str]:
    """
    Digital Earth Africa's floating-debris rule, ported directly.

    Their published sequence:
        1. mask to water (coastline buffer + WOfS frequency >= 0.5)
        2. threshold_fdi  = mean(fdi)  + 3*std(fdi)   over those water pixels
        3. threshold_ndvi = mean(ndvi) + 3*std(ndvi)  same population
        4. debris_map = (fdi > threshold_fdi) & (ndvi > threshold_ndvi)

    Two departures from this module's previous behaviour, both deliberate:

    * NDVI is used INCLUSIVELY, not as a vegetation exclusion. Debris sits above
      the water baseline in NDVI as well as FDI, so requiring both is what gives
      the rule its selectivity -- on the Conakry 2024-01 composite, FDI alone
      flags 102,827 pixels and the AND cuts that to 6,558.
    * NDWI is not used at all, and open water is no longer excluded. Open water
      IS the background the anomaly is measured against; excluding it left the
      statistics computed over 12,972 residual pixels instead of 4.55M.

    SCOPE -- this detects floating debris, NOT plastic specifically. Seaweed,
    algae mats, timber and foam all raise NDVI and FDI together and will be
    flagged. Biermann et al. (2020) separate those classes with a naive Bayes
    classifier in 2-D NDVI-FDI space, which needs a labelled spectral library
    that does not exist for Guinea. Report outputs as floating-debris candidates
    and treat plastic-vs-organic discrimination as out of scope until labels
    exist.

    Reference: Digital Earth Africa, "Detection of Floating Marine Debris",
    https://docs.digitalearthafrica.org/en/latest/sandbox/notebooks/
    Real_world_examples/Floating_marine_debris.html
    """
    n_sigma = n_sigma if n_sigma is not None else pwcfg.DETECTION_CONFIG["threshold_n_sigma"]
    in_water = np.nan_to_num(water, nan=0.0) > 0 if water is not None else np.ones_like(fdi, bool)
    if water is None:
        logger.warning(
            "DE Africa rule invoked without a WATER band; statistics will be computed "
            "over the whole scene, which includes land and will distort both thresholds.")

    valid = in_water & np.isfinite(fdi) & np.isfinite(ndvi)
    if not valid.any():
        return np.zeros_like(fdi, bool), {"fdi": np.nan, "ndvi": np.nan}, "de_africa_no_valid_px"

    f, v = fdi[valid], ndvi[valid]
    t_fdi = float(f.mean() + n_sigma * f.std())
    t_ndvi = float(v.mean() + n_sigma * v.std())

    binary = valid & (fdi > t_fdi) & (ndvi > t_ndvi)
    return binary, {"fdi": t_fdi, "ndvi": t_ndvi, "n_px_population": int(valid.sum())}, \
        f"de_africa_{n_sigma:g}sigma"


def resolve_pixel_area_m2(transform, crs, expected_scale_m=None,
                          tolerance: float = 10.0) -> float:
    """
    Pixel area in m^2, with guards against a broken PROJ install.

    The previous inline version used
    `getattr(crs, "is_geographic", False)`, which silently degrades: if
    proj.db is missing, the CRS object fails to resolve, the getattr default
    of False is taken for a raster that IS EPSG:4326, and degree-sized
    resolution gets multiplied as metres.

        correct : (20/111320 * 111320)^2  = 400 m^2
        broken  : (20/111320)^2           = 3.23e-08 m^2

    ~1.2e10 too small. Every patch then falls below
    DETECTION_CONFIG['min_patch_area_m2'] and is discarded, so the scene
    reports zero hotspots instead of raising. Indistinguishable from a clean
    scene.

    Same failure class as the deforestation temporal hold-out: a plausible
    wrong answer is worse than a crash.
    """
    xres, yres = abs(transform.a), abs(transform.e)

    if crs is None:
        raise ValueError(
            "Raster has no CRS; area cannot be computed. Check the GEE export.")

    try:
        is_geographic = bool(crs.is_geographic)
    except Exception as e:
        raise RuntimeError(
            f"CRS is present but will not resolve ({e}). This is the missing "
            f"proj.db failure. Fix PROJ and restart the kernel; continuing "
            f"would silently discard every hotspot."
        ) from e

    # Cross-check that does NOT depend on PROJ resolving correctly: a
    # resolution below 0.01 is degrees, not metres, whatever the CRS claims.
    looks_like_degrees = xres < 0.01
    if looks_like_degrees and not is_geographic:
        raise ValueError(
            f"Resolution {xres:.3e} looks like degrees but the CRS reports "
            f"projected ({crs}). PROJ is misconfigured — this is exactly the "
            f"case that produces a silent zero-hotspot result.")
    if not looks_like_degrees and is_geographic:
        raise ValueError(
            f"Resolution {xres:.3f} looks like metres but the CRS reports "
            f"geographic ({crs}). Check the export grid.")

    area = ((xres * 111_320.0) * (yres * 111_320.0)
            if is_geographic else xres * yres)

    if expected_scale_m:
        ratio = area / (expected_scale_m ** 2)
        if not (1 / tolerance) < ratio < tolerance:
            raise ValueError(
                f"Pixel area {area:.3e} m^2 is implausible for a "
                f"~{expected_scale_m:g} m raster (off by {ratio:.2e}x). "
                f"crs={crs}, is_geographic={is_geographic}, xres={xres}")

    return area

def confidence_level_for_score(score: float) -> str:
    """Band a [0,1] concentration score into LOW/MEDIUM/HIGH using
    `DETECTION_CONFIG['confidence_band_low_max' / 'high_min']`."""
    cfg = pwcfg.DETECTION_CONFIG
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "LOW"
    if score >= cfg["confidence_band_high_min"]:
        return "HIGH"
    if score >= cfg["confidence_band_low_max"]:
        return "MEDIUM"
    return "LOW"


def vectorize_hotspots(binary_mask: np.ndarray, profile: Dict,
                       acquisition_date: str, cloud_pct: Optional[float] = None,
                       pixel_area_m2: Optional[float] = None,
                       masked_fdi: Optional[np.ndarray] = None,
                       threshold: Optional[float] = None,
                       p_debris: Optional[np.ndarray] = None) -> gpd.GeoDataFrame:
    """
    `p_debris`: when given (the MARIDA debris classifier's per-pixel
    probability, fdi_tool.debris_filter), each patch carries its mean as
    `p_debris`.

    Vectorize a binary hotspot raster into polygons with the same attribute
    schema as the GEE-side `_vectorize_hotspots()` (area, concentration score,
    acquisition metadata, module/source tags), then drop patches below the
    minimum mappable area (GEDH-FR-0503/0504 use this for the popup detail).

    `masked_fdi` + `threshold`: when both are given, `floating_debris_score`
    is computed per patch from FDI intensity (`compute_concentration_score`) so a
    small, intensely bright patch outranks a large, marginal one. Without them,
    falls back to the coarser area-proportional score (kept so callers that only
    have the binary mask still work).
    """
    from scipy.ndimage import label as cc_label

    cfg = pwcfg.DETECTION_CONFIG
    transform = profile["transform"]
    crs = profile.get("crs")
    if pixel_area_m2 is None:
        pixel_area_m2 = resolve_pixel_area_m2(
            transform, crs,
            expected_scale_m=pwcfg.S2_CONFIG.get("download_scale_m"))

    use_fdi_score = masked_fdi is not None and threshold is not None

    # Normalize each patch's mean excess-over-threshold against this scene's own
    # excess distribution (a high percentile across ALL hotspot pixels, not just
    # one patch), rather than against abs(threshold) directly. When the Otsu
    # threshold is small in magnitude, dividing by it saturates almost every
    # patch to 1.0; a scene-relative normalizer keeps the score spread out and
    # comparable to other scenes/months at a similar percentile.
    excess_norm = None
    if use_fdi_score:
        scene_excess = masked_fdi[binary_mask] - threshold
        scene_excess = scene_excess[np.isfinite(scene_excess)]
        if scene_excess.size:
            excess_norm = float(np.percentile(
                scene_excess, cfg["concentration_score_norm_percentile"])) or None

    # Connected-component pixel counts give a robust, CRS-agnostic area
    # (avoids the geometric .area-is-in-degrees^2 trap for geographic rasters).
    def _empty() -> gpd.GeoDataFrame:
        """Typed empty result.

        Needed in TWO places, which is what made this a bug: the obvious case is
        no connected components at all, but a scene can also produce components
        that are every one of them smaller than `min_patch_area_m2` and so all
        filtered out below. That second case left `records` empty, and
        `gpd.GeoDataFrame([], crs=crs)` raises "Assigning CRS to a GeoDataFrame
        without a geometry column" -- which failed the whole Fatala region during
        the first national run rather than returning "no patches here".
        """
        cols = ["geometry", "area_m2", "floating_debris_score", "mean_fdi", "p90_fdi",
                "acquisition_date", "cloud_cover_pct", "confidence_level", "module",
                "source"]
        if p_debris is not None:
            cols.append("p_debris")
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=crs)

    labeled, n_features = cc_label(binary_mask, structure=np.ones((3, 3)))  # 8-connected
    if n_features == 0:
        return _empty()
    pixel_counts = np.bincount(labeled.ravel())  # index 0 = background

    records = []
    for geom, value in rasterio_shapes(labeled.astype("int32"), mask=binary_mask,
                                       transform=transform, connectivity=8):
        label_id = int(value)
        if label_id == 0:
            continue
        area_m2 = float(pixel_counts[label_id]) * pixel_area_m2
        if area_m2 < cfg["min_patch_area_m2"]:
            continue

        mean_fdi = p90_fdi = None
        if use_fdi_score:
            score_info = compute_concentration_score(
                masked_fdi, labeled == label_id, threshold, excess_norm=excess_norm)
            concentration_score = score_info["concentration_score"]
            mean_fdi, p90_fdi = score_info["mean_fdi"], score_info["p90_fdi"]
        else:
            concentration_score = min(area_m2 / cfg["max_expected_patch_m2"], 1.0)

        rec = {
            "geometry": shapely_shape(geom),
            "area_m2": area_m2,
            "floating_debris_score": concentration_score,
            "mean_fdi": mean_fdi,
            "p90_fdi": p90_fdi,
            "acquisition_date": acquisition_date,
            "cloud_cover_pct": cloud_pct,
            "confidence_level": confidence_level_for_score(concentration_score),
            "module": "M2_FDI",
            "source": "Sentinel-2_SR_Harmonized",
        }
        if p_debris is not None:
            vals = p_debris[labeled == label_id]
            vals = vals[np.isfinite(vals)]
            rec["p_debris"] = float(vals.mean()) if vals.size else None
        records.append(rec)

    if not records:
        logger.info(
            "%d connected component(s) found but all were below min_patch_area_m2=%.0f — "
            "returning an empty patch set for this scene.",
            n_features, cfg["min_patch_area_m2"])
        return _empty()

    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf.reset_index(drop=True)

def compute_concentration_score(fdi_array, mask, threshold, excess_norm: Optional[float] = None):
    """
    Concentration from FDI intensity within the patch, not from its size.

    A small, intensely bright patch should outrank a large, marginal one —
    the area-proportional score used before this does the opposite, and sends
    field teams to the biggest blobs rather than the likeliest plastic.

    `excess_norm`: value each patch's mean excess-over-threshold is divided by.
    Pass the scene-level normalizer from `vectorize_hotspots` (a high percentile
    of excess across all hotspot pixels in the same scene) — this scales the
    score to that scene's actual FDI dynamic range. Falling back to
    `abs(threshold)` (the old default) saturates almost every patch to 1.0
    whenever the Otsu threshold is small in magnitude, which collapses the
    ranking this score exists to provide.
    """
    vals = fdi_array[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean_fdi": np.nan, "p90_fdi": np.nan,
                "concentration_score": np.nan}
    excess = vals - threshold          # how far above threshold
    norm = excess_norm if excess_norm else (abs(threshold) + 1e-9)
    return {
        "mean_fdi": float(vals.mean()),
        "p90_fdi": float(np.percentile(vals, 90)),
        "mean_excess": float(excess.mean()),
        "concentration_score": float(np.clip(excess.mean() / norm, 0, 1)),
    }
