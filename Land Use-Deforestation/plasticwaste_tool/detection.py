"""
The DPWMS detection interface — what the backend calls.

This module is deliberately the ONE import surface the Django backend needs for
the National Hotspot Map (GEDH-FR-0503), which requires the marine floating-
debris layer and the terrestrial dumpsite layer to be independently toggleable.
It holds two things and nothing else:

  1. A re-export of `fdi_tool.detection` -- thresholding (Otsu, mean+3sigma,
     fallback), the Digital Earth Africa conjunction rule, vectorisation and
     patch scoring. The implementation moved into the FDI component package;
     the names stay bound here so existing callers, `validation.py` and
     `plasticwaste_analysis.ipynb` among them, are unaffected.

  2. `classify_dumpsite_candidates()`, the dumpsite entry point, whose
     implementation is `dumpsite_tool.scan`.

Keeping both here means the developers' surface is two functions wide and does
not move when either component changes behind it. The backend should NOT import
`fdi_tool` or `dumpsite_tool` directly.

Otsu (1979): "A threshold selection method from gray-level histograms."
IEEE Trans. Systems, Man, and Cybernetics, 9(1), 62-66. Chosen because it is
unsupervised (no labelled plastic training data required at this stage) and is
the method used in the reference mesoscale plastic-detection literature
(e.g. Topouzelis et al. 2019/2020) that this pipeline follows.
"""

import logging
from typing import Optional, Tuple

import numpy as np                                            # noqa: F401
import geopandas as gpd

from . import config as pwcfg
from .fdi_tool import detection as _fdi

# --- component 1: floating debris on water ---------------------------------
# Re-exported, not reimplemented. `import *` would skip anything these modules
# do not list in __all__ (they list nothing), so the names are bound
# explicitly -- an accidental omission then fails loudly at import time rather
# than silently at the call site.
from .fdi_tool.detection import (                             # noqa: F401
    otsu_threshold,
    sigma_threshold,
    threshold_with_fallback,
    hotspot_binary_mask,
    debris_binary_mask_de_africa,
    resolve_pixel_area_m2,
    confidence_level_for_score,
    vectorize_hotspots,
    compute_concentration_score,
)

logger = logging.getLogger(__name__)


# ============================================================================
# MACRO-SCALE DUMPSITE DETECTION
# ============================================================================
# Columns returned by classify_dumpsite_candidates(), whether the model is
# enabled or not. A superset of the original stub schema -- consumers that
# select named columns are unaffected by the additions.
DUMPSITE_SCHEMA = [
    "geometry", "confidence", "tier", "area_ha", "n_windows", "status",
    "detection_date", "imagery_date", "model_version",
    "material_flags", "toxicity_flag",
]


def empty_dumpsite_layer() -> gpd.GeoDataFrame:
    """A correctly typed, empty dumpsite layer."""
    return gpd.GeoDataFrame(columns=DUMPSITE_SCHEMA, geometry="geometry",
                            crs="EPSG:4326")


def classify_dumpsite_candidates(
        bounds: Optional[Tuple[float, float, float, float]] = None,
        *,
        cache_dir: Optional[str] = None,
        budget: Optional[int] = None,
        tier: Optional[str] = None,
        imagery_date: Optional[str] = None,
        image_array: Optional[np.ndarray] = None,
        model_path: Optional[str] = None,
        confirm_source: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Macro-scale illegal-dumpsite candidates for a bbox, as polygons.

    This is the stable interface the backend calls (GEDH-FR-0503 requires the
    dumpsite layer and the mesoscale FDI layer to be independently toggleable
    on the same National Hotspot Map). The implementation lives in
    `plasticwaste_tool.dumpsite_tool.scan`; the backend should not import that
    package directly, so this signature can stay put while the model behind it
    changes.

    METHOD. A two-stage cascade, not a single sweep. Stage 1 scores 250 m cells
    on FREE Esri imagery and discards the country where nothing is happening;
    stage 2 scores 80 m windows inside the cells that survive, on free Esri z17
    imagery with the head fitted on Esri (default) or on licensed Google Static
    Maps z19 imagery with the Google head (`confirm_source`). The heads are
    logistic regressions over frozen ImageNet ResNet-50 features. The 80 m
    footprint is the reason this detects waste rather than settlement -- see
    the dumpsite_tool README.

    (Earlier drafts described a YOLOv8 detector adapted from the Lab's
    Guatemala project. That is not what shipped: Guinea has 111 labelled dump
    points, far too few to train a detector, so the method is a linear probe on
    a frozen backbone.)

    WHAT THE POLYGONS ARE. `status` is always `unverified`. These are
    photo-interpreted candidates from a model whose own ground truth was
    photo-interpretation, never field survey, and `imagery_date` may be well
    over a year older than `detection_date` -- a new row means newly detected,
    NOT newly appeared. Present them as "AI-detected candidates, not
    field-verified".

    Args:
        bounds: (minx, miny, maxx, maxy) in EPSG:4326. Required when enabled.
        cache_dir: where fetched tiles are cached. Cached tiles are free on the
            next run, so a persistent path materially reduces cost.
        budget: hard cap on BILLABLE Google requests in stage 2. Required for
            a Google scan -- see the spend guard below. `budget=0` runs stage 1
            plus whatever stage-2 tiles are already cached, and bills nothing.
            Ignored for an Esri scan, which bills nothing.
        confirm_source: stage-2 imagery, "esri" (free, Esri z17 with the head
            fitted on Esri) or "google" (licensed, billed). Defaults to
            DUMPSITE_MODEL_CONFIG["confirm_source"]. The precision and recall
            published with a scan must be the ones measured for its source.
        tier: `high`, `medium` or `all`; defaults to
            DUMPSITE_MODEL_CONFIG["default_tier"] (`medium`). A display filter, not a
            model change: every row is scored identically and the threshold
            only decides what is returned, so it can be changed without
            re-scanning.
        imagery_date: capture date of the basemap, if known. Recorded, not
            inferred -- the Static Maps API does not report it.
        image_array: accepted and IGNORED. The cascade fetches its own
            geolocated tiles; a bare array has no coordinates to place a
            polygon at. Kept only so the old stub signature does not break.
        model_path: accepted and IGNORED; the frozen heads ship inside the
            package at dumpsite_tool/models/.

    Returns:
        GeoDataFrame with DUMPSITE_SCHEMA columns, EPSG:4326. Empty (typed)
        when the model is disabled.
    """
    cfg = pwcfg.DUMPSITE_MODEL_CONFIG

    if image_array is not None:
        logger.warning("classify_dumpsite_candidates(): image_array is ignored; "
                       "the cascade fetches its own geolocated tiles.")
    if model_path is not None:
        logger.warning("classify_dumpsite_candidates(): model_path is ignored; "
                       "the frozen heads ship inside dumpsite_tool/models/.")

    if not cfg.get("enabled"):
        logger.info("Dumpsite model disabled (config.DUMPSITE_MODEL_CONFIG); "
                    "returning empty macro-scale layer.")
        return empty_dumpsite_layer()

    if bounds is None:
        raise ValueError(
            "classify_dumpsite_candidates() needs bounds=(minx, miny, maxx, maxy) "
            "in EPSG:4326 when the dumpsite model is enabled. It scans an area; "
            "there is no whole-country default, because a national sweep is a "
            "four-figure imagery bill that must be requested explicitly.")

    confirm_source = confirm_source or cfg.get("confirm_source", "google")

    # --- spend guard -------------------------------------------------------
    # A Google stage 2 hits a paid API once per 80 m window. An unbounded scan
    # over a large AOI is thousands of dollars, and a Celery task that does
    # this by accident is a much worse failure than one that refuses to start.
    # An Esri stage 2 is free, so there is nothing to guard.
    if (confirm_source == "google" and budget is None
            and not cfg.get("allow_unbounded_spend", False)):
        raise ValueError(
            "classify_dumpsite_candidates() requires an explicit budget: the cap "
            "on billable Google Static Maps requests for this scan. Expect "
            "roughly 39 stage-2 windows per surviving 250 m cell. Use budget=0 "
            "for a cache-only dry run, or set "
            "DUMPSITE_MODEL_CONFIG['allow_unbounded_spend']=True to opt out.")

    # torch must initialise its native DLLs before the GDAL/GEE stack on
    # Windows. Importing this module has ALREADY loaded geopandas, so the race
    # is lost by the time we get here -- the caller must import torch first.
    # Turn the resulting DLL error into an instruction rather than a mystery.
    try:
        import torch  # noqa: F401
    except OSError as exc:
        raise RuntimeError(
            "torch failed to load its native libraries, which on Windows means "
            "GDAL initialised first. Put `import torch` at the very top of the "
            "calling module, above any plasticwaste_tool import. "
            "Original error: %s" % exc) from exc

    from .dumpsite_tool import scan as dumpsite_scan

    head = dumpsite_scan.confirm_head_for(confirm_source)
    tiers, default_tier = dumpsite_scan.read_tiers(head)
    tier = tier or cfg.get("default_tier") or default_tier
    if tier not in tiers:
        raise ValueError("tier must be one of %s, got %r" % (sorted(tiers), tier))

    cache_dir = cache_dir or cfg.get("cache_dir") or "dumpsite_scan_cache"
    out = dumpsite_scan.scan(bounds, cache_dir, budget=budget,
                             imagery_date=imagery_date,
                             confirm_source=confirm_source)
    if out.empty:
        return empty_dumpsite_layer()

    # The tier is a display filter applied AFTER scanning: re-filtering to a
    # lower tier never requires paying for the imagery again.
    kept = out[out.confidence >= tiers[tier]].copy()
    logger.info("dumpsite scan: %d candidates at tier %r (%d scored at or above "
                "the lowest tier)", len(kept), tier, len(out))
    return kept.reset_index(drop=True)
