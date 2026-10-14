"""
Data loading and grid alignment for the DPWMS Plastic Waste module.

Reuses landuse_tool.utils for the actual raster-alignment and missing-value
machinery (item 3 in the brief) rather than re-implementing it: this module is
a thin, plastic-waste-specific layer on top that (a) knows how to read the
band-stacked composite this package's `api_connector` downloads, (b) rasterizes
Digital Waste Bank point layers (collectors / recyclers / drop-off centres),
and (c) maintains the Monthly FDI Archive index consumed by GEDH-FR-0507.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import geopandas as gpd

from landuse_tool.utils import align_rasters, handle_missing_values
from landuse_tool import utils as landuse_utils

from . import config as pwcfg

logger = logging.getLogger(__name__)

# Band order used by `api_connector.download_composite_bands()`.
DEFAULT_COMPOSITE_BANDS = ["B3", "B4", "B6", "B8", "B8A", "B11", "B12", "FDI", "NDVI", "NDWI", "WATER"]


# ============================================================================
# RASTER LOADING
# ============================================================================
def read_single_band(path: str) -> Tuple[np.ndarray, Dict]:
    """Read a single-band GeoTIFF (e.g. the FDI-only export) + its profile."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    # Belt-and-braces: a masked pixel whose fill doesn't exactly match the
    # file's declared `nodata` tag (e.g. a per-band mismatch introduced by
    # `download_composite_bands()` stacking independently-downloaded bands
    # under one shared profile) would otherwise survive as a huge sentinel
    # and overflow float32 to +/-inf once it hits the FDI formula's `* 10`.
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr, profile


def read_composite(path: str, band_names: Optional[List[str]] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Read a multi-band composite GeoTIFF into a {band_name: array} dict, using
    `DEFAULT_COMPOSITE_BANDS` ordering unless overridden. Used for offline
    QA of the FDI/NDVI/NDWI computation (see `spectral.py`) and for masks
    that need the raw reflectance bands (SCL-derived masks are applied
    server-side already, so raw bands here are post-mask, post-median).
    """
    band_names = band_names or DEFAULT_COMPOSITE_BANDS
    with rasterio.open(path) as src:
        if src.count != len(band_names):
            raise ValueError(
                f"Composite at {path} has {src.count} bands but {len(band_names)} "
                f"names were given ({band_names}). Pass the exact band list used "
                f"in download_composite_bands(). If this file predates the WATER "
                f"band being added to the composite, re-download it once with "
                f"download_composite_bands(..., force_refresh=True) -- the cache "
                f"key is unchanged, so a stale file is returned otherwise.")
        arrays = {name: src.read(i + 1).astype("float32") for i, name in enumerate(band_names)}
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None:
        for name in arrays:
            arrays[name] = np.where(arrays[name] == nodata, np.nan, arrays[name])
    # Belt-and-braces: each band here was downloaded as a SEPARATE tiled GEE
    # request and then stacked under one shared `nodata` tag taken from the
    # first band's file (see `download_composite_bands()`), so a masked pixel
    # whose real fill doesn't exactly match that shared tag can survive the
    # check above as a huge sentinel and overflow float32 to +/-inf once it
    # hits FDI's `* 10` term. Catch anything non-finite regardless of `nodata`.
    for name in arrays:
        arrays[name] = np.where(np.isfinite(arrays[name]), arrays[name], np.nan)
    return arrays, profile


def align_to_reference(rasters: Dict[str, Tuple[np.ndarray, Dict]],
                       reference_key: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Align a dict of {name: (array, profile)} to a common grid, using
    landuse_tool's `align_rasters`. Returns ({name: aligned_array}, ref_profile).
    """
    keys = list(rasters.keys())
    if reference_key is None:
        reference_key = keys[0]
    ordered_keys = [reference_key] + [k for k in keys if k != reference_key]
    pairs = [rasters[k] for k in ordered_keys]
    aligned_pairs = align_rasters(pairs)
    ref_profile = aligned_pairs[0][1]
    aligned = {k: arr for k, (arr, _) in zip(ordered_keys, aligned_pairs)}
    return aligned, ref_profile


# ============================================================================
# DIGITAL WASTE BANK VECTOR LAYERS (collectors / recyclers / drop-off centres)
# ============================================================================
# These come from the Collector web-app / recycler-registration database, NOT
# from GEE — this loader only standardizes the schema and rasterizes them for
# map display / proximity analysis. A real deployment swaps `load_*_from_csv`
# for a call into the shared geospatial DB (HUB-M2 PostGIS tables).
WASTE_BANK_POINT_SCHEMA = ["actor_id", "actor_type", "name", "latitude", "longitude",
                          "registered_date", "status"]


def load_waste_bank_points(csv_path: str) -> gpd.GeoDataFrame:
    """Load collector/recycler/drop-off point locations from a CSV export of
    the Digital Waste Bank table (GEDH-FR-0506) into a GeoDataFrame."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    missing = [c for c in ("latitude", "longitude") if c not in df.columns]
    if missing:
        raise ValueError(f"Waste-bank CSV missing required columns: {missing}")
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    return gdf


def rasterize_recycler_proximity(recycler_gdf: gpd.GeoDataFrame, profile: Dict,
                                 mask: np.ndarray) -> np.ndarray:
    """
    Distance-to-nearest-recycler raster, used to flag hotspots that are far
    from any registered recycler (a practical Waste-Flow planning signal —
    GEDH-FR-0504's "link to the nearest registered recycler").
    Reuses landuse_tool's heatmap/point rasterizer.
    """
    return landuse_utils.rasterize_points_as_heatmap(
        recycler_gdf, profile, mask, decay_factor=0.1, missing_value_handling="mask")


# ============================================================================
# HOTSPOT VECTOR LOADING
# ============================================================================
def load_hotspot_vectors(path: str) -> gpd.GeoDataFrame:
    """Load a hotspot-patch GeoJSON produced by `api_connector.download_fdi_results`."""
    gdf = gpd.read_file(path)
    for col in ("area_m2", "floating_debris_score", "cloud_cover_pct"):
        if col in gdf.columns:
            gdf[col] = gdf[col].astype(float)
    return gdf


# ============================================================================
# MONTHLY FDI ARCHIVE INDEX (GEDH-FR-0507)
# ============================================================================
def load_archive_index(cache_dir: Optional[str] = None) -> Dict[str, str]:
    """Read the {'YYYY-MM': raster_path} index used by the archive slider."""
    cache_dir = cache_dir or pwcfg.PROCESSING_CONFIG["cache_dir"]
    idx_path = os.path.join(cache_dir, pwcfg.ARCHIVE_CONFIG["index_filename"])
    if not os.path.exists(idx_path):
        return {}
    with open(idx_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def update_archive_index(new_entries: Dict[str, str], cache_dir: Optional[str] = None) -> Dict[str, str]:
    """Merge newly-fetched months into the archive index and persist it."""
    cache_dir = cache_dir or pwcfg.PROCESSING_CONFIG["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    idx_path = os.path.join(cache_dir, pwcfg.ARCHIVE_CONFIG["index_filename"])
    index = load_archive_index(cache_dir)
    index.update(new_entries)
    with open(idx_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
    return index
