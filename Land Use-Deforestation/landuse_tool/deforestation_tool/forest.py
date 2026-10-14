"""
Forest baseline & masks.

Two ways to define "forest", both returning boolean rasters on the analysis grid:

  1. Hansen canopy: forest = treecover2000 >= canopy_threshold_pct, and forest
     extent in year Y = forest2000 AND NOT (loss occurred on/before Y). This is
     the Global-Forest-Watch convention and the recommended default.
  2. LULC class: forest = LULC(Y) in FOREST_CLASSES[product]. Uses the Land Use
     module's multitemporal LULC directly and tells you what forest converts TO.

Hansen is one multi-band GEE Image; `fetch_hansen()` downloads the two bands we
need (treecover2000, lossyear) via the SAME primitives api_connector uses for
every other layer, so it lands in Blob cache exactly like the rest.
"""

import logging
import numpy as np

from .. import config as base
from . import config as dcfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetch (reuses api_connector: GEE auth, geometry, tiled/GCS download, cache)
# ---------------------------------------------------------------------------
def fetch_hansen(roi_gdf, out_key="targets/HANSEN_GFC.tif", asset=None, scale=30):
    """
    Download Hansen [treecover2000, lossyear, datamask] over the ROI as a 3-band GeoTIFF.
    Returns the cached local path. Uses api_connector's download stack so it is
    tiled/mosaicked (or GCS-exported) and Blob-cached like every other layer.
    """
    from .. import api_connector
    if not base.PROCESSING_CONFIG.get("gee_authenticated"):
        api_connector.initialize_gee()
    ee = api_connector._EE
    cache = api_connector._cache()
    if cache.exists(out_key):
        return cache.download(out_key)

    asset = asset or dcfg.HANSEN_GFC["asset"]
    ee_roi = api_connector._gdf_to_ee_geometry(roi_gdf)
    # Band 3, datamask (1 = land), gives land areas and forest shares
    # (feature 005, contracts/toolchain.md §3-§4). Files stored before it was
    # added have two bands; read_datamask then answers None.
    img = ee.Image(asset).select(["treecover2000", "lossyear", "datamask"]).clip(ee_roi)

    # Hansen is 30 m; at country scale prefer a coarser working scale to match
    # the analysis grid (TARGET_RESOLUTION). Alignment happens in data_loader.
    scale = max(scale, base.PROCESSING_CONFIG["target_resolution"])
    local = cache.local_path(out_key)
    api_connector._download_ee_image(img, ee_roi, scale, local)
    cache.upload(out_key, local)
    return local


def read_hansen(hansen_path):
    """Read the 2-band Hansen file -> (treecover_pct, lossyear) float arrays."""
    import rasterio
    with rasterio.open(hansen_path) as src:
        treecover = src.read(1).astype("float32")
        lossyear = src.read(2).astype("float32") if src.count >= 2 else None
    return treecover, lossyear


def read_datamask(hansen_path):
    """Hansen `datamask` (band 3: 1 = land, 2 = water, 0 = no data), or None
    for a file stored without it."""
    import rasterio
    with rasterio.open(hansen_path) as src:
        return src.read(3) if src.count >= 3 else None


# ---------------------------------------------------------------------------
# Forest masks
# ---------------------------------------------------------------------------
def forest_mask_from_hansen(treecover_pct, threshold=None):
    """Boolean forest mask from Hansen canopy percentage."""
    threshold = dcfg.DEFOR_PARAMS["canopy_threshold_pct"] if threshold is None else threshold
    return np.asarray(treecover_pct) >= threshold


def forest_mask_from_lulc(lulc_array, product_key):
    """Boolean forest mask from a LULC raster using the product's forest classes."""
    classes = dcfg.FOREST_CLASSES.get(product_key)
    if not classes:
        raise ValueError(f"No forest-class mapping for product '{product_key}'")
    return np.isin(np.asarray(lulc_array), classes)


def decode_hansen_lossyear(lossyear_array, base_year=2000):
    """
    Turn Hansen's lossyear band (0 = no loss, n = loss in year base_year+n) into
    a {year: boolean loss mask} dict for every year that has loss.
    """
    arr = np.asarray(lossyear_array)
    out = {}
    for code in np.unique(arr[arr > 0]):
        out[int(base_year + code)] = (arr == code)
    return out


def forest_extent_by_year(forest2000, lossyear_array, years, base_year=2000):
    """
    Forest extent per year = forest2000 AND NOT cumulative-loss-up-to-year.
    Returns {year: boolean forest mask}. (Hansen 'gain' is 2000-2012 only and is
    intentionally not added back annually, per the GFW convention.)
    """
    arr = np.asarray(lossyear_array)
    forest2000 = np.asarray(forest2000, dtype=bool)
    out = {}
    for y in sorted(years):
        loss_code = y - base_year
        cumulative_loss = (arr > 0) & (arr <= loss_code)
        out[y] = forest2000 & (~cumulative_loss)
    return out


def apply_min_mapping_unit(mask, min_pixels):
    """Optional: drop connected forest/loss patches smaller than min_pixels."""
    if not min_pixels or min_pixels < 2:
        return mask
    from scipy import ndimage
    labelled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(mask), labelled, index=np.arange(1, n + 1))
    keep = np.isin(labelled, np.nonzero(sizes >= min_pixels)[0] + 1)
    return mask & keep
