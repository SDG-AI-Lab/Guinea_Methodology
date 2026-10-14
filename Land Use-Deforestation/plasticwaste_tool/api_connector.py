"""
API connector for the DPWMS Plastic Waste module.

Reuses landuse_tool's shared infrastructure rather than re-implementing it
(item 3 in the brief: "align data loader / API connection with the other
two modules"):
  - GEE service-account authentication  -> landuse_tool.api_connector.initialize_gee
  - Guinea admin boundary (GeoBoundaries) -> landuse_tool.api_connector.get_admin_boundary
  - Blob-Storage cache + Key Vault secrets -> landuse_tool.azure_utils
  - OSM/Geofabrik vector fetch -> landuse_tool.api_connector.fetch_osm_vectors

What THIS module adds on top (DPWMS-specific, ported from the GEE App
prototype `plasticwaste_analysis.js`, translated from client-side JS to the
server-side `earthengine-api` Python client so it can run headless in a
Container App job, exactly like landuse_tool's fetchers):
  - Sentinel-2 SR Harmonized collection filtering + SCL cloud/shadow masking
  - Server-side FDI / NDVI / NDWI computation
  - Server-side Otsu threshold search (histogram-based) with a manual fallback
  - Vectorization of the binary hotspot raster + per-patch attributes
  - Tiled download of the resulting rasters/vectors into local/Blob cache
"""

import os
import logging
import tempfile
from typing import Dict, List, Optional, Tuple

import geopandas as gpd

from landuse_tool.azure_utils import BlobCache, load_secrets_into_env
from landuse_tool import api_connector as landuse_api
from landuse_tool import config as landuse_config

from . import config as pwcfg

logger = logging.getLogger(__name__)

load_secrets_into_env()

_EE = None

# GEE's getDownloadURL caps a single request at 48 MiB (50,331,648 bytes) --
# this is a hard server-side limit, not something we can raise. Leave a
# safety margin under it (GeoTIFF header/tag overhead, edge-tile rounding).
_GEE_REQUEST_BYTE_CAP = 50_331_648
_SAFE_REQUEST_BUDGET_BYTES = 42_000_000


def _download_single_band_safely(image, aoi, scale: int, out_path: str) -> str:
    """
    Download ONE band, staying under GEE's per-request byte cap regardless of
    AOI size, by:

      1. Casting to float32 (`.toFloat()`). GEE keeps computed bands (from
         `.divide()`, `.subtract()`, `.normalizedDifference()`, etc. -- i.e.
         every band this module produces: FDI, NDVI, NDWI, and the
         cloud-masked/rescaled raw S2 bands) as float64 by DEFAULT, silently
         doubling the byte count versus what a naive "4 bytes/pixel" estimate
         assumes. This is the actual reason a single band can still overflow
         the cap even after the earlier per-band split.
      2. Tiling the AOI ourselves with a tile size derived from an explicit
         byte budget (not a fixed pixel count), so it scales correctly for
         any AOI/resolution combination this module uses.

    Deliberately self-contained (does NOT call
    `landuse_tool.api_connector._download_ee_image_tiled`): that function's
    exact signature on your machine doesn't match what a modified copy of
    `landuse_tool` looked like earlier (it doesn't accept `max_dim_px`), and
    since it's a private (`_`-prefixed) helper there's no stable contract to
    rely on. This uses only `ee.Image.getDownloadURL()`, the public/stable
    Earth Engine API, so it can't break again the same way regardless of
    what landuse_tool's internals happen to look like on any given machine.
    """
    import math
    import rasterio
    from rasterio.merge import merge as rasterio_merge

    img_f32 = image.toFloat()
    bytes_per_pixel = 4  # float32, after the cast above
    max_dim_px = max(256, int((_SAFE_REQUEST_BUDGET_BYTES / bytes_per_pixel) ** 0.5))
    crs = landuse_config.PROCESSING_CONFIG.get("target_crs", "EPSG:4326")

    bounds = aoi.bounds().getInfo()["coordinates"][0]
    xs = [c[0] for c in bounds]
    ys = [c[1] for c in bounds]
    west, east, south, north = min(xs), max(xs), min(ys), max(ys)

    deg_per_px = scale / 111_320.0  # approx metres -> degrees at the equator
    span_x_px = (east - west) / deg_per_px
    span_y_px = (north - south) / deg_per_px
    nx = max(1, math.ceil(span_x_px / max_dim_px))
    ny = max(1, math.ceil(span_y_px / max_dim_px))
    logger.info("Tiled download: %dx%d tiles @ %dm, max_dim_px=%d for %s",
               nx, ny, scale, max_dim_px, os.path.basename(out_path))

    tmp_dir = tempfile.mkdtemp(prefix="pw_tiles_", dir=pwcfg.PROCESSING_CONFIG["temp_dir"])
    dx, dy = (east - west) / nx, (north - south) / ny
    tile_files = []
    for i in range(nx):
        for j in range(ny):
            w, e = west + i * dx, west + (i + 1) * dx
            s, n = south + j * dy, south + (j + 1) * dy
            region = _EE.Geometry.Rectangle([w, s, e, n], proj=crs, geodesic=False)
            url = img_f32.getDownloadURL({
                "scale": scale, "crs": crs, "region": region, "format": "GEO_TIFF",
            })
            tile_path = os.path.join(tmp_dir, f"tile_{i}_{j}.tif")
            _download_url_with_retry(url, tile_path)
            if os.path.getsize(tile_path) > 0:
                tile_files.append(tile_path)

    if not tile_files:
        raise RuntimeError(f"No tiles downloaded for {out_path} -- check AOI/date window.")

    srcs = [rasterio.open(f) for f in tile_files]
    mosaic, out_transform = rasterio_merge(srcs)
    out_meta = srcs[0].meta.copy()
    out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2],
                     "transform": out_transform})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    return out_path


def _download_url_with_retry(url: str, out_path: str, retries: int = 3, timeout: int = 120) -> None:
    """Plain requests-based download with retry -- avoids depending on
    landuse_tool's private `_download_with_retry` helper for the same reason
    noted in `_download_single_band_safely`."""
    import time
    import requests

    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("Download attempt %d/%d failed for tile: %s", attempt + 1, retries, e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {out_path} after {retries} attempts: {last_err}")


def _cache() -> BlobCache:
    return BlobCache(
        container=pwcfg.PROCESSING_CONFIG.get("azure_container"),
        prefix=pwcfg.PROCESSING_CONFIG.get("azure_prefix", "plasticwaste"),
        local_root=pwcfg.PROCESSING_CONFIG["cache_dir"],
        connection_string=pwcfg.PROCESSING_CONFIG.get("azure_connection_string"),
        account_url=pwcfg.PROCESSING_CONFIG.get("azure_account_url"),
    )


# ============================================================================
# INITIALIZATION — delegate straight to landuse_tool (one GEE service account,
# one Guinea boundary, shared across both GEDH components per HUB-M2).
# ============================================================================
def initialize_gee() -> bool:
    global _EE
    import ee
    ok = landuse_api.initialize_gee()
    _EE = ee
    return ok


def get_admin_boundary() -> Optional[gpd.GeoDataFrame]:
    return landuse_api.get_admin_boundary()


def aoi_cache_token(aoi_key: str = pwcfg.DEFAULT_AOI_KEY) -> str:
    """
    Short hash of an AOI's bounding box, for inclusion in cache keys.

    Without this the cache key is `{aoi_key}_{start}_{end}`, which does not
    encode the geometry — so editing `AOI_CONFIG` and re-running silently
    returns the raster for the OLD box. That happened: Fatala's box was
    retightened from 2.2% water to a measured 16.4%, the run reported the old
    201,083-pixel water population unchanged, and the region stayed flagged as
    not assessed. A stale cache is worse than a slow one, because the numbers
    look plausible.

    The FDI formula id (S2_CONFIG["fdi_formula"]) is hashed in as well, for the
    same reason: a raster written under an earlier index definition must not be
    returned for the current one.
    """
    import hashlib

    bbox = pwcfg.AOI_CONFIG.get(aoi_key, {}).get("bbox")
    if bbox is None:
        return "nobbox"
    formula = pwcfg.S2_CONFIG.get("fdi_formula", "")
    return hashlib.sha1(repr((tuple(round(float(v), 6) for v in bbox), formula))
                        .encode()).hexdigest()[:8]


def get_aoi_geometry(aoi_key: str = pwcfg.DEFAULT_AOI_KEY):
    """Return an ee.Geometry.BBox for a named AOI (mirrors `getAOI()` in the
    GEE App prototype)."""
    if _EE is None:
        initialize_gee()
    minx, miny, maxx, maxy = pwcfg.get_aoi_bbox(aoi_key)
    return _EE.Geometry.BBox(minx, miny, maxx, maxy)


# ============================================================================
# SENTINEL-2: cloud mask, composite, spectral indices  (server-side / ee ops)
# ============================================================================
def _mask_s2_clouds(image):
    """SCL-based cloud/shadow mask + SR rescale to 0-1 reflectance.
    Ported 1:1 from `maskS2clouds()` in plasticwaste_analysis.js."""
    scl = image.select("SCL")
    keep = pwcfg.S2_CONFIG["scl_clear_classes"]
    clear = scl.eq(keep[0])
    for c in keep[1:]:
        clear = clear.Or(scl.eq(c))
    return image.updateMask(clear).divide(10000).copyProperties(
        image, ["system:time_start"])


def _compute_fdi(image):
    """Floating Debris Index (Biermann et al. 2020, Sci. Rep. 10, 5364).

    FDI = B8 - [B6 + (B11 - B6) * (lambda_B8 - lambda_B4)/(lambda_B11 -
    lambda_B4) * 10]

    Must stay identical to `fdi_tool.spectral.compute_fdi`, the offline path;
    methodology §7.1.2 checks the two agree.
    """
    bands = pwcfg.S2_CONFIG["bands"]
    lam = pwcfg.S2_CONFIG["lambda_nm"]
    b8, b6, b11 = image.select(bands["B8"]), image.select(bands["B6"]), image.select(bands["B11"])
    wl_factor = (lam["B8"] - lam["B4"]) / (lam["B11"] - lam["B4"])
    fdi = b8.subtract(b6.add(b11.subtract(b6).multiply(wl_factor).multiply(10))).rename("FDI")
    return image.addBands(fdi)


def _compute_ndvi(image):
    bands = pwcfg.S2_CONFIG["bands"]
    return image.addBands(image.normalizedDifference([bands["B8"], bands["B4"]]).rename("NDVI"))


def _compute_ndwi(image):
    bands = pwcfg.S2_CONFIG["bands"]
    return image.addBands(image.normalizedDifference([bands["B3"], bands["B8"]]).rename("NDWI"))


def _surface_water_mask(scale_m: Optional[float] = None):
    """
    Binary "this pixel is on permanent surface water" mask, from ESA WorldCover
    class 80.

    This is a POSITIVE constraint, and the NDVI/NDWI pair in
    `_apply_spectral_masks` is not a substitute for it: those two only exclude
    vegetation and open water, so every non-vegetated, non-water pixel
    survives -- bare soil, sand, concrete, port and industrial rooftops. Those
    have strong NIR/SWIR, hence a large FDI excess, so they were landing in the
    HIGH confidence band. Floating debris is by definition on water; requiring
    that explicitly is what separates it from a warehouse roof.

    Known limit, measured rather than assumed: this removes debris candidates
    that sit on LAND, but not ones sitting on structures BUILT OVER water --
    quays, jetties, moored vessels. Two of the five infrastructure false
    positives in the photo-interpreted sample read as water in both WorldCover
    and JRC GSW because they genuinely are over water. Those need a port-area
    exclusion polygon or vessel detection, not a better water mask.
    """
    ee = _EE
    cfg = pwcfg.DETECTION_CONFIG
    scale_m = scale_m or pwcfg.S2_CONFIG["download_scale_m"]
    landcover = ee.ImageCollection(cfg["water_mask_asset"]).first().select("Map")

    water = landcover.eq(cfg["water_mask_class"])
    # units="meters" throughout: with units="pixels" the radius silently
    # rescales with whatever projection the graph happens to be evaluated at.
    if cfg.get("water_buffer_m", 0):
        water = water.focal_max(radius=cfg["water_buffer_m"],
                                kernelType="circle", units="meters")

    built = landcover.eq(cfg["built_mask_class"])
    if cfg.get("built_buffer_m", 0):
        built = built.focal_max(radius=cfg["built_buffer_m"],
                                kernelType="circle", units="meters")

    return water.And(built.Not()).rename("WATER")


def _apply_spectral_masks(image, ndvi_thresh: float, ndwi_thresh: float,
                          water_mask=None):
    """Exclude vegetation (NDVI > thresh) and open water (NDWI > thresh),
    isolating the residual floating-debris signature. Ported from
    `applySpectralMasks()`, plus the positive surface-water constraint from
    `_surface_water_mask()` (see that docstring for why the NDVI/NDWI pair
    alone lets coastal infrastructure through)."""
    veg_mask = image.select("NDVI").lt(ndvi_thresh)
    ndwi_mask = image.select("NDWI").lt(ndwi_thresh)
    keep = veg_mask.And(ndwi_mask)
    if water_mask is not None:
        keep = keep.And(water_mask)
    return image.updateMask(keep)


def _otsu_threshold(image, band: str, aoi, scale: Optional[int] = None):
    """Otsu (1979) unsupervised bimodal threshold on a single-band histogram
    within the AOI. Ported 1:1 from `otsuThreshold()`, including the
    iterative inter-class-variance maximisation (kept server-side / lazy so it
    composes with the rest of the ee graph)."""
    ee = _EE
    scale = scale or pwcfg.DETECTION_CONFIG["otsu_scale_m"]
    histogram = image.select(band).reduceRegion(
        reducer=ee.Reducer.autoHistogram(
            maxBuckets=pwcfg.DETECTION_CONFIG["otsu_histogram_bins"], minBucketWidth=0.0001),
        geometry=aoi, scale=scale, maxPixels=1e9, bestEffort=True)

    hist_array = ee.Array(ee.List(histogram.get(band)))
    counts = hist_array.slice(1, 0, 1).project([0])
    means = hist_array.slice(1, 1, 2).project([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])

    indexes = ee.List.sequence(1, counts.length().get([0]).subtract(1))
    init = ee.Dictionary({"threshold": -1, "maxVar": -1})

    def _iterate(i, prev):
        i = ee.Number(i)
        d = ee.Dictionary(prev)
        w0 = counts.slice(0, 0, i).reduce(ee.Reducer.sum(), [0]).get([0]).divide(total)
        w1 = ee.Number(1).subtract(w0)
        m0 = (means.slice(0, 0, i).multiply(counts.slice(0, 0, i))
              .reduce(ee.Reducer.sum(), [0]).get([0])
              .divide(counts.slice(0, 0, i).reduce(ee.Reducer.sum(), [0]).get([0])))
        m1 = (means.slice(0, i).multiply(counts.slice(0, i))
              .reduce(ee.Reducer.sum(), [0]).get([0])
              .divide(counts.slice(0, i).reduce(ee.Reducer.sum(), [0]).get([0])))
        inter_class_var = w0.multiply(w1).multiply(m0.subtract(m1).pow(2))
        better = inter_class_var.gt(d.get("maxVar"))
        return ee.Dictionary({
            "threshold": better.multiply(means.get([i])).add(
                ee.Number(1).subtract(better).multiply(d.get("threshold"))),
            "maxVar": better.multiply(inter_class_var).add(
                ee.Number(1).subtract(better).multiply(d.get("maxVar"))),
        })

    result = indexes.iterate(_iterate, init)
    return ee.Number(ee.Dictionary(result).get("threshold"))


def _sigma_threshold(image, band: str, aoi, scale: Optional[int] = None):
    """Server-side mean + n*sigma over the masked band — the Digital Earth
    Africa floating-debris rule, mirroring `detection.sigma_threshold` so the
    ee graph and the offline recomputation agree. See that function for the
    reference and for why this replaced Otsu at whole-AOI scope.

    Scale note (matters at national scope): this reduces at `sigma_scale_m`,
    i.e. the scale the threshold is APPLIED at, not the coarse `otsu_scale_m`
    the histogram search uses. See that config key for why the two differ.

    `bestEffort=True` is retained as a guard against an outright failure, but
    it degrades SILENTLY -- past `maxPixels` it raises the reduction scale
    instead of erroring, which reintroduces exactly the sigma-shrinking bias
    described above with no visible signal. `maxPixels` is therefore set high
    enough to cover the coastal AOI at 20 m (~152 M pixels) without triggering
    it. An AOI large enough to exceed 1e10 must be split and run per-region,
    which is the recommended national workflow anyway (per-region statistics
    are more defensible than one pooled coastal threshold).
    """
    ee = _EE
    cfg = pwcfg.DETECTION_CONFIG
    scale = scale or cfg.get("sigma_scale_m") or pwcfg.S2_CONFIG["download_scale_m"]
    stats = image.select(band).reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
        geometry=aoi, scale=scale, maxPixels=1e10, bestEffort=True)
    mean = ee.Number(stats.get(f"{band}_mean"))
    sd = ee.Number(stats.get(f"{band}_stdDev"))
    return mean.add(sd.multiply(cfg["threshold_n_sigma"]))


def _vectorize_hotspots(binary_image, aoi, acquisition_date: str, cloud_pct):
    """Vectorize the binary hotspot raster and attach per-patch attributes
    (area, concentration score, acquisition metadata). Ported from
    `vectorizeHotspots()`."""
    ee = _EE
    cfg = pwcfg.DETECTION_CONFIG
    vectors = binary_image.selfMask().reduceToVectors(
        reducer=ee.Reducer.countEvery(), geometry=aoi, scale=cfg["vectorize_scale_m"],
        maxPixels=1e9, bestEffort=True, geometryType="polygon", eightConnected=False)

    min_count = cfg["min_patch_area_m2"] / (cfg["vectorize_scale_m"] ** 2)
    vectors = vectors.filter(ee.Filter.gte("count", min_count))

    def _attrs(feat):
        area_m2 = feat.geometry().area(1)
        score = area_m2.divide(cfg["max_expected_patch_m2"]).min(1.0)
        return (feat.set("floating_debris_score", score)
                .set("acquisition_date", acquisition_date)
                .set("cloud_cover_pct", cloud_pct)
                .set("confidence_level", "MEDIUM")
                .set("area_m2", area_m2)
                .set("module", "M2_FDI")
                .set("source", "Sentinel-2_SR_Harmonized"))

    return vectors.map(_attrs)


# ============================================================================
# TOP-LEVEL PIPELINE ENTRY POINT (server-side graph construction)
# ============================================================================
def build_fdi_pipeline(start_date: str, end_date: str, aoi,
                       ndvi_thresh: Optional[float] = None,
                       ndwi_thresh: Optional[float] = None,
                       apply_cloud_mask: bool = True) -> Dict:
    """
    Build the full FDI mesoscale-hotspot ee graph for one date window.
    Mirrors `runM2Pipeline()` in plasticwaste_analysis.js, but as a Python
    function so it can be called from a scheduled job / notebook instead of
    only the interactive GEE App.

    Returns a dict of (still-lazy) ee objects; call `.getInfo()` /
    `download_fdi_results()` to materialize.
    """
    if _EE is None:
        initialize_gee()
    ee = _EE
    cfg = pwcfg.S2_CONFIG
    ndvi_thresh = pwcfg.DETECTION_CONFIG["ndvi_threshold"] if ndvi_thresh is None else ndvi_thresh
    ndwi_thresh = pwcfg.DETECTION_CONFIG["ndwi_threshold"] if ndwi_thresh is None else ndwi_thresh

    s2_raw = (ee.ImageCollection(cfg["collection"])
              .filterDate(start_date, end_date)
              .filterBounds(aoi)
              .filter(ee.Filter.lt(cfg["cloud_property"], pwcfg.DETECTION_CONFIG.get(
                  "sar_fallback_cloud_pct", cfg["cloud_threshold_pct"]))))
    collection_size = s2_raw.size()
    cloud_pct = s2_raw.aggregate_mean(cfg["cloud_property"])

    s2 = s2_raw.map(_mask_s2_clouds) if apply_cloud_mask else s2_raw
    composite = s2.median().clip(aoi)
    composite = _compute_fdi(composite)
    composite = _compute_ndvi(composite)
    composite = _compute_ndwi(composite)

    # Carried as a band so the offline path (spectral.apply_spectral_masks via
    # pipeline.run_from_arrays) masks on exactly the same water pixels this
    # graph does, instead of re-deriving them on a possibly different grid.
    water_mask = _surface_water_mask()
    composite = composite.addBands(water_mask.rename("WATER"))

    # Same dispatch as the offline path, so the server-side products and the
    # locally-recomputed ones agree.
    rule = pwcfg.DETECTION_CONFIG.get("detection_rule", "de_africa")

    if rule == "de_africa":
        # Water only -- no NDVI/NDWI exclusion. Open water IS the background the
        # anomaly is measured against. See detection.debris_binary_mask_de_africa.
        water_only = composite.updateMask(water_mask)
        masked_fdi = water_only.select("FDI")
        threshold = _sigma_threshold(masked_fdi, "FDI", aoi)
        threshold_ndvi = _sigma_threshold(water_only.select("NDVI"), "NDVI", aoi)
        hotspot_binary = (masked_fdi.gt(threshold)
                          .And(water_only.select("NDVI").gt(threshold_ndvi))
                          .rename("hotspot"))
    else:
        masked_fdi = _apply_spectral_masks(
            composite, ndvi_thresh, ndwi_thresh, water_mask=water_mask).select("FDI")
        threshold_ndvi = None
        method = pwcfg.DETECTION_CONFIG.get("threshold_method", "sigma")
        if method == "sigma":
            threshold = _sigma_threshold(masked_fdi, "FDI", aoi)
        elif method == "manual":
            threshold = ee.Number(pwcfg.DETECTION_CONFIG["fdi_manual_threshold"])
        else:
            otsu_t = _otsu_threshold(masked_fdi, "FDI", aoi)
            threshold = ee.Number(ee.Algorithms.If(
                otsu_t, otsu_t, pwcfg.DETECTION_CONFIG["fdi_manual_threshold"]))
        hotspot_binary = masked_fdi.gt(threshold).rename("hotspot")
    hotspot_vectors = _vectorize_hotspots(hotspot_binary, aoi, start_date, cloud_pct)

    return {
        "composite": composite,
        "fdi": composite.select("FDI"),
        "ndvi": composite.select("NDVI"),
        "ndwi": composite.select("NDWI"),
        "masked_fdi": masked_fdi,
        "hotspot_binary": hotspot_binary,
        "hotspot_vectors": hotspot_vectors,
        "threshold": threshold,
        "collection_size": collection_size,
        "cloud_pct": cloud_pct,
    }


def download_fdi_results(results: Dict, aoi, output_prefix: str) -> Dict[str, str]:
    """
    Materialize a `build_fdi_pipeline()` result: download the FDI raster
    (tiled, via the same helper landuse_tool uses for country-scale exports)
    and the hotspot vectors (GeoJSON), cache both to Blob/local, and return
    local file paths + scalar metadata.
    """
    cache = _cache()
    out: Dict[str, str] = {}

    fdi_key = f"rasters/{output_prefix}_fdi.tif"
    if cache.exists(fdi_key):
        out["fdi_path"] = cache.download(fdi_key)
    else:
        local = cache.local_path(fdi_key)
        _download_single_band_safely(
            results["fdi"], aoi, pwcfg.S2_CONFIG["download_scale_m"], local)
        cache.upload(fdi_key, local)
        out["fdi_path"] = local

    vectors_key = f"vectors/{output_prefix}_hotspots.geojson"
    if cache.exists(vectors_key):
        out["hotspots_path"] = cache.download(vectors_key)
    else:
        local = cache.local_path(vectors_key)
        geojson = results["hotspot_vectors"].getInfo()
        gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
        gdf.to_file(local, driver="GeoJSON")
        cache.upload(vectors_key, local)
        out["hotspots_path"] = local

    out["threshold"] = results["threshold"].getInfo()
    out["collection_size"] = results["collection_size"].getInfo()
    out["cloud_pct"] = results["cloud_pct"].getInfo()
    return out


# ============================================================================
# OSM / ANCILLARY (waste-flow context: roads for collector routing, POIs for
# candidate drop-off centres) — reuses landuse_tool's OSM fetcher verbatim.
# ============================================================================
def download_composite_bands(results: Dict, aoi, output_prefix: str,
                             bands: Optional[List[str]] = None,
                             force_refresh: bool = False) -> str:
    """
    Download a multi-band GeoTIFF (raw S2 bands + FDI/NDVI/NDWI) for a scene,
    so `spectral.py` can independently recompute the indices offline from the
    raw bands and cross-check the server-side (ee) computation — the same
    "recompute locally, don't just trust the graph" pattern used for
    landuse_tool's driver stack. Not needed for routine runs (the FDI-only
    download in `download_fdi_results` is cheaper); use for QA / validation.

    Implementation note -- GEE per-request size cap
    -------------------------------------------------
    `getDownloadURL` caps a single request at 48 MiB (50,331,648 bytes). The
    deforestation module hits the same cap on its country-scale continuous
    predictors and works around it by COARSENING RESOLUTION (see
    `config.PROCESSING_CONFIG["target_resolution"] = 250` in run_analysis.ipynb's
    Safe Mode cell). That fix is wrong for THIS module: `min_patch_area_m2` is
    500 -- smaller than a single 250 m pixel (62,500 m2) -- so coarsening would
    make FDI patches physically undetectable, not just less precise.

    The actual cause here is different anyway: bundling all `len(bands)` bands
    into ONE request multiplies the byte count by the band count, not the
    pixel count. So instead of coarsening, each band is downloaded as its own
    (small, well-under-cap) request at the SAME fine `download_scale_m`, then
    stacked locally into one multi-band GeoTIFF. Resolution is fully preserved.
    """
    # Same order as data_loader.DEFAULT_COMPOSITE_BANDS, which reads it back.
    bands = bands or ["B3", "B4", "B6", "B8", "B8A", "B11", "B12", "FDI", "NDVI", "NDWI", "WATER"]
    cache = _cache()
    key = f"rasters/{output_prefix}_composite.tif"
    if cache.exists(key) and not force_refresh:
        return cache.download(key)

    local = cache.local_path(key)
    _download_band_stack(results["composite"], aoi, bands, local,
                         band_cache_prefix=f"rasters/{output_prefix}_composite",
                         force_refresh=force_refresh)
    cache.upload(key, local)
    return local


def _download_band_stack(image, aoi, bands: List[str], out_path: str,
                         scale: Optional[int] = None,
                         band_cache_prefix: Optional[str] = None,
                         force_refresh: bool = False) -> str:
    """
    Download `bands` from one ee.Image as separate per-band requests (see
    `download_composite_bands` for why the split is necessary) and stack them
    into a single multi-band GeoTIFF at `out_path`, in the given band order.

    `band_cache_prefix` opts into caching each intermediate single-band file
    under `{prefix}_{band}.tif`; the monthly archive skips it, since only the
    stacked month raster is ever read back.

    Shared by the main composite path and the monthly archive so both produce
    byte-identical stacking behaviour -- notably the single shared `nodata` tag
    taken from the first band, which `data_loader.read_composite` compensates
    for on read.
    """
    import rasterio
    scale = scale or pwcfg.S2_CONFIG["download_scale_m"]
    cache = _cache()
    band_paths: List[str] = []
    for band in bands:
        if band_cache_prefix:
            band_key = f"{band_cache_prefix}_{band}.tif"
            if cache.exists(band_key) and not force_refresh:
                band_paths.append(cache.download(band_key))
                continue
            band_local = cache.local_path(band_key)
        else:
            band_local = f"{os.path.splitext(out_path)[0]}_{band}.tif"
        _download_single_band_safely(image.select([band]), aoi, scale, band_local)
        if band_cache_prefix:
            cache.upload(band_key, band_local)
        band_paths.append(band_local)

    with rasterio.open(band_paths[0]) as ref:
        profile = ref.profile.copy()
    profile.update(count=len(bands))
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, path in enumerate(band_paths, start=1):
            with rasterio.open(path) as src:
                dst.write(src.read(1), i)
            dst.set_band_description(i, bands[i - 1])
    return out_path


def fetch_accessibility_layers(roi_gdf: gpd.GeoDataFrame) -> Dict[str, Optional[str]]:
    """Roads + waterways near the coastal AOI, for Waste-Flow routing context
    (GEDH-FR-0505). Delegates to landuse_tool's OSM fetcher so both modules
    stay on the same OSM/Geofabrik ingestion path and cache convention."""
    return {
        "roads": landuse_api.fetch_osm_vectors(roi_gdf, {"highway": True}, "pw_roads.geojson"),
        "waterways": landuse_api.fetch_osm_vectors(
            roi_gdf, {"waterway": True, "natural": "water"}, "pw_waterways.geojson"),
    }


# ============================================================================
# MONTHLY FDI ARCHIVE (GEDH-FR-0507) — fetch + cache one composite per month
# ============================================================================
def build_monthly_archive(year_months: List[Tuple[int, int]], aoi_key: str = pwcfg.DEFAULT_AOI_KEY,
                          force_refresh: bool = False) -> Dict[str, str]:
    """
    Fetch (or reuse cached) monthly FDI rasters for a list of (year, month)
    tuples. Populates the Monthly FDI Archive sub-view's date-range slider
    (GEDH-FR-0507). Returns {'YYYY-MM': raster_path}.

    Band content depends on `DETECTION_CONFIG['detection_rule']`
    -----------------------------------------------------------
    Under the legacy rule the detection decision is a single threshold on one
    masked FDI band, so a one-band export is sufficient. Under the de_africa
    rule it is a CONJUNCTION over two independently-thresholded indices, so a
    one-band export cannot reproduce it -- the offline side can only apply the
    FDI half, which on Conakry 2024-01 flags 102,827 pixels against the
    6,558 the two-index rule flags. The archive therefore exports the same
    (FDI, NDVI, WATER) stack the main composite path carries and lets
    `detection.debris_binary_mask_de_africa` make the decision locally, so both
    paths run identical logic over an identical pixel population.

    The rule is part of the cache key: a file written under one rule has a
    different band count from the other and would otherwise be silently reused
    and then fail (or, worse, half-apply) on read.
    """
    cache = _cache()
    aoi = get_aoi_geometry(aoi_key)
    rule = pwcfg.DETECTION_CONFIG.get("detection_rule", "de_africa")
    archive_bands = pwcfg.ARCHIVE_CONFIG["bands_de_africa"] if rule == "de_africa" else None
    out: Dict[str, str] = {}
    for year, month in year_months:
        label = f"{year:04d}-{month:02d}"
        # The band count is in the key: a stack exported with fewer bands (the
        # earlier FDI/NDVI/WATER-only archive) must not be returned for this one.
        nb = f"_{len(archive_bands)}b" if archive_bands else ""
        key = (f"{pwcfg.ARCHIVE_CONFIG['cache_prefix']}/"
               f"{aoi_key}_{aoi_cache_token(aoi_key)}_{label}_{rule}{nb}.tif")
        if cache.exists(key) and not force_refresh:
            out[label] = cache.download(key)
            continue
        start = f"{year:04d}-{month:02d}-01"
        end_month, end_year = (month % 12) + 1, year + (1 if month == 12 else 0)
        end = f"{end_year:04d}-{end_month:02d}-01"
        try:
            results = build_fdi_pipeline(start, end, aoi)
            local = cache.local_path(key)
            if archive_bands is None:
                # masked_fdi, not fdi: the raw FDI band has no vegetation/water
                # mask applied, so thresholding it floods most of the AOI as
                # "hotspot" instead of the residual floating-debris signature
                # the main composite path actually detects on.
                _download_single_band_safely(
                    results["masked_fdi"], aoi, pwcfg.S2_CONFIG["download_scale_m"], local)
            else:
                _download_band_stack(results["composite"], aoi, archive_bands, local)
            cache.upload(key, local)
            out[label] = local
        except Exception as e:  # noqa: BLE001
            logger.error("Monthly archive fetch failed for %s: %s", label, e)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    initialize_gee()
    aoi = get_aoi_geometry("conakry")
    res = build_fdi_pipeline(pwcfg.S2_CONFIG["default_start_date"],
                             pwcfg.S2_CONFIG["default_end_date"], aoi)
    paths = download_fdi_results(res, aoi, "conakry_2024-01")
    logger.info("Downloaded: %s", paths)
