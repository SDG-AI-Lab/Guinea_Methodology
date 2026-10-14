"""
API connector for the Land Use tool -- Azure-ready.

What changed vs the original
----------------------------
* GEE now authenticates with a *service account* (headless) instead of the
  interactive ee.Authenticate() flow, which cannot run on a server.
* ROI conversion no longer assumes a single Polygon (Guinea is a MultiPolygon);
  it uses the full geometry via shapely.mapping(union_all()).
* Every GEE fetch selects the correct band and reduces to ONE layer per year.
* Country-scale downloads no longer use getDownloadURL directly (32 MB / 10000px
  limit). Instead the image is either exported to GCS (native resolution) or
  downloaded in tiles and mosaicked (default, no extra setup).
* Proper initializers for OpenTopography (API key), Copernicus (OAuth2),
  WorldPop (REST), GeoBoundaries (two-step), STAC and OSM.
* A multitemporal orchestrator discovers ALL available years for each dataset
  and fetches only the years not already cached (incremental / regular updates).
"""

import os
import glob
import json
import math
import time
import logging
import tempfile
from typing import Optional, Dict, List

import requests
import geopandas as gpd
from shapely.geometry import mapping

from . import config
from .azure_utils import BlobCache, load_secrets_into_env

logger = logging.getLogger(__name__)

# Load secrets (.env locally / Key Vault on Azure) before anything reads a key.
load_secrets_into_env()

# Lazy handles
_EE = None            # earthengine-api module
_OSMNX_READY = False


def _cache() -> BlobCache:
    return BlobCache(
        container=config.PROCESSING_CONFIG.get("azure_container"),
        prefix=config.PROCESSING_CONFIG.get("azure_prefix", "landuse"),
        local_root=config.PROCESSING_CONFIG["cache_dir"],
        connection_string=config.PROCESSING_CONFIG.get("azure_connection_string"),
        account_url=config.PROCESSING_CONFIG.get("azure_account_url"),
    )


# ============================================================================
# INITIALIZATION
# ============================================================================
def initialize_gee() -> bool:
    """
    Headless Earth Engine init using a service account. Requires:
      GEE_PROJECT_ID, GEE_SERVICE_ACCOUNT, GEE_SA_KEY (the JSON *contents*,
      or a file path -- both handled below).
    Falls back to ee.Initialize() if a local ADC/credentials file exists (dev).
    """
    global _EE
    import ee
    _EE = ee
    project = config.PROCESSING_CONFIG.get("gee_project_id")
    sa_email = config.PROCESSING_CONFIG.get("gee_service_account")
    sa_key = config.get_api_key("earthengine")  # GEE_SA_KEY contents (or a path -- see below)
    try:
        if sa_email and sa_key:
            # ee.ServiceAccountCredentials(key_data=...) expects a bare PEM block,
            # NOT the full JSON key file's contents -- passing the raw JSON string
            # directly fails with "Unable to load PEM file... MalformedFraming",
            # since the parser sees '{"type": "service_account"...' instead of a
            # PEM header. Parse the JSON ourselves and extract the actual
            # private_key field.
            pem_key = sa_key
            try:
                pem_key = json.loads(sa_key)["private_key"]
            except (json.JSONDecodeError, KeyError, TypeError):
                pem_key = sa_key  # assume already a bare PEM string
 
            # Normalize: PEM keys stored in .env files are a well-known source of
            # exactly this "MalformedFraming" error, because a multi-line PEM block
            # doesn't survive single-line .env storage/round-tripping cleanly.
            # Handle every common mangling rather than assume one:
            pem_key = pem_key.strip().strip('"').strip("'")          # accidental wrapping quotes
            if "\\n" in pem_key and "\n" not in pem_key:
                pem_key = pem_key.replace("\\n", "\n")                # literal backslash-n, not a real newline
            if "\n" not in pem_key and "-----BEGIN" in pem_key:
                # newlines stripped entirely -- reinsert them around the PEM
                # framing lines and every 64 base64 characters (standard PEM wrap)
                import re as _re
                body = _re.sub(r"-----BEGIN [A-Z ]+-----|-----END [A-Z ]+-----", "", pem_key).strip()
                header = _re.search(r"-----BEGIN [A-Z ]+-----", pem_key).group()
                footer = _re.search(r"-----END [A-Z ]+-----", pem_key).group()
                wrapped_body = "\n".join(body[i:i+64] for i in range(0, len(body), 64))
                pem_key = f"{header}\n{wrapped_body}\n{footer}\n"
            pem_key = pem_key.replace("\r\n", "\n")                   # CRLF -> LF defensively
 
            # Confirmed real-world case: GEE_SA_KEY sometimes ends up holding a
            # file PATH rather than the key content itself (e.g. GEE_SA_KEY was
            # set directly to a path meant for GEE_SA_KEY_FILE, so whatever
            # ".env"-loading logic populates os.environ never resolved it to
            # actual file contents). Read it directly if it looks like a path
            # to an existing file, rather than fail on an otherwise-fixable typo.
            if not pem_key.strip().startswith("-----BEGIN") and os.path.exists(pem_key.strip()):
                logger.warning("GEE_SA_KEY held a file path (%s), not key contents -- reading it "
                              "directly. Fix your .env: rename this variable to GEE_SA_KEY_FILE "
                              "so this doesn't rely on this fallback.", pem_key.strip())
                with open(pem_key.strip(), "r", encoding="utf-8") as fh:
                    sa_key_from_file = fh.read()
                pem_key = json.loads(sa_key_from_file)["private_key"]
 
            if not pem_key.strip().startswith("-----BEGIN"):
                raise ValueError(f"GEE_SA_KEY private_key does not start with a PEM header after "
                                 f"normalization -- first 60 chars: {pem_key[:60]!r}. This needs a "
                                 f"human look at the actual .env value, not another automatic fix.")
 
            creds = ee.ServiceAccountCredentials(sa_email, key_data=pem_key)
            ee.Initialize(creds, project=project)
            logger.info("GEE initialized via service account (%s).", sa_email)
        else:
            ee.Initialize(project=project)  # dev fallback (uses local creds)
            logger.info("GEE initialized via local credentials.")
        config.PROCESSING_CONFIG["gee_authenticated"] = True
        return True
    except Exception as e:
        logger.error("GEE initialization failed: %s", e)
        config.PROCESSING_CONFIG["gee_authenticated"] = False
        raise


def initialize_all(strict: bool = True) -> Dict[str, bool]:
    """
    Validate credentials and initialize services. Returns a status dict.
    If strict, raises on any missing *required* key.
    """
    problems = config.validate_required_keys()
    if problems and strict:
        raise RuntimeError("Configuration problems:\n - " + "\n - ".join(problems))
    for p in problems:
        logger.warning(p)

    status = {"gee": False, "osm": False}
    try:
        status["gee"] = initialize_gee()
    except Exception:
        if strict:
            raise
    status["osm"] = _init_osmnx()
    return status


def _init_osmnx() -> bool:
    global _OSMNX_READY
    try:
        import osmnx as ox
        ox.settings.overpass_url = os.environ.get(
            "OVERPASS_URL", "https://overpass-api.de/api/interpreter")
        ox.settings.requests_timeout = 600
        ox.settings.overpass_rate_limit = True
        ox.settings.use_cache = True
        ox.settings.cache_folder = os.path.join(
            config.PROCESSING_CONFIG["cache_dir"], "osm")
        os.makedirs(ox.settings.cache_folder, exist_ok=True)
        _OSMNX_READY = True
        return True
    except Exception as e:
        logger.warning("osmnx not ready: %s", e)
        return False


def get_copernicus_session():
    """
    Return an OAuth2 requests session for Copernicus Data Space (Sentinel Hub),
    or None if credentials are absent. Uses client-credentials flow.
    """
    cid = config.get_api_key("copernicus_client_id")
    secret = config.get_api_key("copernicus_client_secret")
    if not (cid and secret):
        logger.info("Copernicus credentials not set; skipping CDSE session.")
        return None
    from oauthlib.oauth2 import BackendApplicationClient
    from requests_oauthlib import OAuth2Session
    token_url = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
                 "protocol/openid-connect/token")
    oauth = OAuth2Session(client=BackendApplicationClient(client_id=cid))
    oauth.fetch_token(token_url=token_url, client_secret=secret, include_client_id=True)
    return oauth


# ============================================================================
# BOUNDARY (GeoBoundaries, pinned local files)
# ============================================================================
def get_admin_boundary(level: Optional[str] = None) -> Optional[gpd.GeoDataFrame]:
    """
    A Guinea administrative boundary from the pinned geoBoundaries files in
    the data directory (feature 002, FR-006). No download: the units every
    result is keyed to are the release the hub loaded.

    `level`: "ADM0" (default, national outline used as the analysis mask),
    "ADM1" (regions), "ADM2" (prefectures) or "ADM3" (sub-prefectures).
    Rows carry `shapeID` (the join key) and `shapeName` (for display).
    """
    cfg = config.ADMIN_BOUNDARY_CONFIG
    if level:
        try:
            cfg = cfg["levels"][level]
        except KeyError:
            raise ValueError(
                f"Unknown admin level {level!r}; "
                f"expected one of {sorted(config.ADMIN_BOUNDARY_CONFIG['levels'])}")

    path = cfg["path"]
    try:
        gdf = gpd.read_file(path)
    except Exception as e:
        logger.error("Pinned boundary %s unreadable at %s: %s", level or "ADM0", path, e)
        return None
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf[gdf.geometry.is_valid].to_crs(config.PROCESSING_CONFIG["target_crs"])
    logger.info("Loaded Guinea boundary %s (%d features).", level or "ADM0", len(gdf))
    return gdf


def _gdf_to_ee_geometry(roi_gdf: gpd.GeoDataFrame):
    """Convert a (possibly Multi)Polygon GeoDataFrame to an ee.Geometry safely."""
    roi_wgs = roi_gdf.to_crs("EPSG:4326")
    geom = roi_wgs.geometry.union_all()          # handles MultiPolygon / islands
    return _EE.Geometry(mapping(geom))


# ============================================================================
# GEE: build one clean layer per year, then download
# ============================================================================
def _reduce(collection, reducer: str):
    return {
        "mode": collection.mode,
        "mean": collection.mean,
        "sum": collection.sum,
        "median": collection.median,
        "max": collection.max,
        "min": collection.min,
    }.get(reducer, collection.mode)()


def build_annual_gee_image(cfg: Dict, year: int, ee_roi):
    """
    Return a single-band ee.Image for one year, band selected and reduced.
    Handles three shapes: Image, ImageCollection (one/many images per year),
    and 'band_per_year' collections (e.g. GLC_FCS30D).
    """
    asset = cfg["asset"]
    reducer = cfg.get("reducer") or "mode"

    # Static single Image (e.g. SRTM)
    if cfg.get("asset_type") == "Image":
        img = _EE.Image(asset).select(cfg["band"])
        return img.clip(ee_roi)

    # Static vector layer painted to a binary raster (e.g. WDPA protected
    # areas). No time dimension -- 'year' is accepted but ignored, matching
    # the calling convention of the Image branch above.
    if cfg.get("asset_type") == "FeatureCollection":
        fc = _EE.FeatureCollection(asset).filterBounds(ee_roi)
        img = _EE.Image().byte().paint(fc, 1).rename(cfg.get("band", "zone")).unmask(0)
        return img.clip(ee_roi)

    # Computed spectral index (e.g. NDVI from Sentinel-2), not a pre-existing
    # band -- cloud-masked via the Scene Classification Layer (SCL), then
    # reduced to an annual composite. Added specifically to give the Crops
    # class stronger spectral separation than the generic predictor set
    # provides (confirmed weak point: Crops precision 0.37 in hold-out
    # validation, substantial over-prediction into other classes).
    if cfg.get("asset_type") == "computed_index":
        def _mask_s2_clouds(img):
            scl = img.select("SCL")
            # keep vegetation(4), bare soil(5), water(6), unclassified(7) --
            # drop cloud shadow(3), cloud medium/high prob(8,9), cirrus(10), snow(11)
            good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
            return img.updateMask(good)

        s2 = (_EE.ImageCollection(asset)
              .filterBounds(ee_roi)
              .filterDate(f"{year}-01-01", f"{year}-12-31")
              .filter(_EE.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
              .map(_mask_s2_clouds))

        index_type = cfg.get("index", "NDVI")
        bands = cfg.get("index_bands", ["B8", "B4"])  # NIR, Red for NDVI
        def _add_index(img):
            return img.normalizedDifference(bands).rename(index_type)

        img = s2.map(_add_index).reduce(_EE.Reducer.median()).rename(index_type)
        return img.clip(ee_roi)

    coll = _EE.ImageCollection(asset).filterBounds(ee_roi)

    # band-per-year collections: pick the band for this year
    if cfg.get("band_per_year"):
        band_index = year - cfg["band_base_year"] + 1  # b1 == base_year
        band = f"b{band_index}"
        img = coll.mosaic().select([band]).rename(cfg.get("band", "class"))
        return img.clip(ee_roi)

    # optional metadata filter (e.g. WorldPop country property)
    gee_filter = cfg.get("gee_filter")
    if gee_filter:
        for k, v in gee_filter.items():
            coll = coll.filter(_EE.Filter.eq(k, v))

    coll_dated = coll.filterDate(f"{year}-01-01", f"{year}-12-31").select(cfg["band"])
    count = coll_dated.size().getInfo()
    if count == 0:
        # Genuinely sparse collections (e.g. AGB_NASA: exactly 1 image, total,
        # confirmed via direct test) often don't carry a system:time_start
        # that falls inside the nominal year_hint range at all -- filterDate
        # then silently returns an empty collection, which fails later with
        # no clear indication that date filtering was the actual cause. This
        # only makes sense to fall back on for genuinely tiny collections
        # (a handful of images) where "the year doesn't quite match" is more
        # likely than "we're accidentally averaging in irrelevant data" --
        # for large collections, an empty date-filtered result usually means
        # a real "no data this year" case that should stay empty, not a
        # metadata quirk to paper over.
        total_count = coll.select(cfg["band"]).size().getInfo()
        if 0 < total_count <= 5:
            logger.warning("%s: filterDate(%d) returned 0 images (collection has "
                           "only %d image(s) total, likely without matching "
                           "system:time_start) -- using the full collection "
                           "unfiltered instead, since year_hint on a near-static "
                           "dataset like this is approximate by design.",
                           asset, year, total_count)
            coll_dated = coll.select(cfg["band"])
        else:
            logger.warning("%s: filterDate(%d) returned 0 images out of %d total "
                           "in the collection -- likely genuinely no data for "
                           "this year, not filtered out.", asset, year, total_count)

    img = _reduce(coll_dated, reducer)
    return img.clip(ee_roi)


def discover_available_years(cfg: Dict, ee_roi) -> List[int]:
    """
    Return the sorted list of years the dataset actually provides over the ROI.
    Clamped to MULTITEMPORAL_CONFIG min/max. This is what makes 'all years' and
    'auto-update when the source adds a year' work without hardcoding.
    """
    mt = config.MULTITEMPORAL_CONFIG
    lo, hi = mt["min_year"], mt["max_year"]

    if cfg.get("asset_type") in ("Image", "FeatureCollection"):
        return []  # static, no per-year concept

    if cfg.get("asset_type") == "computed_index":
        # Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED) reliably covers 2017+,
        # matching this platform's 2017-2024 analysis window exactly -- real
        # per-image discovery isn't meaningful here since the index is
        # recomputed fresh from raw bands every year, not selected from a
        # pre-existing per-year band the way other datasets are. year_hint
        # on the config entry defines the range directly.
        a, b = cfg.get("year_hint", (lo, hi))
        return [y for y in range(a, b + 1) if lo <= y <= hi]

    if cfg.get("band_per_year"):
        n = _EE.ImageCollection(cfg["asset"]).first().bandNames().size().getInfo()
        base = cfg["band_base_year"]
        return [y for y in range(base, base + n) if lo <= y <= hi]

    try:
        coll = _EE.ImageCollection(cfg["asset"]).filterBounds(ee_roi)
        gee_filter = cfg.get("gee_filter")
        if gee_filter:
            for k, v in gee_filter.items():
                coll = coll.filter(_EE.Filter.eq(k, v))
        millis = coll.aggregate_array("system:time_start").getInfo() or []
        years = sorted({time.gmtime(m / 1000).tm_year for m in millis})
        result = [y for y in years if lo <= y <= hi]

        # A collection can have images but no system:time_start property at
        # all (common for static reference products, e.g. AGB_NASA's single
        # image) -- aggregate_array then returns an empty list WITHOUT
        # raising, so this returns [] successfully rather than hitting the
        # except block below. That's a silent, different failure mode from
        # an actual exception, and the except block's year_hint fallback
        # never gets a chance to run. Confirmed via a real case: AGB_NASA's
        # collection has exactly 1 image (checked directly), yet this path
        # returned [] with no error, causing it to be silently dropped
        # downstream despite year_hint=(2010, 2010) being available.
        if not result:
            total_count = coll.size().getInfo()
            if total_count > 0:
                logger.warning("%s: found %d image(s) but none has a usable "
                               "system:time_start in range -- falling back to "
                               "year_hint instead of returning no years.",
                               cfg["asset"], total_count)
                a, b = cfg.get("year_hint", (lo, hi))
                return [y for y in range(a, b + 1) if lo <= y <= hi]
        return result
    except Exception as e:
        logger.warning("Year discovery failed for %s (%s); using year_hint.",
                       cfg["asset"], e)
        a, b = cfg.get("year_hint", (lo, hi))
        return [y for y in range(a, b + 1) if lo <= y <= hi]


def _download_ee_image(image, ee_roi, scale: int, out_path: str) -> str:
    """
    Download an ee.Image over ee_roi to out_path (GeoTIFF).

    Strategy:
      * If GCS_EXPORT_BUCKET is set -> Export to GCS then pull (native-res safe).
      * Else -> tile the region so each tile is < 10000 px and < 32 MB, pull each
        via getDownloadURL, and mosaic. Good for <=~30-500 m country scale.
    """
    crs = config.PROCESSING_CONFIG["target_crs"]
    if config.PROCESSING_CONFIG.get("gcs_export_bucket"):
        return _download_ee_image_via_gcs(image, ee_roi, scale, crs, out_path)
    return _download_ee_image_tiled(image, ee_roi, scale, crs, out_path)


def _fetch_tile_with_subdivision(image, w, s, e, n, scale, crs, tmp_dir, tile_name,
                                  tile_files: list, depth: int = 0, max_depth: int = 3):
    """
    Fetch one tile; if GEE still rejects it as too large (the byte-budget
    estimate in _download_ee_image_tiled is approximate -- encoding overhead
    can vary by dataset), split it into 4 quadrants and retry each, rather
    than failing the whole multi-year fetch on one oversized tile. Bottoms
    out at max_depth (16x finer than the original tile) to avoid an infinite
    loop if a tile is oversized for some other reason (e.g. a genuinely
    malformed region).
    """
    region = _EE.Geometry.Rectangle([w, s, e, n], proj=crs, geodesic=False)
    url = image.getDownloadURL({
        "scale": scale, "crs": crs, "region": region, "format": "GEO_TIFF",
    })
    tile_path = os.path.join(tmp_dir, f"{tile_name}.tif")
    try:
        _download_with_retry(url, tile_path, retries=1)  # size errors aren't transient -- don't burn retries on them
        if os.path.getsize(tile_path) > 0:
            tile_files.append(tile_path)
        return
    except Exception as e:
        is_size_error = "request size" in str(e).lower() or "50331648" in str(e)
        if not is_size_error or depth >= max_depth:
            logger.error("Tile %s failed (depth=%d): %s", tile_name, depth, e)
            return
        logger.info("Tile %s still too large at depth %d -- subdividing into 4.",
                   tile_name, depth)
        mid_x, mid_y = (w + e) / 2, (s + n) / 2
        quadrants = [(w, s, mid_x, mid_y), (mid_x, s, e, mid_y),
                     (w, mid_y, mid_x, n), (mid_x, mid_y, e, n)]
        for qi, (qw, qs, qe, qn) in enumerate(quadrants):
            _fetch_tile_with_subdivision(image, qw, qs, qe, qn, scale, crs, tmp_dir,
                                          f"{tile_name}_{qi}", tile_files, depth + 1, max_depth)


def _download_ee_image_tiled(image, ee_roi, scale, crs, out_path,
                             max_bytes: int = 35_000_000, bytes_per_pixel: int = 4) -> str:
    """
    max_dim_px was previously a fixed 8000 px regardless of scale or band
    count -- fine at fine resolutions/small ROIs, but at country scale and
    coarser resolutions (e.g. 100 m WorldPop over all of Guinea) an 8000 px
    tile covers an 800 km span, so Guinea splits into only ~2 tiles and each
    one is still ~100 MB (4143 x 6114 x 4 bytes for a single float32 band) --
    over GEE's 48 MiB (50,331,648 byte) per-request cap even after "tiling".
    Confirmed via a real failure: POPULATION_DENSITY requests reported
    131,392,105 bytes for what raw-pixel math predicts as ~101,325,528 bytes
    -- about 30% GeoTIFF encoding/header overhead on top of raw pixel data,
    which the byte budget below accounts for directly (35 MB default raw
    budget x ~1.3 encoding overhead ~= 45.5 MB, comfortably under the 48 MiB
    cap, rather than budgeting right up against it).

    Fixed by sizing tiles from an actual BYTE budget divided by
    bytes_per_pixel, rather than a fixed pixel count that doesn't account for
    resolution or band depth. Pass a larger bytes_per_pixel for multi-band
    images (e.g. 12 for a 3-band float32 stack) -- default of 4 assumes a
    single float32 band, correct for every current predictor in this
    pipeline.
    """
    import rasterio
    from rasterio.merge import merge

    max_dim_px = max(1, int(math.sqrt(max_bytes / bytes_per_pixel)))

    bounds = ee_roi.bounds().getInfo()["coordinates"][0]
    xs = [c[0] for c in bounds]
    ys = [c[1] for c in bounds]
    west, east, south, north = min(xs), max(xs), min(ys), max(ys)

    deg_per_px = (scale / 111320.0)             # approx metres->deg at equator
    span_x_px = (east - west) / deg_per_px
    span_y_px = (north - south) / deg_per_px
    nx = max(1, math.ceil(span_x_px / max_dim_px))
    ny = max(1, math.ceil(span_y_px / max_dim_px))
    logger.info("Tiled download: %dx%d tiles @ %dm (max_dim_px=%d, byte budget=%d) for %s",
                nx, ny, scale, max_dim_px, max_bytes, os.path.basename(out_path))

    tmp_dir = tempfile.mkdtemp(prefix="ee_tiles_", dir=config.PROCESSING_CONFIG["temp_dir"])
    tile_files = []
    dx = (east - west) / nx
    dy = (north - south) / ny
    for i in range(nx):
        for j in range(ny):
            w = west + i * dx
            e = w + dx
            s = south + j * dy
            n = s + dy
            _fetch_tile_with_subdivision(image, w, s, e, n, scale, crs,
                                          tmp_dir, f"tile_{i}_{j}", tile_files)

    srcs = [rasterio.open(f) for f in tile_files]
    mosaic, transform = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2],
                 "transform": transform})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    return out_path


def _download_ee_image_via_gcs(image, ee_roi, scale, crs, out_path) -> str:
    from google.cloud import storage
    bucket = config.PROCESSING_CONFIG["gcs_export_bucket"]
    name = os.path.splitext(os.path.basename(out_path))[0]
    task = _EE.batch.Export.image.toCloudStorage(
        image=image, description=name, bucket=bucket, fileNamePrefix=f"exports/{name}",
        region=ee_roi, scale=scale, crs=crs, maxPixels=1e13, fileFormat="GeoTIFF")
    task.start()
    logger.info("Started GCS export task '%s'; polling...", name)
    while task.active():
        time.sleep(20)
    state = task.status().get("state")
    if state != "COMPLETED":
        raise RuntimeError(f"GCS export failed: {task.status()}")

    client = storage.Client()
    b = client.bucket(bucket)
    blobs = list(b.list_blobs(prefix=f"exports/{name}"))
    if not blobs:
        raise RuntimeError("Export completed but no output found in GCS.")
    blobs[0].download_to_filename(out_path)     # single-tile case
    return out_path


def _download_with_retry(url: str, dest: str, retries: int = 4) -> None:
    last_error = None
    for attempt in range(retries):
        try:
            r = requests.get(url, stream=True, timeout=300)
            if not r.ok:
                body = r.text[:500] if r.text else ""
                raise requests.exceptions.HTTPError(
                    f"{r.status_code} Client Error for url: {url} -- body: {body}")
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            return
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning("Download retry %d for %s (%s); sleeping %ds",
                           attempt + 1, os.path.basename(dest), e, wait)
            time.sleep(wait)
    # Preserve the last error's actual message (e.g. GEE's "Total request
    # size..." body) in what gets raised -- callers like
    # _fetch_tile_with_subdivision need to see it to distinguish a
    # non-transient size-cap rejection from a transient network error.
    raise RuntimeError(f"Failed to download after {retries} attempts: {url} "
                       f"-- last error: {last_error}") from last_error


# ============================================================================
# OPENTOPOGRAPHY (API key required)
# ============================================================================
def fetch_opentopo_dem(roi_gdf: gpd.GeoDataFrame, output_filename: str,
                       demtype: str = "SRTMGL1") -> Optional[str]:
    api_key = config.get_api_key("opentopography")
    if not api_key:
        logger.error("OPENTOPO_API_KEY missing; cannot fetch DEM.")
        return None

    cache = _cache()
    key = f"predictors/{output_filename}"
    if cache.exists(key):
        return cache.download(key)

    bounds = roi_gdf.to_crs("EPSG:4326").total_bounds
    params = {
        "demtype": demtype,
        "south": bounds[1], "north": bounds[3],
        "west": bounds[0], "east": bounds[2],
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    out = cache.local_path(key)
    try:
        r = requests.get("https://portal.opentopography.org/API/globaldem",
                         params=params, stream=True, timeout=300)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 13):
                f.write(chunk)
        cache.upload(key, out)
        return out
    except Exception as e:
        logger.error("OpenTopography fetch failed: %s", e)
        return None


# ============================================================================
# WORLDPOP (REST fallback; the GEE path in config is preferred)
# ============================================================================
def fetch_worldpop_rest(iso3: str, year: int, output_filename: str) -> Optional[str]:
    cache = _cache()
    key = f"predictors/{output_filename}"
    if cache.exists(key):
        return cache.download(key)
    try:
        meta = requests.get("https://www.worldpop.org/rest/data/pop/wpgp",
                            params={"iso3": iso3}, timeout=60).json()["data"]
        rec = next((r for r in meta if str(r.get("popyear")) == str(year)), None)
        if not rec:
            logger.warning("WorldPop: no %s record for %s.", year, iso3)
            return None
        dl = rec["files"][0].replace("ftp://", "https://")
        out = cache.local_path(key)
        _download_with_retry(dl, out)
        cache.upload(key, out)
        return out
    except Exception as e:
        logger.error("WorldPop REST fetch failed: %s", e)
        return None


def fetch_wfp_prices(output_filename: str) -> Optional[str]:
    """
    Fetch WFP's dedicated Guinea food-price resource from HDX and cache it.
    This is tabular market-point data (not a native raster) -- pair with
    data_loader.rasterize_market_points() to turn it into a predictor surface.

    Uses the CKAN "Guinea - Food Prices" dataset (HDX dataset slug
    "wfp-food-prices-for-guinea", resource id
    "f8e11cd4-630f-44be-afa0-3e7baa5744ca") -- confirmed to be Guinea-only,
    so no country filtering is needed (unlike the 98-country global file).
    Confirmed columns (verified against a real download, 2024-12 vintage):
    date, admin1, admin2, market, market_id, latitude, longitude, category,
    commodity, commodity_id, unit, priceflag, pricetype, currency, price,
    usdprice.

    Resolves the actual file URL via the CKAN Action API (resource_show)
    rather than hardcoding a filename -- HDX resource download filenames/URLs
    can change on re-upload even when the resource id stays stable, and the
    Action API is the documented, stable way to resolve it. The Action API is
    a JSON endpoint distinct from HDX's human-facing browsing pages (which
    have bot-detection); it is intended for exactly this kind of programmatic
    access and does not require authentication for public resources.
    """
    cache = _cache()
    key = f"predictors/{output_filename}"
    if cache.exists(key):
        return cache.download(key)

    resource_id = "f8e11cd4-630f-44be-afa0-3e7baa5744ca"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; GuineaDeforestationPipeline/1.0)"}
    try:
        meta = requests.get("https://data.humdata.org/api/3/action/resource_show",
                            params={"id": resource_id}, headers=headers, timeout=60).json()
        if not meta.get("success"):
            logger.error("WFP prices: CKAN resource_show failed: %s", meta.get("error"))
            return None
        file_url = meta["result"]["url"]

        out = cache.local_path(key)
        _download_with_retry(file_url, out)
        cache.upload(key, out)

        import pandas as pd
        df = pd.read_csv(out, low_memory=False)
        # HDX food-price CSVs commonly have a HXL tag row as row 0 (e.g. "#date",
        # "#geo+lat", ...) immediately under the header -- drop it if present.
        if len(df) and str(df.iloc[0].get("date", "")).startswith("#"):
            df = df.iloc[1:].reset_index(drop=True)
            df.to_csv(out, index=False)
        logger.info("WFP prices: cached %d rows (%d markets, %d commodities).",
                    len(df), df["market"].nunique(), df["commodity"].nunique())
        return out
    except Exception as e:
        logger.error("WFP price fetch failed: %s. If this persists, download the "
                     "'Guinea - Food Prices' CSV manually from "
                     "https://data.humdata.org/dataset/wfp-food-prices-for-guinea -- "
                     "the dataset is public and does not require an account.", e)
        return None


# ============================================================================
# GEE FEATURE COLLECTIONS (vector data hosted on Earth Engine, e.g. WDPA)
# ============================================================================
def fetch_gee_feature_collection(asset: str, roi_gdf: gpd.GeoDataFrame,
                                 output_filename: str) -> Optional[str]:
    """
    Fetch a GEE FeatureCollection (vector data hosted on Earth Engine, e.g.
    WCMC/WDPA/current/polygons) filtered to roi_gdf, and cache it as GeoJSON.

    This is the correct fetch path for PREDICTOR_VECTORS_CONFIG entries with
    source="GEE_VECTOR" -- NOT fetch_osm_vectors, which assumes an OSM
    tags dict that GEE_VECTOR entries don't have (calling it on one throws
    KeyError: 'tags'). See fetch_and_load_api_predictors in data_loader.py
    for the dispatch that routes each vector layer to the right function.

    Filters by the ROI's actual geometry (not just its bounding box), same
    precision standard as every other fetch function in this module.
    """
    cache = _cache()
    key = f"vectors/{output_filename}"
    if cache.exists(key):
        return cache.download(key)
    try:
        roi_geom = _EE.Geometry(mapping(roi_gdf.to_crs("EPSG:4326").geometry.union_all()))
        fc = _EE.FeatureCollection(asset).filterBounds(roi_geom)
        geojson = fc.getInfo()
        if not geojson.get("features"):
            logger.warning("fetch_gee_feature_collection: no features returned for "
                           "%s over this ROI -- check the asset ID and that the ROI "
                           "actually overlaps features in it.", asset)
            return None
        out_path = cache.local_path(key)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(geojson, f)
        cache.upload(key, out_path)
        logger.info("fetch_gee_feature_collection: %d feature(s) from %s cached to %s",
                   len(geojson["features"]), asset, output_filename)
        return out_path
    except Exception as e:
        logger.error("fetch_gee_feature_collection failed for %s: %s", asset, e)
        return None


# ============================================================================
# OSM (osmnx) VECTORS
# ============================================================================
def fetch_osm_vectors(roi_gdf: gpd.GeoDataFrame, tags: Dict,
                      output_filename: str) -> Optional[str]:
    if not _OSMNX_READY and not _init_osmnx():
        return None
    import osmnx as ox
    cache = _cache()
    key = f"vectors/{output_filename}"
    if cache.exists(key):
        return cache.download(key)
    try:
        polygon = roi_gdf.to_crs("EPSG:4326").geometry.union_all()
        gdf = ox.features_from_polygon(polygon, tags)
        if gdf.empty:
            logger.warning("OSM: no features for %s", tags)
            return None
        gdf = gdf.reset_index()
        # keep only serializable columns
        keep = [c for c in gdf.columns
                if not gdf[c].map(lambda x: isinstance(x, (list, dict))).any()]
        out = cache.local_path(key)
        gdf[keep].to_file(out, driver="GeoJSON")
        cache.upload(key, out)
        return out
    except Exception as e:
        logger.error("OSM fetch failed: %s", e)
        return None


# ============================================================================
# MULTITEMPORAL ORCHESTRATION  +  INCREMENTAL UPDATE
# ============================================================================
def _target_years(cfg: Dict, ee_roi) -> List[int]:
    requested = config.MULTITEMPORAL_CONFIG.get("years")
    available = discover_available_years(cfg, ee_roi)
    if requested is None:
        return available
    return [y for y in requested if y in available]


def fetch_gee_dataset_all_years(dataset_key: str, cfg: Dict, roi_gdf: gpd.GeoDataFrame,
                                kind: str = "targets") -> Dict[int, str]:
    """
    Fetch ONE layer per available year for a GEE dataset. Skips years already
    present in cache/Blob (incremental). Returns {year: path}.
    """
    if not config.PROCESSING_CONFIG.get("gee_authenticated"):
        initialize_gee()

    ee_roi = _gdf_to_ee_geometry(roi_gdf)
    cache = _cache()
    scale = max(cfg.get("resolution_m", config.PROCESSING_CONFIG["target_resolution"]),
                config.PROCESSING_CONFIG["target_resolution"])
    force = config.MULTITEMPORAL_CONFIG.get("force_refresh", False)

    out: Dict[int, str] = {}

    # STATIC layers: single fetch, no year loop
    if cfg.get("temporal") == config.TEMPORAL_STATIC:
        key = f"{kind}/{dataset_key}.tif"
        if cache.exists(key) and not force:
            out[0] = cache.download(key)
            return out
        img = build_annual_gee_image(cfg, config.MULTITEMPORAL_CONFIG["max_year"], ee_roi)
        local = cache.local_path(key)
        _download_ee_image(img, ee_roi, scale, local)
        cache.upload(key, local)
        out[0] = local
        return out

    years = _target_years(cfg, ee_roi)
    logger.info("%s: %d available year(s): %s", dataset_key, len(years), years)

    for year in years:
        key = f"{kind}/{dataset_key}_{year}.tif"
        if cache.exists(key) and not force:
            logger.info("  %s %d already cached; skipping.", dataset_key, year)
            out[year] = cache.download(key)
            continue
        try:
            img = build_annual_gee_image(cfg, year, ee_roi)
            local = cache.local_path(key)
            _download_ee_image(img, ee_roi, scale, local)
            cache.upload(key, local)
            out[year] = local
            logger.info("  fetched %s %d", dataset_key, year)
        except Exception as e:
            logger.error("  failed %s %d: %s", dataset_key, year, e)
    return out


def update_all(roi_gdf: Optional[gpd.GeoDataFrame] = None) -> Dict[str, Dict[int, str]]:
    """
    Top-level entry point. Fetches all enabled targets + predictors for every
    available year, fetching ONLY what's missing. Run this on a schedule
    (Azure scheduler -> Container Apps Job) for regular updates when sources add a year.

    Returns: {dataset_key: {year: path}}.
    """
    initialize_all(strict=True)
    if roi_gdf is None:
        roi_gdf = get_admin_boundary()
    if roi_gdf is None:
        raise RuntimeError("Could not obtain Guinea boundary.")

    results: Dict[str, Dict[int, str]] = {}

    # Targets (LULC)
    for k in config.get_enabled_target_rasters():
        cfg = config.TARGET_RASTERS_CONFIG[k]
        if cfg.get("source") == "GEE":
            results[k] = fetch_gee_dataset_all_years(k, cfg, roi_gdf, kind="targets")

    # Predictor rasters
    for k in config.get_enabled_predictor_rasters():
        cfg = config.PREDICTOR_RASTERS_CONFIG[k]
        if cfg.get("source") == "GEE":
            results[k] = fetch_gee_dataset_all_years(k, cfg, roi_gdf, kind="predictors")
        elif cfg.get("source") == "OpenTopography":
            p = fetch_opentopo_dem(roi_gdf, f"{k}.tif", demtype=cfg.get("ot_demtype", "SRTMGL1"))
            if p:
                results[k] = {0: p}
        # 'Derived' (SLOPE/ASPECT) are computed downstream in utils from the DEM.

    # Predictor vectors (static)
    for k in config.get_enabled_predictor_vectors():
        cfg = config.PREDICTOR_VECTORS_CONFIG[k]
        p = fetch_osm_vectors(roi_gdf, cfg["tags"], f"{k}.geojson")
        if p:
            results[k] = {0: p}

    logger.info("Update complete: %d datasets processed.", len(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    update_all()

# ============================================================================
# FETCH GEOFABRIK
# ============================================================================

def fetch_geofabrik_vectors(config_dict, output_dir):
    """
    Downloads and parses the country-wide OpenStreetMap database from Geofabrik.
    """

    from pyrosm import OSM
    
    # 1. Get the URL from your config
    pbf_url = config_dict.get("GEOFABRIK_URL")
    pbf_filename = os.path.join(output_dir, "guinea-latest.osm.pbf")
    
    # 2. Download the file once (if it isn't already cached)
    if not os.path.exists(pbf_filename):
        print(f"Downloading country database from {pbf_url}...")
        response = requests.get(pbf_url, stream=True)
        with open(pbf_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        
    # 3. Initialize Pyrosm
    print("Parsing vector database...")
    osm = OSM(pbf_filename)
    
    # 4. Extract specific layers into GeoDataFrames based on your config tags
    vector_data = {}
    
    if config_dict["ROADS"]["enabled"]:
        # Pyrosm has built-in fast parsers for common layers
        vector_data["ROADS"] = osm.get_network(network_type="driving")
        
    if config_dict["WATERWAYS"]["enabled"]:
        vector_data["WATERWAYS"] = osm.get_data_by_custom_criteria(
            custom_filter={"waterway": True, "natural": ["water"]}
        )
        
    if config_dict["POI"]["enabled"]:
        vector_data["POI"] = osm.get_pois(custom_filter={"amenity": True})
        
    return vector_data