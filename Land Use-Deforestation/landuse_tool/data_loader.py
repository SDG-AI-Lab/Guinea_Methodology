"""
Data Loading Module (Azure-ready)

Responsibilities:
1. Load the Guinea boundary (via GeoBoundaries two-step connector).
2. Load / validate multitemporal target (LULC) rasters.
3. Load predictor rasters (GEE + OpenTopography).
4. RASTERIZE predictor vectors (OSM roads/rail/water/POI) into meaningful raster
   layers -- distance-to, heatmap, or zoning -- so every input to the ML model
   is a raster on the SAME grid.  <-- this is what makes the vectors usable.
5. Assemble an aligned predictor stack (all layers resampled to one reference
   grid) ready for stratified sampling / model training.

Fixes vs the original: standard logging (no logger.success/progress), a
consistent progress_callback(frac, text) signature, docstrings in the right
place, and boundary loading via the connector rather than the metadata URL.
"""
import os
import tempfile
import logging
from contextlib import contextmanager
from typing import Optional, Tuple, Dict, List

import rasterio
import numpy as np
import geopandas as gpd
from rasterio.warp import Resampling
from rasterio.features import rasterize

from . import config
from . import api_connector
from . import utils
from .utils import apply_boundary_mask

logger = logging.getLogger(__name__)


def _noop_progress(frac: float, text: str) -> None:
    """Default progress sink: (frac, text) -> nothing."""
    return None


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================
@contextmanager
def _open_as_raster(file_object_or_path):
    """Open a raster from a Streamlit upload or a path; clean up temp files."""
    temp_filepath = None
    try:
        if hasattr(file_object_or_path, "read"):
            temp_dir = os.environ.get("STREAMLIT_TEMP_DIR", tempfile.gettempdir())
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif", dir=temp_dir) as tmp:
                temp_filepath = tmp.name
                file_object_or_path.seek(0)
                while True:
                    chunk = file_object_or_path.read(16 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
            with rasterio.open(temp_filepath) as src:
                yield src
        else:
            with rasterio.open(str(file_object_or_path)) as src:
                yield src
    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except OSError as e:
                logger.warning("Could not delete temp file %s: %s", temp_filepath, e)


@contextmanager
def _open_as_vector(file_object_or_path):
    """Open a vector file from a Streamlit upload or a path."""
    temp_filepath = None
    try:
        if hasattr(file_object_or_path, "read"):
            temp_dir = os.environ.get("STREAMLIT_TEMP_DIR", tempfile.gettempdir())
            with tempfile.NamedTemporaryFile(delete=False, suffix=".geojson", dir=temp_dir) as tmp:
                temp_filepath = tmp.name
                file_object_or_path.seek(0)
                tmp.write(file_object_or_path.read())
            yield gpd.read_file(temp_filepath)
        else:
            yield gpd.read_file(str(file_object_or_path))
    finally:
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except OSError as e:
                logger.warning("Could not delete temp file %s: %s", temp_filepath, e)


# ============================================================================
# ADMINISTRATIVE BOUNDARY
# ============================================================================
def load_admin_boundary(progress_callback=None) -> Optional[gpd.GeoDataFrame]:
    """Load the Guinea boundary via the GeoBoundaries two-step connector."""
    progress_callback = progress_callback or _noop_progress
    progress_callback(0.3, "Loading Guinea boundary from GeoBoundaries...")
    gdf = api_connector.get_admin_boundary()
    if gdf is None or gdf.empty:
        logger.error("Boundary could not be loaded.")
        return None
    progress_callback(1.0, f"Loaded boundary: {gdf.shape[0]} feature(s)")
    return gdf


def create_boundary_mask(boundary_gdf: gpd.GeoDataFrame, ref_profile: dict) -> np.ndarray:
    """Rasterize a boundary to a boolean mask matching a reference profile."""
    from rasterio.features import rasterize
    if boundary_gdf.crs != ref_profile["crs"]:
        boundary_gdf = boundary_gdf.to_crs(ref_profile["crs"])
    shapes = ((geom, 1) for geom in boundary_gdf.geometry)
    mask = rasterize(
        shapes=shapes,
        out_shape=(ref_profile["height"], ref_profile["width"]),
        transform=ref_profile["transform"],
        fill=0, dtype="uint8",
    )
    return mask.astype(bool)


# ============================================================================
# SPATIAL CONSTRAINTS (scenario zones)
# ============================================================================
# A scenario's spatial constraint is a boolean mask on the analysis grid saying
# where change may not be allocated. It must be applied DURING allocation, not
# by clipping the output afterwards: demand that would have landed in the zone
# has to be reallocated elsewhere, and clipping the result would simply lose it,
# leaving totals that no longer sum to the projected demand.

def admin_zones_to_mask(level: str, names: List[str], ref_profile: dict) -> np.ndarray:
    """Boolean mask covering the named administrative units at `level`."""
    gdf = api_connector.get_admin_boundary(level=level)
    if gdf is None or gdf.empty:
        raise ValueError(f"Could not load administrative boundaries for {level}.")

    # A unit is named by its ID or by its name; the ID is unambiguous.
    level_cfg = config.ADMIN_BOUNDARY_CONFIG["levels"][level]
    fields = [level_cfg["name_field"], level_cfg.get("display_field", level_cfg["name_field"])]
    wanted = {str(n).strip().casefold() for n in names}
    keys = {f: gdf[f].astype(str).str.strip().str.casefold() for f in fields}
    hit = keys[fields[0]].isin(wanted) | keys[fields[1]].isin(wanted)
    selected = gdf[hit]
    if selected.empty:
        raise ValueError(
            f"No {level} unit matched {sorted(names)}. "
            f"Available: {sorted(gdf[fields[1]].astype(str).unique())[:25]}")

    missing = wanted - set(keys[fields[0]][hit]) - set(keys[fields[1]][hit])
    if missing:
        # Named but not found is a user error worth surfacing, not a silent
        # partial constraint that quietly protects less than was asked for.
        raise ValueError(f"These {level} units were not found: {sorted(missing)}")

    return create_boundary_mask(selected, ref_profile)


def geometry_to_mask(geometry: Dict, ref_profile: dict,
                     crs: str = "EPSG:4326") -> np.ndarray:
    """Boolean mask from a GeoJSON geometry, Feature or FeatureCollection."""
    from shapely.geometry import shape as shapely_shape

    kind = (geometry or {}).get("type")
    if not kind:
        raise ValueError("spatial constraint geometry has no 'type' field.")

    if kind == "FeatureCollection":
        gdf = gpd.GeoDataFrame.from_features(geometry["features"], crs=crs)
    elif kind == "Feature":
        gdf = gpd.GeoDataFrame.from_features([geometry], crs=crs)
    else:
        gdf = gpd.GeoDataFrame({"geometry": [shapely_shape(geometry)]}, crs=crs)

    if gdf.empty:
        raise ValueError("spatial constraint geometry is empty.")
    return create_boundary_mask(gdf, ref_profile)


def point_buffer_to_mask(lat: float, lon: float, radius_m: float,
                         ref_profile: dict) -> np.ndarray:
    """Boolean mask for a circle of `radius_m` around a point.

    Buffered in a projected CRS: buffering degrees would give an ellipse whose
    size varies with latitude and is not the requested radius anywhere.
    """
    if radius_m <= 0:
        raise ValueError("spatial constraint radius_m must be positive.")
    point = gpd.GeoDataFrame(
        {"geometry": gpd.points_from_xy([lon], [lat])}, crs="EPSG:4326")
    metric = point.estimate_utm_crs()
    buffered = point.to_crs(metric).buffer(radius_m).to_frame("geometry")
    buffered = buffered.set_crs(metric)
    return create_boundary_mask(buffered, ref_profile)


def build_constraint_mask(spec: Optional[Dict], ref_profile: dict) -> Optional[np.ndarray]:
    """Boolean mask from a spatial-constraint spec, or None when unconstrained.

    Accepted forms:
        {"mode": "admin",   "level": "ADM2", "names": ["Boke", "Kindia"]}
        {"mode": "geojson", "geometry": {...}}          # geometry/Feature/FC
        {"mode": "buffer",  "lat": 9.6, "lon": -13.6, "radius_m": 5000}

    `"invert": true` flips the meaning from "change may not happen here" to
    "change may happen ONLY here", which is the other half of what a scenario
    builder needs and costs one line rather than a second code path.
    """
    if not spec:
        return None

    mode = spec.get("mode")
    if mode == "admin":
        mask = admin_zones_to_mask(spec["level"], spec["names"], ref_profile)
    elif mode == "geojson":
        mask = geometry_to_mask(spec["geometry"], ref_profile,
                                crs=spec.get("crs", "EPSG:4326"))
    elif mode == "buffer":
        mask = point_buffer_to_mask(float(spec["lat"]), float(spec["lon"]),
                                    float(spec["radius_m"]), ref_profile)
    else:
        raise ValueError(
            f"Unknown spatial constraint mode {mode!r}; expected "
            f"'admin', 'geojson' or 'buffer'.")

    if spec.get("invert"):
        mask = ~mask
    if not mask.any():
        raise ValueError(
            "Spatial constraint covers no pixels of the analysis grid. Check "
            "the geometry's CRS and that it overlaps Guinea.")
    logger.info("Spatial constraint (%s%s): %d pixels.",
                mode, ", inverted" if spec.get("invert") else "", int(mask.sum()))
    return mask


# ============================================================================
# TARGET RASTER (LULC) LOADING
# ============================================================================
def load_targets(target_files: List, align_boundary: bool = True, progress_callback=None):
    """Validate multitemporal LULC rasters and build a validity + boundary mask."""
    progress_callback = progress_callback or _noop_progress
    if not target_files:
        logger.warning("No target files provided.")
        return None

    progress_callback(0.05, f"Validating {len(target_files)} target file(s)...")
    for idx, f in enumerate(target_files):
        try:
            with _open_as_raster(f) as src:
                logger.info("Target %d: %sx%s nodata=%s",
                            idx, src.width, src.height, src.nodata)
        except Exception as e:
            logger.error("Invalid target file %d: %s", idx, e)
            return None
    logger.info("All %d target(s) valid.", len(target_files))

    profiles_dict, masks_dict = {}, {}
    latest_file = target_files[-1]
    with _open_as_raster(latest_file) as src:
        ref_profile = src.profile
        nodata = src.nodata
        mask = np.empty((src.height, src.width), dtype=bool)
        total_blocks = sum(1 for _ in src.block_windows(1))
        for block_idx, (_, window) in enumerate(src.block_windows(1)):
            window_data = src.read(1, window=window)
            mask[window.row_off:window.row_off + window.height,
                 window.col_off:window.col_off + window.width] = (window_data != nodata)
            progress_callback(0.1 + 0.8 * (block_idx + 1) / total_blocks,
                              "Creating validity mask...")
        profiles_dict["latest"] = ref_profile
        masks_dict["latest"] = mask

    boundary_mask = None
    if align_boundary:
        boundary_gdf = load_admin_boundary()
        if boundary_gdf is not None:
            boundary_mask = create_boundary_mask(boundary_gdf, ref_profile)

    progress_callback(1.0, "Target rasters loaded.")
    return profiles_dict, masks_dict, boundary_mask


def fetch_and_load_api_targets(boundary_gdf: Optional[gpd.GeoDataFrame] = None,
                                target_keys: Optional[List[str]] = None,
                                progress_callback=None):
    """
    Fetch enabled GEE targets for ALL available years (one layer/year), then run
    the newest year's raster through load_targets to build masks.

    target_keys: optional explicit subset (e.g. [config.LAND_USE_TARGET_DATASET])
    so a module fetches only the target product it actually uses, not every
    LULC product enabled in TARGET_RASTERS_CONFIG (ESRI_LULC, MODIS_MCD12Q1,
    and any others enabled by default would otherwise ALL be fetched).

    progress_callback: optional (frac, text) -> None, required by the Django
    Celery tasks (deforestation_api/landuse_api tasks.py) that call this with
    progress_callback= to report job progress; kept optional here so scripts
    invoking this outside that context don't need to pass one.

    Returns: (per_dataset_year_paths, (profiles, masks, boundary_mask))
    """
    progress_callback = progress_callback or _noop_progress

    if boundary_gdf is None:
        progress_callback(0.0, "Loading Guinea boundary...")
        boundary_gdf = api_connector.get_admin_boundary()

    # target_keys narrows the set before the progress denominator is taken, so
    # the reported fraction covers only the datasets actually being fetched.
    enabled = [k for k in config.get_enabled_target_rasters()
               if config.TARGET_RASTERS_CONFIG[k].get("source") == "GEE"
               and (target_keys is None or k in target_keys)]

    # Reserve the first half of the progress range for fetching, the second half
    # for load_targets, which reports 0..1 over its own work.
    per_dataset: Dict[str, Dict[int, str]] = {}
    for idx, key in enumerate(enabled):
        progress_callback(0.5 * idx / len(enabled),
                          f"Fetching {key} ({idx + 1}/{len(enabled)})...")
        per_dataset[key] = api_connector.fetch_gee_dataset_all_years(
            key, config.TARGET_RASTERS_CONFIG[key], boundary_gdf, kind="targets")

    all_paths = [p for d in per_dataset.values() for p in d.values()]
    masks = load_targets(
        all_paths, align_boundary=True,
        progress_callback=lambda f, t: progress_callback(0.5 + 0.5 * f, t),
    ) if all_paths else None
    progress_callback(1.0, "API targets fetched and loaded.")
    return per_dataset, masks


# ============================================================================
# GRID ALIGNMENT  (all ML inputs must share one grid)
# ============================================================================
def load_and_align_raster(path: str, ref_profile: dict,
                          categorical: bool = False) -> np.ndarray:
    """
    Read a raster and resample it onto the reference grid (CRS, resolution,
    extent). Categorical layers (LULC) use nearest-neighbour; continuous layers
    use bilinear. This is what lets a 10 m ESRI layer, a 500 m MODIS layer and a
    100 m population layer all line up pixel-for-pixel.
    """
    resampling = Resampling.nearest if categorical else Resampling.bilinear
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        src_profile = src.profile
    if (src_profile["width"], src_profile["height"]) == (
            ref_profile["width"], ref_profile["height"]) and \
       src_profile["crs"] == ref_profile["crs"]:
        return arr  # already aligned
    return utils.resample_raster(arr, src_profile, ref_profile, resampling=resampling)


# ============================================================================
# PREDICTOR RASTER LOADING
# ============================================================================
def load_predictors(predictor_files: List, ref_profile: dict,
                    boundary_mask: Optional[np.ndarray] = None,
                    categorical: bool = False, progress_callback=None):
    """
    Load predictor rasters, resampling each onto the reference grid (so mismatched
    native resolutions are aligned instead of rejected), then apply the boundary.
    """
    progress_callback = progress_callback or _noop_progress
    if not predictor_files:
        logger.warning("No predictor files provided.")
        return None
    if not ref_profile:
        logger.error("Cannot align predictors without a reference profile.")
        return None

    predictors: Dict[str, np.ndarray] = {}
    errors: List[str] = []
    progress_callback(0.0, f"Aligning {len(predictor_files)} predictor(s)...")

    for idx, f in enumerate(predictor_files):
        name = f.name if hasattr(f, "name") else str(f)
        try:
            data = load_and_align_raster(f, ref_profile, categorical=categorical)
            if boundary_mask is not None:
                data = apply_boundary_mask(data, boundary_mask)
            predictors[os.path.basename(name)] = data
        except Exception as e:
            errors.append(f"{name}: {e}")
        progress_callback((idx + 1) / len(predictor_files), f"Aligned {name}")

    if errors:
        logger.error("Predictor alignment errors:\n%s", "\n".join(errors))
        if not predictors:
            return None

    progress_callback(1.0, f"Loaded {len(predictors)}/{len(predictor_files)} predictors")
    return predictors or None


def build_coarsest_reference_profile(predictor_files: List, boundary_gdf=None,
                                      max_resolution_m: float = 100.0) -> dict:
    """
    Build a reference profile (for use as ref_profile in load_predictors)
    matching the COARSEST resolution among the given predictor files, capped
    at max_resolution_m (default 100 m, matching this platform's documented
    default -- Methodology_Deforestation_Guinea.docx Section 4.1). A resolution
    cap is more principled and self-maintaining than a hand-curated exclusion
    list of "which layers to ignore when picking the coarsest" -- it doesn't
    need updating every time a new coarse layer is added to the pipeline.

    CRS FIX: earlier versions of this function read src.transform.a/.e
    directly as metres regardless of the file's actual CRS. For a geographic
    (lat/lon) CRS, that value is in DEGREES, not metres -- e.g. MODIS's 250 m
    native resolution is ~0.00225 degrees, which printed as "0.0 m" and was
    silently treated as smaller than a 30 m DEM, corrupting the "pick the
    coarsest" comparison. This is now converted to an approximate metre
    resolution using the standard 111,320 m/degree latitude conversion (and
    cos(latitude) for longitude) before any comparison happens.

    Confirmed necessary via two real, separate failures: including an ~11 km
    climate layer unfiltered collapsed the grid to 54x77 pixels; treating
    MODIS's degree-based pixel size as metres directly gave a similarly wrong
    (if coincidentally larger) result. Both are fixed by the combination of
    correct-units comparison + an explicit cap.

    boundary_gdf: if given, the profile's extent is clipped to this boundary
    (e.g. Guinea's national polygon) rather than the raw union of input file
    extents -- prevents the country-boundary inflation bug documented in
    Methodology_Deforestation_Guinea.docx Section 9.3 (raster grids commonly
    extend well beyond the country of interest when fetched as a bounding
    box) from recurring for any future layer added to this pipeline.
    """
    import rasterio
    import math

    def _pixel_size_m(src):
        """Return (approx) pixel size in metres, correctly handling both
        geographic (degrees) and projected (already metres) CRS."""
        xres, yres = abs(src.transform.a), abs(src.transform.e)
        if src.crs and src.crs.is_geographic:
            b = src.bounds
            mean_lat = (b.top + b.bottom) / 2.0
            m_per_deg_lat = 111_320.0
            m_per_deg_lon = 111_320.0 * math.cos(math.radians(mean_lat))
            return max(xres * m_per_deg_lon, yres * m_per_deg_lat)
        return max(xres, yres)  # already metres (projected CRS)

    resolutions = []
    crs = None
    bounds = None
    for f in predictor_files:
        path = f.name if hasattr(f, "name") else str(f)
        with rasterio.open(path) as src:
            res_m = _pixel_size_m(src)
            resolutions.append((res_m, path))
            if crs is None:
                crs = src.crs
            b = src.bounds
            bounds = b if bounds is None else rasterio.coords.BoundingBox(
                min(bounds.left, b.left), min(bounds.bottom, b.bottom),
                max(bounds.right, b.right), max(bounds.top, b.top))

    if not resolutions:
        raise ValueError("No predictor files given -- cannot determine coarsest resolution.")

    coarsest_res, coarsest_source = max(resolutions, key=lambda r: r[0])
    coarsest_res_capped = min(coarsest_res, max_resolution_m)
    logger.info("Coarsest input resolution: %.1f m (from %s); capped at %.1f m -> using %.1f m.",
                coarsest_res, coarsest_source, max_resolution_m, coarsest_res_capped)

    if boundary_gdf is not None:
        b = boundary_gdf.total_bounds  # [minx, miny, maxx, maxy]
        bounds = rasterio.coords.BoundingBox(b[0], b[1], b[2], b[3])

    # bounds are in the input files' CRS (degrees if geographic); convert the
    # chosen metre resolution back to that CRS's units for the transform.
    if crs and crs.is_geographic:
        mean_lat = (bounds.top + bounds.bottom) / 2.0
        xres_native = coarsest_res_capped / (111_320.0 * math.cos(math.radians(mean_lat)))
        yres_native = coarsest_res_capped / 111_320.0
    else:
        xres_native = yres_native = coarsest_res_capped

    width = max(1, int((bounds.right - bounds.left) / xres_native))
    height = max(1, int((bounds.top - bounds.bottom) / yres_native))
    transform = rasterio.transform.from_origin(bounds.left, bounds.top, xres_native, yres_native)

    return {
        "driver": "GTiff", "dtype": "float32", "nodata": None,
        "width": width, "height": height, "count": 1,
        "crs": crs, "transform": transform,
    }


def rasterize_market_points(csv_path: str, commodity: str, target_year: int,
                             ref_profile: dict, idw_power: float = 2.0,
                             max_neighbors: int = 8,
                             price_column: str = "usdprice") -> Optional[np.ndarray]:
    """
    Turn WFP market-point price observations into a continuous predictor
    surface via inverse-distance weighting (IDW) -- a reasonable default
    given WFP market coverage is typically sparse and unevenly distributed
    (dense near cities, sparse in remote areas), which is exactly the
    situation IDW handles better than assuming a fixed grid of observations.

    Columns confirmed against a real download of the Guinea-specific HDX
    resource (see api_connector.fetch_wfp_prices): date, admin1, admin2,
    market, market_id, latitude, longitude, category, commodity,
    commodity_id, unit, priceflag, pricetype, currency, price, usdprice.
    `date` is a single column (not separate year/month) -- parsed here, not
    assumed to already be split. Defaults to usdprice rather than the raw
    `price` column so values are comparable across markets and time without
    a separate currency-normalisation step (local `price` is in whatever
    currency that market reports in); pass price_column="price" plus your
    own FX handling if usdprice's exchange-rate source is not suitable.

    Returns a 2D array on ref_profile's grid, or None if no usable data.
    """
    import pandas as pd
    df = pd.read_csv(csv_path, low_memory=False)
    required = {"date", "latitude", "longitude", "commodity", price_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"rasterize_market_points: expected columns {missing} not found -- "
                         f"actual columns: {list(df.columns)}. Source schema may have changed; "
                         f"see api_connector.fetch_wfp_prices docstring for the confirmed vintage.")

    df["_year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    sub = df[(df["commodity"].str.strip().str.lower() == commodity.strip().lower())
             & (df["_year"] == target_year)]
    sub = sub.dropna(subset=["latitude", "longitude", price_column])
    if sub.empty:
        logger.warning("rasterize_market_points: no %s observations for %d.", commodity, target_year)
        return None

    # Average multiple observations (e.g. different months, or a market
    # reporting more than once) sharing the same coordinate within the year,
    # via groupby then a clean re-extraction -- avoids relying on the
    # groupby result's row order matching any other array's order.
    grouped = sub.groupby(["latitude", "longitude"], as_index=False)[price_column].mean()
    pts_lat = grouped["latitude"].to_numpy()
    pts_lon = grouped["longitude"].to_numpy()
    pts_val = grouped[price_column].to_numpy()

    h, w = ref_profile["height"], ref_profile["width"]
    transform = ref_profile["transform"]
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    grid_x, grid_y = rasterio.transform.xy(transform, rows, cols)
    grid_x, grid_y = np.asarray(grid_x), np.asarray(grid_y)

    out = np.full((h, w), np.nan, dtype="float32")
    flat_gx, flat_gy = grid_x.ravel(), grid_y.ravel()
    for i in range(len(flat_gx)):
        d = np.sqrt((pts_lon - flat_gx[i]) ** 2 + (pts_lat - flat_gy[i]) ** 2)
        d = np.where(d == 0, 1e-6, d)
        nearest = np.argsort(d)[:max_neighbors]
        weights = 1.0 / (d[nearest] ** idw_power)
        out.ravel()[i] = np.sum(weights * pts_val[nearest]) / np.sum(weights)

    logger.info("rasterize_market_points: %s %d interpolated from %d market observations.",
               commodity, target_year, len(sub))
    return out.reshape(h, w)


def fetch_period_specific_predictor(predictor_key: str, target_year: int,
                                     available_years: List[int],
                                     fetch_fn, output_dir: str,
                                     _cache: Optional[Dict] = None) -> str:
    """
    Resolve and fetch (or reuse a cached fetch of) whichever available_years
    entry is closest to target_year, instead of reusing one fixed snapshot
    year for every analysis period.

    Confirmed contributing factor (Methodology_Deforestation_Guinea.docx
    Section 9.1) in why the dist_forest_edge fix's improvement was smaller for
    older historical hold-out splits: population/infrastructure predictors
    were a single fixed-year snapshot reused unchanged across every
    calibration/test window, so the snapshot was progressively more
    out-of-date for splits further from that snapshot year.

    predictor_key: cache key, e.g. "WORLDPOP".
    target_year:   the year this predictor is needed for (a calibration or
                   test window boundary).
    available_years: years the underlying source actually publishes for.
    fetch_fn:      callable(year, output_path) -> filepath, e.g.
                   functools.partial(api_connector.fetch_worldpop_rest, "GIN").
    _cache:        pass the same dict across calls in one session to avoid
                   redundant fetches when two target_years round to the same
                   available_year.

    Returns: filepath for the resolved year.
    """
    cache = {} if _cache is None else _cache
    available_years = sorted(available_years)

    import bisect
    pos = bisect.bisect_left(available_years, target_year)
    if pos == 0:
        resolved_year = available_years[0]
    elif pos == len(available_years):
        resolved_year = available_years[-1]
    else:
        before, after = available_years[pos - 1], available_years[pos]
        resolved_year = before if (target_year - before) <= (after - target_year) else after

    cache_key = (predictor_key, resolved_year)
    if cache_key not in cache:
        # BUG FIX (recurring -- see project history): fetch_fn (e.g.
        # fetch_worldpop_rest) expects a bare FILENAME, not a full path -- it
        # constructs its own cache path internally via
        # BlobCache.local_path(f"predictors/{output_filename}"), joining that
        # with its OWN cache root. Passing an already-absolute path here gets
        # joined a second time, producing a malformed nested path (confirmed:
        # crashes with "...cache/predictors/c:" on Windows). output_dir is
        # intentionally unused now -- the cache mechanism owns where the file
        # actually lives.
        filename = f"{predictor_key.lower()}_{resolved_year}.tif"
        cache[cache_key] = fetch_fn(resolved_year, filename)
        if resolved_year != target_year:
            logger.info("%s for %d: nearest available year is %d (gap: %d yr)",
                        predictor_key, target_year, resolved_year, abs(target_year - resolved_year))
    return cache[cache_key]


# ============================================================================
# VECTOR -> RASTER  (roads/rail/water -> distance-to; POI -> heatmap)
# ============================================================================
def rasterize_vector_predictors(vector_files: Dict[str, str], ref_profile: dict,
                                boundary_mask: Optional[np.ndarray] = None,
                                progress_callback=None) -> Dict[str, np.ndarray]:
    """
    Convert fetched vector layers into aligned raster predictors using the
    rasterization_mode declared per layer in config.PREDICTOR_VECTORS_CONFIG:
        distance -> distance-to surface (roads, railways, waterways)
        heatmap  -> density/decay surface (points of interest)
        zoning   -> burn a per-feature attribute value (multi-category, e.g.
                    administrative zones)
        binary_zone -> simple in/out mask, no attribute needed (e.g. WDPA
                    protected areas)

    Every output is on the reference grid with NaN outside the boundary, so they
    can be stacked directly with the raster predictors for ML.

    Args:
        vector_files: {layer_key: path_to_geojson}
    Returns:
        {f"{layer_key}_{mode}": np.ndarray}
    """
    progress_callback = progress_callback or _noop_progress
    if boundary_mask is None:
        boundary_mask = np.ones((ref_profile["height"], ref_profile["width"]), dtype=bool)

    mv_method = config.get_missing_value_strategy("vector_based").get("method", "mask")
    out: Dict[str, np.ndarray] = {}

    keys = list(vector_files.items())
    for i, (key, path) in enumerate(keys):
        cfg = config.PREDICTOR_VECTORS_CONFIG.get(key, {})
        mode = cfg.get("rasterization_mode", "distance")
        try:
            if mode == "binary_zone":
                # Simple in/out mask (e.g. WDPA protected areas) -- every
                # feature burns 1, everything else 0. Deliberately NOT routed
                # through the general "zoning" path below, which expects a
                # multi-category attribute field to split by; WDPA here is a
                # single binary condition, and forcing it through the
                # multi-category path would either crash (no attribute) or
                # produce a meaningless single-category "zone" split. Same
                # simple pattern as create_boundary_mask.
                gdf = gpd.read_file(path)
                gdf = gdf[gdf.geometry.is_valid].to_crs(ref_profile["crs"])
                rasterized = rasterize(
                    shapes=((geom, 1) for geom in gdf.geometry),
                    out_shape=(ref_profile["height"], ref_profile["width"]),
                    transform=ref_profile["transform"],
                    fill=0, all_touched=True, dtype="float32",
                )
                out[f"{key}_zone"] = np.where(boundary_mask, rasterized, np.nan)
                logger.info("Rasterized %s -> binary zoning mask (%d features)", key, len(gdf))

            elif mode == 'zoning' and cfg.get("attribute"):
                gdf = gpd.read_file(path)
                gdf = gdf[gdf.geometry.is_valid].to_crs(ref_profile['crs'])
                unique_vals = gdf[cfg.get("attribute")].dropna().unique()
                
                for val in unique_vals:
                    shapes = ((geom, 1) for geom, attr in zip(gdf.geometry, gdf[cfg.get("attribute")]) if attr == val)
                    rasterized = rasterize(
                        shapes=shapes,
                        out_shape=(ref_profile['height'], ref_profile['width']),
                        transform=ref_profile['transform'],
                        fill=0,
                        all_touched=True,
                        dtype='float32'
                    )
                    layer_name = f"{key}_zone_{val}"
                    out[layer_name] = np.where(boundary_mask, rasterized, np.nan)
                    logger.info("Rasterized %s -> zoning mask %s", key, layer_name)

            else:
                # Keep existing logic for 'distance' and 'heatmap'
                arr = utils.process_vector_layer(
                    vector_path=path,
                    profile=ref_profile,
                    mask=boundary_mask,
                    mode=mode,
                    attribute=cfg.get("attribute"),
                    max_distance=cfg.get("max_distance"),
                    missing_value_handling=mv_method,
                )
                out[f"{key}_{mode}"] = arr
                logger.info("Rasterized %s -> %s surface %s", key, mode, arr.shape)
                
        except Exception as e:
            logger.error("Failed to rasterize %s: %s", key, e)

    return out


def fetch_and_load_api_predictors(boundary_gdf: Optional[gpd.GeoDataFrame] = None,
                                  ref_profile: Optional[dict] = None,
                                  boundary_mask: Optional[np.ndarray] = None,
                                  progress_callback=None,
                                  raster_keys: Optional[List[str]] = None,
                                  vector_keys: Optional[List[str]] = None):
    """
    Fetch enabled predictors (GEE annual per-year, OpenTopography DEM, and OSM
    vectors via Geofabrik or Overpass), then -- if a ref_profile is given -- align
    the rasters and RASTERIZE the vectors so the returned predictor dict is fully
    raster and grid-aligned.

    raster_keys / vector_keys: optional explicit subsets (e.g.
    config.LAND_USE_PREDICTOR_RASTERS / config.LAND_USE_PREDICTOR_VECTORS) so a
    module can fetch only its own justified predictor set rather than every
    predictor enabled globally. Defaults to everything enabled, unchanged
    behaviour if not passed.

    Returns: (raster_year_paths, vector_paths, loaded_predictor_arrays_or_None)
    """
    progress_callback = progress_callback or _noop_progress
    if boundary_gdf is None:
        boundary_gdf = api_connector.get_admin_boundary()

    # 1. Raster predictors (GEE + OpenTopography)
    raster_paths: Dict[str, Dict[int, str]] = {}
    for key in config.get_enabled_predictor_rasters(raster_keys):
        cfg = config.PREDICTOR_RASTERS_CONFIG[key]
        if cfg.get("source") == "GEE":
            raster_paths[key] = api_connector.fetch_gee_dataset_all_years(
                key, cfg, boundary_gdf, kind="predictors")
        elif cfg.get("source") == "OpenTopography":
            p = api_connector.fetch_opentopo_dem(
                boundary_gdf, f"{key}.tif", demtype=cfg.get("ot_demtype", "SRTMGL1"))
            if p:
                raster_paths[key] = {0: p}

    # 2. Vector predictors — check cache first (skips the whole-country PBF
    #    parse entirely when cached; fetch_geofabrik_vectors is bypassed because
    #    it returns GeoDataFrames, not paths, which breaks rasterize_vector_predictors)
    #
    #    Dispatches by source: OSM tags-based layers (roads/waterways/POI) go
    #    through fetch_osm_vectors; GEE FeatureCollection layers (e.g. WDPA
    #    protected areas, source="GEE_VECTOR") go through
    #    fetch_gee_feature_collection instead. Calling fetch_osm_vectors on a
    #    GEE_VECTOR entry is exactly what threw KeyError: 'tags' before this
    #    fix -- that entry's config shape has no "tags" key at all.
    vector_paths: Dict[str, str] = {}
    for key in config.get_enabled_predictor_vectors(vector_keys):
        cfg = config.PREDICTOR_VECTORS_CONFIG.get(key)
        if not isinstance(cfg, dict):
            logger.warning("PREDICTOR_VECTORS_CONFIG[%r] is not a dict (got %r) -- "
                           "skipping rather than crashing the whole predictor fetch.",
                           key, type(cfg).__name__)
            continue
        source = cfg.get("source", "OSM")
        if source == "GEE_VECTOR":
            p = api_connector.fetch_gee_feature_collection(cfg["asset"], boundary_gdf, f"{key}.geojson")
        elif "tags" in cfg:
            p = api_connector.fetch_osm_vectors(boundary_gdf, cfg["tags"], f"{key}.geojson")
        else:
            logger.warning("PREDICTOR_VECTORS_CONFIG[%r] has source=%r but no 'tags' key -- "
                           "don't know how to fetch this. Skipping.", key, source)
            continue
        if p:
            vector_paths[key] = p

    # 3. Assemble an aligned, all-raster predictor stack
    loaded = None
    if ref_profile:
        loaded = {}
        # newest year of each raster predictor, aligned to the grid
        latest_raster = {}
        for name, paths in raster_paths.items():
            if paths:
                latest_raster[name] = paths[max(paths.keys())]
        aligned = load_predictors(list(latest_raster.values()), ref_profile,
                                  boundary_mask, categorical=False,
                                  progress_callback=progress_callback)
        if aligned:
            loaded.update(aligned)

        # rasterized vectors
        vec_rasters = rasterize_vector_predictors(vector_paths, ref_profile,
                                                  boundary_mask, progress_callback)
        loaded.update(vec_rasters)
        loaded = loaded or None

    return raster_paths, vector_paths, loaded


# ============================================================================
# ASSEMBLE A FULL PREDICTOR STACK FOR ONE SNAPSHOT YEAR
# ============================================================================
def assemble_predictor_stack(year: int,
                             raster_paths: Dict[str, Dict[int, str]],
                             vector_paths: Dict[str, str],
                             ref_profile: dict,
                             boundary_mask: Optional[np.ndarray] = None,
                             progress_callback=None) -> Dict[str, np.ndarray]:
    """
    Build the ML-ready predictor stack for a given snapshot year: every raster
    predictor is resampled to the reference grid (nearest available year for
    sparse layers), and every vector predictor is rasterized. All arrays share
    the reference grid and are masked to the boundary.

    Args:
        year: snapshot year to assemble.
        raster_paths: {predictor_key: {year: path}} from fetch_and_load_api_predictors.
        vector_paths: {vector_key: geojson_path}.
    Returns:
        {predictor_name: np.ndarray}  -- all same shape as ref_profile.
    """
    progress_callback = progress_callback or _noop_progress
    stack: Dict[str, np.ndarray] = {}

    # --- raster predictors, choosing the right year ---
    for name, paths in raster_paths.items():
        if not paths:
            continue
        if 0 in paths:                       # static (e.g. DEM)
            chosen = paths[0]
        elif year in paths:                  # exact annual match
            chosen = paths[year]
        else:                                # sparse -> nearest available year
            nearest = min(paths.keys(), key=lambda y: abs(y - year))
            chosen = paths[nearest]
            logger.info("%s: using nearest year %d for %d", name, nearest, year)
        try:
            arr = load_and_align_raster(chosen, ref_profile, categorical=False)
            if boundary_mask is not None:
                arr = apply_boundary_mask(arr, boundary_mask)
            stack[name] = arr
        except Exception as e:
            logger.error("Could not align %s: %s", name, e)

    # --- derived terrain (slope/aspect) if DEM present and enabled ---
    if "DEM" in stack or "DEM_GEE" in stack:
        dem = stack.get("DEM", stack.get("DEM_GEE"))
        if config.PREDICTOR_RASTERS_CONFIG.get("SLOPE", {}).get("enabled"):
            stack["SLOPE"] = utils.calculate_slope(dem, ref_profile)
        if config.PREDICTOR_RASTERS_CONFIG.get("ASPECT", {}).get("enabled"):
            stack["ASPECT"] = utils.calculate_aspect(dem, ref_profile)

    # --- vector predictors -> rasters ---
    stack.update(rasterize_vector_predictors(vector_paths, ref_profile,
                                             boundary_mask, progress_callback))

    logger.info("Assembled predictor stack for %d: %d layers.", year, len(stack))
    return stack


# ============================================================================
# BRIDGE: write grid-aligned predictor files for training / prediction
# ============================================================================
# training.py and prediction.py read predictor *files* window-by-window, indexing
# every layer with the target grid's (row, col). That only works if every predictor
# file is already aligned to the target grid. These helpers turn the aligned,
# in-memory stack into GeoTIFFs on the reference grid and return their paths.
def save_predictor_stack(stack: Dict[str, np.ndarray], ref_profile: dict,
                         out_dir: str) -> List[str]:
    """Write an aligned predictor stack {name: array} to GeoTIFFs. Returns paths."""
    os.makedirs(out_dir, exist_ok=True)
    profile = dict(ref_profile)
    profile.update(count=1, dtype="float32", nodata=np.nan)
    paths: List[str] = []
    for name, arr in stack.items():
        safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name)
        path = os.path.join(out_dir, f"{safe}.tif")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(np.asarray(arr, dtype="float32"), 1)
        paths.append(path)
    logger.info("Wrote %d aligned predictor files to %s", len(paths), out_dir)
    return paths


def prepare_predictor_files(year: int,
                            raster_paths: Dict[str, Dict[int, str]],
                            vector_paths: Dict[str, str],
                            ref_profile: dict,
                            boundary_mask: Optional[np.ndarray] = None,
                            out_dir: Optional[str] = None,
                            progress_callback=None) -> List[str]:
    """
    Produce a list of predictor GeoTIFFs aligned to the target grid for `year`,
    ready to pass straight to training.train_transition_models(predictor_files=...)
    and prediction.generate_suitability_atlas(predictor_files=...).

    This is the bridge from the fetch/align layer to the file-based ML modules:
    it aligns each raster predictor (nearest year for sparse layers), derives
    slope/aspect, rasterizes the vectors, and writes everything on one grid.

    Note: assemble_predictor_stack holds all layers in memory; at 100 m country
    scale that is a few GB. Raise TARGET_RESOLUTION or tile if memory is tight.
    """
    out_dir = out_dir or os.path.join(
        config.PROCESSING_CONFIG["output_dir"], f"predictors_{year}")
    stack = assemble_predictor_stack(year, raster_paths, vector_paths,
                                     ref_profile, boundary_mask, progress_callback)
    return save_predictor_stack(stack, ref_profile, out_dir)


# ============================================================================
# TRAINING DATA SAMPLING
# ============================================================================
def sample_training_data_stratified(
    target_file,
    predictor_dict: Dict[str, np.ndarray],
    boundary_mask: Optional[np.ndarray] = None,
    samples_per_class: int = 2000,
    progress_callback=None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Stratified random sampling for a balanced training set."""
    progress_callback = progress_callback or _noop_progress
    if not predictor_dict:
        logger.error("No predictors available for sampling.")
        return None
    try:
        progress_callback(0.0, "Starting stratified sampling...")
        class_locations: Dict = {}
        with _open_as_raster(target_file) as lc_src:
            nodata = lc_src.nodata
            total_blocks = sum(1 for _ in lc_src.block_windows(1))
            for block_idx, (_, window) in enumerate(lc_src.block_windows(1)):
                lc_window = lc_src.read(1, window=window)
                valid_mask = (lc_window != nodata) & np.isfinite(lc_window)
                for cls in np.unique(lc_window[valid_mask]):
                    rows, cols = np.where(lc_window == cls)
                    coords = list(zip(rows + window.row_off, cols + window.col_off))
                    class_locations.setdefault(cls, []).extend(coords)
                progress_callback(0.4 * (block_idx + 1) / total_blocks,
                                  "Scanning land cover classes...")
        if not class_locations:
            logger.error("No valid land cover classes found.")
            return None
        logger.info("Found %d classes.", len(class_locations))

        X_samples, y_samples = [], []
        num_classes = len(class_locations)
        for class_idx, (cls, locations) in enumerate(class_locations.items()):
            n = min(samples_per_class, len(locations))
            idxs = np.random.choice(len(locations), n, replace=False)
            for i in idxs:
                r, c = locations[i]
                try:
                    vals = [arr[r, c] for arr in predictor_dict.values()
                            if not np.isnan(arr[r, c])]
                    if len(vals) == len(predictor_dict):
                        X_samples.append(vals)
                        y_samples.append(cls)
                except (IndexError, ValueError):
                    continue
            progress_callback(0.4 + 0.6 * (class_idx + 1) / num_classes,
                              f"Sampled class {cls}")
        if not X_samples:
            logger.error("Sampling produced an empty dataset.")
            return None
        progress_callback(1.0, "Sampling complete.")
        return (np.array(X_samples, dtype=np.float32),
                np.array(y_samples, dtype=np.int32))
    except Exception as e:
        logger.error("Sampling error: %s", e)
        return None