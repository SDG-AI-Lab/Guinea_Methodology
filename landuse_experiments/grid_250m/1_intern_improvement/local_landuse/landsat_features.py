"""Download dated, QA-screened historical Landsat predictors from public COGs.

Inputs are January-April of the requested starting year, never future years.
QA is decoded at native resolution; a 500m cell requires >=99% clear native
support. Reflectance is read through COG overviews/average reprojection, so this
is a coarse predictive composite, not a new reference land-cover product.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time
from threading import Lock
import urllib.parse
import urllib.request

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject

from .comparison import DEFAULT_DATA, read_raster
from .models import NODATA

LOGGER = logging.getLogger(__name__)
API = "https://planetarycomputer.microsoft.com/api/stac/v1"
BANDS = ("red", "nir08", "swir16", "swir22")
FEATURE_NAMES = list(BANDS) + ["ndvi", "ndmi", "nbr"]
TOKEN_CACHE = {}
TOKEN_LOCK = Lock()


def request_json(url, payload=None):
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, data=json.dumps(payload).encode() if payload else None,
                                              headers={"Content-Type": "application/json", "Cache-Control": "no-cache"})
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 5))


def sign_url(href):
    parts = urllib.parse.urlsplit(href)
    key = (parts.netloc, parts.path.split('/')[1])
    with TOKEN_LOCK:
        token, expires = TOKEN_CACHE.get(key, (None, 0))
        if expires < time.time() + 300:
            signed = request_json("https://planetarycomputer.microsoft.com/api/sas/v1/sign?" +
                urllib.parse.urlencode({"href": href, "cache_bust": int(time.time())}))["href"]
            query = urllib.parse.urlsplit(signed).query
            values = urllib.parse.parse_qs(query)
            expires = datetime.fromisoformat(values["se"][0].replace("Z", "+00:00")).timestamp()
            if expires <= time.time() + 60:
                raise RuntimeError("Signing API returned an expired/near-expiry token")
            if values.get("sr") != ["c"]:
                return signed
            token = query
            TOKEN_CACHE[key] = (token, expires)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, token, ""))


def download_asset(item, band, raw_dir):
    """Stream one full COG to a temporary local cache, avoiding fragile range I/O."""
    path = raw_dir / f"{item['id']}_{band}.tif"
    if path.exists():
        with rasterio.open(path):
            pass
        return path
    partial = path.with_suffix(".download")
    for attempt in range(5):
        try:
            request = urllib.request.Request(sign_url(item["assets"][band]["href"]),
                                             headers={"Cache-Control": "no-cache"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as stream:
                if response.status != 200:
                    raise IOError("Expected a complete asset response")
                expected = int(response.headers.get("Content-Length", 0))
                if expected > 300_000_000:
                    raise ValueError("Unexpectedly large scene asset")
                count = 0
                while chunk := response.read(1_048_576):
                    stream.write(chunk)
                    count += len(chunk)
                if expected and count != expected:
                    raise IOError("Incomplete asset response")
            with rasterio.open(partial) as src:
                if src.count != 1:
                    raise ValueError("Expected a single-band asset")
            partial.replace(path)
            return path
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 5))


def clear_native(qa, saturation):
    """Cloud/fill/shadow/snow bits 0-5; water retained; no saturated/dropped pixels."""
    return (qa != 0) & ((qa & 63) == 0) & (saturation == 0)


def spectral_indices(reflectance):
    red, nir, swir1, swir2 = reflectance
    indices = []
    for other in (red, swir1, swir2):
        denominator = nir + other
        indices.append(np.divide(nir - other, denominator,
            out=np.full_like(nir, np.nan), where=np.abs(denominator) > .02))
    return np.stack([red, nir, swir1, swir2, *indices]).astype(np.float32)


def discover(year, reference, land_mask, scenes_per_tile=2, platform=None):
    bounds = rasterio.transform.array_bounds(reference["height"], reference["width"], reference["transform"])
    platform = platform or ("landsat-5" if year < 2013 else "landsat-8")
    query = dict(collections=["landsat-c2-l2"], bbox=list(bounds),
        datetime=f"{year}-01-01T00:00:00Z/{year}-04-30T23:59:59Z", limit=1000,
        query={"eo:cloud_cover": {"lt": 20}, "platform": {"eq": platform}})
    page = request_json(API + "/search", query)
    items = list(page.get("features", []))
    while next_link := next((link for link in page.get("links", []) if link.get("rel") == "next"), None):
        page = request_json(next_link["href"], next_link.get("body") if next_link.get("method") == "POST" else None)
        items.extend(page.get("features", []))
    groups = {}
    for item in items:
        if int(item["properties"]["datetime"][:4]) != year:
            raise ValueError("Catalog returned an image outside the requested year")
        if not all(b in item["assets"] for b in (*BANDS, "qa_pixel", "qa_radsat")):
            continue
        inside = geometry_mask([item["geometry"]], out_shape=land_mask.shape,
                               transform=reference["transform"], invert=True)
        overlap = int(np.sum(inside & land_mask))
        if overlap < 1000:
            continue
        props = item["properties"]
        key = (props["landsat:wrs_path"], props["landsat:wrs_row"])
        groups.setdefault(key, []).append(item)
    selected = []
    for key in sorted(groups):
        selected.extend(sorted(groups[key], key=lambda i: (i["properties"]["eo:cloud_cover"], i["id"]))[:scenes_per_tile])
    if not selected:
        if platform == "landsat-5":
            LOGGER.info("No Landsat 5 scenes for %d; checking same-year Landsat 7", year)
            return discover(year, reference, land_mask, scenes_per_tile, platform="landsat-7")
        raise ValueError(f"No suitable scenes for {year}")
    return query, selected


def average_clear_native(values, clear, transform, crs, reference):
    """Exclude native gaps/clouds before averaging, without gap interpolation."""
    masked = np.where(clear > 0, values, np.nan).astype(np.float32)
    result = np.full((reference["height"], reference["width"]), np.nan, dtype=np.float32)
    reproject(masked, result, src_transform=transform, src_crs=crs,
        dst_transform=reference["transform"], dst_crs=reference["crs"],
        src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average)
    return result


def extract_scene(item, reference, land_mask, scene_dir):
    destination = scene_dir / f"{item['id']}.tif"
    if destination.exists():
        with rasterio.open(destination) as src:
            if src.tags().get("scene_id") != item["id"] or src.count != len(FEATURE_NAMES):
                raise ValueError("Cached scene metadata mismatch")
        return destination
    shape = (reference["height"], reference["width"])
    raw_dir = scene_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    raw_paths = []

    def local_asset(band):
        path = download_asset(item, band, raw_dir)
        raw_paths.append(path)
        return path
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="60",
                      GDAL_HTTP_MAX_RETRY="2", GDAL_HTTP_RETRY_DELAY="2", GDAL_CACHEMAX=128_000_000):
        with rasterio.open(local_asset("qa_pixel")) as src:
            qa = src.read(1)
            transform, crs = src.transform, src.crs
        with rasterio.open(local_asset("qa_radsat")) as src:
            if src.transform != transform or src.crs != crs or src.shape != qa.shape:
                raise ValueError("QA grids disagree")
            clear = clear_native(qa, src.read(1)).astype(np.float32)
        del qa
        fraction = np.full(shape, -1., dtype=np.float32)
        reproject(clear, fraction, src_transform=transform, src_crs=crs,
                  dst_transform=reference["transform"], dst_crs=reference["crs"],
                  resampling=Resampling.average, dst_nodata=-1)
        slc_off = item["properties"]["platform"] == "landsat-7"
        # Landsat 7's native scan-line gaps must be excluded before aggregation.
        # A 500m cell needs at least half its area observed and clear; no values
        # are synthesized in the gaps. Other sensors retain the original method.
        minimum_support = .5 if slc_off else .99
        good = (fraction >= minimum_support) & land_mask
        reflectance = []
        for band in BANDS:
            asset = item["assets"][band]
            metadata = asset["raster:bands"][0]
            with rasterio.open(local_asset(band)) as src:
                if slc_off:
                    if src.transform != transform or src.crs != crs or src.shape != clear.shape:
                        raise ValueError("Landsat band and native QA grids disagree")
                    values = average_clear_native(src.read(1), clear, transform, crs, reference)
                else:
                    with WarpedVRT(src, crs=reference["crs"], transform=reference["transform"],
                        height=shape[0], width=shape[1], dtype="float32", nodata=np.nan,
                        resampling=Resampling.average) as vrt:
                        values = vrt.read(1)
            values = values * metadata["scale"] + metadata["offset"]
            values[~good] = np.nan
            reflectance.append(values)
        del clear
        features = spectral_indices(reflectance)
        valid = np.isfinite(features).all(axis=0)
        features[:, ~valid] = np.nan
    profile = reference.copy()
    profile.update(count=len(FEATURE_NAMES), dtype="float32", nodata=np.nan,
                   compress="deflate", predictor=3)
    temporary = destination.with_suffix(".partial.tif")
    with rasterio.open(temporary, "w", **profile) as dst:
        dst.write(features)
        for i, name in enumerate(FEATURE_NAMES, 1):
            dst.set_band_description(i, name)
        dst.update_tags(scene_id=item["id"], acquisition_time=item["properties"]["datetime"],
                        clear_native_fraction_threshold=str(minimum_support), collection="landsat-c2-l2",
                        aggregation="native_QA_masked_average" if slc_off else "overview_average_with_clear_support")
    temporary.replace(destination)
    # Only discard this job's raw downloads after the derived scene is saved.
    for path in raw_paths:
        path.unlink()
    LOGGER.info("Scene %s: %d clear land pixels", item["id"], int(valid.sum()))
    return destination


def fetch_year(year, output_dir, reference_path=DEFAULT_DATA / "Guinea_LandCover_2010.tif",
               workers=2, scenes_per_tile=2):
    root = Path(output_dir) / str(year)
    root.mkdir(parents=True, exist_ok=True)
    scene_dir = root / "scenes"
    scene_dir.mkdir(exist_ok=True)
    land, reference = read_raster(reference_path, categorical=True)
    land_mask = land != NODATA
    catalog = root / "catalog.json"
    if catalog.exists():
        saved = json.loads(catalog.read_text())
        if saved["year"] != year or saved["scenes_per_tile"] != scenes_per_tile:
            raise ValueError("Cached discovery settings differ")
        items = saved["items"]
    else:
        query, items = discover(year, reference, land_mask, scenes_per_tile)
        catalog.write_text(json.dumps(dict(year=year, scenes_per_tile=scenes_per_tile,
                                          query=query, items=items), indent=2))
    LOGGER.info("Year %d: %d scenes selected", year, len(items))
    completed, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(extract_scene, i, reference, land_mask, scene_dir): i for i in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                failed.append(dict(scene_id=item["id"], error_type=type(error).__name__))
                LOGGER.error("Scene %s failed: %s", item["id"], type(error).__name__)
    if not completed:
        raise RuntimeError("No scene could be read; caches retained for retry")
    shape = land.shape
    sums = np.zeros((len(FEATURE_NAMES), *shape), dtype=np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(shape, dtype=np.uint16)
    for path in sorted(completed):
        with rasterio.open(path) as src:
            values = src.read()
        good = np.isfinite(values).all(axis=0)
        counts += good
        filled = np.nan_to_num(values, nan=0).astype(np.float64)
        sums += filled
        squares += filled ** 2
    mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    second = np.divide(squares, counts, out=np.full_like(squares, np.nan), where=counts > 0)
    std = np.sqrt(np.maximum(0, second - mean ** 2))
    composite = np.concatenate([mean, std, counts[None]]).astype(np.float32)
    descriptions = ["mean_" + n for n in FEATURE_NAMES] + ["std_" + n for n in FEATURE_NAMES] + ["clear_observation_count"]
    profile = reference.copy()
    profile.update(count=len(descriptions), dtype="float32", nodata=np.nan, compress="deflate", predictor=3)
    composite_path = root / "landsat_dry_season.tif"
    with rasterio.open(composite_path, "w", **profile) as dst:
        dst.write(composite)
        for i, name in enumerate(descriptions, 1):
            dst.set_band_description(i, name)
        dst.update_tags(year=str(year), temporal_window="January-April", source="landsat-c2-l2")
    coverage = float(np.mean(counts[land_mask] > 0))
    manifest = dict(year=year, source="Microsoft Planetary Computer / USGS Landsat Collection 2 Level 2",
        date_window=[f"{year}-01-01", f"{year}-04-30"], requested_scenes=len(items),
        completed_scenes=len(completed), failed=failed, land_coverage_fraction=coverage,
        reference=str(reference_path), reference_sha256=hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
        composite_sha256=hashlib.sha256(composite_path.read_bytes()).hexdigest(),
        features=descriptions, platforms=sorted({i["properties"]["platform"] for i in items}),
        notes=["No acquisitions after the feature year.",
        "Landsat 5/8: >=99% native clear support; reflectance averaged separately using COG overviews.",
        "Landsat 7 fallback: >=50% native clear area; gaps/clouds masked before native reflectance averaging; no gap interpolation.",
        "Landsat 5 preferred before 2013, Landsat 7 if unavailable, Landsat 8 thereafter; sensor/aggregation differences require validation.",
        "Missing spectral pixels remain missing and must not change the evaluation mask."])
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    LOGGER.info("Year %d complete: %d/%d scenes, land coverage %.2f%%", year, len(completed), len(items), 100*coverage)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2005, 2010, 2015])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scenes-per-tile", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for year in args.years:
        fetch_year(year, args.output_dir, workers=args.workers, scenes_per_tile=args.scenes_per_tile)


if __name__ == "__main__":
    main()
