"""Dated MODIS seasonal NDVI/EVI features, downloaded without Earth Engine.

Twelve Terra composites per tile/year, all ending within the starting year.
Native 500m QA is applied before nearest-neighbor reprojection to avoid averaging
across adjacent vegetation pixels.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject

from .comparison import DEFAULT_DATA, read_raster
from .landsat_features import API, download_asset, request_json
from .models import NODATA

LOGGER = logging.getLogger(__name__)
ASSETS = ["500m_16_days_NDVI", "500m_16_days_EVI"]
QA = "500m_16_days_VI_Quality"
TARGET_DAYS = [8 + 30 * i for i in range(12)]


def valid_qa(qa):
    return ((qa & 3) <= 1) & (((qa >> 2) & 15) <= 11) & (((qa >> 6) & 3) < 3) & ((qa & ((1 << 8) | (1 << 10) | (1 << 14) | (1 << 15))) == 0)


def discover(year, profile, land):
    bbox = list(rasterio.transform.array_bounds(*land.shape, profile["transform"]))
    query = dict(collections=["modis-13A1-061"], bbox=bbox,
        datetime=f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z", limit=1000,
        query={"platform": {"eq": "terra"}})
    page = request_json(API + "/search", query)
    items = list(page.get("features", []))
    while link := next((l for l in page.get("links", []) if l.get("rel") == "next"), None):
        body = link.get("body")
        if link.get("merge") and body:
            body = dict(query, **body)
        page = request_json(link["href"], body if link.get("method") == "POST" else None)
        items.extend(page.get("features", []))
    groups = {}
    for item in items:
        props = item["properties"]
        if props.get("platform") != "terra":
            continue
        if props["start_datetime"][:4] != str(year) or props["end_datetime"][:4] != str(year):
            continue
        inside = geometry_mask([item["geometry"]], land.shape, profile["transform"], invert=True)
        if np.sum(inside & land) < 1000:
            continue
        tile = (props["modis:horizontal-tile"], props["modis:vertical-tile"])
        groups.setdefault(tile, []).append(item)
    selected = []
    for tile in sorted(groups):
        for period, target_day in enumerate(TARGET_DAYS):
            chosen = min(groups[tile], key=lambda i: (abs(datetime.fromisoformat(
                i["properties"]["start_datetime"].replace("Z", "+00:00")).timetuple().tm_yday-target_day), i["id"]))
            selected.append(dict(item=chosen, period=period))
    if not selected:
        raise ValueError("No Terra composites found for the requested year")
    return dict(year=year, query=query, selected=selected)


def extract(entry, root, profile, land):
    item, period = entry["item"], entry["period"]
    path = root / f"{period:02d}_{item['id']}.tif"
    if path.exists():
        with rasterio.open(path) as src:
            if src.tags().get("scene_id") != item["id"] or src.count != 2:
                raise ValueError("Cached phenology scene mismatch")
        return path
    raw_dir = root / "raw"
    raw_dir.mkdir(exist_ok=True)
    downloaded = []
    qa_path = download_asset(item, QA, raw_dir)
    downloaded.append(qa_path)
    with rasterio.open(qa_path) as src:
        good = valid_qa(src.read(1))
        transform, crs = src.transform, src.crs
    result = np.full((2, *land.shape), np.nan, dtype=np.float32)
    for band, asset in enumerate(ASSETS):
        raw = download_asset(item, asset, raw_dir)
        downloaded.append(raw)
        with rasterio.open(raw) as src:
            if src.transform != transform or src.crs != crs or src.shape != good.shape:
                raise ValueError("MODIS QA and reflectance grids differ")
            values = src.read(1).astype(np.float32)
            valid = good & (values >= -2000) & (values <= 10000)
            values[~valid] = np.nan
            values *= item["assets"][asset]["raster:bands"][0]["scale"]
        reproject(values, result[band], src_transform=transform, src_crs=crs,
            dst_transform=profile["transform"], dst_crs=profile["crs"],
            src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
        result[band, ~land] = np.nan
    temporary = path.with_suffix(".partial.tif")
    with rasterio.open(temporary, "w", **dict(profile, count=2, dtype="float32", nodata=np.nan, compress="deflate", predictor=3)) as dst:
        dst.write(result)
        dst.set_band_description(1, "ndvi")
        dst.set_band_description(2, "evi")
        dst.update_tags(scene_id=item["id"], period=str(period),
                        start=item["properties"]["start_datetime"], end=item["properties"]["end_datetime"])
    temporary.replace(path)
    for raw in downloaded:
        raw.unlink()
    LOGGER.info("MODIS %s period %d: %d valid land pixels", item["id"], period, int(np.isfinite(result[0]).sum()))
    return path


def fetch_year(year, output_dir, reference_path=DEFAULT_DATA / "Guinea_LandCover_2010.tif", workers=2):
    root = Path(output_dir) / str(year)
    root.mkdir(parents=True, exist_ok=True)
    scenes = root / "scenes"
    scenes.mkdir(exist_ok=True)
    landcover, profile = read_raster(reference_path, categorical=True)
    land = landcover != NODATA
    catalog = root / "catalog.json"
    if catalog.exists():
        manifest = json.loads(catalog.read_text())
    else:
        manifest = discover(year, profile, land)
        catalog.write_text(json.dumps(manifest, indent=2))
    LOGGER.info("MODIS %d: %d tile/composites", year, len(manifest["selected"]))
    successful, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = {executor.submit(extract, e, scenes, profile, land): e for e in manifest["selected"]}
        for future in as_completed(tasks):
            try:
                successful.append(future.result())
            except Exception as error:
                failed.append(dict(scene_id=tasks[future]["item"]["id"], error_type=type(error).__name__))
                LOGGER.error("MODIS %s failed: %s", tasks[future]["item"]["id"], type(error).__name__)
    if not successful:
        raise RuntimeError("No MODIS composites acquired")
    # Mosaic only within the same seasonal period. Never mix final-year data.
    cube = np.full((24, *land.shape), np.nan, dtype=np.float32)
    for path in sorted(successful):
        with rasterio.open(path) as src:
            period, values = int(src.tags()["period"]), src.read()
        for b in range(2):
            layer = cube[period * 2 + b]
            update = np.isfinite(values[b]) & ~np.isfinite(layer)
            layer[update] = values[b][update]
    names = [f"{name}_period_{p+1:02d}" for p in range(12) for name in ("ndvi", "evi")]
    destination = root / "seasonal_vegetation.tif"
    with rasterio.open(destination, "w", **dict(profile, count=24, dtype="float32", nodata=np.nan, compress="deflate", predictor=3)) as dst:
        dst.write(cube)
        for i, name in enumerate(names, 1):
            dst.set_band_description(i, name)
        dst.update_tags(year=str(year), source="MOD13A1.061 Terra", latest_allowed_date=f"{year}-12-31")
    count = np.isfinite(cube[::2]).sum(axis=0)
    report = dict(year=year, source="MOD13A1.061 Terra via Microsoft Planetary Computer", features=names,
        target_days=TARGET_DAYS, completed=len(successful), requested=len(manifest["selected"]), failed=failed,
        coverage_any=float(np.mean(count[land] > 0)), coverage_six_periods=float(np.mean(count[land] >= 6)),
        period_coverage=[float(np.mean(np.isfinite(cube[2*p])[land])) for p in range(12)],
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        notes=["All source composites start and end within the requested starting year.",
               "Native MODIS QA, no high aerosols, adjacent/mixed cloud, snow or shadow.",
               "Nearest-neighbor reprojection to preserve native 500m pixel identity.",
               "Missing periods remain missing and must not reduce the evaluation footprint."])
    (root / "manifest.json").write_text(json.dumps(report, indent=2))
    LOGGER.info("MODIS %d: %.2f%% any coverage; %.2f%% at least six periods", year, 100*report["coverage_any"], 100*report["coverage_six_periods"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2010, 2015, 2005])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for year in args.years:
        fetch_year(year, args.output_dir, workers=args.workers)


if __name__ == "__main__":
    main()
