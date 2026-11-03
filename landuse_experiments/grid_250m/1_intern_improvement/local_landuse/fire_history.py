"""Dated MODIS burned area, downloaded without Earth Engine.

Fire is the driver behind the savanna/grassland relabelling that dominates
observed change in this landscape, so annual burn extent and burn timing are
recorded per starting year. Twelve monthly MCD64A1 granules per tile and year,
all beginning and ending inside the requested year, are combined into one annual
burned-fraction and burn-day raster on the analysis grid.

Burn_Date is 0 for unburned, 1-366 for the ordinal day of burn, and negative for
cells the algorithm could not map (missing data or water). Unmapped cells stay
missing; they never become unburned zeros.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
ASSET = "Burn_Date"
BANDS = ("burned_fraction", "first_burn_day", "mapped_fraction")


def discover(year, profile, land):
    bbox = list(rasterio.transform.array_bounds(*land.shape, profile["transform"]))
    query = dict(collections=["modis-64A1-061"], bbox=bbox,
                 datetime=f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z", limit=1000)
    page = request_json(API + "/search", query)
    items = list(page.get("features", []))
    while link := next((l for l in page.get("links", []) if l.get("rel") == "next"), None):
        body = link.get("body")
        if link.get("merge") and body:
            body = dict(query, **body)
        page = request_json(link["href"], body if link.get("method") == "POST" else None)
        items.extend(page.get("features", []))
    selected = []
    for item in items:
        props = item["properties"]
        if props["start_datetime"][:4] != str(year) or props["end_datetime"][:4] != str(year):
            continue
        inside = geometry_mask([item["geometry"]], land.shape, profile["transform"], invert=True)
        if np.sum(inside & land) < 1000:
            continue
        selected.append(dict(item=item, month=int(props["start_datetime"][5:7])))
    if not selected:
        raise ValueError(f"No burned-area granules found inside {year}")
    return dict(year=year, query=query, selected=sorted(selected, key=lambda e: (e["month"], e["item"]["id"])))


def extract(entry, root, profile, land):
    """One granule to burned fraction, earliest burn day and mapped fraction."""
    item, month = entry["item"], entry["month"]
    path = root / f"{month:02d}_{item['id']}.tif"
    if path.exists():
        with rasterio.open(path) as src:
            if src.tags().get("scene_id") != item["id"] or src.count != 3:
                raise ValueError("Cached burned-area granule mismatch")
        return path
    raw_dir = root / "raw"
    raw_dir.mkdir(exist_ok=True)
    raw = download_asset(item, ASSET, raw_dir)
    with rasterio.open(raw) as src:
        values = src.read(1).astype(np.float32)
        transform, crs = src.transform, src.crs
    mapped = values >= 0
    burned = np.where(mapped, (values > 0).astype(np.float32), np.nan)
    day = np.where(values > 0, values, np.nan)
    result = np.full((3, *land.shape), np.nan, dtype=np.float32)
    # Area-weighted so a partly burned analysis cell keeps its true burned share.
    reproject(burned, result[0], src_transform=transform, src_crs=crs,
              dst_transform=profile["transform"], dst_crs=profile["crs"],
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average)
    reproject(day, result[1], src_transform=transform, src_crs=crs,
              dst_transform=profile["transform"], dst_crs=profile["crs"],
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.min)
    reproject(mapped.astype(np.float32), result[2], src_transform=transform, src_crs=crs,
              dst_transform=profile["transform"], dst_crs=profile["crs"],
              src_nodata=None, dst_nodata=np.nan, resampling=Resampling.average)
    result[:, ~land] = np.nan
    temporary = path.with_suffix(".partial.tif")
    with rasterio.open(temporary, "w", **dict(profile, count=3, dtype="float32", nodata=np.nan,
                                              compress="deflate", predictor=3)) as dst:
        dst.write(result)
        for i, name in enumerate(BANDS, 1):
            dst.set_band_description(i, name)
        dst.update_tags(scene_id=item["id"], month=str(month),
                        start=item["properties"]["start_datetime"], end=item["properties"]["end_datetime"])
    temporary.replace(path)
    raw.unlink()
    LOGGER.info("Fire %s month %d: %.3f%% of land burned", item["id"], month,
                100 * float(np.nanmean(np.nan_to_num(result[0])[land])))
    return path


def fetch_year(year, output_dir, reference_path=DEFAULT_DATA / "Guinea_LandCover_2010.tif", workers=3):
    root = Path(output_dir) / str(year)
    scenes = root / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    landcover, profile = read_raster(reference_path, categorical=True)
    land = landcover != NODATA
    catalog = root / "catalog.json"
    manifest = (json.loads(catalog.read_text()) if catalog.exists()
                else discover(year, profile, land))
    if not catalog.exists():
        catalog.write_text(json.dumps(manifest, indent=2))
    LOGGER.info("Fire %d: %d monthly granules", year, len(manifest["selected"]))
    successful, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = {executor.submit(extract, e, scenes, profile, land): e for e in manifest["selected"]}
        for future in as_completed(tasks):
            try:
                successful.append(future.result())
            except Exception as error:
                failed.append(dict(scene_id=tasks[future]["item"]["id"], error_type=type(error).__name__))
                LOGGER.error("Fire %s failed: %s", tasks[future]["item"]["id"], type(error).__name__)
    if not successful:
        raise RuntimeError("No burned-area granules acquired")
    burned = np.zeros(land.shape, dtype=np.float32)
    day = np.full(land.shape, np.nan, dtype=np.float32)
    mapped = np.zeros(land.shape, dtype=np.float32)
    months = np.zeros(land.shape, dtype=np.float32)
    for path in sorted(successful):
        with rasterio.open(path) as src:
            values = src.read()
        known = np.isfinite(values[2])
        # Annual burned share is the sum of monthly shares; MCD64A1 maps a cell once.
        burned[known] += np.nan_to_num(values[0])[known]
        mapped[known] += values[2][known]
        months[known] += 1
        earlier = np.isfinite(values[1]) & (~np.isfinite(day) | (values[1] < day))
        day[earlier] = values[1][earlier]
    burned = np.where(months > 0, np.minimum(burned, 1.), np.nan)
    mapped = np.where(months > 0, mapped / np.maximum(months, 1), np.nan)
    cube = np.stack([burned, np.where(np.isfinite(day), day, 0).astype(np.float32), mapped])
    cube[:, ~land] = np.nan
    destination = root / "annual_fire.tif"
    with rasterio.open(destination, "w", **dict(profile, count=3, dtype="float32", nodata=np.nan,
                                                compress="deflate", predictor=3)) as dst:
        dst.write(cube)
        for i, name in enumerate(BANDS, 1):
            dst.set_band_description(i, name)
        dst.update_tags(year=str(year), source="MCD64A1.061", latest_allowed_date=f"{year}-12-31")
    report = dict(year=year, source="MCD64A1.061 via Microsoft Planetary Computer", features=list(BANDS),
                  completed=len(successful), requested=len(manifest["selected"]), failed=failed,
                  coverage_mapped=float(np.mean(np.isfinite(burned[land]))),
                  burned_land_fraction=float(np.nanmean(burned[land])),
                  months_observed_median=float(np.median(months[land])),
                  sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                  notes=["Every granule begins and ends inside the requested year.",
                         "Unmapped and water cells stay missing rather than becoming unburned.",
                         "Burned share is area-weighted; burn day is the earliest in the year.",
                         "Missing fire data must not reduce the evaluation footprint."])
    (root / "manifest.json").write_text(json.dumps(report, indent=2))
    LOGGER.info("Fire %d: %.2f%% mapped, %.2f%% of land burned", year,
                100 * report["coverage_mapped"], 100 * report["burned_land_fraction"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for year in args.years:
        fetch_year(year, args.output_dir, workers=args.workers)


if __name__ == "__main__":
    main()
