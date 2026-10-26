"""Seasonal MODIS vegetation on the ESRI 250 m grid, and the features built from it.

Source: MOD13Q1.061 (Terra, 16-day NDVI/EVI, native 250 m) from Microsoft Planetary
Computer, no account needed. The download follows the intern's
local_landuse.modis_phenology (same QA rule, twelve composites per year nearest
to the same target days, nearest-neighbour reprojection), with the 250 m product
and the ESRI grid instead of the 500 m product and the MODIS grid.

Features (see notebook 07 for the literature behind each):
  phenology of the year      annual mean, amplitude, dry- and wet-season mean NDVI, EVI mean
  spectral change evidence   change of those against the three previous years, and the
                             departure of the year's seasonal profile from the previous
                             years' profile (the year itself, and the year before)
  trend                      slope of annual mean NDVI over the last five years

Run as a script to download:  python -m esri_seasonal --years 2018 2019 ... 2024
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

import nb_helpers as H
import esri_methods as E

import sys
if str(H.INTERN) not in sys.path:          # the intern's package: 1_intern_improvement/local_landuse
    sys.path.insert(0, str(H.INTERN))
from local_landuse.landsat_features import API, download_asset, request_json
from local_landuse.modis_phenology import TARGET_DAYS, valid_qa

LOGGER = logging.getLogger(__name__)
COLLECTION = "modis-13Q1-061"
ASSETS = ["250m_16_days_NDVI", "250m_16_days_EVI"]
QA = "250m_16_days_VI_Quality"
ROOT = H.DATA / "esri_seasonal"
SCALE, NODATA = 10000, -32768
DRY = [0, 1, 2, 3, 11]      # periods centred in Dec-Apr (Guinea's dry season)
WET = [5, 6, 7, 8, 9]       # Jun-Oct
FEATURE_NAMES = ["NDVI annual mean", "NDVI amplitude (p90-p10)", "NDVI dry-season mean", "NDVI wet-season mean",
                 "EVI annual mean", "Change in NDVI mean vs 3 previous years",
                 "Change in NDVI amplitude vs 3 previous years", "Change in dry-season NDVI vs 3 previous years",
                 "Seasonal profile departure, this year", "Seasonal profile departure, last year",
                 "NDVI mean trend, 5 years", "Valid composites this year"]


# --------------------------------------------------------------------------
# Download

def _items(year, bbox):
    """All Terra composites of `year`, searched month by month (small pages, no paging needed).
    Terra is selected by product id (MOD13Q1; Aqua is MYD13Q1): the `platform` property is
    missing on most 2024 items, so filtering on it silently drops them."""
    items = {}
    for month in range(1, 13):
        last = (datetime(year + month // 12, month % 12 + 1, 1) - __import__("datetime").timedelta(seconds=1))
        query = dict(collections=[COLLECTION], bbox=bbox, limit=100,
                     datetime=f"{year}-{month:02d}-01T00:00:00Z/{last:%Y-%m-%dT%H:%M:%S}Z")
        page = request_json(API + "/search", query)
        if len(page.get("features", [])) >= 100:
            raise RuntimeError(f"{year}-{month:02d}: search may be truncated")
        items.update({i["id"]: i for i in page.get("features", []) if i["id"].startswith("MOD13Q1.")})
    return [i for i in items.values() if i["properties"]["start_datetime"][:4] == str(year)
            and i["properties"]["end_datetime"][:4] == str(year)]


def _select(items):
    tiles = {}
    for item in items:
        p = item["properties"]
        tiles.setdefault((p["modis:horizontal-tile"], p["modis:vertical-tile"]), []).append(item)
    day = lambda i: datetime.fromisoformat(i["properties"]["start_datetime"].replace("Z", "+00:00")).timetuple().tm_yday
    return [dict(item=min(group, key=lambda i: (abs(day(i) - target), i["id"])), period=period)
            for tile, group in sorted(tiles.items()) for period, target in enumerate(TARGET_DAYS)]


def _extract(entry, scenes, profile, land):
    item, period = entry["item"], entry["period"]
    path = scenes / f"{period:02d}_{item['id']}.tif"
    if path.exists():
        return path
    raw_dir = scenes / f"raw_{period:02d}"     # one per period: a composite can be nearest to two periods
    raw_dir.mkdir(exist_ok=True)
    qa_path = download_asset(item, QA, raw_dir)
    with rasterio.open(qa_path) as src:
        good, transform, crs = valid_qa(src.read(1)), src.transform, src.crs
    out = np.full((2, *land.shape), np.nan, dtype=np.float32)
    downloaded = [qa_path]
    for band, asset in enumerate(ASSETS):
        raw = download_asset(item, asset, raw_dir)
        downloaded.append(raw)
        with rasterio.open(raw) as src:
            v = src.read(1).astype(np.float32)
        v[~(good & (v >= -2000) & (v <= 10000))] = np.nan
        reproject(v, out[band], src_transform=transform, src_crs=crs, dst_transform=profile["transform"],
                  dst_crs=profile["crs"], src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
    out[:, ~land] = np.nan
    stored = np.where(np.isfinite(out), out, NODATA).astype(np.int16)    # MODIS VI are already x10000
    tmp = path.with_suffix(".partial.tif")
    with rasterio.open(tmp, "w", **dict(profile, count=2, dtype="int16", nodata=NODATA, compress="deflate")) as dst:
        dst.write(stored)
        dst.update_tags(scene_id=item["id"], period=str(period))
    tmp.replace(path)
    for raw in downloaded:
        raw.unlink()
    return path


def fetch_year(year, workers=4):
    """Twelve-period NDVI/EVI cube for `year` on the ESRI grid: data/esri_seasonal/<year>/cube.tif."""
    root = ROOT / str(year)
    cube_path = root / "cube.tif"
    if cube_path.exists():
        return cube_path
    scenes = root / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    profile = E.grid()
    land = E.deployed_mask(profile)
    bbox = list(rasterio.transform.array_bounds(*land.shape, profile["transform"]))
    selected = _select(_items(year, bbox))
    (root / "catalog.json").write_text(json.dumps([dict(id=s["item"]["id"], period=s["period"]) for s in selected], indent=1))
    LOGGER.info("%d: %d tile composites", year, len(selected))
    done = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(_extract, s, scenes, profile, land): s for s in selected}
        for fut in as_completed(tasks):
            done.append(fut.result())      # any failure stops the year: no silent gaps
    cube = np.full((24, *land.shape), NODATA, dtype=np.int16)
    for path in sorted(done):
        with rasterio.open(path) as src:
            period, values = int(src.tags()["period"]), src.read()
        for b in range(2):
            layer = cube[period * 2 + b]
            fill = (values[b] != NODATA) & (layer == NODATA)
            layer[fill] = values[b][fill]
    with rasterio.open(cube_path, "w", **dict(profile, count=24, dtype="int16", nodata=NODATA, compress="deflate")) as dst:
        dst.write(cube)
        for i in range(24):
            dst.set_band_description(i + 1, f"{('ndvi', 'evi')[i % 2]}_period_{i // 2 + 1:02d}")
        dst.update_tags(year=str(year), source="MOD13Q1.061 Terra via Microsoft Planetary Computer",
                        latest_allowed_date=f"{year}-12-31")
    ndvi_ok = (cube[::2] != NODATA)[:, land]
    report = dict(year=year, composites=len(done), coverage_any=float(ndvi_ok.any(0).mean()),
                  coverage_six_periods=float((ndvi_ok.sum(0) >= 6).mean()),
                  period_coverage=[float(c) for c in ndvi_ok.mean(1)])
    (root / "manifest.json").write_text(json.dumps(report, indent=2))
    for p in done:
        p.unlink()
    LOGGER.info("%d done: coverage any %.1f%%, >=6 periods %.1f%%", year, 100 * report["coverage_any"],
                100 * report["coverage_six_periods"])
    return cube_path


# --------------------------------------------------------------------------
# Features

_CUBES, _FEATS = {}, {}


def cube(year):
    """(ndvi, evi), each (12, H, W) float32 with NaN where QA failed."""
    if year not in _CUBES:
        with rasterio.open(ROOT / str(year) / "cube.tif") as src:
            c = src.read().astype(np.float32)
        c[c == NODATA] = np.nan
        c /= SCALE
        _CUBES[year] = (c[0::2], c[1::2])
    return _CUBES[year]


def _nanpct(stack, q):
    """Per-pixel percentile over axis 0 ignoring NaN (nearest rank; np.nanpercentile is far
    too slow on 12 x 8.4 million values)."""
    ordered = np.sort(stack, axis=0)                       # NaN sort last
    n = np.isfinite(stack).sum(0)
    k = np.clip(np.round(q / 100 * (n - 1)), 0, None).astype(np.intp)
    out = np.take_along_axis(ordered, k[None], 0)[0]
    return np.where(n > 0, out, np.nan)


def _summary(year):
    import warnings
    ndvi, evi = cube(year)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dict(mean=np.nanmean(ndvi, 0), amp=_nanpct(ndvi, 90) - _nanpct(ndvi, 10),
                    dry=np.nanmean(ndvi[DRY], 0), wet=np.nanmean(ndvi[WET], 0), evi=np.nanmean(evi, 0),
                    n=np.isfinite(ndvi).sum(0).astype(np.float32))


def _departure(year):
    """RMS difference between the year's 12-period NDVI profile and the mean profile of
    the three previous years (a CCDC-style change magnitude, without model fitting)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = np.nanmean(np.stack([cube(y)[0] for y in (year - 3, year - 2, year - 1)]), 0)
        return np.sqrt(np.nanmean((cube(year)[0] - ref) ** 2, 0))


def seasonal_features(year, to_class=None, mask=None):
    """Twelve layers from MODIS maps of year-4 .. year only. Missing values are filled
    with the Guinea median; the last layer says how many composites were valid.
    Signature matches esri_methods.run_deployed(transition_features=...)."""
    import warnings
    if year not in _FEATS:
        s = {y: _summary(y) for y in range(year - 4, year + 1)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prev = {k: np.nanmean(np.stack([s[y][k] for y in (year - 3, year - 2, year - 1)]), 0)
                    for k in ("mean", "amp", "dry")}
            means = np.stack([s[y]["mean"] for y in range(year - 4, year + 1)])
            x = np.arange(5, dtype=np.float32) - 2
            ok = np.isfinite(means)
            xm = np.where(ok, x[:, None, None], np.nan)
            slope = np.nansum((xm - np.nanmean(xm, 0)) * (means - np.nanmean(means, 0)), 0) / \
                    np.nansum((xm - np.nanmean(xm, 0)) ** 2, 0)
        cur = s[year]
        layers = [cur["mean"], cur["amp"], cur["dry"], cur["wet"], cur["evi"], cur["mean"] - prev["mean"],
                  cur["amp"] - prev["amp"], cur["dry"] - prev["dry"], _departure(year), _departure(year - 1),
                  slope, cur["n"]]
        g = H.guinea_mask(E.grid())
        out = []
        for layer in layers:
            layer = layer.astype(np.float32)
            bad = ~np.isfinite(layer)
            layer[bad] = np.nanmedian(layer[g & ~bad])
            out.append(layer)
        _FEATS[year] = np.stack(out)
        _CUBES.clear()                # ~770 MB per year; features are what is kept
    f = _FEATS[year]
    return f if mask is None else np.where(mask[None], f, 0).astype(np.float32)


def seasonal_plus_history(year, to_class, mask):
    return np.vstack([seasonal_features(year, to_class, mask), E.history_features(year, to_class, mask)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2018, 2025)))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for year in args.years:
        fetch_year(year, args.workers)


if __name__ == "__main__":
    main()
