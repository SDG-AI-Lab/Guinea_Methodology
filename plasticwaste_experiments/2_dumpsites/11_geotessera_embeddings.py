"""
GeoTessera test, step 1: TESSERA embeddings at the benchmark's labeled points.

TESSERA (Feng et al. 2026, CVPR; arXiv 2506.20380) turns a year of Sentinel-1 and
Sentinel-2 into 128 numbers per 10 m pixel. Its precomputed global embeddings are
read here through the geotessera library (MIT), Zarr store, so only the sampled
pixels are downloaded.

For each of the 2,441 points in outputs/benchmark/pool.parquet:
  pixel    the 10 m pixel under the point (128 numbers)
  ctx150   mean and standard deviation over a 5 x 5 grid at 30 m spacing, about the
           150 m window the adopted DINOv3 model uses (256 numbers)

Needs Python >= 3.12 (geotessera); run with the separate environment:
    "E:/SDG AI Lab/Guinea/national_scan_2026/venv_tessera/Scripts/python" 2_dumpsites/11_geotessera_embeddings.py 2024

Output: outputs/geotessera/tessera_<year>.npz (pixel, ctx150, status, uid)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from geotessera import GeoTesseraZarr

HERE = Path(__file__).resolve().parent
POOL = HERE / "outputs" / "benchmark" / "pool.parquet"
OUT = HERE / "outputs" / "geotessera"
OUT.mkdir(parents=True, exist_ok=True)
STEP_M, HALF = 30.0, 2          # 5 x 5 offsets at 30 m: -60..+60 m, about a 150 m window


def offsets(lat):
    k = np.arange(-HALF, HALF + 1) * STEP_M
    dy, dx = np.meshgrid(k, k, indexing="ij")
    return (dx.ravel() / (111_320 * np.cos(np.radians(lat))), dy.ravel() / 110_540)


def main(year):
    pool = pd.read_parquet(POOL)
    gt = GeoTesseraZarr()
    print("years in store:", getattr(gt, "years", "?"), flush=True)

    # fail fast: the first 20 points must come back with values before the full run.
    # Missing pixels (water, no data, outside) come back as NaN.
    first = np.asarray(gt.sample_points(list(zip(pool.lon[:20].astype(float), pool.lat[:20].astype(float))),
                                        year=year, progress=False))
    print("first 20 valid:", int((~np.isnan(first).any(1)).sum()), flush=True)
    if np.isnan(first).any(1).all():
        raise SystemExit(f"no embeddings for {year} at the first points; check coverage before the full run")

    pixel = np.asarray(gt.sample_points(list(zip(pool.lon.astype(float), pool.lat.astype(float))), year=year),
                       dtype=np.float32)
    status = np.where(np.isnan(pixel).any(1), "missing", "valid")
    print("pixel", pixel.shape, pd.Series(status).value_counts().to_dict(), flush=True)

    ctx = np.zeros((len(pool), 256), np.float32)
    for i0 in range(0, len(pool), 200):
        part = pool.iloc[i0:i0 + 200]
        pts = []
        for lo, la in zip(part.lon, part.lat):
            ox, oy = offsets(la)
            pts += list(zip(lo + ox, la + oy))
        E = np.asarray(gt.sample_points(pts, year=year), dtype=np.float32).reshape(len(part), -1, 128)
        ctx[i0:i0 + len(part)] = np.concatenate([np.nanmean(E, 1), np.nanstd(E, 1)], 1)
        print(f"ctx150 {i0 + len(part)}/{len(pool)}", flush=True)

    np.savez(OUT / f"tessera_{year}.npz", pixel=pixel, ctx150=ctx,
             status=np.array(status), uid=pool.uid.to_numpy().astype(str))
    print("saved", OUT / f"tessera_{year}.npz")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024)
