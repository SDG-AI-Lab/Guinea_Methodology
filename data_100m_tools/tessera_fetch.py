"""TESSERA embeddings on the 100 m cells of the sample blocks (tessera_blocks.py).

For each block and year, reads the 10 m embeddings of the block (geotessera, Zarr store, v1.1;
dequantised float32, 128 numbers per pixel; NaN where there is no data) and averages the 10 x 10
pixels around each 100 m cell centre: one 128-number vector per cell, NaN if the cell has no
valid pixel. Saved as STORE/tessera/<year>/<block>.npy (float16, cells x 128, in the order of
blocks_cells.npz). Existing files are skipped, so the run can be restarted.

Needs Python >= 3.12 (geotessera):
    "E:/SDG AI Lab/Guinea/national_scan_2026/venv_tessera/Scripts/python" data_100m_tools/tessera_fetch.py 2017 2018 ...
    add --block NAME to fetch one block only (the first, timed check).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from pyproj import Transformer

STORE = Path(os.environ.get("GUINEA_DATA_100M", r"E:\SDG AI Lab\Guinea\data_100m"))
OUT = STORE / "tessera"
PAD = 0.002                                   # degrees around the block, so edge cells get whole windows
ZONE_EDGE = -12.0                             # UTM zones 28 | 29


def cell_means(mosaic, transform, crs, lon, lat):
    """Mean of the 10 x 10 pixels (100 m) centred on each cell centre."""
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    inv = ~transform
    col, row = inv * (np.asarray(x), np.asarray(y))
    r0 = np.floor(row - 5 + 0.5).astype(int); c0 = np.floor(col - 5 + 0.5).astype(int)
    out = np.full((lon.size, mosaic.shape[2]), np.nan, np.float32)
    k = np.arange(10)
    for i in range(0, lon.size, 2000):
        rr = np.clip(r0[i:i + 2000, None] + k, 0, mosaic.shape[0] - 1)
        cc = np.clip(c0[i:i + 2000, None] + k, 0, mosaic.shape[1] - 1)
        win = mosaic[rr[:, :, None], cc[:, None, :]]            # n x 10 x 10 x 128
        with np.errstate(all="ignore"):
            out[i:i + 2000] = np.nanmean(win.reshape(win.shape[0], 100, -1), 1)
    return out


def main(years, only=None):
    from geotessera import GeoTesseraZarr
    gt = GeoTesseraZarr()
    blocks = json.loads((OUT / "blocks.json").read_text())["blocks"]
    cells = np.load(OUT / "blocks_cells.npz")
    for year in years:
        (OUT / str(year)).mkdir(parents=True, exist_ok=True)
        for b in blocks:
            name = b["name"]
            if only and name != only:
                continue
            path = OUT / str(year) / f"{name}.npy"
            if path.exists():
                continue
            t = time.time()
            w, s, e, n = b["bbox"]
            # TESSERA is stored per UTM zone (28 west of -12 degrees, 29 east); read_region reads one
            # zone only, so the padding must not cross -12 degrees.
            wp = max(w - PAD, ZONE_EDGE + 1e-6) if w >= ZONE_EDGE else w - PAD
            ep = min(e + PAD, ZONE_EDGE - 1e-6) if e <= ZONE_EDGE else e + PAD
            mosaic, transform, crs = gt.read_region((wp, s - PAD, ep, n + PAD), year=year)
            v = cell_means(mosaic, transform, crs, cells[f"{name}_lon"], cells[f"{name}_lat"])
            np.save(path, v.astype(np.float16))
            print(f"{year} {name}: {v.shape[0]} cells, {np.isnan(v).any(1).mean():.1%} without data, "
                  f"{time.time() - t:.0f} s", flush=True)
            del mosaic


if __name__ == "__main__":
    args = sys.argv[1:]
    only = args[args.index("--block") + 1] if "--block" in args else None
    years = [int(a) for a in args if a.isdigit()]
    main(years, only)
