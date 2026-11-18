"""TESSERA embeddings at the GLanCE reference points (studies/dataset_checks, notebook 02).

For each of the 565 points, in its label's year as notebook 02 reads the products (End_Year,
at most 2020; TESSERA starts in 2017, so points labelled before 2017 get no embedding, as
Esri gets no map there):
  pixel   the 10 m pixel under the point
  cell    the mean of the 10 x 10 pixels of the platform's 100 m cell holding the point
          (as data_100m_tools/tessera_fetch.py builds the block cells)

Needs Python >= 3.12 (geotessera):
    "E:/SDG AI Lab/Guinea/national_scan_2026/venv_tessera/Scripts/python" landuse_experiments/grid_100m/scripts/tessera_glance_points.py
Output: STORE/tessera/glance_points.npz (pid, year, pixel, cell)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]                        # d:/Guinea_Methodology
STORE = Path(os.environ.get("GUINEA_DATA_100M", r"E:\SDG AI Lab\Guinea\data_100m"))
GLANCE = ROOT / "studies" / "dataset_checks" / "output" / "02_glance_points.parquet"


def main():
    from geotessera import GeoTesseraZarr
    g = pd.read_parquet(GLANCE)
    g["pid"] = np.arange(len(g))
    g["year"] = g["End_Year"].clip(upper=2020)
    g = g[g["year"] >= 2017].reset_index(drop=True)
    t = json.loads((STORE / "grid.json").read_text())["transform"]
    col = np.floor((g.Lon - t[2]) / t[0]); row = np.floor((g.Lat - t[5]) / t[4])
    clon, clat = t[2] + (col + 0.5) * t[0], t[5] + (row + 0.5) * t[4]
    gt = GeoTesseraZarr()
    pixel = np.full((len(g), 128), np.nan, np.float32); cell = pixel.copy()
    k = (np.arange(10) - 4.5) * 10
    for year in sorted(g.year.unique()):
        m = np.flatnonzero(g.year == year)
        pixel[m] = gt.sample_points(list(zip(g.Lon[m], g.Lat[m])), year=int(year), progress=False)
        pts = [(clon[i] + dx / (111320 * np.cos(np.radians(clat[i]))), clat[i] + dy / 110540)
               for i in m for dy in k for dx in k]
        E = np.asarray(gt.sample_points(pts, year=int(year), progress=False), np.float32).reshape(m.size, 100, 128)
        with np.errstate(all="ignore"):
            cell[m] = np.nanmean(E, 1)
        print(year, m.size, "points", flush=True)
    np.savez(STORE / "tessera" / "glance_points.npz", pid=g.pid.to_numpy(), year=g.year.to_numpy(),
             pixel=pixel, cell=cell)
    print("saved", (~np.isnan(pixel).any(1)).sum(), "of", len(g), "points with a pixel embedding")


if __name__ == "__main__":
    main()
