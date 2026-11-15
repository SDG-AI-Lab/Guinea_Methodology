"""
Does the published detector stay off Guinea's tidal flats?

Reads the detector's probability maps for the two clear dates of January 2024 and
compares them with the tide mask, at the decision threshold stored with the published
weights. The model is built for water, so land is counted apart and left out of the
share on flats. For each date it reports the debris pixels on water and flats, the share
of those on flats, and the same share for the whole box, because a detector with no
skill would land there at that rate. It also reports how much the two dates overlap.
The forest rejected earlier had over 90 % on flats and a swing of 29 times between
these dates.

Staying off the flats says it avoids the sediment problem. It cannot show that what
remains is debris, because no confirmed debris exists for these waters.
"""
import os
import sys
from pathlib import Path

os.environ.pop("PROJ_LIB", None)  # PostGIS points these at its own copy, which rasterio cannot read
os.environ.pop("GDAL_DATA", None)

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT  # noqa: E402

from scipy import ndimage as ndi  # noqa: E402

OUT = ROOT / "7_published_detector" / "outputs"
MASK = ROOT / "1_floating_debris" / "outputs" / "07_conakry_tide_mask.tif"
DATES = ("2024-01-11", "2024-01-26")
THRESHOLD = 0.0639  # unetpp seed 1: model.threshold, stored with the weights


def main():
    with rasterio.open(MASK) as s:
        tidal, occurrence = s.read(1) > 0, s.read(2)
        grid = {"crs": s.crs, "transform": s.transform, "shape": (s.height, s.width)}
    land = ~tidal & (occurrence == 0)
    flat = ~land & (tidal | (occurrence < 90))
    maps, rows = [], []
    for date in DATES:
        with rasterio.open(OUT / f"conakry_{date}_prediction.tif") as src:
            on_grid = np.zeros(grid["shape"], "float32")
            reproject(src.read(1).astype("float32"), on_grid, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=grid["transform"], dst_crs=grid["crs"], resampling=Resampling.max)
        if on_grid.max() > 1.5:  # stored as 0-255
            on_grid = on_grid / 255.0
        on_land = int((on_grid[land] >= THRESHOLD).sum())
        debris = (on_grid >= THRESHOLD) & ~land
        maps.append(debris)
        _, patches = ndi.label(debris, structure=np.ones((3, 3), bool))
        rows.append({"date": date, "pixels_20m": int(debris.sum()), "patches": patches, "pixels_on_land": on_land,
                     "share_on_flats": float(flat[debris].mean()) if debris.any() else None,
                     "box_share_flats": float(flat[~land].mean())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "02_scores.csv", index=False)
    print(t.round(3).to_string())
    low, high = t.pixels_20m.min(), t.pixels_20m.max()
    print("larger date over smaller:", round(high / low, 1) if low else "no pixels on one date")
    print("overlap of the dates:", round(float((maps[0] & maps[1]).sum() / (maps[0] | maps[1]).sum()), 3))


if __name__ == "__main__":
    main()
