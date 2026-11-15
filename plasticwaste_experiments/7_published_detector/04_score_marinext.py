"""
Does MariNeXt stay off Guinea's tidal flats, and what does it call them?

Same checks as 02 for the other published detector, on the same two dates and tide mask:
debris pixels on water and flats, the share of those on flats against the share of the
box that is flats (what a detector with no skill would give), pixels on land, the swing
between the dates and their overlap. Debris is the class MariNeXt gives the highest
probability, as in the authors' evaluation, so no threshold is chosen here.

MADOS adds one check the other detector cannot give: the class MariNeXt assigns to the
tidal flats. Sediment-laden, turbid or shallow water there would mean it recognises the
material that the other methods mistake for debris.

Staying off the flats cannot show that what remains is debris: no confirmed debris exists
for these waters.
"""
import json
import os
import sys
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage as ndi

OUT = Path(__file__).resolve().parent / "outputs"
MASK = Path(__file__).resolve().parents[1] / "1_floating_debris" / "outputs" / "07_conakry_tide_mask.tif"
DATES = ("2024-01-11", "2024-01-26")
LABELS = ["Marine Debris", "Dense Sargassum", "Sparse Floating Algae", "Natural Organic Material", "Ship",
          "Oil Spill", "Marine Water", "Sediment-Laden Water", "Foam", "Turbid Water", "Shallow Water",
          "Waves & Wakes", "Oil Platform", "Jellyfish", "Sea snot"]


def main():
    tag = "_rhorc" if len(sys.argv) > 1 and sys.argv[1] == "rhorc" else ""  # which run of 03 to score
    with rasterio.open(MASK) as s:
        tidal, occurrence = s.read(1) > 0, s.read(2)
        grid = {"crs": s.crs, "transform": s.transform, "shape": (s.height, s.width)}
    land = ~tidal & (occurrence == 0)
    flat = ~land & (tidal | (occurrence < 90))
    maps, rows, flat_classes = [], [], {}
    for date in DATES:
        with rasterio.open(OUT / f"conakry_{date}_marinext{tag}.tif") as src:
            cls = np.full(grid["shape"], 255, "uint8")
            # mode is the nearest 20 m equivalent of 'any debris in the cell' for a class map:
            # debris is rare, so it is counted first with max on a 0/1 layer, and the class map with mode
            deb10 = (src.read(1) == 0).astype("uint8")
            deb = np.zeros(grid["shape"], "uint8")
            reproject(deb10, deb, src_transform=src.transform, src_crs=src.crs, dst_transform=grid["transform"],
                      dst_crs=grid["crs"], resampling=Resampling.max)
            reproject(src.read(1), cls, src_transform=src.transform, src_crs=src.crs, dst_transform=grid["transform"],
                      dst_crs=grid["crs"], resampling=Resampling.mode, src_nodata=255, dst_nodata=255)
        on_land = int((deb.astype(bool) & land).sum())
        debris = deb.astype(bool) & ~land
        maps.append(debris)
        _, patches = ndi.label(debris, structure=np.ones((3, 3), bool))
        rows.append({"date": date, "pixels_20m": int(debris.sum()), "patches": patches, "pixels_on_land": on_land,
                     "share_on_flats": float(flat[debris].mean()) if debris.any() else None,
                     "box_share_flats": float(flat[~land].mean())})
        v = cls[flat & (cls != 255)]
        counts = np.bincount(v, minlength=15)[:15] / max(len(v), 1)
        flat_classes[date] = {LABELS[i]: round(float(x), 3) for i, x in enumerate(counts) if x >= 0.005}
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"04_scores_marinext{tag}.csv", index=False)
    print(t.round(3).to_string())
    low, high = t.pixels_20m.min(), t.pixels_20m.max()
    swing = round(high / low, 1) if low else None
    overlap = float((maps[0] & maps[1]).sum() / max((maps[0] | maps[1]).sum(), 1))
    print("larger date over smaller:", swing if swing else "no pixels on one date")
    print("overlap of the dates:", round(overlap, 3))
    print("classes on the tidal flats:", json.dumps(flat_classes, indent=1))
    (OUT / f"04_marinext_summary{tag}.json").write_text(json.dumps(
        {"swing": swing, "overlap": round(overlap, 3), "flat_classes": flat_classes}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
