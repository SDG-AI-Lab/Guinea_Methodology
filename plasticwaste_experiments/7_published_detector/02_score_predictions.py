"""
Does the published detector stay off Guinea's tidal flats?

Reads the detector's probability maps for the two clear dates of January 2024 and
compares them with the tide mask. The rule, fixed before running: it is worth a place
beside the current method if, at probability 0.5,

  1. fewer than half its debris pixels lie on flats or water under 90 %,
  2. the two dates agree on the amount: the larger count is under 3 times the smaller.

The forest rejected earlier failed on both: over 90 % on flats, and a swing of 29 times
between these dates. A pass says it avoids the sediment problem. It cannot show that
what remains is debris, because no confirmed debris exists for these waters.
"""
import json
import sys
from pathlib import Path

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
THRESHOLD = 0.5


def main():
    with rasterio.open(MASK) as s:
        flat = (s.read(1) > 0) | (s.read(2) < 90)
        grid = {"crs": s.crs, "transform": s.transform, "shape": (s.height, s.width)}
    rows = []
    for date in DATES:
        with rasterio.open(OUT / f"conakry_{date}_prediction.tif") as src:
            on_grid = np.zeros(grid["shape"], "float32")
            reproject(src.read(1).astype("float32"), on_grid, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=grid["transform"], dst_crs=grid["crs"], resampling=Resampling.max)
        if on_grid.max() > 1.5:  # stored as 0-255
            on_grid = on_grid / 255.0
        debris = on_grid >= THRESHOLD
        _, patches = ndi.label(debris, structure=np.ones((3, 3), bool))
        rows.append({"date": date, "pixels_20m": int(debris.sum()), "patches": patches,
                     "share_on_flats": float(flat[debris].mean()) if debris.any() else None})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "02_scores.csv", index=False)
    print(t.round(3).to_string())
    low, high = t.pixels_20m.min(), t.pixels_20m.max()
    rule = {"under_half_on_flats": bool((t.share_on_flats.fillna(0) < .5).all()),
            "dates_agree": bool(low > 0 and high / low < 3)}
    rule["worth_adding"] = all(rule.values())
    (OUT / "02_rule.json").write_text(json.dumps(rule, indent=1), encoding="utf-8")
    print(json.dumps(rule, indent=1))


if __name__ == "__main__":
    main()
