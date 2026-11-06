"""
Flag the Conakry patches instead of deleting them.

Removing tidal flats outright leaves almost no pixels, so this keeps every patch and
gives it a label:

  on tidal flat    at least half its pixels on mapped flats or water under 90 %
  does not move    still above the FDI threshold on both clear dates of the month,
                   or back within a pixel in another month
  both, or no flag

Patches with no flag are the ones worth a person's look. The January patches on the
map are part of this test. Imagery is the cached Conakry set used in `1_floating_debris`.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1_floating_debris"))
from common import ROOT, RASTERS  # noqa: E402

P = importlib.import_module("02_conakry_persistence")  # its detector and products
from scipy import ndimage as ndi  # noqa: E402

OUT = ROOT / "5_floating_debris_flags" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MASK = ROOT / "1_floating_debris" / "outputs" / "07_conakry_tide_mask.tif"
EIGHT = np.ones((3, 3), bool)
MEDIANS = {"2024-01": "2024-01 median", "2023-11": "2023-11 median"}
OTHER = {"2024-01": "2023-11", "2023-11": "2024-01"}
DATES = ("2024-01-11 single", "2024-01-26 single")  # the clear dates of January


def load(name):
    path, names = P.PRODUCTS[name]
    bands, profile = P.DL.read_composite(str(path), band_names=names) if names else P.DL.read_composite(str(path))
    return bands, profile


def above_threshold(name):
    bands, _ = load(name)
    fdi = P.SP.compute_fdi(bands)
    water = np.nan_to_num(bands["WATER"], nan=0.0) > 0
    return fdi > float(np.nanmean(fdi[water]) + 3 * np.nanstd(fdi[water]))


def main():
    with rasterio.open(MASK) as s:
        flat = (s.read(1) > 0) | (s.read(2) < 90)
    kept, transforms = {}, {}
    for month, name in MEDIANS.items():
        bands, profile = load(name)
        kept[month] = P.detect(bands)["kept"]
        transforms[month] = profile["transform"]
    high = [above_threshold(n) for n in DATES]

    rows = []
    for month, mask in kept.items():
        labels, n = ndi.label(mask, structure=EIGHT)
        near_other = ndi.binary_dilation(kept[OTHER[month]], structure=EIGHT)
        for i, part in enumerate(ndi.find_objects(labels), start=1):
            px = labels[part] == i
            r, c = np.argwhere(px).mean(axis=0) + (part[0].start, part[1].start)
            x, y = transforms[month] * (c + .5, r + .5)
            tide = flat[part][px].mean() >= .5
            recurs = bool(near_other[part][px].any())
            both_dates = month == "2024-01" and all(h[part][px].any() for h in high)
            fixed = recurs or both_dates
            rows.append({"month": month, "patch": i, "pixels": int(px.sum()), "x": x, "y": y,
                         "share_on_flat": float(flat[part][px].mean()), "on_tidal_flat": bool(tide),
                         "recurs_other_month": recurs, "on_both_clear_dates": bool(both_dates),
                         "does_not_move": bool(fixed),
                         "flag": "both" if tide and fixed else "on tidal flat" if tide
                         else "does not move" if fixed else "no flag"})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "01_patch_flags.csv", index=False)
    s = t.groupby(["month", "flag"]).agg(patches=("patch", "size"), pixels=("pixels", "sum")).reset_index()
    s.to_csv(OUT / "01_flag_summary.csv", index=False)
    print(s.to_string())
    print("x and y are in the image's coordinate system, UTM or lon/lat as cached. Patches with no flag:")
    print(t[t.flag == "no flag"][["month", "patch", "pixels", "x", "y"]].to_string())


if __name__ == "__main__":
    main()
