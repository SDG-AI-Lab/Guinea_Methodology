"""Chips of the three clusters holding the 26 January patches, and of the 26 January
single-date detections: Sentinel-2 (near-infrared, red, green) on each product, and
Esri imagery for the same ground. For looking, not for scoring."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, RASTERS  # noqa: E402

from plasticwaste_tool.geo_env import fix_proj  # noqa: E402
fix_proj()
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

from plasticwaste_tool import data_loader as DL  # noqa: E402
from plasticwaste_tool.fdi_tool import debris_filter as DF  # noqa: E402
from plasticwaste_tool.fdi_tool import detection as D  # noqa: E402
from plasticwaste_tool.fdi_tool import spectral as SP  # noqa: E402

OUT = ROOT / "1_floating_debris" / "outputs"
CL = {"A north of the peninsula": (-13.6315, 9.640), "B south-east shore": (-13.5825, 9.5435),
      "C Dubreka side": (-13.430, 9.408)}
HALF = 0.012
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
PRODUCTS = {"Jan median": "2024-01-01_2024-02-01", "11 Jan": "2024-01-11_2024-01-12", "26 Jan": "2024-01-26_2024-01-27"}


def masks(tag):
    bands, _ = DL.read_composite(str(RASTERS / f"conakry_a410b95f_{tag}_composite.tif"))
    fdi, ndvi = SP.compute_fdi(bands), SP.compute_ndvi(bands)
    rule, _, _ = D.debris_binary_mask_de_africa(fdi, ndvi, bands["WATER"])
    kept, _, _ = DF.apply(rule, bands)
    return bands, kept


def esri(lon, lat, half):
    r = requests.get(ESRI, params={"bbox": f"{lon - half},{lat - half},{lon + half},{lat + half}",
                                   "bboxSR": 4326, "imageSR": 4326, "size": "520,520",
                                   "format": "png", "f": "image"}, timeout=60)
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))


def main():
    src = rasterio.open(RASTERS / "conakry_a410b95f_2024-01-01_2024-02-01_composite.tif")
    res = {k: masks(v) for k, v in PRODUCTS.items()}
    ys, xs = np.nonzero(res["26 Jan"][1])
    lons, lats = rasterio.transform.xy(src.transform, ys, xs)
    lons, lats = np.array(lons), np.array(lats)
    d = pd.DataFrame({"lon": lons.round(3), "lat": lats.round(3)}).value_counts().head(8)
    print("26 Jan kept, densest 0.001-degree cells:\n", d)
    print("26 Jan kept bounds", lons.min(), lons.max(), lats.min(), lats.max(), np.median(lons), np.median(lats))
    places = dict(CL)
    places["D 26 Jan detections"] = (float(np.median(lons)), float(np.median(lats)))
    fig, ax = plt.subplots(len(places), 4, figsize=(13, 3.2 * len(places)))
    for r, (name, (lon, lat)) in enumerate(places.items()):
        w = from_bounds(lon - HALF, lat - HALF, lon + HALF, lat + HALF, src.transform)
        win = w.round_offsets().round_lengths()
        y0, x0, h, wd = int(win.row_off), int(win.col_off), int(win.height), int(win.width)
        for c, (p, (bands, kept)) in enumerate(res.items()):
            s = np.dstack([bands[b][y0:y0 + h, x0:x0 + wd] for b in ("B8", "B4", "B3")])
            s = np.clip(s / 0.25, 0, 1)
            ax[r, c].imshow(np.nan_to_num(s), interpolation="nearest")
            ky, kx = np.nonzero(kept[y0:y0 + h, x0:x0 + wd])
            ax[r, c].scatter(kx, ky, s=5, facecolors="none", edgecolors="yellow", linewidths=0.6)
            ax[r, c].set_title(f"{name[:1]} {p}: {len(kx)} px", fontsize=9)
            ax[r, c].axis("off")
        ax[r, 3].imshow(esri(lon, lat, HALF))
        ax[r, 3].set_title(f"Esri imagery ({lon:.3f}, {lat:.3f})", fontsize=9)
        ax[r, 3].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "03_clusters.png", dpi=110)


if __name__ == "__main__":
    main()
