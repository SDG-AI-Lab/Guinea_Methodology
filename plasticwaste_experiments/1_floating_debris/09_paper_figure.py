"""Figure for the short paper: published patches and the tidal flats under them."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, RASTERS  # noqa: E402

from plasticwaste_tool.geo_env import fix_proj  # noqa: E402
fix_proj()
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

import importlib  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
L = importlib.import_module("03_look_at_clusters")

OUT = ROOT / "1_floating_debris" / "outputs"
HALF = 0.012


def main():
    src = rasterio.open(L.RASTERS / "conakry_a410b95f_2024-01-01_2024-02-01_composite.tif")
    res = {k: L.masks(v) for k, v in L.PRODUCTS.items()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    fig, ax = plt.subplots(1, 4, figsize=(7.1, 1.95))

    def s2(a, lon, lat, product, title, mark=True):
        w = from_bounds(lon - HALF, lat - HALF, lon + HALF, lat + HALF, src.transform).round_offsets().round_lengths()
        y0, x0, h, wd = int(w.row_off), int(w.col_off), int(w.height), int(w.width)
        bands, kept = res[product]
        s = np.clip(np.dstack([bands[b][y0:y0 + h, x0:x0 + wd] for b in ("B8", "B4", "B3")]) / 0.25, 0, 1)
        a.imshow(np.nan_to_num(s), interpolation="nearest")
        if mark:
            ky, kx = np.nonzero(kept[y0:y0 + h, x0:x0 + wd])
            a.scatter(kx, ky, s=2.2, c="#ffe600", linewidths=0)
        a.set_title(title, fontsize=7.5, loc="left", pad=2)
        a.axis("off")

    a_lon, a_lat = L.CL["A north of the peninsula"]
    c_lon, c_lat = L.CL["C Dubreka side"]
    s2(ax[0], a_lon, a_lat, "Jan median", "(a) Patches, January median")
    ax[1].imshow(L.esri(a_lon, a_lat, HALF))
    ax[1].set_title("(b) Same place, Esri imagery", fontsize=7.5, loc="left", pad=2)
    ax[1].axis("off")
    s2(ax[2], c_lon, c_lat, "11 Jan", "(c) Flat exposed, 11 January", mark=False)
    s2(ax[3], c_lon, c_lat, "26 Jan", "(d) Detections, 26 January")
    fig.subplots_adjust(left=0.002, right=0.998, top=0.9, bottom=0.005, wspace=0.03)
    fig.savefig(OUT / "09_figure_tidal_flats.png", dpi=300)


if __name__ == "__main__":
    main()
