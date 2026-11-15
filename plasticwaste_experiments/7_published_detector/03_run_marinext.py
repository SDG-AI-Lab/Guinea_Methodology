"""
Run MariNeXt (Kikaki et al. 2024, trained on MADOS) over the same two Conakry dates.

MADOS has classes the earlier detectors never saw: sediment-laden, turbid and shallow
water, foam, waves. If the false detections on the tidal flats come from sediment, a model
that knows sediment should put those pixels in a sediment class rather than in debris.

Inputs follow the authors' evaluation code (github gkakogeorgiou/mados, MIT licence):
11 bands (B1-B8A, B11, B12; no B9), reflectance standardised by their band means and
deviations, 240-pixel windows, the five published runs. The five runs' class
probabilities are averaged. Windows overlap and only their centres are kept.

One difference: the authors used Acolite Rayleigh-corrected reflectance; these scenes are
Sentinel-2 L2A surface reflectance (the same scenes the other detector was given).

Uses the GPU environment plus MariNeXt's own copy of mmsegmentation and pure-Python mmcv
1.6.0, kept outside it (E:\\SDG AI Lab\\Guinea\\marinext_test):

    "E:\\SDG AI Lab\\Guinea\\national_scan_2026\\venv_gpu\\Scripts\\python" 7_published_detector/03_run_marinext.py
"""
import glob
import os
import sys
from pathlib import Path

os.environ.pop("PROJ_LIB", None)  # PostGIS points these at its own copy, which rasterio cannot read
os.environ.pop("GDAL_DATA", None)

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window

MX = Path(r"E:\SDG AI Lab\Guinea\marinext_test")
sys.path[:0] = [str(MX / "libs"), str(MX / "repo" / "marinext")]
from marinext_wrapper import MariNext  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
DATES = ("2024-01-11", "2024-01-26")
BANDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]  # of the 12 stored bands: drop B9
MEAN = np.array([0.0582676, 0.05223386, 0.04381474, 0.0357083, 0.03412902, 0.03680401,
                 0.03999107, 0.03566642, 0.03965081, 0.0267993, 0.01978944], "float32")
STD = np.array([0.03240627, 0.03432253, 0.0354812, 0.0375769, 0.03785412, 0.04992323,
                0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115], "float32")
SIZE, KEEP = 240, 192
MARGIN = (SIZE - KEEP) // 2
DEBRIS = 0  # 'Marine Debris' is the first of the 15 MADOS classes


def models(dev):
    out = []
    for f in sorted(glob.glob(str(MX / "weights" / "*" / "*.pth"))):
        m = MariNext(11, 15)
        ck = torch.load(f, map_location="cpu", weights_only=True)
        ck = {k.replace("decoder", "decode_head"): v for k, v in ck.items() if "proj1" not in k and "proj2" not in k}
        m.load_state_dict(ck, strict=True)
        out.append(m.eval().to(dev))
    if len(out) != 5:
        raise SystemExit(f"expected 5 MariNeXt runs, found {len(out)}")
    return out


def predict(nets, path, target, dev):
    with rasterio.open(path) as src:
        prof = {**src.profile, "count": 2, "dtype": "uint8", "nodata": 255}
        with rasterio.open(target, "w", **prof) as dst:
            rows, cols = range(0, src.height, KEEP), range(0, src.width, KEEP)
            for k, (r, c) in enumerate((r, c) for r in rows for c in cols):
                top, left = max(r - MARGIN, 0), max(c - MARGIN, 0)
                bottom, right = min(r + KEEP + MARGIN, src.height), min(c + KEEP + MARGIN, src.width)
                a = src.read(window=Window(left, top, right - left, bottom - top)).astype("float32")[BANDS] * 1e-4
                empty = (a == 0).all(0)
                a[:, empty] = MEAN[:, None]  # the authors fill missing pixels with the band means
                pad = (MARGIN - (r - top), SIZE - (bottom - top) - (MARGIN - (r - top)))
                side = (MARGIN - (c - left), SIZE - (right - left) - (MARGIN - (c - left)))
                a = np.pad(a, ((0, 0), pad, side), mode="reflect")
                x = torch.from_numpy((a - MEAN[:, None, None]) / STD[:, None, None])[None].to(dev)
                with torch.no_grad():
                    p = sum(F.softmax(F.interpolate(n(x), size=(SIZE, SIZE), mode="bilinear"), 1) for n in nets) / len(nets)
                p = p[0].cpu().numpy()
                if not np.isfinite(p).all():
                    raise SystemExit(f"non-finite output in window {k} of {path.name}")
                h, w = min(KEEP, src.height - r), min(KEEP, src.width - c)
                cls = p.argmax(0)[MARGIN:MARGIN + h, MARGIN:MARGIN + w].astype("uint8")
                prob = (p[DEBRIS, MARGIN:MARGIN + h, MARGIN:MARGIN + w] * 254).astype("uint8")
                dst.write(np.stack([cls, prob]), window=Window(c, r, w, h))
                if k % 100 == 0:
                    print(path.name, k, "of", len(rows) * len(cols), flush=True)


def main():
    """Default input: the L2A scenes. With 'rhorc': Acolite reflectance from 06, as MADOS was made."""
    global BANDS
    rhorc = len(sys.argv) > 1 and sys.argv[1] == "rhorc"
    if rhorc:
        BANDS = list(range(11))  # 06 already writes the 11 MariNeXt bands
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = models(dev)
    for date in DATES:
        src = OUT / (f"conakry_{date}_rhorc.tif" if rhorc else f"conakry_{date}.tif")
        dst = OUT / f"conakry_{date}_marinext{'_rhorc' if rhorc else ''}.tif"
        predict(nets, src, dst, dev)
        print("wrote", date, flush=True)


if __name__ == "__main__":
    main()
