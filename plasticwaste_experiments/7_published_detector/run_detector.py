"""
Run the published detector over the Conakry images. Uses the detector's own environment.

Its command line tool pads the edge of each window with zeros. At the model's own
threshold that leaves a grid of false detections along the window edges. So this runs
the same weights on overlapping windows and keeps only the centre of each one.

    D:\\guinea_data\\mdd_env\\Scripts\\python 7_published_detector/run_detector.py
"""
import os
from pathlib import Path

os.environ.pop("PROJ_LIB", None)  # PostGIS points these at its own copy, which rasterio cannot read
os.environ.pop("GDAL_DATA", None)

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

OUT = Path(__file__).resolve().parent / "outputs"
DATES = ("2024-01-11", "2024-01-26")
SIZE, KEEP = 608, 480  # the window the model sees, and its centre that is kept
MARGIN = (SIZE - KEEP) // 2


def predict(model, path, target):
    with rasterio.open(path) as src, rasterio.open(target, "w", **{**src.profile, "count": 1, "dtype": "uint8"}) as dst:
        rows = range(0, src.height, KEEP)
        cols = range(0, src.width, KEEP)
        for k, (r, c) in enumerate((r, c) for r in rows for c in cols):
            top, left = max(r - MARGIN, 0), max(c - MARGIN, 0)
            bottom, right = min(r + KEEP + MARGIN, src.height), min(c + KEEP + MARGIN, src.width)
            a = src.read(window=Window(left, top, right - left, bottom - top)).astype("float32") * 1e-4
            pad = (MARGIN - (r - top), SIZE - (bottom - top) - (MARGIN - (r - top)))
            side = (MARGIN - (c - left), SIZE - (right - left) - (MARGIN - (c - left)))
            a = np.pad(a, ((0, 0), pad, side), mode="reflect")
            with torch.no_grad():
                p = torch.sigmoid(model(torch.from_numpy(a[None]))).squeeze().numpy()
            h, w = min(KEEP, src.height - r), min(KEEP, src.width - c)
            dst.write((p[MARGIN:MARGIN + h, MARGIN:MARGIN + w] * 255).astype("uint8"), 1, window=Window(c, r, w, h))
            if k % 20 == 0:
                print(path.name, k, "of", len(rows) * len(cols), flush=True)


def main():
    model = torch.hub.load("marccoru/marinedebrisdetector", "unetpp", 1, trust_repo=True).eval()
    print("threshold stored with the weights", float(model.threshold))
    for date in DATES:
        predict(model, OUT / f"conakry_{date}.tif", OUT / f"conakry_{date}_prediction.tif")


if __name__ == "__main__":
    main()
