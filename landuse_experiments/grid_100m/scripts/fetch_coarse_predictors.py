"""The coarse yearly predictors for a year the platform did not fetch (2020, for notebook 05's fresh
window), made as main's config.PREDICTOR_RASTERS_CONFIG makes them (asset, band, annual reducer) on the
grid of the platform's 2021 file. Re-making 2021 first checks the recipe against the platform's file.
Usage: python fetch_coarse_predictors.py 2021 2020   (writes only years whose file is missing)"""

import sys
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[1] / "data_100m_tools"))
import grid100 as G                                                 # noqa: E402

SRC = HERE.parent / "data" / "esri_predictors_src"
RECIPE = {"PRECIPITATION": ("UCSB-CHG/CHIRPS/DAILY", "precipitation", "sum"),
          "SOIL_MOISTURE": ("ECMWF/ERA5_LAND/MONTHLY_AGGR", "volumetric_soil_water_layer_1", "mean"),
          "NDVI": ("MODIS/061/MOD13Q1", "NDVI", "mean")}
BLOCK = 600                                                         # rows per request (computePixels size limit)


def make(name, year):
    ee = G.earth_engine()
    asset, band, reducer = RECIPE[name]
    images = ee.ImageCollection(asset).filterDate(f"{year}-01-01", f"{year + 1}-01-01").select(band)
    image = getattr(images, reducer)().toFloat()
    with rasterio.open(SRC / f"{name}_2021.tif") as ref:
        t, (h, w), profile = ref.transform, ref.shape, ref.profile
    rows = []
    for r0 in range(0, h, BLOCK):
        n = min(BLOCK, h - r0)
        grid = {"dimensions": {"width": w, "height": n}, "crsCode": "EPSG:4326",
                "affineTransform": {"scaleX": t.a, "shearX": 0, "translateX": t.c,
                                    "shearY": 0, "scaleY": t.e, "translateY": t.f + r0 * t.e}}
        px = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": grid})
        rows.append(px[band].astype("float64"))
    values = np.vstack(rows)
    return np.where(np.isfinite(values), values, 0.0), profile      # the platform's files hold 0 where no data


def check(name):
    """Largest difference between the re-made 2021 layer and the platform's file."""
    made, _ = make(name, 2021)
    with rasterio.open(SRC / f"{name}_2021.tif") as ref:
        platform = ref.read(1)
    diff = np.abs(made - platform)
    rel = diff.max() / np.abs(platform).max()
    print(f"{name} 2021: {(diff <= 1e-3 * np.abs(platform).max()).mean():.1%} of pixels equal, "
          f"largest difference {diff.max():.4g} ({rel:.2%} of the largest value)", flush=True)
    return rel < 0.01


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]]
    for name in RECIPE:
        if 2021 in years and not check(name):
            sys.exit(f"{name}: the recipe does not reproduce the platform's 2021 file; nothing written")
        for year in (y for y in years if y != 2021):
            path = SRC / f"{name}_{year}.tif"
            if not path.exists():
                values, profile = make(name, year)
                with rasterio.open(path, "w", **profile) as dst:
                    dst.write(values, 1)
                print(f"wrote {path.name}: mean {np.nanmean(values):.4g}", flush=True)
