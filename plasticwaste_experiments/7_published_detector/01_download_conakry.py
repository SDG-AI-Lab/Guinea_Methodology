"""
Download the 12 Sentinel-2 bands the published detector needs, for Conakry.

The cached composites hold fewer bands, so this fetches the two clear dates of January
2024 from Earth Engine in small pieces and joins them. About 600 MB a date, on the
same box and projection as the tide mask.
"""
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import PROJECT, ROOT  # noqa: E402

import ee  # noqa: E402

OUT = ROOT / "7_published_detector" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MASK = ROOT / "1_floating_debris" / "outputs" / "07_conakry_tide_mask.tif"
DATES = ("2024-01-11", "2024-01-26")
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
PIECES = 6  # each side, so each piece stays under Earth Engine's download limit


def project():
    for line in (PROJECT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("GEE_PROJECT_ID="):
            return line.split("=", 1)[1].strip().strip('"')


def main():
    ee.Initialize(project=project())
    with rasterio.open(MASK) as s:
        west, south, east, north = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
        crs = s.crs.to_string()
    lons, lats = np.linspace(west, east, PIECES + 1), np.linspace(south, north, PIECES + 1)
    for date in DATES:
        final = OUT / f"conakry_{date}.tif"
        if final.exists():
            print("have", final.name)
            continue
        day = ee.Date(date)
        box = ee.Geometry.Rectangle([west, south, east, north], None, False)
        image = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(box)
                 .filterDate(day, day.advance(1, "day")).mosaic().select(BANDS).toUint16())
        folder = OUT / f"pieces_{date}"
        folder.mkdir(exist_ok=True)
        files = []
        for i in range(PIECES):
            for j in range(PIECES):
                path = folder / f"{i}_{j}.tif"
                if not path.exists():
                    region = ee.Geometry.Rectangle([lons[i], lats[j], lons[i + 1], lats[j + 1]], None, False)
                    url = image.getDownloadURL({"region": region, "scale": 10, "crs": crs, "format": "GEO_TIFF"})
                    urllib.request.urlretrieve(url, path)
                    print(date, path.name, flush=True)
                files.append(path)
        sources = [rasterio.open(f) for f in files]
        mosaic, transform = merge(sources)
        profile = {**sources[0].profile, "height": mosaic.shape[1], "width": mosaic.shape[2],
                   "transform": transform, "compress": "deflate"}
        for s in sources:
            s.close()
        with rasterio.open(final, "w", **profile) as dst:
            dst.write(mosaic)
        shutil.rmtree(folder)
        print("wrote", final.name, mosaic.shape)


if __name__ == "__main__":
    main()
