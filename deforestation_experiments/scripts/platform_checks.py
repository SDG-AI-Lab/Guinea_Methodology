"""Reference figures from the platform, to check the 100 m layers against (read only).

  platform_composition()  the platform's published land-cover areas per year (public GET of
                          /api/landuse/composition/), saved as data/platform_composition.parquet
  bbox_class_areas(year)  the area of each Esri class as the platform reads it (mode of the
                          year's tiles at 100 m), summed over the grid's whole rectangle, on Earth
                          Engine: what the composition would be if it were not cut to Guinea
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]                  # deforestation_experiments/
DATA = HERE / "data" / "checks"
sys.path.insert(0, str(HERE.parent / "data_100m_tools"))
import grid100 as G                                           # noqa: E402
import fetch as F                                             # noqa: E402

PLATFORM = "http://172.205.219.243"


def platform_composition(rerun=False):
    path = DATA / "platform_composition.parquet"
    if path.exists() and not rerun:
        return pd.read_parquet(path)
    import requests
    payload = requests.get(f"{PLATFORM}/api/landuse/composition/?from=2017&to=2025&lang=en", timeout=60).json()
    rows = [{"year": r["year"], "code": c["code"], "class": c["label"]["text"], "area_ha": c["area_ha"]}
            for r in payload["rows"] for c in r["classes"]]
    table = pd.DataFrame(rows).assign(data_date=payload.get("data_date"))
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


def bbox_class_areas(year, rerun=False):
    path = DATA / f"bbox_class_areas_{year}.parquet"
    if path.exists() and not rerun:
        return pd.read_parquet(path)
    ee = G.earth_engine()
    g = G.grid()
    t = g["transform"]
    west, north = t[2], t[5]
    east, south = west + g["width"] * t[0], north + g["height"] * t[4]
    roi = ee.Geometry(json.loads(G.BOUNDARY.read_text())["features"][0]["geometry"])
    classes = (ee.ImageCollection(F.ESRI).filterBounds(roi).filterDate(f"{year}-01-01", f"{year}-12-31")
               .select("b1").mode().unmask(0))
    box = ee.Geometry.Rectangle([west, south, east, north], proj="EPSG:4326", geodesic=False)
    result = ee.Image.pixelArea().divide(1e4).addBands(classes).reduceRegion(
        ee.Reducer.sum().group(1, "code"), box, crs="EPSG:4326", crsTransform=t, maxPixels=1e10, tileScale=4).getInfo()
    table = pd.DataFrame([{"code": int(r["code"]), "area_ha": r["sum"]} for r in result["groups"]]).assign(year=year)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table
