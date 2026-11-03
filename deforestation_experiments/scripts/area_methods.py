"""Notebook 01: how much forest and loss area each way of putting Hansen on a
grid gives, against Hansen's own 30 m pixels.

Ways of putting Hansen on a coarser grid, each cell then counting whole or in part:

  one value per cell  Earth Engine reads canopy and loss year at 100 m from its pyramids
                      (canopy: mean; loss year: most common value among the pixels that
                      have one), and the whole cell counts as forest and as lost. The
                      deforestation notebook's download: data/hansen_100m.tif.
  majority            a cell is forest, or lost in a year, if over half of its 30 m pixels are.
  per-cell share      each cell keeps the share of its 30 m pixels that are forest (canopy
                      above 30 %) and lost in each year; areas are share x cell area.

The 30 m truth is the stage-1 table (studies/output_datasets/hansen_loss_by_h3_res6_polygon_year.parquet)
and, in the test areas, Hansen's pixels summed on Earth Engine.
Earth Engine results are small tables, saved once as data/*.parquet.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]                  # deforestation_experiments/
sys.path.insert(0, str(HERE / "scripts"))
import defor_data as DD                                      # noqa: E402

DATA = HERE / "data" / "checks"
HANSEN = "UMD/hansen/global_forest_change_2025_v1_13"
NATIVE = [0.00025, 0, -180, 0, -0.00025, 80]                 # Hansen's 30 m grid
THRESHOLD = 30
YEARS = list(range(2013, 2025))

# Test areas of 0.5° x 0.5° (about 3,000 km²), one per kind of landscape
TEST_AREAS = {
    "Forest-savanna, north-west (loss hotspot)": (-13.0, 10.25, -12.5, 10.75),
    "Forest-savanna, east": (-9.5, 9.5, -9.0, 10.0),
    "Lowland forests, south-east": (-9.25, 7.75, -8.75, 8.25),
    "Montane forests (Fouta Djallon)": (-12.5, 11.0, -12.0, 11.5),
}


def grid_transform(cell_deg):
    """crsTransform of a grid with the origin of the deforestation notebook's download and cells of `cell_deg` degrees."""
    t = DD.grid()["transform"]
    return [cell_deg, 0, t.c, 0, -cell_deg, t.f]


CELL_DEG = {"100 m": DD.grid()["transform"].a / 2.5,
            "50 m": DD.grid()["transform"].a / 5}


# --------------------------------------------------------------------------
# Test areas, on Earth Engine

def _bands(ee):
    """Forest 2000 and loss of each year, as 0/1 bands on Hansen's own pixels (no-loss filled with 0)."""
    h = ee.Image(HANSEN)
    forest = h.select("treecover2000").gt(THRESHOLD)
    loss_year = h.select("lossyear").unmask(0)
    return ee.Image.cat([forest.rename("forest_2000")] +
                        [forest.And(loss_year.eq(y - 2000)).rename(str(y)) for y in YEARS])


def _sum(ee, image, region, transform):
    km2 = image.multiply(ee.Image.pixelArea()).divide(1e6)
    return km2.reduceRegion(ee.Reducer.sum(), region, crs="EPSG:4326", crsTransform=transform,
                            maxPixels=1e10, tileScale=4).getInfo()


def test_area_sums(rerun=False):
    """Forest 2000 and yearly loss (km²) in each test area: 30 m pixels, per-cell share and majority."""
    path = DATA / "test_area_sums.parquet"
    if path.exists() and not rerun:
        return pd.read_parquet(path)
    ee = DD.earth_engine()
    bands = _bands(ee)
    rows = []
    for area, (w, s, e, n) in TEST_AREAS.items():
        region = ee.Geometry.Rectangle([w, s, e, n], proj="EPSG:4326", geodesic=False)
        runs = {("30 m pixels", "30 m"): _sum(ee, bands, region, NATIVE)}
        for size, deg in CELL_DEG.items():
            transform = grid_transform(deg)
            share = bands.reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject(crs="EPSG:4326", crsTransform=transform)
            runs[("Per-cell share", size)] = _sum(ee, share, region, transform)
            runs[("Majority", size)] = _sum(ee, share.gt(0.5), region, transform)
        for (method, size), values in runs.items():
            rows.append({"area": area, "method": method, "cell": size, **{k: float(v or 0) for k, v in values.items()}})
        print("done:", area, flush=True)
    table = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


def asset_check(rerun=False):
    """Pyramiding policy of Hansen's canopy and loss-year bands, and the share of 30 m pixels that
    have a value in one test area (no-loss pixels have no loss year: they are masked)."""
    path = DATA / "hansen_asset_check.parquet"
    if path.exists() and not rerun:
        return pd.read_parquet(path)
    ee = DD.earth_engine()
    policy = {b["id"]: b.get("pyramidingPolicy") for b in ee.data.getAsset(HANSEN)["bands"]}
    h = ee.Image(HANSEN)
    w, s, e, n = TEST_AREAS["Forest-savanna, east"]
    region = ee.Geometry.Rectangle([w, s, e, n], proj="EPSG:4326", geodesic=False)
    valid = ee.Image.cat([h.select(b).mask().rename(b) for b in ("treecover2000", "lossyear")]).reduceRegion(
        ee.Reducer.mean(), region, crs="EPSG:4326", crsTransform=NATIVE, maxPixels=1e9).getInfo()
    table = pd.DataFrame([{"Band": b, "Pyramiding policy": policy[b], "Pixels with a value (%)": 100 * valid[b]}
                          for b in ("treecover2000", "lossyear")])
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return table


# --------------------------------------------------------------------------
# One value per cell, from the experiments' files

def cell_area_km2(profile):
    """Area of each cell of an EPSG:4326 grid (km²), as the experiments compute it."""
    t = profile["transform"]
    lat = t.f + (np.arange(profile["height"]) + 0.5) * t.e
    row = (abs(t.a) * 111_320.0 * np.cos(np.radians(lat))) * (abs(t.e) * 110_574.0) / 1e6
    return np.repeat(row[:, None], profile["width"], axis=1).astype("float32")


def local_grids():
    """The Hansen download on disk, one value per 100 m cell."""
    return {"100 m": ("hansen_100m", DD.fine_grid(DD.grid()))}


def single_value_test_area_sums():
    """Forest 2000 and yearly loss (km²) of one value per cell in each test area (cell centres inside)."""
    rows = []
    for size, (name, profile) in local_grids().items():
        bands, area = DD.read(name), cell_area_km2(profile)
        t = profile["transform"]
        lon = t.c + (np.arange(profile["width"]) + 0.5) * t.a
        lat = t.f + (np.arange(profile["height"]) + 0.5) * t.e
        for test, (w, s, e, n) in TEST_AREAS.items():
            box = np.ix_((lat > s) & (lat < n), (lon > w) & (lon < e))
            forest = bands["treecover2000"][box] > THRESHOLD
            year, a = bands["lossyear"][box], area[box]
            rows.append({"area": test, "method": "One value per cell", "cell": size, "forest_2000": float(a[forest].sum()),
                         **{str(y): float(a[forest & (year == y - 2000)].sum()) for y in YEARS}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# By hexagon: the stage-1 30 m table against one value per cell

TRUTH = DD.STUDIES / "output_datasets" / "hansen_loss_by_h3_res6_polygon_year.parquet"


def hexagons_inside():
    """H3 resolution-6 hexagons (about 36 km²) wholly inside Guinea for both boundaries: the stage-1
    table's land covers the whole hexagon, and the hexagon lies inside geoBoundaries' Guinea."""
    import geopandas as gpd
    import h3
    from shapely.geometry import Polygon, shape
    truth = pd.read_parquet(TRUTH)
    land = truth.groupby("h3_id").land_km2.sum()
    full = land[land >= 0.98 * pd.Series({h: h3.cell_area(h, "km^2") for h in land.index})].index
    guinea = shape(json.loads(DD.BOUNDARY.read_text())["features"][0]["geometry"])
    polygons = gpd.GeoDataFrame({"h3_id": list(full)}, crs=4326,
                                geometry=[Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(h)]) for h in full])
    return polygons[polygons.within(guinea)].reset_index(drop=True)


def hex_comparison(rerun=False):
    """Forest 2000 and yearly loss (km²) per hexagon: 30 m pixels, and one value per cell at 100 m."""
    path = DATA / "hex_comparison.parquet"
    if path.exists() and not rerun:
        return pd.read_parquet(path)
    from rasterio.features import rasterize
    hexes = hexagons_inside()
    truth = pd.read_parquet(TRUTH)
    truth = truth[truth.h3_id.isin(hexes.h3_id)]
    loss = truth[truth.year.between(YEARS[0], YEARS[-1])].pivot_table(
        index="h3_id", columns="year", values="forest2000_gt30_km2", aggfunc="sum")
    frames = [truth.groupby("h3_id").forest2000_gt30_km2.sum().rename("forest_2000").to_frame()
              .join(loss.rename(columns=str)).assign(source="30 m pixels")]
    for size, (name, profile) in local_grids().items():
        ids = rasterize(((g, i + 1) for i, g in enumerate(hexes.geometry)), out_shape=(profile["height"], profile["width"]),
                        transform=profile["transform"], fill=0, dtype="int32")
        bands, area = DD.read(name), cell_area_km2(profile)
        forest = bands["treecover2000"] > THRESHOLD
        n = len(hexes) + 1
        table = {"forest_2000": np.bincount(ids[forest], weights=area[forest], minlength=n)}
        for y in YEARS:
            lost = forest & (bands["lossyear"] == y - 2000)
            table[str(y)] = np.bincount(ids[lost], weights=area[lost], minlength=n)
        frame = pd.DataFrame(table).iloc[1:]
        frame.index = pd.Index(hexes.h3_id, name="h3_id")
        frames.append(frame.assign(source=f"One value per cell, {size}"))
    out = pd.concat(frames).fillna(0).reset_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out
