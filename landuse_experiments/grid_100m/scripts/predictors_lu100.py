"""The land-use predictors on the platform's 100 m grid, as the platform's main branch builds them
(config.LAND_USE_PREDICTOR_RASTERS / _VECTORS, data_loader.assemble_predictor_stack, commit
6710b3d), from files on this laptop. No Earth Engine.

  Name (main's file name)  Source here                                   Year used for `year`
  DEM_GEE                  SRTM at 100 m (store: dem)                    static
  POPULATION_DENSITY       WorldPop at 100 m (store: worldpop)           min(year, 2020)
  PRECIPITATION            CHIRPS annual total, native 0.05°             min(year, 2025)
  SOIL_MOISTURE            ERA5-Land annual mean, native 0.1°            min(year, 2025)
  AGB_NASA                 NASA/ORNL biomass 2010                        2010
  NDVI                     MOD13Q1 annual mean, 250 m                    min(year, 2024)
  SLOPE, ASPECT            from DEM_GEE with the platform's functions    static
  ROADS_distance, WATERWAYS_distance
                           OpenStreetMap cells (store: osm, the platform's tags, all_touched)
                           -> distance in cells (process_vector_layer)  static
  POI_heatmap              OpenStreetMap amenities -> exp(-d / (0.1 max d))  static
  PROTECTED_AREAS_zone     WDPA (store: wdpa_year): 1 inside, 0 outside  current polygons

The year rule is data_loader._snapshot_years (the year, capped by each source's year_hint). The
coarse rasters are the 250 m experiments' inputs (landuse_experiments/data/esri_predictors_src),
the files the platform fetched; they are read onto the 100 m grid by nearest neighbour, as Earth
Engine reads a coarser image at a finer scale (Google Earth Engine, Scale guide). Terrain,
distances and points of interest are the deforestation work's functions
(deforestation_experiments/scripts/predictors.py, the platform's utils.py).

Two differences from the platform, both small: WDPA here is Guinea's designated, inscribed and
established areas painted at cell centres; the platform takes every polygon touching Guinea with
all_touched. OpenStreetMap is the extract of 2026-10-27, not the platform's cached one.
"""

import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]                         # landuse_experiments/grid_100m
ROOT = HERE.parent
sys.path.insert(0, str(ROOT.parent / "data_100m_tools"))
sys.path.insert(0, str(ROOT.parent / "deforestation_experiments" / "scripts"))
import grid100 as G                                                 # noqa: E402
import predictors as D                                              # noqa: E402

SRC = ROOT / "data" / "esri_predictors_src"
NAMES = ["DEM_GEE", "POPULATION_DENSITY", "PRECIPITATION", "SOIL_MOISTURE", "AGB_NASA", "NDVI",
         "SLOPE", "ASPECT", "ROADS_distance", "WATERWAYS_distance", "POI_heatmap", "PROTECTED_AREAS_zone"]
LABELS = {"DEM_GEE": "Elevation", "POPULATION_DENSITY": "Population density", "PRECIPITATION": "Precipitation",
          "SOIL_MOISTURE": "Soil moisture", "AGB_NASA": "Above-ground biomass", "NDVI": "NDVI",
          "SLOPE": "Slope", "ASPECT": "Aspect", "ROADS_distance": "Distance to roads",
          "WATERWAYS_distance": "Distance to waterways", "POI_heatmap": "Points of interest",
          "PROTECTED_AREAS_zone": "Protected areas"}
YEAR_HINT = {"POPULATION_DENSITY": (2000, 2020), "PRECIPITATION": (1981, 2025), "SOIL_MOISTURE": (1981, 2025),
             "AGB_NASA": (2010, 2010), "NDVI": (2017, 2024)}
COARSE_FILE = {"PRECIPITATION": "PRECIPITATION_{y}.tif", "SOIL_MOISTURE": "SOIL_MOISTURE_{y}.tif",
               "AGB_NASA": "AGB_NASA_{y}.tif", "NDVI": "NDVI_{y}.tif"}
STATIC_FROM_DEFORESTATION = {"DEM_GEE": "Elevation", "SLOPE": "Slope", "ASPECT": "Aspect",
                             "ROADS_distance": "Distance to roads", "WATERWAYS_distance": "Distance to waterways",
                             "POI_heatmap": "Points of interest"}


def year_used(name, year):
    """data_loader._snapshot_years: the year, capped by the source's year_hint (None = static)."""
    if name not in YEAR_HINT:
        return None
    first, last = YEAR_HINT[name]
    return min(last, year) if min(last, year) >= first else None


@lru_cache(maxsize=None)
def _coarse(name, y):
    return G.resample(SRC / COARSE_FILE[name].format(y=y), resampling="nearest")


def predictor(name, year):
    """One predictor at `year` -> float32 values over the cells inside Guinea (NaN where missing)."""
    if name in STATIC_FROM_DEFORESTATION:
        return D.static(STATIC_FROM_DEFORESTATION[name])
    if name == "POPULATION_DENSITY":
        y = str(year_used(name, year))
        return G.read_layer("worldpop", [y], scaled=True)[y].astype("float32")
    if name == "PROTECTED_AREAS_zone":
        return (G.read_layer("wdpa_year", ["status_year"])["status_year"] > 0).astype("float32")
    if name in COARSE_FILE:
        values = _coarse(name, year_used(name, year))
        return np.where(np.isfinite(values), values, np.nan).astype("float32")
    raise KeyError(name)


def stack(year, names=NAMES, timings=None):
    """(n_cells, len(names)) float32, columns in `names` order."""
    out = np.empty((G.cells().size, len(names)), dtype="float32")
    for j, name in enumerate(names):
        t0 = time.time()
        out[:, j] = predictor(name, year)
        if timings is not None:
            timings[name] = time.time() - t0
    return out


# --------------------------------------------------------------------------
# Comparison with the 250 m experiments' predictors

def agreement_with_250m(year, sample=400_000, seed=0):
    """Spearman correlation between each 100 m predictor and the 250 m experiments' value of the
    same predictor (esri_methods.predictor_stack, dated `year`) at the 250 m pixel each 100 m cell's
    centre falls in, over a random sample of cells; and the share of cells with no value."""
    import pandas as pd
    from scipy.stats import spearmanr
    sys.path.insert(0, str(ROOT / "scripts"))
    import esri_methods as E
    prof = E.grid()
    mask = np.ones((prof["height"], prof["width"]), dtype=bool)
    s250 = E.predictor_stack(prof, mask, year=year)                     # (12, H, W), the 250 m order
    order250 = dict(zip(E.PREDICTORS, range(len(E.PREDICTORS))))
    same = {"DEM_GEE": "Elevation", "POPULATION_DENSITY": "Population density", "PRECIPITATION": "Precipitation",
            "SOIL_MOISTURE": "Soil moisture", "AGB_NASA": "AGB", "NDVI": "NDVI", "SLOPE": "Slope",
            "ASPECT": "Aspect", "ROADS_distance": "Roads (distance)", "WATERWAYS_distance": "Waterways (distance)",
            "POI_heatmap": "POI (density)", "PROTECTED_AREAS_zone": "Protected areas"}
    # 250 m pixel of each 100 m cell, on the full 250 m raster
    t = prof["transform"]
    lon, lat = G.lon_lat()
    col = np.floor((lon - t.c) / t.a).astype("int64")
    row = np.floor((lat - t.f) / t.e).astype("int64")
    rng = np.random.default_rng(seed)
    pick = rng.choice(lon.size, sample, replace=False)
    rows = []
    for name in NAMES:
        v100 = predictor(name, year)
        v250 = s250[order250[same[name]]][row[pick], col[pick]]
        a = v100[pick]
        ok = np.isfinite(a) & np.isfinite(v250)
        rho = spearmanr(a[ok], v250[ok]).correlation if np.ptp(a[ok]) > 0 else np.nan
        rows.append({"predictor": LABELS[name], "file": name, "year used": year_used(name, year) or "static",
                     "no value (% of cells)": 100 * float(np.isnan(v100).mean()),
                     "Spearman with 250 m": float(rho)})
    return pd.DataFrame(rows).set_index("predictor")


# --------------------------------------------------------------------------
# Files for the platform's own code

LANDUSE = G.STORE / "landuse"                                       # inputs and runs of this folder
# A copy of the inputs on D: is read when present: on 2026-10-29 the E: drive dropped out twice
# during runs and the platform's code skipped the maps it could not read.
INPUTS = HERE / "data" / "inputs" if (HERE / "data" / "inputs").exists() else LANDUSE


def write_esri(year):
    """The platform's target file: the class at 100 m, 0 (no data) outside Guinea."""
    import rasterio
    path = INPUTS / "esri" / f"esri_{year}.tif"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        cls = G.read_layer(f"esri_{year}", ["class"])["class"]
        with rasterio.open(path, "w", **G.profile("uint8", 0)) as dst:
            dst.write(G.to_grid(cls, fill=0, dtype="uint8"), 1)
    return str(path)


def write_predictors(year):
    """The platform's predictor files for `year` (save_predictor_stack: one float32 GeoTIFF per
    layer named as main names it, NaN outside Guinea), in NAMES order."""
    import rasterio
    folder = INPUTS / f"predictors_{year}"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in NAMES:
        path = folder / f"{name}.tif"
        if not path.exists():
            with rasterio.open(path, "w", **G.profile("float32", np.nan)) as dst:
                dst.write(G.to_grid(predictor(name, year), fill=np.nan, dtype="float32"), 1)
        paths.append(str(path))
    return paths
