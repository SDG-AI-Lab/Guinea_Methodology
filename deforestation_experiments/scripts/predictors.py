"""The deforestation predictors on the platform's 100 m grid, as values over the cells inside
Guinea (float32), built from the shared 100 m store (data_100m_tools) and, for inputs whose
sources are coarser than 100 m, from files fetched once from Earth Engine (defor_data.py).

Each predictor is made as the platform makes it (data_100m_tools/platform_utils.py, a copy of
the platform's utils.py on main):

  Elevation, Slope, Aspect   SRTM read at 100 m; slope and aspect from it (calculate_slope, _aspect)
  Distance to roads, railways, waterways
                             OpenStreetMap cells -> distance in cells (process_vector_layer, "distance")
  Points of interest         OpenStreetMap amenities -> exp(-distance / (0.1 x largest distance))
                             (process_vector_layer, "heatmap")
  Population density         WorldPop, people per 100 m cell, the latest year at or before `year`

and the others: distance to roads (GRIP4),
to built-up area (GHSL above 1,000 m² of a cell, latest epoch at or before `year`), to mines;
protected area at `year`; ecoregion; and, resampled when used, rainfall, its 1991-2020 normal,
land-surface temperature, soil moisture, night lights, travel time and fire history.

The predictors that come from Hansen (distance to prior and recent loss, local loss rate, forest
edge) depend on which cells count as forest or lost: they are made in model.py (notebook 04).
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]                     # deforestation_experiments/
sys.path.insert(0, str(HERE.parent / "data_100m_tools"))
import grid100 as G                                            # noqa: E402
import platform_utils as PU                                    # noqa: E402

COARSE_DATA = HERE / "data"                                    # inputs from sources coarser than 100 m


def _inside2d():
    return G.inside_mask()


def _distance_cells(mask2d):
    """process_vector_layer's distance: Euclidean distance in cells to the nearest marked cell."""
    from scipy.ndimage import distance_transform_edt
    return G.from_grid(distance_transform_edt(mask2d == 0).astype("float32"))


def _distance_m(mask2d):
    """The platform's distance (drivers.distance_to): metres at 111,320 m per degree."""
    return _distance_cells(mask2d) * np.float32(G.grid()["cell_deg"] * 111_320.0)


def _mask(layer, column):
    return G.read_layer(layer, [column], as_grid=True, fill=0)[column] > 0


@lru_cache(maxsize=1)
def _dem2d():
    """Elevation with NaN outside Guinea, as the platform's DEM clipped to the outline: its
    handle_missing_values fills the ring just outside, so border cells get a slope."""
    return G.read_layer("dem", ["elevation"], as_grid=True, fill=np.nan)["elevation"].astype("float32")


def _platform_profile():
    p = G.profile()
    p["nodata"] = None
    return p


@lru_cache(maxsize=None)
def static(name):
    """A predictor that does not change with the year -> float32 values over the cells inside Guinea."""
    if name == "Elevation":
        return G.from_grid(_dem2d())
    if name == "Slope":
        return G.from_grid(PU.calculate_slope(_dem2d(), _platform_profile()).astype("float32"))
    if name == "Aspect":
        return G.from_grid(PU.calculate_aspect(_dem2d(), _platform_profile()).astype("float32"))
    if name in ("Distance to roads", "Distance to railways", "Distance to waterways"):
        key = {"Distance to roads": "roads", "Distance to railways": "railways", "Distance to waterways": "waterways"}[name]
        return _distance_cells(_mask("osm", key))
    if name == "Points of interest":
        distance = _distance_cells(_mask("osm", "poi"))
        largest = distance.max() if distance.max() > 0 else 1.0
        return np.exp(-distance / (largest * 0.1)).astype("float32")
    if name == "Distance to roads (GRIP4)":
        return _distance_m(_mask("grip4_roads", "road"))
    if name == "Distance to mines (2019)":
        return _distance_m(_mask("mines", "mine"))
    if name == "Ecoregion":
        return G.read_layer("ecoregions", ["ecoregion"])["ecoregion"].astype("float32")
    if name == "Precipitation (1991–2020 normal)":
        return G.resample(COARSE_DATA / "chirps_normal.tif", "1991-2020")
    if name == "Travel time to cities (2015)":
        values = G.resample(COARSE_DATA / "travel_time_2015.tif", "2015")
        return np.where(values >= 0, values, np.nan).astype("float32")
    raise KeyError(name)


COARSE_ANNUAL = {"Land-surface temperature": "lst_annual", "Precipitation": "chirps_annual",
                 "Soil moisture": "soil_moisture_annual", "Night lights": "nightlights"}


def _closest_year(columns, year):
    years = sorted(int(c) for c in columns)
    return str(max([y for y in years if y <= year], default=years[0]))


def at_year(name, year):
    """A predictor measured at the start of a window (`year`) -> float32 values over the cells inside Guinea."""
    if name == "Population density":
        column = _closest_year(G.layers()["worldpop"]["columns"], year)
        return G.read_layer("worldpop", [column], scaled=True)[column].astype("float32")
    if name in COARSE_ANNUAL:
        import rasterio
        with rasterio.open(COARSE_DATA / f"{COARSE_ANNUAL[name]}.tif") as src:
            column = _closest_year(src.descriptions, year)
        return G.resample(COARSE_DATA / f"{COARSE_ANNUAL[name]}.tif", column)
    if name == "Distance to built-up area":
        epoch = max([e for e in (2010, 2015, 2020) if e <= year], default=2010)
        built = G.read_layer("ghsl_built", [str(epoch)], as_grid=True, fill=0)[str(epoch)] > 1000
        return _distance_m(built)
    if name == "Protected area":
        status = G.read_layer("wdpa_year", ["status_year"])["status_year"]
        return ((status > 0) & (status <= year)).astype("float32")
    if name == "Fire history (years burned of 5)":
        import rasterio
        with rasterio.open(COARSE_DATA / "burned.tif") as src:
            names = list(src.descriptions)
        return sum(G.resample(COARSE_DATA / "burned.tif", str(y), "nearest") for y in range(year - 4, year + 1)
                   if str(y) in names).astype("float32")
    return static(name)


BUILT_HERE = ["Elevation", "Slope", "Aspect", "Distance to roads", "Distance to railways", "Distance to waterways",
              "Points of interest", "Population density", "Distance to built-up area", "Protected area",
              "Distance to roads (GRIP4)", "Distance to mines (2019)", "Ecoregion"]
RESAMPLED = ["Land-surface temperature", "Precipitation", "Precipitation (1991–2020 normal)", "Soil moisture",
             "Night lights", "Travel time to cities (2015)", "Fire history (years burned of 5)"]
