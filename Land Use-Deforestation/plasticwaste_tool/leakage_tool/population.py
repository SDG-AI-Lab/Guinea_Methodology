"""
Population on a 1 km grid, classified urban/rural, summarised by ADM1.

THE CORRECTION THAT MATTERS HERE

`gin_pd_2020_1km_UNadj.tif` is a population DENSITY surface -- persons per
square kilometre -- not a count of people per pixel. The two are easy to
confuse because the grid is described as "1 km", which invites the assumption
that summing the pixels gives the population.

It does not, and the error is not small. Summing the raster gives 15.70
million. Guinea's 2020 population was about 13.1 million (UN World Population
Prospects 2022). The 20% discrepancy is geometry: the grid is 0.00833 degrees,
which is 1 km north-south but only about 0.92 km east-west at Guinea's
latitudes, so a pixel covers roughly 0.85 km2, not 1.0. Density must be
multiplied by each pixel's TRUE area, which varies with latitude.

Getting this wrong would have inflated every downstream mass estimate by a
fifth -- silently, and in the direction that makes the problem look worse.
`total_population` below therefore does the area weighting explicitly and
`validate_against_census` checks the result against an independent figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent          # Land Use-Deforestation/
DATA = PROJECT / "data"

POP_RASTER = DATA / "gin_pd_2020_1km_UNadj.tif"
ADM1 = DATA / "geoBoundaries-GIN-ADM1.geojson"

# UN World Population Prospects 2022, Guinea, 2020. Used only as an
# independent check on the area weighting above, never as an input.
UN_POPULATION_2020 = 13_132_795
POP_REFERENCE_YEAR = 2020

# WGS84 ellipsoid, for exact pixel areas.
_R_EQ = 6378137.0
_FLAT = 1 / 298.257223563
_E2 = _FLAT * (2 - _FLAT)


def _pixel_area_km2(lat_deg: np.ndarray, dlat: float, dlon: float) -> np.ndarray:
    """
    Area of a graticule cell centred on each latitude, on the WGS84 ellipsoid.

    Uses the authalic (equal-area) formulation rather than the spherical
    cos(lat) approximation: the difference is about 0.3% at Guinea's
    latitudes, which is negligible here, but the exact form costs nothing and
    removes one thing a reviewer would otherwise have to take on trust.
    """
    lat = np.radians(lat_deg)
    half = np.radians(abs(dlat)) / 2.0
    lo, hi = lat - half, lat + half

    def _q(phi):
        s = np.sin(phi)
        e = np.sqrt(_E2)
        return (s / (1 - _E2 * s ** 2)
                + (1 / (2 * e)) * np.log((1 + e * s) / (1 - e * s)))

    zone = (np.pi * _R_EQ ** 2 * (1 - _E2) / 1e6) * (_q(hi) - _q(lo))
    return zone * (abs(dlon) / 360.0)


def load_population(path: Optional[Path] = None) -> Dict:
    """
    Read the density grid and convert it to people per pixel.

    Returns a dict with `density` (persons/km2), `area_km2`, `people`
    (density * area, i.e. an actual headcount per pixel), the affine
    `transform`, `crs`, and a boolean `valid` mask.
    """
    import rasterio

    path = Path(path or POP_RASTER)
    if not path.exists():
        raise FileNotFoundError(
            f"population raster not found at {path}. Expected WorldPop "
            f"gin_pd_2020_1km_UNadj.tif (unconstrained, UN-adjusted density).")

    with rasterio.open(path) as src:
        density = src.read(1).astype("float64")
        transform, crs = src.transform, src.crs
        nodata = src.nodata
        h, w = src.shape

    valid = np.isfinite(density)
    if nodata is not None:
        valid &= density != nodata
    valid &= density >= 0
    density = np.where(valid, density, 0.0)

    # Latitude of each ROW centre, then broadcast across columns.
    rows = np.arange(h) + 0.5
    lats = transform.f + rows * transform.e          # e is negative (north-up)
    area_col = _pixel_area_km2(lats, transform.e, transform.a)
    area_km2 = np.repeat(area_col[:, None], w, axis=1)

    return {"density": density, "area_km2": area_km2,
            "people": density * area_km2, "valid": valid,
            "transform": transform, "crs": crs, "shape": (h, w),
            "path": str(path)}


def total_population(pop: Dict) -> float:
    return float(pop["people"][pop["valid"]].sum())


def validate_against_census(pop: Dict, tol: float = 0.05) -> Dict:
    """
    Compare the area-weighted total against UN WPP, and show what the naive
    sum would have given. Reported, not enforced -- WorldPop and WPP are
    independent products and small disagreement is expected.
    """
    weighted = total_population(pop)
    naive = float(pop["density"][pop["valid"]].sum())
    return {
        "area_weighted_total": weighted,
        "naive_pixel_sum": naive,
        "un_wpp_2020": UN_POPULATION_2020,
        "relative_error": weighted / UN_POPULATION_2020 - 1.0,
        "naive_relative_error": naive / UN_POPULATION_2020 - 1.0,
        "within_tolerance": abs(weighted / UN_POPULATION_2020 - 1.0) <= tol,
        "mean_pixel_area_km2": float(
            pop["area_km2"][pop["valid"]].mean()),
    }


def classify_urban(pop: Dict, threshold_per_km2: float) -> np.ndarray:
    """
    Boolean urban mask by population density, per DEGURBA.

    Applied to the DENSITY band, not the headcount band -- density is what the
    definition is written in terms of, and using headcount would make the
    classification depend on the latitude-varying pixel area, so identical
    settlements would classify differently north and south.

    DEGURBA proper also requires spatial CONTIGUITY (a cluster of >=5,000
    people in contiguous cells). That is not applied here: the contiguity rule
    exists to delineate settlement boundaries for reporting, whereas this mask
    only has to route each pixel to one of two collection rates. The
    simplification is recorded rather than hidden, and the threshold itself is
    sampled across the DEGURBA range (300-1500/km2) so the sensitivity is
    visible in the output.
    """
    return pop["valid"] & (pop["density"] >= threshold_per_km2)


def urban_rural_population(pop: Dict, threshold_per_km2: float) -> Tuple[float, float]:
    urban = classify_urban(pop, threshold_per_km2)
    p = pop["people"]
    return float(p[urban].sum()), float(p[pop["valid"] & ~urban].sum())


def zonal_by_adm1(pop: Dict, threshold_per_km2: float,
                  adm1_path: Optional[Path] = None):
    """
    Urban and rural population per ADM1 region.

    `all_touched=False`: a pixel belongs to the region containing its centre,
    so no one is counted twice along a shared boundary. National totals from
    this function must reconcile with `total_population` up to pixels whose
    centre falls outside every polygon (coastline and border effects); the
    reconciliation is returned rather than assumed.
    """
    import geopandas as gpd
    import pandas as pd
    from rasterio.features import rasterize

    adm1_path = Path(adm1_path or ADM1)
    gdf = gpd.read_file(adm1_path).to_crs("EPSG:4326").reset_index(drop=True)
    name_col = "shapeName" if "shapeName" in gdf.columns else gdf.columns[0]
    # The same row's shapeID: the key the hub joins on (feature 002, R12).
    id_col = "shapeID" if "shapeID" in gdf.columns else None

    zones = rasterize(
        ((geom, i + 1) for i, geom in enumerate(gdf.geometry)),
        out_shape=pop["shape"], transform=pop["transform"],
        fill=0, dtype="int32", all_touched=False)

    urban = classify_urban(pop, threshold_per_km2)
    people = np.where(pop["valid"], pop["people"], 0.0)

    rows = []
    ids = list(gdf[id_col]) if id_col else [None] * len(gdf)
    for i, (name, unit_id) in enumerate(zip(gdf[name_col], ids), start=1):
        m = zones == i
        pu = float(people[m & urban].sum())
        pr = float(people[m & ~urban].sum())
        rows.append({"adm1": name, "adm1_id": unit_id, "pop_urban": pu, "pop_rural": pr,
                     "pop_total": pu + pr,
                     "urban_share": (pu / (pu + pr)) if (pu + pr) > 0 else 0.0,
                     "area_km2": float(pop["area_km2"][m].sum())})

    df = pd.DataFrame(rows).sort_values("pop_total", ascending=False)
    df.attrs["unassigned_population"] = (
        total_population(pop) - df["pop_total"].sum())
    df.attrs["zone_raster"] = zones
    df.attrs["adm1_names"] = list(gdf[name_col])
    return df
