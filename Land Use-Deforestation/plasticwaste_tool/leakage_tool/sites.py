"""
Per-site plastic mass: giving the mobilisation score physical units.

THE PROBLEM THIS SOLVES, AND THE ONE IT DOES NOT

`characterisation_tool/mobilisation.py` ranks detected dumpsites by how likely
their contents are to reach a watercourse. Its mass term is `mass_proxy`, a
dimensionless 1/2/4 by size class, so the resulting `mobilisation_risk` is an
ordinal score on a 0-100 scale. It can say site A outranks site B. It cannot
say how many tonnes either of them contributes, which is the question a
clean-up budget is actually written against.

This module attaches tonnes. It does so by CATCHMENT ALLOCATION: the
population whose waste plausibly arrives at a site, run through the same mass
balance as the national model, and apportioned among the sites competing for
that population.

WHAT THIS IS NOT

It is not a measurement of the plastic at a site, and no remote-sensing
product could make it one. The extent work established that these sites are
about 14 m across -- sub-pixel at Sentinel-2's 10 m bands and far below
EnMAP's or EMIT's ground sample distance -- and the hyperspectral work found
no C-H absorption signal attributable to polymer at these scales. There is
therefore no spectral route from an image to a mass, and inventing one by
multiplying area by an assumed depth and density would produce a number whose
precision is entirely fictitious.

What this module does instead is defensible for the opposite reason: it never
looks at the pixels at all. The mass comes from demography and the published
waste chain; the satellite contributes only the LOCATION, which is the thing
it genuinely measures. Satellite detection tells you where material is; a
gravimetric sorting survey would tell you what share of it is plastic; this
model tells you how much the surrounding population generates. Keeping those
three provenances separate is the point.

The consequence is a real limitation, stated rather than buried: two adjacent
sites with the same catchment receive the same mass estimate regardless of how
different they look. The estimate is a property of the CATCHMENT, refined by
the site's size class and transport probability, not an observation of the
site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import massbalance as MB
from . import parameters as P

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent
MOBILISATION = TOOL / "runs" / "mobilisation" / "mobilisation_risk.csv"

# Radius of the population catchment assigned to a site.
#
# 5 km is the distance band used for "waste generated within reach of an
# unmanaged disposal point" in the UNEP/IUCN hotspotting guidance, and is
# consistent with the observation in Kaza et al. (2018) that informal disposal
# in low-income cities is predominantly served by headloading, handcart and
# short motorised trips rather than by municipal haulage. It is a convention,
# not a measurement, so `catchment_sensitivity` reports the effect of
# changing it.
DEFAULT_CATCHMENT_KM = 5.0


def load_mobilisation(path: Optional[Path] = None):
    import pandas as pd
    path = Path(path or MOBILISATION)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run characterisation_tool/mobilisation.py "
            f"first -- this module adds units to its output, it does not "
            f"recompute the transport terms.")
    return pd.read_csv(path)


def catchment_population(pop: Dict, lons, lats,
                         radius_km: float = DEFAULT_CATCHMENT_KM) -> np.ndarray:
    """
    Population within `radius_km` of each site, from the 1 km headcount grid.

    Distances are computed on the WGS84 sphere rather than in raster index
    space: at Guinea's latitude a degree of longitude is about 0.985 of a
    degree of latitude, so a circle in index space is a noticeably eccentric
    ellipse on the ground, and the error would fall entirely on east-west
    extent.
    """
    people = np.where(pop["valid"], pop["people"], 0.0)
    h, w = pop["shape"]
    t = pop["transform"]

    rows = np.arange(h) + 0.5
    cols = np.arange(w) + 0.5
    pix_lat = t.f + rows * t.e
    pix_lon = t.c + cols * t.a

    R = 6371.0088
    lat_grid = np.repeat(pix_lat[:, None], w, axis=1)
    lon_grid = np.repeat(pix_lon[None, :], h, axis=0)

    out = np.zeros(len(lons), dtype=float)
    for i, (lo, la) in enumerate(zip(np.asarray(lons), np.asarray(lats))):
        # Equirectangular approximation: exact enough well under 100 km, and
        # it avoids a haversine over the full grid for every site.
        dy = np.radians(lat_grid - la) * R
        dx = np.radians(lon_grid - lo) * R * np.cos(np.radians(la))
        out[i] = people[(dx * dx + dy * dy) <= radius_km ** 2].sum()
    return out


def allocate(pop: Dict, sites=None,
             radius_km: float = DEFAULT_CATCHMENT_KM,
             scenario: str = P.DEFAULT_SCENARIO,
             urban_threshold: Optional[float] = None):
    """
    Tonnes per year of plastic entering the environment at each site, and the
    share of that reaching a watercourse.

    THE SHARED-CATCHMENT CORRECTION

    Site catchments overlap. Two sites 3 km apart with 5 km radii claim much of
    the same population, and simply summing their independently-computed
    catchments would count those people twice -- inflating the national total
    above what the same chain produces for the whole country.

    So each site's catchment population is divided by the number of sites
    claiming it, via a competition weight computed on the grid. The result is
    that per-site masses sum to the population-weighted total for the covered
    area, and never exceed the national figure.
    """
    import pandas as pd

    from .population import classify_urban

    sites = load_mobilisation() if sites is None else sites
    thr = float(urban_threshold if urban_threshold is not None
                else P.URBAN_DENSITY_THRESHOLD.value)
    params = {k: p.value for k, p in P.all_parameters(scenario).items()}

    lons = sites["lon"].to_numpy(float)
    lats = sites["lat"].to_numpy(float)

    people = np.where(pop["valid"], pop["people"], 0.0)
    urban = classify_urban(pop, thr)
    h, w = pop["shape"]
    t = pop["transform"]
    R = 6371.0088
    lat_grid = np.repeat((t.f + (np.arange(h) + 0.5) * t.e)[:, None], w, axis=1)
    lon_grid = np.repeat((t.c + (np.arange(w) + 0.5) * t.a)[None, :], h, axis=0)

    masks = []
    for lo, la in zip(lons, lats):
        dy = np.radians(lat_grid - la) * R
        dx = np.radians(lon_grid - lo) * R * np.cos(np.radians(la))
        masks.append((dx * dx + dy * dy) <= radius_km ** 2)

    # How many sites claim each pixel; 0 outside every catchment.
    claims = np.zeros((h, w), dtype=float)
    for m in masks:
        claims += m
    share = np.divide(1.0, claims, out=np.zeros_like(claims), where=claims > 0)

    rows = []
    for i, m in enumerate(masks):
        pu = float((people * share)[m & urban].sum())
        pr = float((people * share)[m & ~urban].sum())
        gross = float(people[m].sum())
        r = MB.chain(pu, pr, **params)
        rows.append({
            "point_id": sites["point_id"].iloc[i],
            "lon": lons[i], "lat": lats[i],
            "region": sites["region"].iloc[i] if "region" in sites else None,
            "size_class": sites["size_class"].iloc[i] if "size_class" in sites else None,
            "catchment_pop_gross": gross,
            "catchment_pop_attributed": pu + pr,
            "catchment_urban_share": (pu / (pu + pr)) if (pu + pr) > 0 else 0.0,
            "plastic_t_yr": r["plastic_t_yr"],
            "to_environment_t_yr": r["to_environment_t_yr"],
        })

    df = pd.DataFrame(rows)

    # Transport: prefer the site-specific probability from the mobilisation
    # model over the national p_to_water constant. `p_to_river` there is a
    # relative probability, not an annual flux fraction, so it is rescaled to
    # preserve the national mean rather than used as a fraction directly --
    # otherwise the two models would disagree on the national total purely
    # because of a units mismatch.
    if "p_to_river" in sites.columns:
        p_rel = sites["p_to_river"].to_numpy(float)
        scale = params["p_to_water"] / np.nanmean(p_rel) if np.nanmean(p_rel) > 0 else 0.0
        df["p_to_water_site"] = p_rel * scale
        df["transport_source"] = "mobilisation model (slope, drainage, runoff)"
    else:
        df["p_to_water_site"] = params["p_to_water"]
        df["transport_source"] = "national constant"

    df["to_water_t_yr"] = df["to_environment_t_yr"] * df["p_to_water_site"]

    for col in ("mobilisation_risk", "ocean_leakage_risk", "mass_proxy",
                "dist_river_m", "flood_ever"):
        if col in sites.columns:
            df[col] = sites[col].to_numpy()

    df.attrs["double_counted_population"] = float(
        df["catchment_pop_gross"].sum() - df["catchment_pop_attributed"].sum())
    df.attrs["radius_km"] = radius_km
    df.attrs["scenario"] = scenario
    return df.sort_values("to_water_t_yr", ascending=False)


def national_coverage(df, pop: Dict, scenario: str = P.DEFAULT_SCENARIO) -> Dict:
    """
    What share of national leakage do the detected sites' catchments account
    for? Reported because it bounds how much of the problem a site-based
    clean-up strategy could possibly address.
    """
    from .population import urban_rural_population

    pu, pr = urban_rural_population(pop, float(P.URBAN_DENSITY_THRESHOLD.value))
    nat = MB.deterministic(pu, pr, scenario)
    covered = float(df["to_environment_t_yr"].sum())
    return {
        "national_to_environment_t_yr": nat["to_environment_t_yr"],
        "sites_to_environment_t_yr": covered,
        "share_of_national": covered / nat["to_environment_t_yr"],
        "n_sites": len(df),
        "catchment_radius_km": df.attrs.get("radius_km"),
        "population_in_catchments": float(df["catchment_pop_attributed"].sum()),
        "national_population": pu + pr,
        "population_share": float(
            df["catchment_pop_attributed"].sum()) / (pu + pr),
        "caveat": (
            "The site inventory is a LABELLED SAMPLE from a national random "
            "grid, not a census of Guinea's dumpsites, so this share is a "
            "property of the sample and must not be read as 'the rest of the "
            "country has no sites'. It answers a narrower question: of the "
            "national leakage flow, how much originates from populations "
            "living near a site we have actually confirmed."),
    }


def catchment_sensitivity(pop: Dict, sites=None,
                          radii=(2.0, 5.0, 10.0, 20.0),
                          scenario: str = P.DEFAULT_SCENARIO):
    """The catchment radius is a convention; show what it costs."""
    import pandas as pd
    sites = load_mobilisation() if sites is None else sites
    rows = []
    for r in radii:
        df = allocate(pop, sites, radius_km=r, scenario=scenario)
        cov = national_coverage(df, pop, scenario)
        rows.append({
            "radius_km": r,
            "pop_in_catchments": cov["population_in_catchments"],
            "population_share": cov["population_share"],
            "sites_to_env_t_yr": cov["sites_to_environment_t_yr"],
            "share_of_national": cov["share_of_national"],
            "double_counted_pop": df.attrs["double_counted_population"],
            "median_site_t_yr": float(df["to_environment_t_yr"].median()),
        })
    return pd.DataFrame(rows)
