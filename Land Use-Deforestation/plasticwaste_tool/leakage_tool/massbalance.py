"""
The waste chain: people -> MSW -> plastic -> uncollected -> unburned ->
environment -> water.

Deterministic arithmetic only. Uncertainty propagation lives in
`uncertainty.py`, and the separation is deliberate: this module must be
simple enough to check by hand, because a Monte Carlo wrapped around a wrong
chain produces a confident wrong answer with error bars.

THE CHAIN, AND WHY IT HAS THE STEPS IT HAS

    generated      = population * rate * 365.25            [t/yr MSW]
    plastic        = generated * plastic_share             [t/yr plastic]
    uncollected    = plastic * (1 - collection)            urban/rural split
    burned         = uncollected * open_burn_fraction      DESTROYED, not leaked
    to_environment = (uncollected - burned) * leak_to_env
                   + collected * litter_rate               [t/yr]
    to_water       = to_environment * p_to_water           [t/yr]

The step that distinguishes this from a naive estimate is OPEN BURNING.
"Uncollected" is routinely treated as a synonym for "leaked", which it is not:
burning destroys plastic as macro-debris. Skipping that step roughly doubles
the answer. It is included here even though the evidence for its magnitude is
regional rather than Guinean, and the output labels it as such.

WHAT THIS MODEL IS AND IS NOT

It is a spatially-resolved estimate of plastic waste GENERATION and its
mismanaged share, driven by population. It is not a measurement of plastic in
the environment. Satellite detection measures where material is and how much
of it there is; ground gravimetric sorting measures what share of it is
plastic; terrain and hydrology models say where it goes. This module is the
bookkeeping that connects the second to the third. It cannot validate itself
against the first, and Section `cross_check_against_detections` says so
explicitly rather than implying agreement.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import parameters as P


def _collection_rate(urban_fraction: float, c_urban: float,
                     c_rural: float) -> float:
    return urban_fraction * c_urban + (1.0 - urban_fraction) * c_rural


def check_collection_consistency(urban_fraction: float,
                                 c_urban: Optional[float] = None,
                                 c_rural: Optional[float] = None) -> Dict:
    """
    Do the surveyed urban and rural collection rates, weighted by the urban
    population share this model derives, reproduce the separately-reported
    NATIONAL collection rate?

    This is a genuine out-of-sample check. The national figure (9.7%) is a
    third entry in the same survey and is never used as an input, so
    agreement is evidence that the urban/rural classification is routing
    people to roughly the right rate -- and disagreement would mean the
    density threshold is wrong.
    """
    c_urban = P.COLLECTION_URBAN.value if c_urban is None else c_urban
    c_rural = P.COLLECTION_RURAL.value if c_rural is None else c_rural
    implied = _collection_rate(urban_fraction, c_urban, c_rural)
    reported = P.COLLECTION_NATIONAL_REPORTED
    return {"urban_fraction": urban_fraction,
            "implied_national_collection": implied,
            "reported_national_collection": reported,
            "ratio": implied / reported if reported else float("nan"),
            "agrees_within_20pct": abs(implied / reported - 1.0) <= 0.20}


def chain(pop_urban: float, pop_rural: float, *,
          msw_rate: float, plastic_share: float,
          collection_urban: float, collection_rural: float,
          open_burn_fraction: float, leak_to_env: float,
          litter_rate_collected: float, p_to_water: float,
          **_ignored) -> Dict[str, float]:
    """
    One evaluation of the chain for one population, in tonnes per year.

    Accepts scalars or numpy arrays interchangeably, so the same function
    serves a single region, a vector of Monte Carlo draws, and a raster of
    pixels without a second implementation to keep in step.
    """
    kg_to_t = P.DAYS_PER_YEAR / P.KG_PER_TONNE

    msw_urban = pop_urban * msw_rate * kg_to_t
    msw_rural = pop_rural * msw_rate * kg_to_t
    msw = msw_urban + msw_rural

    plastic_urban = msw_urban * plastic_share
    plastic_rural = msw_rural * plastic_share
    plastic = plastic_urban + plastic_rural

    uncollected = (plastic_urban * (1.0 - collection_urban)
                   + plastic_rural * (1.0 - collection_rural))
    collected = plastic - uncollected

    burned = uncollected * open_burn_fraction
    unburned = uncollected - burned

    to_env = unburned * leak_to_env + collected * litter_rate_collected
    contained = plastic - burned - to_env

    to_water = to_env * p_to_water

    return {
        "msw_t_yr": msw,
        "plastic_t_yr": plastic,
        "collected_t_yr": collected,
        "uncollected_t_yr": uncollected,
        "burned_t_yr": burned,
        "to_environment_t_yr": to_env,
        "contained_t_yr": contained,
        "to_water_t_yr": to_water,
        # Diagnostics, not new information -- they make the result legible
        # without the reader recomputing ratios by hand.
        "mismanaged_share_of_plastic": _safe_div(to_env, plastic),
        "plastic_kg_per_capita_yr": _safe_div(
            plastic * P.KG_PER_TONNE, (pop_urban + pop_rural)),
        "leak_kg_per_capita_yr": _safe_div(
            to_env * P.KG_PER_TONNE, (pop_urban + pop_rural)),
    }


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.asarray(b) > 0, np.asarray(a, float)
                        / np.where(np.asarray(b) > 0, b, 1.0), np.nan)


def check_mass_balance(result: Dict[str, float], tol: float = 1e-6) -> bool:
    """Every tonne of plastic is burned, leaked, or contained. No exceptions."""
    lhs = np.asarray(result["plastic_t_yr"], dtype=float)
    rhs = (np.asarray(result["burned_t_yr"], float)
           + np.asarray(result["to_environment_t_yr"], float)
           + np.asarray(result["contained_t_yr"], float))
    return bool(np.all(np.abs(lhs - rhs) <= tol * np.maximum(lhs, 1.0)))


def deterministic(pop_urban: float, pop_rural: float,
                  scenario: str = P.DEFAULT_SCENARIO) -> Dict[str, float]:
    """Central estimate: every parameter at its central value."""
    params = {k: p.value for k, p in P.all_parameters(scenario).items()}
    out = chain(pop_urban, pop_rural, **params)
    out["scenario"] = scenario
    return out


def urban_split_sensitivity(pop, scenario: str = P.DEFAULT_SCENARIO):
    """
    How much does the answer move across the DEGURBA threshold range?

    Reported because the urban/rural split is the only place where a
    definitional choice -- not a measurement -- changes the result, and a
    reader deserves to see its size rather than be told it is small.
    """
    import pandas as pd

    from .population import urban_rural_population

    rows = []
    for thr in (300, 500, 1000, 1500):
        pu, pr = urban_rural_population(pop, thr)
        r = deterministic(pu, pr, scenario)
        cc = check_collection_consistency(pu / (pu + pr))
        rows.append({
            "threshold_per_km2": thr,
            "urban_share": pu / (pu + pr),
            "implied_national_collection": cc["implied_national_collection"],
            "reported_national_collection": cc["reported_national_collection"],
            "collection_ratio": cc["ratio"],
            "plastic_t_yr": r["plastic_t_yr"],
            "to_environment_t_yr": r["to_environment_t_yr"],
            "to_water_t_yr": r["to_water_t_yr"],
        })
    return pd.DataFrame(rows)


def per_pixel_leakage(pop: Dict, threshold_per_km2: float,
                      scenario: str = P.DEFAULT_SCENARIO) -> np.ndarray:
    """
    Tonnes per year of plastic reaching the environment, per 1 km pixel.

    This is the layer the map is drawn from. Because every parameter except
    the collection rate is national and constant, the SHAPE of this surface is
    the population surface with an urban/rural step in it -- it adds no
    independent spatial information. Stated plainly here because a smooth
    national raster invites the reader to believe it was measured
    pixel-by-pixel, and it was not.
    """
    from .population import classify_urban

    params = {k: p.value for k, p in P.all_parameters(scenario).items()}
    urban = classify_urban(pop, threshold_per_km2)
    people = np.where(pop["valid"], pop["people"], 0.0)

    kg_to_t = P.DAYS_PER_YEAR / P.KG_PER_TONNE
    plastic = people * params["msw_rate"] * kg_to_t * params["plastic_share"]
    coll = np.where(urban, params["collection_urban"], params["collection_rural"])

    uncollected = plastic * (1.0 - coll)
    unburned = uncollected * (1.0 - params["open_burn_fraction"])
    return (unburned * params["leak_to_env"]
            + (plastic - uncollected) * params["litter_rate_collected"])


def cross_check_against_detections(to_env_t_yr: float,
                                   n_sites: int,
                                   n_fdi_patches: int) -> Dict:
    """
    State the relationship between this model and the detection layers
    WITHOUT implying it is a validation.

    The two cannot validate each other. This model estimates an annual FLOW of
    plastic into the environment, in tonnes. The detection layers report a
    COUNT of locations where material is visible at one moment, with no mass
    attached -- the extent work established that dumpsites are ~14 m across,
    sub-pixel at Sentinel-2 resolution, so no mass can be inverted from them.
    A flow and a count have no arithmetic relationship, and dividing one by
    the other produces a number with units but no meaning.

    What CAN be said is a consistency statement about implied per-site mass,
    which is returned here explicitly labelled as such.
    """
    implied = to_env_t_yr / n_sites if n_sites else float("nan")
    return {
        "to_environment_t_yr": to_env_t_yr,
        "n_detected_sites": n_sites,
        "n_fdi_patches": n_fdi_patches,
        "implied_t_yr_per_detected_site": implied,
        "interpretation": (
            "If every tonne entering the environment passed through a detected "
            f"site, each of the {n_sites} sites would handle "
            f"{implied:,.0f} t/yr. It does not: detection is incomplete, "
            "much leakage is diffuse littering that never forms a visible "
            "accumulation, and the site inventory is a labelled sample rather "
            "than a census. The figure is an UPPER BOUND on per-site "
            "throughput under a maximally concentrated assumption, useful for "
            "sanity-checking order of magnitude and nothing else."),
        "is_validation": False,
    }
