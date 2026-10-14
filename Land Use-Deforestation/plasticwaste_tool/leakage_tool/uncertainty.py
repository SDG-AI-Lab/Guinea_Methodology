"""
Uncertainty propagation by Monte Carlo, with the correlation structure that
quadrature would get wrong.

WHY NOT ADD ERRORS IN QUADRATURE

The obvious approach -- give every region an error bar and combine them as
sqrt(sum of squares) -- assumes the errors are independent. Here they are not,
and the dependence is total rather than incidental.

Every region's estimate is driven by the SAME national generation rate, the
SAME plastic share, the SAME collection survey, the SAME open-burning
fraction. If the true plastic share is 15% rather than 4%, it is 15% in Boke
AND in Kankan AND in Conakry simultaneously. The regional errors are not
independent draws that partially cancel; they move in lockstep.

Quadrature would therefore report a national error bar substantially narrower
than the truth -- the more regions you divide the country into, the more
spurious cancellation you manufacture, and dividing a country into more pieces
cannot make its total better known.

THE FIX, WHICH IS ALSO THE WHOLE DESIGN

Draw each national parameter ONCE per Monte Carlo iteration and apply that one
draw to every region and every pixel in that iteration. Regional totals then
sum exactly to the national total within each iteration, and the spread across
iterations carries the correlation correctly by construction. The sum of the
regional 5th percentiles will NOT equal the national 5th percentile -- that is
the correct behaviour, and `check_additivity` below demonstrates it rather
than treating it as a bug to be smoothed away.

WHAT THE INTERVAL DOES AND DOES NOT COVER

It covers the stated ranges of the parameters. It does not cover structural
error: that the chain has the right steps, that a population-driven model is
the right model, or that What a Waste's modelled generation rate is unbiased.
Those are not quantifiable from the available evidence and the output says so
rather than implying the interval is a complete account of what could be wrong.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from . import massbalance as MB
from . import parameters as P

DEFAULT_N = 20_000
DEFAULT_SEED = 20240119   # fixed so every run of the notebook is reproducible

# Reported quantiles. 5/95 rather than 2.5/97.5 because the lognormal ranges
# were themselves specified as 90% intervals -- quoting a 95% interval would
# extrapolate the tails further than the sources support.
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def draw_parameters(n: int = DEFAULT_N,
                    scenario: str = P.DEFAULT_SCENARIO,
                    seed: int = DEFAULT_SEED) -> Dict[str, np.ndarray]:
    """
    n joint draws of the national parameter set.

    Parameters are drawn independently of each other -- there is no evidence
    in the sources for a correlation between, say, the generation rate and the
    burning fraction. The correlation this module exists to handle is across
    SPACE (regions sharing one parameter value), not between parameters, and
    that is enforced by the caller reusing these same n draws everywhere.
    """
    rng = np.random.default_rng(seed)
    return {k: p.draw(n, rng) for k, p in P.all_parameters(scenario).items()}


def national(pop, n: int = DEFAULT_N,
             scenario: str = P.DEFAULT_SCENARIO,
             seed: int = DEFAULT_SEED) -> Dict:
    """
    National totals with a credible interval.

    The urban/rural threshold is itself sampled, so each iteration reclassifies
    the population. Because re-running the zonal split 20,000 times would be
    prohibitive, the population is pre-computed on a small grid of thresholds
    and interpolated -- the urban share is smooth and monotone in the
    threshold, so interpolation introduces error far below the width of any
    interval reported here.
    """
    from .population import urban_rural_population

    d = draw_parameters(n, scenario, seed)

    grid = np.array([300., 500., 750., 1000., 1250., 1500.])
    shares = np.array([
        (lambda t: t[0] / (t[0] + t[1]))(urban_rural_population(pop, g))
        for g in grid])
    total_pop = sum(urban_rural_population(pop, grid[0]))

    urban_share = np.interp(d["urban_density_threshold"], grid, shares)
    pop_urban = total_pop * urban_share
    pop_rural = total_pop * (1.0 - urban_share)

    res = MB.chain(pop_urban, pop_rural, **d)
    res["_draws"] = d
    res["_n"] = n
    res["_scenario"] = scenario
    res["_total_population"] = total_pop
    return res


def summarise(res: Dict, keys: Optional[Iterable[str]] = None):
    """Quantiles of each output quantity, as a table."""
    import pandas as pd

    keys = keys or ["msw_t_yr", "plastic_t_yr", "uncollected_t_yr",
                    "burned_t_yr", "to_environment_t_yr", "to_water_t_yr",
                    "leak_kg_per_capita_yr"]
    rows = []
    for k in keys:
        v = np.asarray(res[k], dtype=float)
        q = np.quantile(v, QUANTILES)
        rows.append({"quantity": k, "mean": v.mean(),
                     **{f"p{int(100 * p)}": qi for p, qi in zip(QUANTILES, q)},
                     "ratio_p95_p5": q[-1] / q[0] if q[0] > 0 else np.nan})
    return pd.DataFrame(rows)


def by_adm1(pop, threshold_or_draws=None, n: int = DEFAULT_N,
            scenario: str = P.DEFAULT_SCENARIO,
            seed: int = DEFAULT_SEED, adm1_path=None):
    """
    Per-region totals under the SHARED national draws.

    Each region is evaluated with the same n parameter vectors, so region
    totals are perfectly rank-correlated across iterations. This is the point
    of the module; see `check_additivity`.
    """
    import pandas as pd

    from .population import zonal_by_adm1

    d = draw_parameters(n, scenario, seed)
    # Regions are split at the central threshold: sampling the threshold
    # per-region as well would require re-running zonal statistics per draw
    # for negligible gain, since the national sensitivity analysis already
    # shows the split moves the answer by under 3%.
    thr = float(P.URBAN_DENSITY_THRESHOLD.value)
    z = zonal_by_adm1(pop, thr, adm1_path=adm1_path)

    rows, per_region_draws = [], {}
    for r in z.itertuples():
        out = MB.chain(r.pop_urban, r.pop_rural, **d)
        per_region_draws[r.adm1] = out["to_environment_t_yr"]
        q = np.quantile(out["to_environment_t_yr"], QUANTILES)
        rows.append({
            "adm1": r.adm1,
            "adm1_id": getattr(r, "adm1_id", None),
            "pop_total": r.pop_total,
            "urban_share": r.urban_share,
            "plastic_t_yr_p50": np.quantile(out["plastic_t_yr"], 0.5),
            "to_env_t_yr_p5": q[0],
            "to_env_t_yr_p50": q[2],
            "to_env_t_yr_p95": q[-1],
            "to_water_t_yr_p50": np.quantile(out["to_water_t_yr"], 0.5),
            "leak_kg_per_capita_p50": np.quantile(
                out["leak_kg_per_capita_yr"], 0.5),
        })

    df = pd.DataFrame(rows).sort_values("to_env_t_yr_p50", ascending=False)
    df.attrs["per_region_draws"] = per_region_draws
    df.attrs["unassigned_population"] = z.attrs["unassigned_population"]
    return df


def check_additivity(df) -> Dict:
    """
    Demonstrate that correlated intervals do not add, and quantify by how much.

    Returns the national p5/p95 obtained by summing regions WITHIN each
    iteration (correct) alongside the sum of the regional p5/p95 values
    (what a naive table would show), and the quadrature interval (what
    independence would give). The gap between them is the size of the error
    this module exists to avoid.
    """
    draws = df.attrs["per_region_draws"]
    stacked = np.vstack([np.asarray(v, float) for v in draws.values()])
    correct = stacked.sum(axis=0)

    q_correct = np.quantile(correct, [0.05, 0.5, 0.95])
    naive_p5 = float(df["to_env_t_yr_p5"].sum())
    naive_p95 = float(df["to_env_t_yr_p95"].sum())

    med = np.median(stacked, axis=1)
    half_widths = (np.quantile(stacked, 0.95, axis=1)
                   - np.quantile(stacked, 0.05, axis=1)) / 2.0
    quad = float(np.sqrt((half_widths ** 2).sum()))
    quad_p5, quad_p95 = med.sum() - quad, med.sum() + quad

    correct_w = q_correct[2] - q_correct[0]
    return {
        "national_p5_correct": q_correct[0],
        "national_p50_correct": q_correct[1],
        "national_p95_correct": q_correct[2],
        "sum_of_regional_p5": naive_p5,
        "sum_of_regional_p95": naive_p95,
        "quadrature_p5": quad_p5,
        "quadrature_p95": quad_p95,
        "correct_interval_width": correct_w,
        "quadrature_interval_width": quad_p95 - quad_p5,
        "quadrature_understates_by": (
            correct_w / (quad_p95 - quad_p5) if quad_p95 > quad_p5 else np.nan),
        "note": ("Summing regional p5/p95 happens to reproduce the correct "
                 "interval here BECAUSE the parameters are perfectly shared -- "
                 "that is the correlated case, not a coincidence. The "
                 "quadrature interval is the one that is wrong, and the ratio "
                 "above is the factor by which it understates the true "
                 "national uncertainty."),
    }


def contribution_to_variance(res: Dict, target: str = "to_environment_t_yr"):
    """
    Which parameter is the estimate actually limited by?

    Rank correlation between each input draw and the output, squared and
    normalised -- a standard global sensitivity measure for a monotone model
    (Saltelli, A. et al. (2008), Global Sensitivity Analysis: The Primer,
    Wiley). Spearman rather than Pearson because the chain is multiplicative,
    so the relationships are monotone but not linear.

    This is what tells a programme manager whether to fund a waste
    characterisation survey or a collection-coverage survey.
    """
    import pandas as pd
    from scipy.stats import spearmanr

    y = np.asarray(res[target], float)
    rows = []
    for k, v in res["_draws"].items():
        v = np.asarray(v, float)
        if np.allclose(v, v[0]):
            rho = 0.0            # fixed parameter contributes no variance
        else:
            rho = spearmanr(v, y).statistic
        rows.append({"parameter": k, "spearman_rho": rho, "rho_squared": rho ** 2})
    df = pd.DataFrame(rows)
    tot = df["rho_squared"].sum()
    df["share_of_variance"] = df["rho_squared"] / tot if tot > 0 else np.nan
    return df.sort_values("share_of_variance", ascending=False)
