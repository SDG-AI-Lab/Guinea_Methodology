"""
Future deforestation scenarios.

Two ingredients:
  * HOW MUCH forest is lost   -> project_future_loss (extrapolate historical rate)
  * WHERE it is lost          -> allocate_future_loss (highest-risk forest pixels,
                                 using the risk map from drivers.py)

Policy levers let the analyst explore alternatives:
  * protected areas          -> mask risk to 0 (no loss allowed there)
  * new roads / frontiers    -> raise risk near a new accessibility layer
  * governance / rate factor -> scale the projected demand up or down

This mirrors the Land Use module's suitability+allocation design (risk == the
"suitability" of a forest pixel to be cleared), so the same mental model and, if
desired, prediction.run_allocation_simulation can be reused.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HOW MUCH: project future annual loss
# ---------------------------------------------------------------------------
def project_future_loss(loss_ha_by_year, future_years, method="mean_rate",
                        forest_ha_by_year=None):
    """
    Project annual loss (ha) for `future_years`.
      * 'mean_rate' : mean of recent annual loss (ha).
      * 'linear'    : linear trend of annual loss extrapolated forward.
      * 'pct'       : constant % of standing forest (needs forest_ha_by_year).
    Returns {year: projected_loss_ha}.
    """
    years = sorted(loss_ha_by_year)
    vals = np.array([loss_ha_by_year[y] for y in years], dtype="float64")
    if method == "linear" and len(years) >= 2:
        x = np.array(years, dtype="float64")
        a, b = np.polyfit(x, vals, 1)
        return {int(y): max(0.0, float(a * y + b)) for y in future_years}
    if method == "pct" and forest_ha_by_year:
        # average historical %/yr applied to the last known forest area
        fy = sorted(forest_ha_by_year)
        pct = np.mean([loss_ha_by_year[y] / forest_ha_by_year[y - 1]
                       for y in years if (y - 1) in forest_ha_by_year
                       and forest_ha_by_year[y - 1] > 0])
        standing = forest_ha_by_year[fy[-1]]
        out = {}
        for y in future_years:
            loss = pct * standing
            out[int(y)] = float(loss)
            standing -= loss
        return out
    # default: mean of the most recent up-to-5 years
    recent = vals[-5:]
    mean_loss = float(recent.mean()) if recent.size else 0.0
    return {int(y): mean_loss for y in future_years}


# ---------------------------------------------------------------------------
# Policy levers on the risk map
# ---------------------------------------------------------------------------
def apply_policy(risk_map, protected_mask=None, new_frontier_mask=None,
                 frontier_boost=0.3, profile=None):
    """
    Adjust the risk map for a scenario:
      * protected_mask     -> risk set to 0 (deforestation prevented).
      * new_frontier_mask  -> risk raised near a new road/settlement (proximity
                              decay), capped at 1.
    """
    risk = np.array(risk_map, dtype="float32", copy=True)
    if protected_mask is not None:
        risk[np.asarray(protected_mask, dtype=bool)] = 0.0
    if new_frontier_mask is not None:
        from .drivers import distance_to
        d = distance_to(new_frontier_mask, profile)
        # exponential decay boost within ~5 km
        boost = frontier_boost * np.exp(-d / 5000.0)
        risk = np.clip(risk + np.nan_to_num(boost), 0.0, 1.0)
    return risk


# ---------------------------------------------------------------------------
# WHERE: allocate projected loss to the highest-risk forest pixels
# ---------------------------------------------------------------------------
def allocate_future_loss(risk_map, forest_mask, demand_ha, pixel_area_ha,
                         rate_factor=1.0, protected_mask=None):
    """
    Clear the projected number of pixels from the highest-risk standing forest.

    Returns (future_forest_mask, cleared_mask). demand_ha is total ha to clear
    over the projection horizon; rate_factor scales it (e.g. 0.5 under strong
    governance, 1.5 under accelerated expansion).

    protected_mask: HARD exclusion from the candidate pool, not just score-based
    deprioritization. Confirmed necessary: apply_policy() setting risk=0.0 inside
    protected_mask is a FINITE value, so those pixels remained in the candidate
    pool via `np.isfinite(risk)` and could still be selected once positive-risk
    candidates ran out -- observed directly (Guinea pilot): the protected-area
    scenario allocated MORE clearing inside protected zones (808,182 ha) than
    business-as-usual (720,398 ha), the opposite of what protection should do,
    because zeroing an area's risk enlarges the "tied at the bottom" pool that
    gets swept in once demand exceeds available positive-risk forest elsewhere.
    If demand still exceeds capacity outside protected_mask even with the hard
    exclusion, this now reports that explicitly (a real, defensible finding --
    "current demand exceeds what protection alone can contain") rather than
    silently allocating into protected areas anyway.
    """
    risk = np.asarray(risk_map, dtype="float32")
    forest = np.asarray(forest_mask, dtype=bool)
    per_px = float(np.mean(pixel_area_ha)) if not np.isscalar(pixel_area_ha) else float(pixel_area_ha)
    n_clear = int(round((demand_ha * rate_factor) / max(per_px, 1e-9)))

    eligible = forest & np.isfinite(risk)
    if protected_mask is not None:
        eligible = eligible & ~np.asarray(protected_mask, dtype=bool)

    candidates = np.argwhere(eligible)
    if n_clear <= 0 or candidates.size == 0:
        return forest.copy(), np.zeros_like(forest)

    if protected_mask is not None and n_clear > len(candidates):
        logger.warning("Demand (%d px) exceeds available non-protected forest (%d px) -- "
                       "capping at capacity rather than allocating into protected areas. "
                       "This means projected demand exceeds what full protection-area "
                       "enforcement alone can contain; report this explicitly.",
                       n_clear, len(candidates))
        n_clear = len(candidates)

    scores = risk[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(scores)[::-1][:n_clear]        # highest risk first
    cleared = np.zeros_like(forest)
    sel = candidates[order]
    cleared[sel[:, 0], sel[:, 1]] = True
    future_forest = forest & ~cleared
    logger.info("Allocated %d cleared pixels (~%.0f ha)", cleared.sum(), cleared.sum() * per_px)
    return future_forest, cleared


def run_scenario(risk_map, forest_mask, loss_ha_by_year, future_years,
                 pixel_area_ha, forest_ha_by_year=None, method="mean_rate",
                 rate_factor=1.0, protected_mask=None, new_frontier_mask=None,
                 profile=None):
    """
    End-to-end scenario: project demand, apply policy to risk, allocate loss.
    Returns dict with projected_loss (per year), total demand, future_forest and
    cleared masks.
    """
    projected = project_future_loss(loss_ha_by_year, future_years, method,
                                    forest_ha_by_year)
    total_demand = float(sum(projected.values()))
    adj_risk = apply_policy(risk_map, protected_mask, new_frontier_mask, profile=profile)
    future_forest, cleared = allocate_future_loss(
        adj_risk, forest_mask, total_demand, pixel_area_ha, rate_factor,
        protected_mask=protected_mask)
    return {
        "projected_loss_ha": projected,
        "total_demand_ha": round(total_demand * rate_factor, 2),
        "future_forest_mask": future_forest,
        "cleared_mask": cleared,
        "adjusted_risk": adj_risk,
    }