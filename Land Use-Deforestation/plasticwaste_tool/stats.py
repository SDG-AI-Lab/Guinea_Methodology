"""
Summary statistics for the DPWMS Plastic Waste module: hotspot-patch
statistics, monthly time-series aggregation (for the Monthly FDI Archive and
National Overview KPI cards), and Digital Waste Bank supply-chain KPIs.

No dependency on landuse_tool: pixel-area is computed directly in
`detection.vectorize_hotspots` (via connected-component pixel counts, CRS-
aware), and this module just consumes the resulting `area_m2` column --
nothing here needs a shared area-calculation helper.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd

from . import config as pwcfg

logger = logging.getLogger(__name__)

M2_PER_HA = 10_000.0


# ============================================================================
# PATCH-LEVEL STATISTICS  (feeds GEDH-FR-0503 / 0504 detail popup)
# ============================================================================
def hotspot_summary(hotspot_gdf: gpd.GeoDataFrame) -> Dict:
    """Aggregate stats for one scene's hotspot vectors: count, total area,
    mean concentration score, distribution by confidence level."""
    if hotspot_gdf is None or hotspot_gdf.empty:
        return {"n_patches": 0, "total_area_ha": 0.0, "mean_concentration_score": np.nan}
    area_ha = hotspot_gdf["area_m2"] / M2_PER_HA
    return {
        "n_patches": int(len(hotspot_gdf)),
        "total_area_ha": round(float(area_ha.sum()), 3),
        "mean_area_ha": round(float(area_ha.mean()), 4),
        "max_area_ha": round(float(area_ha.max()), 4),
        "mean_concentration_score": round(float(hotspot_gdf["floating_debris_score"].mean()), 3),
        "by_confidence": hotspot_gdf["confidence_level"].value_counts().to_dict()
                        if "confidence_level" in hotspot_gdf.columns else {},
    }


def sar_fallback_flag(cloud_pct: Optional[float]) -> bool:
    """SAR-fallback recommendation flag (mirrors the GEE App's `sarAlert`,
    referenced in the original prototype as FR-0207): when optical cloud
    cover exceeds the configured threshold, the optical FDI product is
    unreliable for that window and a SAR-based product should be used
    instead."""
    if cloud_pct is None:
        return False
    return cloud_pct > pwcfg.DETECTION_CONFIG["sar_fallback_cloud_pct"]


# ============================================================================
# MONTHLY / TEMPORAL AGGREGATION (GEDH-FR-0507, GEDH-FR-0302 KPI card)
# ============================================================================
def monthly_time_series(monthly_hotspots: Dict[str, gpd.GeoDataFrame]) -> pd.DataFrame:
    """
    Build a month-by-month summary table from {'YYYY-MM': hotspot_gdf}, for
    the Monthly FDI Archive comparison view and the "Active Plastic Hotspots"
    / "Plastic Mass Tracked" National Overview KPI cards (GEDH-FR-0302).
    """
    rows = []
    for label in sorted(monthly_hotspots):
        gdf = monthly_hotspots[label]
        summ = hotspot_summary(gdf)
        rows.append({"year_month": label, **summ})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["month_over_month_area_change_pct"] = df["total_area_ha"].pct_change() * 100
    return df


def month_over_month_difference(gdf_a: gpd.GeoDataFrame, gdf_b: gpd.GeoDataFrame) -> Dict:
    """
    Simple two-month comparison for the "Comparer avec" overlay (US-HUB-10):
    net change in hotspot count/area between two archived months. Uses total
    area rather than per-patch matching (patches are not guaranteed to persist
    with a stable ID between months for a mesoscale, tide/current-driven
    signal like floating debris — matching individual patches month-to-month
    would overstate precision that the data doesn't support).
    """
    a, b = hotspot_summary(gdf_a), hotspot_summary(gdf_b)
    return {
        "area_change_ha": round(b["total_area_ha"] - a["total_area_ha"], 3),
        "patch_count_change": b["n_patches"] - a["n_patches"],
        "from_area_ha": a["total_area_ha"],
        "to_area_ha": b["total_area_ha"],
    }


# ============================================================================
# DIGITAL WASTE BANK KPIs (GEDH-FR-0506, role-gated view)
# ============================================================================
def waste_bank_kpis(waste_bank_gdf: gpd.GeoDataFrame,
                    collection_events_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Supply-chain KPIs for the Digital Waste Bank sub-view: total plastic
    tracked this month (kg), active collectors, registered recyclers.
    `collection_events_df` (columns: actor_id, event_date, mass_kg) comes from
    the Collector app backend, not GEE; pass None to get only the actor-count
    KPIs.
    """
    out = {
        "active_collectors": int((waste_bank_gdf["actor_type"] == "collector").sum())
                            if "actor_type" in waste_bank_gdf.columns else 0,
        "registered_recyclers": int((waste_bank_gdf["actor_type"] == "recycler").sum())
                               if "actor_type" in waste_bank_gdf.columns else 0,
    }
    if collection_events_df is not None and not collection_events_df.empty:
        this_month = pd.to_datetime(collection_events_df["event_date"]).dt.to_period("M")
        current_period = pd.Timestamp.today().to_period("M")
        out["plastic_tracked_this_month_kg"] = round(
            float(collection_events_df.loc[this_month == current_period, "mass_kg"].sum()), 1)
    else:
        out["plastic_tracked_this_month_kg"] = 0.0
    return out


# ============================================================================
# MATERIAL / TOXICITY BREAKDOWN (GEDH-FR-0504 popup — schema-only until the
# Collector-app material classifier output is wired in)
# ============================================================================
def material_distribution(material_counts: Optional[Dict[str, int]] = None) -> Dict:
    """Normalize a raw {material: count} dict into the popup schema
    (percentages + toxicity flag), defaulting to 'UNCLASSIFIED' when no
    material classifier output is available yet for a given hotspot."""
    if not material_counts:
        return {"distribution_pct": {"UNCLASSIFIED": 100.0}, "toxicity_flag": False}
    total = sum(material_counts.values()) or 1
    dist = {k: round(100.0 * v / total, 1) for k, v in material_counts.items()}
    toxic = any(k in pwcfg.TOXICITY_FLAG_MATERIALS and v > 0 for k, v in material_counts.items())
    return {"distribution_pct": dist, "toxicity_flag": toxic}