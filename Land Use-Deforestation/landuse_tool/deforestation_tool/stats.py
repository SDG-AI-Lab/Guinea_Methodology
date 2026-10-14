"""
Deforestation statistics & time series (the numbers behind the graphs).

All areas are returned in hectares (1 px area depends on the grid). For a
geographic grid (EPSG:4326) pixel area varies with latitude, so we compute it
from the raster transform when a rasterio profile is supplied; otherwise we fall
back to a nominal resolution in metres.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

M2_PER_HA = 10_000.0


# ---------------------------------------------------------------------------
# Pixel area
# ---------------------------------------------------------------------------
def pixel_area_ha(profile=None, resolution_m=None, shape=None, latitude=None):
    """
    Per-pixel area in hectares.
    - Projected CRS (metres): constant = (xres*yres)/1e4.
    - Geographic CRS (degrees): approximate with cos(latitude) correction.
    Returns either a scalar (metres CRS / nominal) or a 2D array (geographic).
    """
    if profile is not None:
        t = profile["transform"]
        xres, yres = abs(t.a), abs(t.e)
        crs = profile.get("crs")
        is_geographic = bool(crs and getattr(crs, "is_geographic", False))
        if not is_geographic:
            return (xres * yres) / M2_PER_HA
        # geographic: degrees -> metres, latitude-dependent
        h, w = shape or (profile["height"], profile["width"])
        rows = np.arange(h)
        lat0 = t.f  # top
        lats = lat0 + (rows + 0.5) * t.e  # t.e is negative
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lats))
        col_area = (xres * m_per_deg_lon) * (yres * m_per_deg_lat) / M2_PER_HA
        return np.repeat(col_area[:, None], w, axis=1)
    # nominal fallback
    r = resolution_m or 100.0
    return (r * r) / M2_PER_HA


def _area_of_mask(mask, area):
    mask = np.asarray(mask, dtype=bool)
    if np.isscalar(area):
        return float(mask.sum() * area)
    return float(np.asarray(area)[mask].sum())


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------
def annual_deforestation_area(loss_by_year, profile=None, resolution_m=None):
    """{year: hectares lost} from a {year: boolean loss mask} dict."""
    area = pixel_area_ha(profile, resolution_m,
                         shape=next(iter(loss_by_year.values())).shape
                         if loss_by_year else None)
    return {int(y): round(_area_of_mask(m, area), 2) for y, m in sorted(loss_by_year.items())}


def forest_cover_trajectory(forest_by_year, profile=None, resolution_m=None):
    """{year: forest hectares} from a {year: boolean forest mask} dict."""
    area = pixel_area_ha(profile, resolution_m,
                         shape=next(iter(forest_by_year.values())).shape
                         if forest_by_year else None)
    return {int(y): round(_area_of_mask(m, area), 2) for y, m in sorted(forest_by_year.items())}


def deforestation_rate(forest_ha_by_year):
    """
    Annual deforestation rate (% of previous year's forest lost).
    Returns {year: percent}. Uses the forest-cover trajectory.

    None, not 0.0, when the previous year's forest is 0 or missing, or this
    year's is missing: no rate was measured, and a 0.0 would read as "no
    loss" (feature 005, doc 70 K24).
    """
    years = sorted(forest_ha_by_year)
    rates = {}
    for prev, cur in zip(years, years[1:]):
        f0, f1 = forest_ha_by_year[prev], forest_ha_by_year[cur]
        rates[cur] = round(100.0 * (f0 - f1) / f0, 3) if f0 and f1 is not None else None
    return rates


def _land(datamask):
    """Hansen `datamask` land pixels (1 = land, 2 = water, 0 = no data)."""
    return np.asarray(datamask) == 1


def _share(numerator, denominator):
    """100 x numerator / denominator; None for a missing or zero denominator."""
    if numerator is None or not denominator:
        return None
    return 100.0 * numerator / denominator


def summary_table(loss_by_year, forest_by_year, profile=None, resolution_m=None, datamask=None):
    """Tidy DataFrame: year, forest_ha, loss_ha, rate_pct, roi_land_ha,
    forest_pct (ready to plot/export).

    `datamask` is Hansen's band on the analysis grid; with it, `roi_land_ha`
    is the ROI's land area and `forest_pct = 100 x forest_ha / roi_land_ha`
    (contracts/toolchain.md §4). Without it both are None.
    """
    loss = annual_deforestation_area(loss_by_year, profile, resolution_m)
    forest = forest_cover_trajectory(forest_by_year, profile, resolution_m)
    rate = deforestation_rate(forest)
    land = None
    if datamask is not None:
        area = pixel_area_ha(profile, resolution_m, shape=np.asarray(datamask).shape)
        land = round(_area_of_mask(_land(datamask), area), 2)
    years = sorted(set(loss) | set(forest))
    return pd.DataFrame([{
        "year": y,
        "forest_ha": forest.get(y),
        "loss_ha": loss.get(y),
        "rate_pct": rate.get(y),
        "roi_land_ha": land,
        "forest_pct": _share(forest.get(y), land),
    } for y in years])


# ---------------------------------------------------------------------------
# Forest -> what?  conversion matrix
# ---------------------------------------------------------------------------
def conversion_matrix(deforested_mask, lulc_t2, class_names=None,
                      profile=None, resolution_m=None):
    """
    Area (ha) of forest converted to each destination LULC class at t2.
    Returns a one-row-per-class DataFrame sorted by area.
    """
    area = pixel_area_ha(profile, resolution_m, shape=np.asarray(lulc_t2).shape)
    m = np.asarray(deforested_mask, dtype=bool)
    dest = np.asarray(lulc_t2)[m]
    rows = []
    for cls in np.unique(dest):
        cls_mask = np.zeros_like(np.asarray(lulc_t2), dtype=bool)
        cls_mask[m] = (dest == cls)
        name = (class_names or {}).get(int(cls), f"Class {int(cls)}")
        rows.append({"to_class": int(cls), "to_name": name,
                     "area_ha": round(_area_of_mask(cls_mask, area), 2)})
    df = pd.DataFrame(rows).sort_values("area_ha", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Vegetation-index trend (robust)
# ---------------------------------------------------------------------------
def vi_trend(vi_by_year, mask=None):
    """
    Per-pixel linear NDVI trend (slope per year) via least squares, plus the
    area-mean slope. Negative = greening loss / degradation.

    vi_by_year: {year: 2D NDVI array}. Returns (slope_map, mean_slope).
    """
    years = np.array(sorted(vi_by_year), dtype="float64")
    stack = np.stack([np.asarray(vi_by_year[int(y)], dtype="float64") for y in years])
    n = len(years)
    if n < 2:
        return np.zeros_like(stack[0]), 0.0
    x = years - years.mean()
    denom = float((x ** 2).sum())
    # slope = sum(x*(y-ybar)) / sum(x^2), vectorized over pixels
    ymean = stack.mean(axis=0)
    slope = np.tensordot(x, stack - ymean, axes=(0, 0)) / denom
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        mean_slope = float(np.nanmean(slope[m])) if m.any() else 0.0
    else:
        mean_slope = float(np.nanmean(slope))
    return slope.astype("float32"), round(mean_slope, 5)


# ---------------------------------------------------------------------------
# Zonal breakdown (per administrative unit)
# ---------------------------------------------------------------------------
def zonal_deforestation(loss_mask, zones_gdf, profile, name_field="shapeName",
                        metric_name="loss_ha"):
    """
    Hectares of a boolean mask per administrative zone. Rasterizes each zone to
    the grid.
    zones_gdf: GeoDataFrame of admin polygons (e.g. GeoBoundaries ADM1/ADM2).

    `metric_name` names the output column. The function measures whatever mask
    it is handed, so the same call computes forest extent (`forest_ha`) as well
    as loss; it is called once per metric because one call returns one mask's
    area.
    """
    from rasterio.features import rasterize
    area = pixel_area_ha(profile, shape=np.asarray(loss_mask).shape)
    zones = zones_gdf.to_crs(profile["crs"]) if zones_gdf.crs != profile["crs"] else zones_gdf
    m = np.asarray(loss_mask, dtype=bool)
    rows = []
    for i, row in zones.reset_index(drop=True).iterrows():
        zmask = rasterize([(row.geometry, 1)], out_shape=m.shape,
                          transform=profile["transform"], fill=0, dtype="uint8").astype(bool)
        rows.append({"zone": row.get(name_field, f"zone_{i}"),
                     metric_name: round(_area_of_mask(m & zmask, area), 2)})
    return pd.DataFrame(rows).sort_values(metric_name, ascending=False).reset_index(drop=True)


def zone_land_area(datamask, zones_gdf, profile, name_field="shapeName"):
    """{zone: land hectares}: the zone's Hansen `datamask` = 1 pixels on the
    analysis grid (contracts/toolchain.md §3). The same grid as the forest
    and loss areas, so a share never mixes a raster area with a vector one."""
    table = zonal_deforestation(_land(datamask), zones_gdf, profile, name_field=name_field,
                                metric_name="zone_land_ha")
    return {row["zone"]: row["zone_land_ha"] for row in table.to_dict(orient="records")}


def zonal_shares(per_zone_year, land_ha_by_zone):
    """Per zone and year, `zone_land_ha`, `forest_pct` and `loss_pct`
    (contracts/toolchain.md §3).

    per_zone_year: {zone: {year: {"forest_ha": ha | None, "loss_ha": ha | None}}}
    land_ha_by_zone: {zone: ha} from `zone_land_area`.

    forest_pct = 100 x forest_ha / zone_land_ha;
    loss_pct = 100 x loss_ha(year) / forest_ha(year - 1), the definition of the
    national `rate_pct`. A missing or zero denominator gives None, and so does
    a missing year before: the forest of an earlier year is not used.
    """
    shares = {}
    for zone, by_year in per_zone_year.items():
        land = land_ha_by_zone.get(zone)
        shares[zone] = {}
        for year, metrics in by_year.items():
            previous = (by_year.get(year - 1) or {}).get("forest_ha")
            shares[zone][year] = {
                "zone_land_ha": land,
                "forest_pct": _share(metrics.get("forest_ha"), land),
                "loss_pct": _share(metrics.get("loss_ha"), previous),
            }
    return shares


# ---------------------------------------------------------------------------
# Prefecture risk summary (feature 005, contracts/toolchain.md §1)
# ---------------------------------------------------------------------------
def risk_zone_summary(risk, forest_mask, zones_gdf, profile, id_field="shapeID"):
    """
    Per zone, the risk surface summarised over its forest:

      zone_id           the zone's `id_field` (shapeID)
      forest_ha         area of `forest_mask` in the zone
      scored_forest_ha  area of forest with a finite risk value
      mean_p            sum(risk x pixel area) / sum(pixel area) over the scored
                        forest pixels; NaN when none is scored
      scored_share      scored_forest_ha / forest_ha; NaN when forest_ha = 0

    `risk` is what drivers.predict_risk_map returns (NaN outside scored
    forest); `forest_mask` is the mask it scored (pipeline `forest_now`). No
    threshold is applied (doc 70 S41). Areas are the latitude-corrected pixel
    areas of `pixel_area_ha`.
    """
    from rasterio.features import rasterize
    risk = np.asarray(risk, dtype="float64")
    forest = np.asarray(forest_mask, dtype=bool)
    area = pixel_area_ha(profile, shape=forest.shape)
    area = np.broadcast_to(np.asarray(area, dtype="float64"), forest.shape)
    zones = zones_gdf.to_crs(profile["crs"]) if zones_gdf.crs != profile["crs"] else zones_gdf
    scored = forest & np.isfinite(risk)
    rows = []
    for i, row in zones.reset_index(drop=True).iterrows():
        zmask = rasterize([(row.geometry, 1)], out_shape=forest.shape,
                          transform=profile["transform"], fill=0, dtype="uint8").astype(bool)
        forest_ha = float(area[forest & zmask].sum())
        in_zone = scored & zmask
        scored_ha = float(area[in_zone].sum())
        mean_p = float((risk[in_zone] * area[in_zone]).sum() / scored_ha) if scored_ha > 0 else float("nan")
        rows.append({
            "zone_id": row.get(id_field, f"zone_{i}"),
            "forest_ha": forest_ha,
            "scored_forest_ha": scored_ha,
            "mean_p": mean_p,
            "scored_share": scored_ha / forest_ha if forest_ha > 0 else float("nan"),
        })
    return pd.DataFrame(rows, columns=["zone_id", "forest_ha", "scored_forest_ha", "mean_p", "scored_share"])
