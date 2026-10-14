"""
Deployment-readiness helpers for the DPWMS mesoscale FDI layer.

These address the four gaps between "the detection works on Conakry" and "this
layer can be published nationally under the GEDH SRS":

  1. `port_exclusion_zones` / `flag_port_patches`
     The dominant irreducible false-positive class. Quays, jetties and moored
     vessels sit ON water, so every water product classifies them correctly as
     water and no water mask removes them. Six of twenty confirmed negatives in
     the Conakry review were this class, and it grows with AOI extent.

  2. `water_fraction_for_bbox`
     Lets a candidate AOI be tested BEFORE a full run. Fatala's configured box
     contained 2.2% water, which inflated its mean+3*sigma threshold to 3.5x
     Conakry's and produced an uninformative zero. Checking the water fraction
     of a proposed box costs one cheap reduceRegion.

  3. `sample_for_review`
     Stratified review sampling for regions too large to enumerate. The Conakry
     review is a full enumeration of its patches (26 in January 2024); a
     region with hundreds of patches is sampled instead.

  4. `attribute_admin`
     The SRS Custom Report Builder filters by prefecture, and the National
     Overview reports a hotspot count per area. Patches carry no administrative
     attribution until this is applied.

All four operate on the vectorized patch layer, offline, so they are testable
without Earth Engine except where noted.
"""
# ---------------------------------------------------------------------------
# MOVED HERE from plasticwaste_tool/deployment.py by 24_build_component_packages.py.
# This file is now CANONICAL and is hand-maintained: the old top-level name is
# a re-export shim pointing back at it, not a second copy. Edit this file.
# ---------------------------------------------------------------------------

import logging
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import config as pwcfg

logger = logging.getLogger(__name__)

__all__ = ["port_exclusion_zones", "flag_port_patches", "water_fraction_for_bbox",
           "sample_for_review", "attribute_admin", "fixed_threshold_series",
           "check_aoi_overlaps", "deduplicate_mosaic"]


# ============================================================================
# 0. AOI HYGIENE  (overlapping regions double-count the national KPI)
# ============================================================================
def check_aoi_overlaps(region_keys: Optional[List[str]] = None) -> List[Dict]:
    """
    Report pairs of regional AOIs whose bounding boxes overlap.

    Regions are detected independently and then mosaicked, so any patch inside
    two boxes is produced twice. The SRS makes "Active Plastic Hotspots (count)"
    a headline national indicator, and a double-counted patch corrupts it
    directly. Measured on the first national run, conakry and mangrove overlapped
    by roughly 980 km2 and duplicated 8 patches.

    Returns one entry per overlapping pair; an empty list means the AOIs are
    clean.
    """
    from shapely.geometry import box

    keys = region_keys or [k for k in pwcfg.AOI_CONFIG if k != "full"]
    out = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ga = box(*pwcfg.AOI_CONFIG[a]["bbox"])
            gb = box(*pwcfg.AOI_CONFIG[b]["bbox"])
            inter = ga.intersection(gb)
            if not inter.is_empty and inter.area > 0:
                out.append({"region_a": a, "region_b": b,
                            "overlap_deg2": round(inter.area, 5),
                            "overlap_pct_of_a": round(100 * inter.area / ga.area, 1),
                            "overlap_pct_of_b": round(100 * inter.area / gb.area, 1)})
    if out:
        logger.warning(
            "%d overlapping AOI pair(s): %s. Patches in the shared area are detected "
            "once per region and inflate the national count. deduplicate_mosaic() "
            "removes them, but non-overlapping boxes are the correct fix.",
            len(out), ", ".join(f"{o['region_a']}x{o['region_b']}" for o in out))
    return out


def deduplicate_mosaic(mosaic: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, Dict]:
    """
    Remove patches that the same feature produced in two overlapping regions.

    Assignment rule: a duplicated patch is kept for the region whose AOI CENTRE
    is nearest its centroid. Deterministic, and it keeps the detection made by
    the region that covers that location most centrally — which is also the
    region whose water statistics are most representative of it. Arbitrary
    rules like "first region wins" would make the national count depend on
    dictionary ordering.

    Returns (deduplicated, report).
    """
    from shapely.geometry import box

    if mosaic is None or mosaic.empty or "region" not in mosaic.columns:
        return mosaic, {"n_before": 0, "n_removed": 0, "n_after": 0}

    n_before = len(mosaic)
    gdf = mosaic.reset_index(drop=True).copy()

    # Centroids and the AOI-centre distances are computed in a projected CRS.
    # Taking them in degrees would warn and, more importantly, would compare
    # distances on an anisotropic grid -- a degree of longitude is ~1.6% shorter
    # than a degree of latitude at 10 deg N, which can flip a near-tie.
    utm = gdf.estimate_utm_crs()
    proj = gdf.to_crs(utm)
    cent = proj.geometry.centroid
    gdf["_cx"], gdf["_cy"] = cent.x.values, cent.y.values

    centre_gdf = gpd.GeoDataFrame(
        {"region": [k for k in pwcfg.AOI_CONFIG if k != "full"]},
        geometry=[box(*v["bbox"]).centroid
                  for k, v in pwcfg.AOI_CONFIG.items() if k != "full"],
        crs="EPSG:4326").to_crs(utm)
    centres = dict(zip(centre_gdf["region"], centre_gdf.geometry))

    # Group spatially coincident patches across DIFFERENT regions.
    joined = gpd.sjoin(gdf[["geometry", "region"]], gdf[["geometry", "region"]],
                       how="inner", predicate="intersects")
    cross = joined[joined["region_left"] != joined["region_right"]]

    drop = set()
    for left_idx, right_idx in zip(cross.index, cross["index_right"]):
        if left_idx in drop or right_idx in drop:
            continue
        best = None
        for idx in (left_idx, right_idx):
            r = gdf.at[idx, "region"]
            c = centres.get(r)
            if c is None:
                continue
            d = ((gdf.at[idx, "_cx"] - c.x) ** 2 + (gdf.at[idx, "_cy"] - c.y) ** 2) ** 0.5
            if best is None or d < best[1]:
                best = (idx, d)
        if best is not None:
            drop.add(right_idx if best[0] == left_idx else left_idx)

    out = gdf.drop(index=list(drop)).drop(columns=["_cx", "_cy"]).reset_index(drop=True)
    report = {"n_before": n_before, "n_removed": len(drop), "n_after": len(out)}
    if drop:
        logger.warning(
            "Removed %d duplicate patch(es) produced by overlapping AOIs "
            "(%d -> %d). Fix the AOI bounding boxes to stop this recurring.",
            len(drop), n_before, len(out))
    return gpd.GeoDataFrame(out, crs=mosaic.crs), report


# ============================================================================
# 1. PORT / HARBOUR EXCLUSION
# ============================================================================
def port_exclusion_zones(aoi_gdf: gpd.GeoDataFrame,
                         buffer_m: Optional[float] = None,
                         output_filename: str = "pw_port_zones.geojson"
                         ) -> Optional[gpd.GeoDataFrame]:
    """
    Port, harbour and over-water structure polygons from OpenStreetMap, buffered.

    OSM rather than a hand-drawn polygon for three reasons: it is reproducible by
    anyone re-running the pipeline, it updates as Guinea's ports are mapped, and
    it extends to Kamsar and Benty without new manual work. The trade-off is OSM
    coverage, which is uneven in Guinea -- `flag_port_patches` therefore reports
    how many patches it caught so that poor coverage is visible rather than
    silently producing an empty exclusion.

    Returns a single-column GeoDataFrame of buffered exclusion geometries in the
    input CRS, or None if OSM returned nothing.
    """
    from .. import api_connector as api

    buffer_m = pwcfg.DETECTION_CONFIG["port_buffer_m"] if buffer_m is None else buffer_m
    tags = pwcfg.DETECTION_CONFIG["port_osm_tags"]

    path = api.landuse_api.fetch_osm_vectors(aoi_gdf, tags, output_filename)
    if not path:
        logger.warning(
            "OSM returned no port/harbour features for this AOI. Guinea's OSM coverage "
            "is uneven; if the AOI contains a known port, draw a manual polygon and pass "
            "it to flag_port_patches directly rather than assuming there is nothing to "
            "exclude.")
        return None

    gdf = gpd.read_file(path)
    if gdf.empty:
        logger.warning("OSM port layer is empty for this AOI.")
        return None

    # Buffer in metres, which requires a projected CRS. estimate_utm_crs picks the
    # right UTM zone for the AOI rather than assuming one.
    utm = gdf.estimate_utm_crs()
    buffered = gdf.to_crs(utm).buffer(buffer_m)
    out = gpd.GeoDataFrame(geometry=buffered, crs=utm).dissolve().to_crs(gdf.crs)
    logger.info("Port exclusion: %d OSM feature(s) -> %.2f km2 buffered at %.0f m",
                len(gdf), out.to_crs(utm).area.sum() / 1e6, buffer_m)
    return out


def flag_port_patches(hotspots: gpd.GeoDataFrame,
                      port_zones: Optional[gpd.GeoDataFrame],
                      drop: Optional[bool] = None) -> Tuple[gpd.GeoDataFrame, Dict]:
    """
    Tag each patch with `in_port_zone`, optionally dropping the flagged ones.

    FLAGS by default rather than deleting. A published layer that silently drops
    detections cannot be audited, and the operator loses the ability to see
    whether the exclusion is doing something sensible or swallowing real debris
    in a working harbour. Set `drop=True` (or PW_PORT_DROP=true) once the
    exclusion has been reviewed on a real scene.

    Returns (gdf, report).
    """
    drop = pwcfg.DETECTION_CONFIG["port_drop_flagged"] if drop is None else drop
    gdf = hotspots.copy()

    if port_zones is None or port_zones.empty:
        gdf["in_port_zone"] = False
        return gdf, {"n_patches": int(len(gdf)), "n_in_port": 0,
                     "pct_in_port": 0.0, "dropped": False,
                     "note": "No port zones supplied — nothing flagged."}

    zones = port_zones.to_crs(gdf.crs)
    union = zones.geometry.union_all() if hasattr(zones.geometry, "union_all") \
        else zones.geometry.unary_union
    gdf["in_port_zone"] = gdf.geometry.intersects(union)

    n_in = int(gdf["in_port_zone"].sum())
    report = {
        "n_patches": int(len(gdf)),
        "n_in_port": n_in,
        "pct_in_port": round(100.0 * n_in / len(gdf), 1) if len(gdf) else 0.0,
        "buffer_m": pwcfg.DETECTION_CONFIG["port_buffer_m"],
        "dropped": bool(drop),
    }
    if drop:
        gdf = gdf[~gdf["in_port_zone"]].reset_index(drop=True)
    return gdf, report


# ============================================================================
# 2. AOI SANITY CHECK  (the Fatala problem, before the run rather than after)
# ============================================================================
def water_fraction_for_bbox(bbox: Tuple[float, float, float, float],
                            scale_m: Optional[int] = None) -> Dict:
    """
    Water share of a candidate bounding box, and the thresholds it would imply.

    Run this BEFORE committing an AOI. Fatala's configured box
    (-14.20, 10.10, -13.60, 10.60) contains 2.2% water: the mean+3*sigma
    thresholds computed on that small, heterogeneous population came out 3.5x
    (FDI) and 2.7x (NDVI) the Conakry values, and the region detected nothing.
    A nil return from an anomaly detector is only informative when the
    background population is large enough for the threshold to be meaningful.

    Costs one reduceRegion over the water mask — far cheaper than a full run.
    Requires an initialized Earth Engine session.
    """
    from .. import api_connector as api

    if api._EE is None:
        api.initialize_gee()
    ee = api._EE
    scale_m = scale_m or pwcfg.S2_CONFIG["download_scale_m"]

    west, south, east, north = bbox
    region = ee.Geometry.Rectangle([west, south, east, north])
    water = api._surface_water_mask()

    stats = water.rename("w").reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
        geometry=region, scale=scale_m, maxPixels=1e10, bestEffort=True).getInfo()

    frac = stats.get("w_mean")
    n_total = stats.get("w_count")
    water_px = int(frac * n_total) if (frac is not None and n_total) else 0
    minimum = pwcfg.DETECTION_CONFIG["min_water_fraction"]

    out = {
        "bbox": bbox,
        "water_fraction": round(frac, 4) if frac is not None else None,
        "water_px_approx": water_px,
        "water_km2_approx": round(water_px * scale_m ** 2 / 1e6, 1),
        "min_water_fraction": minimum,
        "usable": bool(frac is not None and frac >= minimum),
    }
    out["verdict"] = (
        f"USABLE — {frac:.1%} water clears the {minimum:.0%} minimum."
        if out["usable"] else
        f"TOO LITTLE WATER — {(frac or 0):.1%} against a {minimum:.0%} minimum. The "
        f"anomaly threshold would be derived from a small, likely heterogeneous "
        f"population and a nil return would be uninformative. Move the box over the "
        f"water body of interest.")
    return out


# ============================================================================
# 3. REVIEW SAMPLING  (the mangrove problem)
# ============================================================================
def sample_for_review(hotspots: gpd.GeoDataFrame, n_per_band: Optional[int] = 20,
                      region: Optional[str] = None,
                      random_state: int = 42) -> gpd.GeoDataFrame:
    """
    Stratified sample of patches for photo-interpretation, balanced by confidence.

    The Conakry review enumerates all of its patches (26 in January 2024), which
    removes sampling uncertainty entirely and is the right choice when the
    count allows it. A region with hundreds of patches is not practical to
    review by eye, so a stratified sample is used instead and the resulting
    precision carries a sampling interval that a full enumeration does not.

    Stratify by confidence band rather than sampling uniformly: the bands are the
    thing being validated, and HIGH is much rarer than MEDIUM, so a uniform
    sample would leave HIGH almost unmeasured.

    `n_per_band=None` enumerates everything.
    """
    gdf = hotspots if region is None else hotspots[hotspots.get("region") == region]
    if gdf.empty:
        logger.warning("No patches to sample%s.", f" for region {region}" if region else "")
        return gdf

    if n_per_band is None:
        return gdf.reset_index(drop=True)

    parts = []
    for band, sub in gdf.groupby("confidence_level"):
        take = min(n_per_band, len(sub))
        parts.append(sub.sample(n=take, random_state=random_state))
        if take < n_per_band:
            logger.info("Band %s has only %d patches; taking all.", band, len(sub))
    out = pd.concat(parts).sample(frac=1.0, random_state=random_state)
    return gpd.GeoDataFrame(out, crs=gdf.crs).reset_index(drop=True)


# ============================================================================
# 4. ADMINISTRATIVE ATTRIBUTION  (SRS: prefecture filters, per-area KPIs)
# ============================================================================
def fetch_admin_units(level: str = "ADM2") -> Optional[gpd.GeoDataFrame]:
    """
    Guinea administrative units at `level` (ADM1 regions, ADM2 prefectures).

    `landuse_tool.get_admin_boundary()` defaults to ADM0 — the national
    outline, one feature named "Guinea". That is correct for clipping a
    country-scale raster and useless for the SRS report filters, which select
    *prefectures*, so this asks for the level actually needed.

    Read from the pinned geoBoundaries files the hub loaded (feature 002,
    research F2), never downloaded: rows carry the `shapeID` every stored
    reference joins on.
    """
    from landuse_tool import api_connector as landuse_api

    try:
        gdf = landuse_api.get_admin_boundary(level=level)
    except ValueError as e:
        logger.error("Admin units %s: %s", level, e)
        return None
    if gdf is None:
        return None
    logger.info("Loaded %d %s unit(s) for Guinea.", len(gdf), level)
    return gdf.to_crs("EPSG:4326")


def attribute_admin(hotspots: gpd.GeoDataFrame,
                    admin: Optional[gpd.GeoDataFrame] = None,
                    name_field: Optional[str] = None,
                    level: str = "ADM2",
                    max_distance_km: float = 25.0) -> gpd.GeoDataFrame:
    """
    Attach the administrative unit each patch belongs to, for SRS report filters.

    Two things a naive implementation gets wrong, both measured on the first
    national run:

    * LEVEL. `get_admin_boundary()` returns ADM0 — every patch attributed to
      "Guinea", which no prefecture filter can use. This defaults to ADM2.
    * PREDICATE. A strict within-polygon join attributed only 36 of 526 patches,
      because these are MARINE detections and 93% of them sit offshore of every
      land polygon. A hotspot two kilometres off Boffa belongs to Boffa for
      reporting purposes. This therefore joins to the NEAREST unit within
      `max_distance_km`, and records that distance so a reviewer can see which
      attributions are direct and which are proximity-based.

    Adds `admin_name` and `admin_distance_km` (0.0 when the patch is inside the
    unit). Patches beyond the cutoff keep a null name rather than being forced
    onto a distant prefecture.
    """
    if admin is None:
        admin = fetch_admin_units(level)
    if admin is None or admin.empty:
        logger.warning("No administrative units available; admin_name left null.")
        out = hotspots.copy()
        out["admin_name"], out["admin_distance_km"] = None, None
        return out

    if name_field is None:
        for cand in ("shapeName", "ADM2_FR", "ADM1_FR", "NAME_2", "NAME_1", "name"):
            if cand in admin.columns:
                name_field = cand
                break
    if name_field is None:
        logger.warning("No name field in the admin layer (columns: %s).", list(admin.columns))
        out = hotspots.copy()
        out["admin_name"], out["admin_distance_km"] = None, None
        return out

    # Nearest-neighbour distances must be metric, so both sides go to UTM.
    utm = hotspots.estimate_utm_crs()
    keep = [name_field, "geometry"] + (["shapeID"] if "shapeID" in admin.columns else [])
    adm = admin.to_crs(utm)[keep].rename(
        columns={name_field: "admin_name", "shapeID": "admin_id"})

    pts = hotspots.to_crs(utm).copy()
    pts["geometry"] = pts.geometry.centroid
    pts = pts[["geometry"]]

    joined = gpd.sjoin_nearest(pts, adm, how="left", distance_col="_dist_m",
                               max_distance=max_distance_km * 1000)
    joined = joined[~joined.index.duplicated(keep="first")].reindex(pts.index)

    out = hotspots.copy()
    out["admin_name"] = joined["admin_name"].values
    # The unit's shapeID: the join key (FR-007); admin_name is for display.
    out["admin_id"] = joined["admin_id"].values if "admin_id" in joined else None
    out["admin_distance_km"] = (joined["_dist_m"].values / 1000).round(2)

    n_named = int(out["admin_name"].notna().sum())
    n_inside = int((out["admin_distance_km"] == 0).sum())
    logger.info("Admin attribution (%s): %d/%d patches named, %d inside a unit, "
                "%d attributed by proximity within %.0f km.",
                level, n_named, len(out), n_inside, n_named - n_inside, max_distance_km)
    return out


# ============================================================================
# 5. COMPARABLE MONTHLY SERIES  (SRS: Monthly FDI Archive slider, national KPI)
# ============================================================================
def fixed_threshold_series(monthly_masked_fdi: Dict[str, np.ndarray],
                           monthly_ndvi: Dict[str, np.ndarray],
                           monthly_water: Dict[str, np.ndarray],
                           t_fdi: Optional[float] = None,
                           t_ndvi: Optional[float] = None) -> pd.DataFrame:
    """
    Re-detect every month under ONE fixed threshold pair, for a like-for-like series.

    The per-scene mean+3*sigma threshold is correct for detection but makes the
    monthly counts non-comparable: Conakry's 44 -> 0 -> 26 patches (November
    2023 to January 2024) came from FDI thresholds of +0.06969, +0.06981 and
    +0.06804 and NDVI thresholds of +0.11487, +0.14795 and +0.09524, so the
    series mixes real change with a moving criterion. The SRS asks for an "Active Plastic Hotspots
    (count)" headline KPI and a month-by-month archive slider, and neither is
    meaningful on a moving threshold.

    Defaults to the MEDIAN of the per-month thresholds, so the reference is drawn
    from the observed range rather than invented. Report BOTH series: the
    per-scene one is the better detector, the fixed one is the better trend.
    """
    from scipy import ndimage

    import math
    cfg = pwcfg.DETECTION_CONFIG
    min_px = max(1, math.ceil(cfg["min_patch_area_m2"] /
                              (pwcfg.S2_CONFIG["download_scale_m"] ** 2)))
    n_sigma = cfg["threshold_n_sigma"]

    per_month = {}
    for label in monthly_masked_fdi:
        f, v = monthly_masked_fdi[label], monthly_ndvi[label]
        w = monthly_water.get(label)
        in_water = np.nan_to_num(w, nan=0.0) > 0 if w is not None else np.ones_like(f, bool)
        valid = in_water & np.isfinite(f) & np.isfinite(v)
        per_month[label] = (f, v, valid,
                            float(f[valid].mean() + n_sigma * f[valid].std()),
                            float(v[valid].mean() + n_sigma * v[valid].std()))

    if t_fdi is None:
        t_fdi = float(np.median([p[3] for p in per_month.values()]))
    if t_ndvi is None:
        t_ndvi = float(np.median([p[4] for p in per_month.values()]))

    rows = []
    for label, (f, v, valid, own_f, own_v) in per_month.items():
        mask = valid & (f > t_fdi) & (v > t_ndvi)
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n:
            sizes = np.bincount(lab.ravel())[1:]
            keep = sizes >= min_px
            n_patches, px = int(keep.sum()), int(sizes[keep].sum())
        else:
            n_patches, px = 0, 0
        rows.append({
            "year_month": label,
            "own_threshold_fdi": round(own_f, 5),
            "own_threshold_ndvi": round(own_v, 5),
            "fixed_threshold_fdi": round(t_fdi, 5),
            "fixed_threshold_ndvi": round(t_ndvi, 5),
            "n_patches_fixed": n_patches,
            "area_ha_fixed": round(px * pwcfg.S2_CONFIG["download_scale_m"] ** 2 / 10_000, 2),
        })
    return pd.DataFrame(rows).sort_values("year_month").reset_index(drop=True)
