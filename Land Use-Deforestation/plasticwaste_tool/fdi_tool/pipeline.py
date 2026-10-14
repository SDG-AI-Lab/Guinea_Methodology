"""
Plastic Waste (mesoscale FDI hotspot) analysis orchestrator.

`run_from_arrays(...)` ties the modules together on aligned band arrays and
returns a results dict — deliberately independent of the fetch layer
(`api_connector`) so it is fully testable offline, mirroring
`deforestation_tool.pipeline.run_from_arrays`:

  raw bands -> spectral indices -> spectral masks -> Otsu/manual threshold
  -> hotspot binary mask -> vectorize -> stats -> (optional) validation

A thin `run_from_gee(...)` wrapper drives the same flow starting from a live
Earth Engine fetch, and a synthetic self-test (`python -m
plasticwaste_tool.pipeline`) exercises the whole offline chain.
"""
# ---------------------------------------------------------------------------
# MOVED HERE from plasticwaste_tool/pipeline.py by 24_build_component_packages.py.
# This file is now CANONICAL and is hand-maintained: the old top-level name is
# a re-export shim pointing back at it, not a second copy. Edit this file.
# ---------------------------------------------------------------------------

import logging
from typing import Dict, List, Optional

import numpy as np

from . import spectral as SP
from . import detection as D
from . import debris_filter as DF
from .. import stats as ST
from .. import validation as V
from .. import config as pwcfg

logger = logging.getLogger(__name__)


def assess_water_population(water_px, total_px, aoi_kind=None, cfg=None):
    """
    Decide whether a run's water population can carry a mean+3sigma threshold.

    Split out of `run_from_arrays` so the decision can be tested without a
    raster stack, a CRS or an Earth Engine fetch -- the rule is arithmetic and
    deserves a test that does not depend on PROJ being installed correctly.

    TWO guards, because they fail in different situations and a river defeats
    the first one on its own.

    FRACTION catches a bounding box drawn in the wrong place: mostly land, so
    the "water" population is really a mixture and sigma is inflated. That is
    the Fatala case, 2.2% water and a threshold 3.5x Conakry's.

    COUNT catches a population too small to estimate a mean and a standard
    deviation from, whatever share of the window it occupies. The relative
    standard error of a standard deviation is about 1/sqrt(2n): 7% at n=100,
    2.2% at n=1000.

    A river corridor is the case that separates them. It is only ~7% water by
    area, because the corridor must be wide enough to absorb HydroRIVERS
    centreline error (HydroSHEDS 15 arc-sec, a few hundred metres), yet it can
    carry hundreds of thousands of genuine water pixels. That is a sound
    population and a failing fraction, so for a corridor the fraction guard is
    advisory and the count guard decides. For a coastal box both hold and
    behaviour is unchanged.

    A THIRD case is neither of those: no water population at all. Both guards
    above are written `x is not None and x < floor`, so when the water mask is
    absent entirely they evaluate False and the tile is reported as ASSESSED
    with a NaN threshold -- missing data silently presented as a measurement of
    zero debris. The river corridor run surfaced this: 15 of 69 tiles reported
    `assessed=True` with `water_px=None` and `threshold_fdi=NaN`. Absence of a
    water mask is the strongest possible reason not to assess, so it is checked
    explicitly rather than left to fall through the numeric comparisons.

    Returns {water_fraction, low_water_fraction, low_water_count,
             no_water_mask, low_water_warning, assessed}.
    """
    cfg = cfg or pwcfg.DETECTION_CONFIG
    frac = (water_px / total_px) if (water_px and total_px) else None
    is_corridor = (aoi_kind == "river_corridor")

    # Distinguish "the mask says zero water" from "there is no mask": both give
    # an unusable population, and neither may be reported as a measurement.
    no_mask = water_px is None or total_px is None or not water_px

    low_frac = frac is not None and frac < cfg["min_water_fraction"]
    low_count = water_px is not None and water_px < cfg.get("min_water_px", 0)

    # `low_water_warning` keeps its original meaning for every existing caller:
    # "do not report this count as a measurement".
    warn = bool(no_mask or low_count or (low_frac and not is_corridor))
    return {"water_fraction": frac,
            "low_water_fraction": bool(low_frac),
            "low_water_count": bool(low_count),
            "no_water_mask": bool(no_mask),
            "low_water_warning": warn,
            "assessed": not warn}


def run_from_arrays(bands: Dict[str, np.ndarray], profile: Dict,
                    acquisition_date: str, cloud_pct: Optional[float] = None,
                    ndvi_thresh: Optional[float] = None, ndwi_thresh: Optional[float] = None,
                    reference_points=None, progress_callback=None,
                    aoi_kind: Optional[str] = None) -> Dict:
    """
    Full mesoscale hotspot analysis on aligned Sentinel-2 band arrays.

    Required:
        bands     : {'B3','B4','B6','B8','B11'(,'B8A','B12')} -> aligned arrays
        profile   : rasterio profile of the analysis grid
        acquisition_date : ISO date string for the scene / composite window

    Optional:
        cloud_pct         : mean CLOUDY_PIXEL_PERCENTAGE for the window (drives
                            the SAR-fallback flag)
        reference_points  : GeoDataFrame of known plastic/dumpsite points ->
                            enables `hotspot_reference_validation`
    """
    pcb = progress_callback or (lambda f, t: None)
    results: Dict = {}

    pcb(0.10, "Computing spectral indices (FDI/NDVI/NDWI)")
    fdi = SP.compute_fdi(bands)
    ndvi = SP.compute_ndvi(bands)
    ndwi = SP.compute_ndwi(bands)
    results["fdi"], results["ndvi"], results["ndwi"] = fdi, ndvi, ndwi
    # The grid the arrays are on, so a caller can write them out (GEDH
    # feature 002 publishes the monthly composite). Carried, not computed.
    results["profile"] = profile

    water = bands.get("WATER")
    if water is None:
        logger.warning(
            "No WATER band in the composite — coastal land will not be excluded, which "
            "produces infrastructure false positives. Re-download the composite with "
            "force_refresh=True to pick up the WATER band.")

    rule = pwcfg.DETECTION_CONFIG.get("detection_rule", "de_africa")

    if rule == "de_africa":
        pcb(0.30, "Masking to water")
        # Statistics are taken over ALL water pixels, so masked_fdi here is the
        # population, not a pre-filtered subset (see D.debris_binary_mask_de_africa).
        in_water = np.nan_to_num(water, nan=0.0) > 0 if water is not None else np.ones_like(fdi, bool)
        masked_fdi = np.where(in_water, fdi, np.nan)

        pcb(0.50, "Thresholding FDI and NDVI at mean + n*sigma")
        hotspot_mask, thresholds, method = D.debris_binary_mask_de_africa(fdi, ndvi, water)
        threshold = thresholds["fdi"]
        results["threshold_ndvi"] = thresholds["ndvi"]
        results["threshold_population_px"] = thresholds.get("n_px_population")

        # Rule first, then the MARIDA debris classifier on the rule's pixels
        # (fdi_tool/debris_filter.py). The thresholds above are untouched: they
        # are still taken over the whole water population.
        if DF.enabled():
            pcb(0.55, "Filtering detections with the MARIDA debris classifier")
            results["hotspot_binary_rule"] = hotspot_mask
            hotspot_mask, p_debris, results["debris_filter"] = DF.apply(hotspot_mask, bands)
            results["p_debris"] = p_debris

        # NDVI is (NIR-Red)/(NIR+Red), so it is bounded to [-1, 1] BY
        # CONSTRUCTION. A mean+3sigma threshold outside that interval is
        # therefore not a statement about vegetation -- it is proof that the
        # pixels called "water" are not water. In the river corridor run this
        # caught 4 tiles whose NDVI thresholds came back at 1.01, 1.11, 1.14
        # and 55.68: each had cleared the 1,000 px count floor but occupied
        # under 0.2% of its window, so the population was dominated by nodata
        # and scene-edge fill, where NIR+Red approaches zero and the ratio
        # diverges. The check needs no calibration and no literature: it is an
        # algebraic bound on the index itself.
        ndvi_thr = thresholds.get("ndvi")
        implausible = (ndvi_thr is None or not np.isfinite(ndvi_thr)
                       or abs(ndvi_thr) > 1.0)
        results["threshold_implausible"] = bool(implausible)
        if implausible:
            results["low_water_warning"] = True
            logger.warning(
                "NDVI threshold %s is outside the [-1, 1] interval NDVI is "
                "algebraically confined to. The water population is not water "
                "(most likely nodata or scene-edge fill), so the FDI threshold "
                "derived from it is meaningless and the patch count must be "
                "reported as NOT ASSESSED.", ndvi_thr)
    else:
        pcb(0.30, "Applying vegetation / water masks (legacy rule)")
        masked_fdi = SP.apply_spectral_masks(fdi, ndvi, ndwi, ndvi_thresh, ndwi_thresh,
                                             water=water)
        pcb(0.50, "Thresholding FDI")
        hotspot_mask, threshold, method = D.hotspot_binary_mask(masked_fdi)

    results["masked_fdi"] = masked_fdi
    results["hotspot_binary"] = hotspot_mask
    results["threshold"] = threshold
    results["threshold_method"] = method
    results["detection_rule"] = rule

    # Water share of the analysis window, and the low-water guard.
    #
    # A window that is mostly land gives the anomaly rule a small, heterogeneous
    # water population; sigma inflates and the run can return zero patches for
    # want of SENSITIVITY rather than want of debris. Recording the fraction is
    # what separates "nothing there" from "could not look", and the Fatala
    # region returned a nil result at 2.2% water for exactly this reason.
    #
    # This lives HERE, not in run_national, because run_from_arrays is the one
    # place every path goes through -- run_from_gee, run_national and
    # run_monthly_batch all land here. It was previously applied only inside
    # run_national's per-region loop, which left single-region and monthly-
    # archive runs completely unguarded while the methodology described the
    # guard as a property of "the pipeline". Both are now true.
    #
    # It costs nothing: both terms are already computed above, so there is no
    # extra Earth Engine reduction.
    # Carried so the guard below can tell a river corridor from a coastal box.
    # A corridor is legitimately low-fraction; a misplaced box is not, and the
    # two are indistinguishable from the fraction alone.
    if aoi_kind:
        results["aoi_kind"] = aoi_kind

    water_px = results.get("threshold_population_px")
    total_px = int(fdi.size) if fdi is not None else None
    verdict = assess_water_population(water_px, total_px, aoi_kind)
    results.update({k: v for k, v in verdict.items() if k != "assessed"})
    min_px = pwcfg.DETECTION_CONFIG.get("min_water_px", 0)
    low_frac = verdict["low_water_fraction"]
    low_count = verdict["low_water_count"]
    frac = verdict["water_fraction"]
    is_corridor = (aoi_kind == "river_corridor")

    if low_count:
        logger.warning(
            "Water population is only %s px, below the %s px floor. A mean+3sigma "
            "threshold estimated from this few pixels is unstable (relative "
            "standard error of sigma is about %.1f%%), so the patch count must be "
            "reported as NOT ASSESSED rather than as a measurement.",
            f"{water_px:,}", f"{min_px:,}",
            100.0 / (2.0 * max(water_px, 1)) ** 0.5)
    elif low_frac and is_corridor:
        logger.info(
            "River corridor is %.1f%% water (%s px). Below the %.0f%% fraction "
            "guard, but that guard is calibrated for compact coastal boxes: the "
            "corridor is wide because the centreline is imprecise, not because "
            "the window is misplaced. The %s-pixel population is ample, so this "
            "is assessed.",
            100 * frac, f"{water_px:,}",
            100 * pwcfg.DETECTION_CONFIG["min_water_fraction"], f"{water_px:,}")
    elif low_frac:
        logger.warning(
            "Analysis window is only %.1f%% water (%s px). Its threshold is derived "
            "from a small, likely heterogeneous population and its patch count "
            "should be reported as NOT ASSESSED, not as a measurement. Check the "
            "bounding box actually covers the water body of interest.",
            100 * frac, f"{water_px:,}")

    pcb(0.65, "Vectorizing hotspot patches")
    hotspot_gdf = D.vectorize_hotspots(hotspot_mask, profile, acquisition_date, cloud_pct,
                                       masked_fdi=masked_fdi, threshold=threshold,
                                       p_debris=results.get("p_debris"))
    results["hotspot_vectors"] = hotspot_gdf

    pcb(0.80, "Computing summary statistics")
    results["summary"] = ST.hotspot_summary(hotspot_gdf)
    results["sar_fallback_recommended"] = ST.sar_fallback_flag(cloud_pct)
    # Sensitivity must probe the rule actually in force. Under de_africa the
    # deployed detector is the FDI-and-NDVI conjunction; perturbing FDI alone
    # describes a 33x larger population than the rule selects (see
    # V.threshold_sensitivity_de_africa).
    if rule == "de_africa":
        # With the debris filter on, the published detector is rule-then-
        # classifier, so the perturbed rule output is filtered the same way.
        post = (lambda m: DF.apply(m, bands)[0]) if DF.enabled() else None
        results["threshold_sensitivity"] = V.threshold_sensitivity_de_africa(
            fdi, ndvi, water, post_filter=post)
    else:
        results["threshold_sensitivity"] = V.threshold_sensitivity_analysis(masked_fdi)

    if reference_points is not None:
        pcb(0.90, "Validating against reference points")
        results["reference_validation"] = V.hotspot_reference_validation(
            hotspot_gdf, reference_points)

    pcb(1.0, "Plastic waste hotspot analysis complete")
    return results


def run_from_gee(start_date: str, end_date: str, aoi_key: str = pwcfg.DEFAULT_AOI_KEY,
                 output_prefix: Optional[str] = None, **kwargs) -> Dict:
    """Convenience wrapper: fetch a Sentinel-2 composite via `api_connector`,
    download the multi-band raster, load it, then `run_from_arrays(...)`."""
    from .. import api_connector as api
    from .. import data_loader as DL

    api.initialize_gee()
    aoi = api.get_aoi_geometry(aoi_key)
    gee_results = api.build_fdi_pipeline(start_date, end_date, aoi)
    # The AOI bbox hash belongs in the cache key: without it, editing AOI_CONFIG
    # and re-running silently returns the raster for the previous box.
    prefix = output_prefix or f"{aoi_key}_{api.aoi_cache_token(aoi_key)}_{start_date}_{end_date}"
    local_path = api.download_composite_bands(gee_results, aoi, prefix)
    bands, profile = DL.read_composite(local_path)
    cloud_pct = gee_results["cloud_pct"].getInfo()
    return run_from_arrays(bands, profile, start_date, cloud_pct=cloud_pct, **kwargs)


NATIONAL_REGION_KEYS = ["conakry", "konkoure", "fatala", "mangrove"]


def run_national(start_date: str, end_date: str,
                 region_keys: Optional[List[str]] = None,
                 output_prefix: Optional[str] = None, **kwargs) -> Dict:
    """
    National coastal run (GEDH-FR-0503 National Hotspot Map): one detection per
    region, mosaicked, with each region's thresholds carried on its patches.

    Why per-region and not one pass over `AOI_CONFIG['full']`
    --------------------------------------------------------
    `mean + n*sigma` is an anomaly threshold measured against a background
    population. Taking it over every water pixel on the coast pools Conakry's
    turbid, port-influenced water with clear offshore water and the
    Konkoure/Fatala mangrove estuaries, producing one compromise number: too
    high in clear water (misses), too low in turbid water (false alarms), and
    regional detections that are not comparable with each other. Digital Earth
    Africa apply the rule per scene/region for this reason.

    Each region therefore gets its own `t_fdi` / `t_ndvi`, recorded per patch
    (`region`, `threshold_fdi`, `threshold_ndvi`, `threshold_population_px`) so
    the published layer carries the provenance of its own detection rule rather
    than implying a single national threshold.

    Secondary benefit: the coastal AOI is ~152 M pixels at 20 m, and
    `run_from_arrays` holds several float32 arrays of that size at once. Four
    regional runs stay comfortably in memory where one national pass would not.

    Regions that fail (no imagery, download error) are recorded in `failed`
    rather than aborting the run -- a monthly scheduled job should publish the
    regions that did work and report the ones that did not.
    """
    import pandas as pd

    region_keys = region_keys or NATIONAL_REGION_KEYS
    per_region: Dict[str, Dict] = {}
    failed: Dict[str, str] = {}

    for key in region_keys:
        try:
            logger.info("National run: region %s (%s to %s)", key, start_date, end_date)
            res = run_from_gee(start_date, end_date, aoi_key=key,
                               output_prefix=output_prefix and f"{output_prefix}_{key}",
                               **kwargs)
            # water_fraction / low_water_warning are set by run_from_arrays
            # for every path now, so there is nothing to compute here. The
            # region name is logged alongside for readability, since the
            # warning raised downstream does not know which region it is in.
            if res.get("low_water_warning"):
                logger.warning("Region %s is below the minimum water fraction; "
                               "its count is NOT ASSESSED.", key)

            gdf = res["hotspot_vectors"]
            if not gdf.empty:
                gdf = gdf.copy()
                gdf["region"] = key
                gdf["threshold_fdi"] = res["threshold"]
                gdf["threshold_ndvi"] = res.get("threshold_ndvi")
                gdf["threshold_population_px"] = res.get("threshold_population_px")
                res["hotspot_vectors"] = gdf
            per_region[key] = res
        except Exception as e:  # noqa: BLE001
            logger.error("National run: region %s failed: %s", key, e)
            failed[key] = str(e)

    frames = [r["hotspot_vectors"] for r in per_region.values()
              if not r["hotspot_vectors"].empty]
    mosaic, dedup = None, {"n_before": 0, "n_removed": 0, "n_after": 0}
    if frames:
        import geopandas as gpd

        from . import deployment as DEP
        mosaic = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                  crs=frames[0].crs)
        # Overlapping AOIs detect the same feature once per region, inflating the
        # national count that the SRS publishes as a headline indicator.
        mosaic, dedup = DEP.deduplicate_mosaic(mosaic)

    return {
        "per_region": per_region,
        "failed": failed,
        "hotspot_vectors": mosaic,
        "summary": ST.hotspot_summary(mosaic) if mosaic is not None else {},
        "regional_thresholds": {
            k: {"fdi": r["threshold"], "ndvi": r.get("threshold_ndvi"),
                "population_px": r.get("threshold_population_px"),
                "water_fraction": (round(r["water_fraction"], 4)
                                   if r.get("water_fraction") is not None else None),
                "assessed": not r.get("low_water_warning", False),
                "method": r["threshold_method"],
                "n_patches": r["summary"]["n_patches"]}
            for k, r in per_region.items()},
        "not_assessed": [k for k, r in per_region.items()
                         if r.get("low_water_warning", False)],
        "deduplication": dedup,
        "aoi_overlaps": DEP.check_aoi_overlaps(region_keys) if frames else [],
        "scope_note": (
            "National COASTAL and estuarine coverage, not the full land area of "
            "Guinea: FDI is defined only on water. Thresholds are per-region, so "
            "patch counts are comparable within a region across months, and "
            "between regions only with the regional threshold quoted alongside."),
    }


def run_monthly_batch(monthly_bands: Dict[str, Dict[str, np.ndarray]], profile: Dict,
                      cloud_pct_by_month: Optional[Dict[str, float]] = None) -> Dict:
    """Run `run_from_arrays` for each month in a {'YYYY-MM': bands} dict and
    return both the per-month results and the aggregated time series /
    Otsu-fallback-rate statistics (for the Monthly FDI Archive view)."""
    per_month, methods = {}, []
    for label, bands in monthly_bands.items():
        cloud_pct = (cloud_pct_by_month or {}).get(label)
        res = run_from_arrays(bands, profile, label, cloud_pct=cloud_pct)
        per_month[label] = res
        methods.append(res["threshold_method"])

    return {
        "per_month": per_month,
        "time_series": ST.monthly_time_series(
            {label: r["hotspot_vectors"] for label, r in per_month.items()}),
        "otsu_fallback_rate": V.otsu_fallback_rate(methods),
    }


# ===========================================================================
# Synthetic self-test
# ===========================================================================
# Median reflectances of MARIDA's training pixels (Kikaki et al. 2022): class
# "Marine Water" (86,877 px) and class "Marine Debris" (1,943 px). The fixture
# is built from them so the self-test exercises the debris classifier
# (fdi_tool.debris_filter) on spectra it has actually seen, not invented ones.
_MARIDA_WATER = {"B3": 0.0195, "B4": 0.0122, "B6": 0.0090, "B8": 0.0077,
                 "B8A": 0.0086, "B11": 0.0041, "B12": 0.0029}
_MARIDA_DEBRIS = {"B3": 0.0570, "B4": 0.0555, "B6": 0.0392, "B8": 0.0539,
                  "B8A": 0.0439, "B11": 0.0184, "B12": 0.0103}


def _synthetic_scene(H=200, W=200, seed=0):
    """Build a synthetic Sentinel-2-like band set: two patches with MARIDA's
    median marine-debris spectrum in water with MARIDA's median marine-water
    spectrum, beside a vegetated shoreline strip."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    veg_strip = xx < W * 0.15  # shoreline vegetation on the west edge
    veg = {"B3": 0.08, "B4": 0.05, "B6": 0.25, "B8": 0.35, "B8A": 0.34,
           "B11": 0.10, "B12": 0.05}

    bands = {}
    for b, w in _MARIDA_WATER.items():
        water_px = w + rng.normal(0, 0.0005, (H, W))
        bands[b] = np.where(veg_strip, veg[b], water_px).astype("float32")

    # Two debris patches. Debris raises both FDI and NDVI over seawater, which
    # is what both references describe (Digital Earth Africa require NDVI
    # ABOVE its own mean+3*sigma; Biermann et al. 2020 find debris high on
    # both axes of the NDVI-FDI plane).
    cy1, cx1, r1 = int(H * 0.5), int(W * 0.6), 6
    cy2, cx2, r2 = int(H * 0.7), int(W * 0.75), 4
    for cy, cx, r in [(cy1, cx1, r1), (cy2, cx2, r2)]:
        patch = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        for b, d in _MARIDA_DEBRIS.items():
            bands[b][patch] = d

    # WATER band, as the real composite now carries: everything except the
    # shoreline vegetation strip. The DE Africa rule takes its mean and sigma
    # over this population, so it has to exist for the fixture to exercise the
    # default path at all.
    bands["WATER"] = (~veg_strip).astype("float32")
    return bands


def _synthetic_test():
    import sys

    # Two environment repairs, both needed before the first rasterio CRS call
    # and neither of which touches the analysis. Without them this self-test
    # fails on a clean Windows checkout for reasons unrelated to the pipeline,
    # which is worse than useless -- a green tick is only informative if a red
    # one means the code is wrong.
    #
    # 1. PROJ. A machine-wide PROJ_LIB pointing at another GDAL install (a
    #    PostGIS one, typically) makes CRS.from_epsg(4326) raise
    #    "Cannot find proj.db". geo_env.fix_proj repoints it at rasterio's own
    #    bundled copy for this process only.
    # 2. Console encoding. The tick characters below are not encodable in
    #    cp1252, so on a default Windows console the test dies in its own
    #    reporting after the analysis has already passed.
    from ..geo_env import fix_proj
    fix_proj()
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):   # already wrapped, or not a TTY
            pass

    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    import geopandas as gpd
    from shapely.geometry import Point

    GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
    H = W = 200
    profile = {"crs": CRS.from_epsg(4326), "width": W, "height": H,
              "transform": from_bounds(-13.90, 9.30, -13.85, 9.35, W, H)}
    bands = _synthetic_scene(H, W)

    # Reference point sitting on top of the larger synthetic debris patch.
    lon, lat = -13.90 + (0.6 * 0.05), 9.35 - (0.5 * 0.05)
    ref_points = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(lon, lat)], crs="EPSG:4326")

    print("=" * 60)
    print(" PLASTIC WASTE (FDI HOTSPOT) MODULE SELF-TEST (synthetic)")
    print("=" * 60)

    ok = True
    try:
        res = run_from_arrays(bands, profile, "2024-01-15", cloud_pct=12.0,
                              reference_points=ref_points)
        print(f"{GREEN}\u2713{RESET} spectral indices computed "
             f"(FDI range: {np.nanmin(res['fdi']):.4f} to {np.nanmax(res['fdi']):.4f})")
        print(f"{GREEN}\u2713{RESET} threshold: {res['threshold']:.5f} ({res['threshold_method']})")
        print(f"{GREEN}\u2713{RESET} patches detected: {res['summary']['n_patches']}")
        print(f"{GREEN}\u2713{RESET} summary: {res['summary']}")
        print(f"{GREEN}\u2713{RESET} SAR fallback recommended: {res['sar_fallback_recommended']}")
        print(f"{GREEN}\u2713{RESET} threshold sensitivity: {res['threshold_sensitivity']}")
        print(f"{GREEN}\u2713{RESET} reference validation: {res['reference_validation']}")
        assert res["summary"]["n_patches"] >= 1, "expected at least one synthetic patch detected"
    except Exception:
        import traceback; traceback.print_exc()
        ok = False
    print("=" * 60)
    print(" RESULT:", f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _synthetic_test()
