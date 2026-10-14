"""
Deforestation analysis orchestrator.

`run_from_arrays(...)` ties the modules together on aligned rasters (the output
of the fetch + data_loader alignment step) and returns a results dict:

  forest masks -> detection -> statistics -> driver model -> risk map -> scenario

It deliberately takes already-aligned arrays so it is independent of the fetch
layer and fully testable offline. A thin `run_from_files(...)` wrapper and a
synthetic self-test (`python -m landuse_tool.deforestation.pipeline`) are included.
"""

import logging
import numpy as np

from . import forest as F
from . import detection as D
from . import stats as S
from . import drivers as DR
from . import scenario as SC
from . import config as dcfg

logger = logging.getLogger(__name__)


def run_from_arrays(treecover, lossyear, profile, years,
                    lulc_by_year=None, lulc_product="ESRI_LULC",
                    ndvi_by_year=None, shared_predictors=None,
                    roads_mask=None, fire_mask=None,
                    class_names=None, future_years=None,
                    canopy_threshold=None, progress_callback=None, datamask=None):
    """
    Full deforestation analysis on aligned inputs.

    Required:
        treecover : Hansen canopy-2000 percentage array
        lossyear  : Hansen lossyear array (0 = none, n = year 2000+n)
        profile   : rasterio profile of the analysis grid
        years     : list of analysis years to report

    Optional (enable extra outputs):
        lulc_by_year     : {year: LULC array}  -> conversion "forest -> what"
        ndvi_by_year     : {year: NDVI array}  -> degradation + VI trend
        shared_predictors: {name: array}       -> driver model + risk map
        future_years     : list                -> scenario projection
        datamask         : Hansen datamask on the grid (land = 1) -> land
                           area and forest share (feature 005, toolchain.md §2, §4)

    Feature 005 result keys (contracts/toolchain.md §2): `importance_model`
    ("descriptive", or "risk" when the descriptive model cannot be fitted),
    `datamask` (as given) and `forest_now` (the forest mask the risk map scored).
    """
    pcb = progress_callback or (lambda f, t: None)
    results = {}

    # 1. Forest masks (Hansen convention) ----------------------------------
    pcb(0.05, "Building forest masks")
    forest2000 = F.forest_mask_from_hansen(treecover, canopy_threshold)
    forest_by_year = F.forest_extent_by_year(forest2000, lossyear, years)
    results["forest_by_year"] = forest_by_year

    # 2. Detection ---------------------------------------------------------
    pcb(0.20, "Detecting annual forest loss")
    loss_by_year = D.annual_loss_from_hansen(lossyear, years)
    results["loss_by_year"] = loss_by_year

    if lulc_by_year:
        y0, y1 = min(years), max(years)
        f_t1 = F.forest_mask_from_lulc(lulc_by_year[y0], lulc_product)
        f_t2 = F.forest_mask_from_lulc(lulc_by_year[y1], lulc_product)
        deforested = D.deforestation_from_lulc(f_t1, f_t2)
        results["deforested_lulc"] = deforested
        results["conversion"] = D.conversion_classes(deforested, lulc_by_year[y1])

    if ndvi_by_year and len(ndvi_by_year) >= 2:
        yrs = sorted(ndvi_by_year)
        results["degradation"] = D.degradation_from_vi(
            ndvi_by_year[yrs[0]], ndvi_by_year[yrs[-1]],
            forest_by_year[yrs[-1]] if yrs[-1] in forest_by_year else forest2000,
            drop=dcfg.DEFOR_PARAMS["vi_degradation_drop"])

    # 3. Statistics --------------------------------------------------------
    pcb(0.40, "Computing statistics")
    results["summary"] = S.summary_table(loss_by_year, forest_by_year, profile=profile, datamask=datamask)
    if datamask is not None:
        results["datamask"] = datamask

    if ndvi_by_year:
        slope, mean_slope = S.vi_trend(ndvi_by_year, mask=forest2000)
        results["vi_trend_map"], results["vi_trend_mean"] = slope, mean_slope

    if lulc_by_year and "deforested_lulc" in results:
        results["conversion_matrix"] = S.conversion_matrix(
            results["deforested_lulc"], lulc_by_year[max(years)],
            class_names=class_names, profile=profile)

    # 4. Driver model + risk map ------------------------------------------
    risk = None
    if shared_predictors:
        pcb(0.60, "Fitting driver / risk model")
        y_last = max(years)
        forest_now = forest_by_year[y_last]
        results["forest_now"] = forest_now
        prior_loss = D.cumulative_loss(lossyear, up_to_year=min(years))

        # Commodity price surface (Oil (palm), confirmed adopted -- Methodology
        # Section 9.1/6.9, 3/3 historical splits improved). Fetched here so both
        # the descriptive and adopted driver stacks get it automatically, rather
        # than needing every caller to remember to wire it in by hand. Graceful
        # fallback (warn + None) if the fetch fails -- a network hiccup on this
        # one optional predictor should not crash the whole pipeline run.
        commodity_price_arr = None
        wfp_cfg = getattr(dcfg, "WFP_PRICES_SOURCE", {})
        if wfp_cfg.get("enabled"):
            try:
                from .. import api_connector, data_loader
                wfp_csv = api_connector.fetch_wfp_prices(output_filename="wfp_guinea_prices.csv")
                if wfp_csv:
                    commodity_price_arr = data_loader.rasterize_market_points(
                        wfp_csv, wfp_cfg["commodity"], wfp_cfg["reference_year"], profile)
            except Exception as e:
                logger.warning("Commodity price fetch/rasterize failed (%s) -- proceeding "
                              "without it for this run.", e)

        # TWO configurations, two purposes -- do not conflate them (see
        # drivers.build_driver_stack docstring and Methodology_Deforestation_
        # Guinea.docx Section 9.1/Table 4 caption for the full rationale):
        #   (a) descriptive_stack: dist_forest_edge INCLUDED, for driver
        #       importance / attribution only (Table 4). Confirmed dominant
        #       (~0.69 importance) within its own calibration window; this is
        #       legitimate as a description of where loss has concentrated,
        #       but must never be used for forward prediction (see (b)).
        #   (b) driver_stack (adopted): dist_forest_edge EXCLUDED (function
        #       default). This is the only configuration validated by a true
        #       temporal hold-out (AUC 0.686, stable across 3 historical
        #       splits) and is the one used for risk_map / scenario below.
        #
        # Confirmed-rejected candidates (NDVI, burned area -- Section 6.5,
        # hold-out AUC 0.479/0.496, below chance) are excluded here explicitly,
        # by whatever key name they arrive under in shared_predictors. This is
        # deliberately a filter local to THIS module, not a global config
        # change: shared_predictors is populated from the same base
        # landuse_tool.config the Land Use Tool also uses, and that sibling
        # module has its own, separately-tested, legitimately-adopted NDVI
        # predictor (registered under the plain key "NDVI", distinct from this
        # module's own "MODIS_NDVI" candidate-test registration) -- disabling
        # NDVI globally would risk breaking that module's validated
        # configuration. Confirmed necessary: a real run showed NDVI_2024.tif
        # present in the adopted driver importances (weight 0.0245) even after
        # MODIS_NDVI/MODIS_BURNED were disabled in this module's own config,
        # because it was arriving under the Land Use Tool's "NDVI" key instead.
        _REJECTED_CANDIDATE_KEY_FRAGMENTS = ("ndvi", "burned")
        shared_predictors_filtered = {
            k: v for k, v in shared_predictors.items()
            if not any(frag in k.lower() for frag in _REJECTED_CANDIDATE_KEY_FRAGMENTS)
        }
        _excluded = set(shared_predictors) - set(shared_predictors_filtered)
        if _excluded:
            logger.info("Excluded confirmed-rejected candidate predictor(s) from the "
                       "deforestation driver stack (not from shared_predictors itself, "
                       "which other modules may still use): %s", sorted(_excluded))

        # Slope/aspect: documented as adopted predictors since early in this
        # project but never actually computed for this pipeline -- the
        # derivation functions (utils.calculate_slope/calculate_aspect) exist
        # in the PARENT landuse_tool package (not this deforestation_tool
        # subpackage -- confirmed the correct import is "from .. import utils",
        # not "from . import utils", matching how forest.py already reaches
        # the parent package for config via "from .. import config as base").
        # They're already used correctly in the land-cover training pathway
        # (data_loader.assemble_predictor_stack), but shared_predictors here
        # comes from fetch_and_load_api_predictors, a different pathway that
        # never called them -- confirmed absent from every driver-importance
        # output this session despite being documented as included since early
        # in this project. Wired in here, at the same point NDVI/burned area
        # are filtered, so both descriptive and adopted driver stacks get them.
        _dem_key = next((k for k in shared_predictors_filtered if "DEM" in k.upper()), None)
        if _dem_key is not None:
            try:
                from .. import utils as _landuse_utils
                dem_arr = shared_predictors_filtered[_dem_key]
                dem_profile = {"transform": profile["transform"] if isinstance(profile, dict) else profile.transform,
                               "nodata": None}
                shared_predictors_filtered["SLOPE"] = _landuse_utils.calculate_slope(dem_arr, dem_profile)
                shared_predictors_filtered["ASPECT"] = _landuse_utils.calculate_aspect(dem_arr, dem_profile)
                logger.info("Derived SLOPE/ASPECT from %s and added to the driver stack.", _dem_key)
            except Exception as e:
                logger.warning("Could not derive SLOPE/ASPECT from %s (%s) -- proceeding "
                              "without them rather than failing the whole pipeline run.",
                              _dem_key, e)
        else:
            logger.warning("No DEM key found in shared_predictors -- cannot derive SLOPE/ASPECT.")

        descriptive_stack = DR.build_driver_stack(
            shared_predictors_filtered, forest_now, prior_loss,
            roads_mask=roads_mask, fire_mask=fire_mask, profile=profile,
            include_forest_edge=True, commodity_price_surface=commodity_price_arr)

        driver_stack = DR.build_driver_stack(
            shared_predictors_filtered, forest_now, prior_loss,
            roads_mask=roads_mask, fire_mask=fire_mask, profile=profile,
            commodity_price_surface=commodity_price_arr)

        # positives: loss during the analysis window; negatives: stayed forest
        loss_window = D.cumulative_loss(lossyear, up_to_year=y_last) & ~prior_loss
        stable_forest = forest_now & ~loss_window

        X, y, names = DR.sample_training(
            loss_window, stable_forest, driver_stack,
            samples_per_class=dcfg.DEFOR_PARAMS["samples_per_class"])

        if X is not None:
            model, metrics = DR.fit_risk_model(
                X, y, model_type=dcfg.DEFOR_PARAMS["driver_model"])

            # Descriptive attribution (Table 4) -- separate model, separate
            # purpose, includes dist_forest_edge. NOT used for risk_map/scenario.
            X_desc, y_desc, names_desc = DR.sample_training(
                loss_window, stable_forest, descriptive_stack,
                samples_per_class=dcfg.DEFOR_PARAMS["samples_per_class"])
            if X_desc is not None:
                model_desc, _ = DR.fit_risk_model(
                    X_desc, y_desc, model_type=dcfg.DEFOR_PARAMS["driver_model"])
                results["driver_importance"] = DR.driver_importance(model_desc, names_desc)
                results["importance_model"] = "descriptive"
            else:
                # Too small a descriptive sample: the risk model ranks them,
                # and the record says so (research F6; FR-023).
                results["driver_importance"] = DR.driver_importance(model, names)
                results["importance_model"] = "risk"

            results["risk_metrics"] = metrics
            risk = DR.predict_risk_map(model, driver_stack, names, forest_now)
            results["risk_map"] = risk
            # Expose the fitted model so the risk map can be VALIDATED: to
            # back-test, re-predict risk over the forest standing at the START
            # of the validation window and compare with observed loss.
            results["risk_model"] = model
            results["driver_names"] = names
            results["driver_stack"] = driver_stack
            results["loss_window_mask"] = loss_window
            logger.info("Driver importances: %s", results["driver_importance"])

    # 5. Scenario ----------------------------------------------------------
    if future_years and risk is not None:
        pcb(0.85, "Projecting future scenario")
        loss_ha = S.annual_deforestation_area(loss_by_year, profile=profile)
        forest_ha = S.forest_cover_trajectory(forest_by_year, profile=profile)
        area = S.pixel_area_ha(profile, shape=treecover.shape)
        results["scenario"] = SC.run_scenario(
            risk, forest_by_year[max(years)], loss_ha, future_years,
            pixel_area_ha=area, forest_ha_by_year=forest_ha, profile=profile)

    pcb(1.0, "Deforestation analysis complete")
    return results


def run_from_files(hansen_path, profile, years, **kwargs):
    """Convenience wrapper: read Hansen 2-band file, then run_from_arrays."""
    treecover, lossyear = F.read_hansen(hansen_path)
    return run_from_arrays(treecover, lossyear, profile, years, **kwargs)


# ===========================================================================
# Synthetic self-test
# ===========================================================================
def _synthetic_test():
    import sys
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"
    np.random.seed(0)
    H = W = 200
    yy, xx = np.mgrid[0:H, 0:W]
    profile = {"crs": CRS.from_epsg(4326), "width": W, "height": H,
               "transform": from_bounds(-13.7, 9.5, -13.6, 9.6, W, H)}

    # Hansen-like: canopy high in one half; loss concentrated near "roads" (centre)
    treecover = np.where(xx < W * 0.7, 80.0, 10.0).astype("float32")
    dist_center = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
    lossyear = np.zeros((H, W), dtype="float32")
    for k, yr_code in enumerate([18, 19, 20, 21, 22], start=1):   # 2018..2022
        band = (treecover >= 30) & (dist_center < 20 + k * 12) & (dist_center >= 20 + (k - 1) * 12)
        lossyear[band & (np.random.rand(H, W) < 0.7)] = yr_code

    years = [2018, 2019, 2020, 2021, 2022]
    lulc = {2018: np.where(treecover >= 30, 2, 5).astype("uint8"),   # 2=Trees,5=Crops
            2022: np.where((treecover >= 30) & (lossyear == 0), 2, 5).astype("uint8")}
    ndvi = {2018: (treecover / 100.0).astype("float32"),
            2022: (treecover / 100.0 - (lossyear > 0) * 0.3).astype("float32")}

    predictors = {
        "slope": (np.abs(np.gradient(treecover)[0]) * 5).astype("float32"),
        "population": np.random.exponential(3, (H, W)).astype("float32"),
        "ROADS_distance": dist_center.astype("float32"),
        "precipitation": (1500 - yy).astype("float32"),
    }
    roads_mask = dist_center < 3

    print("=" * 56)
    print(" DEFORESTATION MODULE SELF-TEST (synthetic)")
    print("=" * 56)
    ok = True
    try:
        res = run_from_arrays(
            treecover, lossyear, profile, years,
            lulc_by_year=lulc, ndvi_by_year=ndvi, shared_predictors=predictors,
            roads_mask=roads_mask, class_names={2: "Trees", 5: "Cropland"},
            future_years=[2023, 2024, 2025])
        print(f"{GREEN}\u2713{RESET} forest masks / detection / stats")
        print("   annual loss (ha):",
              {y: v for y, v in list(res["summary"][['year', 'loss_ha']].values.tolist())})
        print(f"{GREEN}\u2713{RESET} conversion matrix:")
        print(res["conversion_matrix"].to_string(index=False))
        print(f"{GREEN}\u2713{RESET} VI trend mean (should be <=0):", res["vi_trend_mean"])
        print(f"{GREEN}\u2713{RESET} driver importances:", res.get("driver_importance"))
        print(f"{GREEN}\u2713{RESET} risk model CV:", res.get("risk_metrics"))
        sc = res.get("scenario", {})
        print(f"{GREEN}\u2713{RESET} scenario projected loss (ha):", sc.get("projected_loss_ha"))
        print(f"{GREEN}\u2713{RESET} scenario cleared pixels:", int(sc.get("cleared_mask").sum()) if sc else 0)
    except Exception as e:
        import traceback; traceback.print_exc()
        ok = False

    print("=" * 56)
    print(" RESULT:", f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}")
    print("=" * 56)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _synthetic_test()