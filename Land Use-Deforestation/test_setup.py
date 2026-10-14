#!/usr/bin/env python3
"""
Staged smoke test for the land-use data pipeline.

Run from the repository root:
    python test_setup.py

It checks each moving part independently over a TINY area near Conakry (so it is
fast, cheap, and never hits the GEE 32 MB / rate limits), and prints PASS / FAIL
/ SKIP for each stage. A stage failing does not stop the others, so you see the
full picture in one run.

Stages:
  0. Config & credentials     (offline)
  1. GeoBoundaries boundary    (keyless network)
  2. Earth Engine auth         (needs GEE service account)
  3. Tiny GEE layer fetch      (needs stage 2)
  4. OpenTopography DEM        (needs OPENTOPO_API_KEY)
  5. OSM vector fetch          (keyless network)
"""

import os
import sys
import logging

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s %(name)s: %(message)s")

# Make the package importable when run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
results = {}


def ok(stage):    results[stage] = "PASS";  print(f"{GREEN}✓ PASS{RESET}  {stage}")
def bad(stage, e): results[stage] = "FAIL"; print(f"{RED}✗ FAIL{RESET}  {stage}\n         {e}")
def skip(stage, why): results[stage] = "SKIP"; print(f"{YELLOW}– SKIP{RESET}  {stage}  ({why})")


# A tiny test polygon (~0.1 deg box) near Conakry, Guinea.
def _tiny_roi():
    import geopandas as gpd
    from shapely.geometry import box
    return gpd.GeoDataFrame(
        {"name": ["test"]},
        geometry=[box(-13.75, 9.55, -13.65, 9.65)],
        crs="EPSG:4326",
    )


def stage0_config():
    stage = "0. Config & credentials"
    try:
        from landuse_tool.azure_utils import load_secrets_into_env
        load_secrets_into_env()
        from landuse_tool import config
        
        print("   Enabled targets    :", config.get_enabled_target_rasters())
        print("   Enabled predictors :", config.get_enabled_predictor_rasters())
        problems = config.validate_required_keys()
        for p in problems:
            print(f"   {YELLOW}missing:{RESET} {p.split('.')[0]}")
        # config itself loading is the pass condition; missing keys are warnings.
        ok(stage)
        return not problems  # True if all required keys present
    except Exception as e:
        bad(stage, e)
        return False


def stage1_boundary():
    stage = "1. GeoBoundaries boundary"
    try:
        from landuse_tool import api_connector
        gdf = api_connector.get_admin_boundary()
        assert gdf is not None and not gdf.empty, "empty/None boundary"
        print(f"   Guinea boundary: {len(gdf)} feature(s), bounds={gdf.total_bounds.round(2)}")
        ok(stage)
    except Exception as e:
        bad(stage, e)


def stage2_gee_auth(keys_present):
    stage = "2. Earth Engine auth"
    from landuse_tool import config
    if not (config.PROCESSING_CONFIG.get("gee_project_id")
            and config.get_api_key("earthengine")):
        skip(stage, "no GEE service account configured")
        return False
    try:
        from landuse_tool import api_connector
        api_connector.initialize_gee()
        import ee
        _ = ee.Number(1).getInfo()  # round-trip
        ok(stage)
        return True
    except Exception as e:
        bad(stage, e)
        return False


def stage3_gee_fetch(gee_ok):
    stage = "3. Tiny GEE layer fetch"
    if not gee_ok:
        skip(stage, "stage 2 did not pass")
        return
    try:
        import rasterio
        import numpy as np
        from landuse_tool import config, api_connector
        roi = _tiny_roi()
        ee_roi = api_connector._gdf_to_ee_geometry(roi)
        cfg = config.TARGET_RASTERS_CONFIG["ESRI_LULC"]
        img = api_connector.build_annual_gee_image(cfg, 2020, ee_roi)
        out = os.path.join(config.PROCESSING_CONFIG["temp_dir"], "smoke_esri_2020.tif")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        api_connector._download_ee_image(img, ee_roi, scale=100, out_path=out)
        with rasterio.open(out) as src:
            arr = src.read(1)
        print(f"   Fetched ESRI 2020: shape={arr.shape}, classes={np.unique(arr)[:10]}")
        ok(stage)
    except Exception as e:
        bad(stage, e)


def stage4_opentopo():
    stage = "4. OpenTopography DEM"
    from landuse_tool import config, api_connector
    if not config.get_api_key("opentopography"):
        skip(stage, "no OPENTOPO_API_KEY")
        return
    try:
        import rasterio
        p = api_connector.fetch_opentopo_dem(_tiny_roi(), "smoke_dem.tif")
        assert p and os.path.exists(p), "no file returned"
        with rasterio.open(p) as src:
            print(f"   DEM: {src.width}x{src.height}, dtype={src.dtypes[0]}")
        ok(stage)
    except Exception as e:
        bad(stage, e)


def stage5_osm():
    stage = "5. OSM vector fetch"
    try:
        import geopandas as gpd
        from landuse_tool import api_connector
        p = api_connector.fetch_osm_vectors(_tiny_roi(), {"amenity": True}, "smoke_poi.geojson")
        if not p:
            skip(stage, "no POIs in tiny box (not an error)")
            return
        print(f"   OSM features: {len(gpd.read_file(p))}")
        ok(stage)
    except Exception as e:
        bad(stage, e)



def stage6_rasterize_vector():
    stage = "6. Vector -> raster (ML readiness)"
    try:
        import numpy as np
        from landuse_tool import config, api_connector, data_loader, utils
        # Fetch a tiny OSM layer, then rasterize it to a small reference grid.
        roi = _tiny_roi()
        vpath = api_connector.fetch_osm_vectors(roi, {"highway": True}, "smoke_roads.geojson")
        if not vpath:
            skip(stage, "no roads in tiny box (not an error)")
            return
        # Build a small reference profile (~100 m) over the ROI.
        from rasterio.transform import from_bounds
        w, s, e, n = roi.total_bounds
        width, height = 120, 120
        ref_profile = {
            "crs": "EPSG:4326", "width": width, "height": height,
            "transform": from_bounds(w, s, e, n, width, height),
            "height_m": 100,
        }
        rasters = data_loader.rasterize_vector_predictors(
            {"ROADS": vpath}, ref_profile)
        arr = next(iter(rasters.values()))
        finite = np.isfinite(arr).sum()
        print(f"   distance-to-roads raster: shape={arr.shape}, finite px={finite}")
        assert arr.shape == (height, width) and finite > 0, "empty raster"
        ok(stage)
    except Exception as e:
        bad(stage, e)


def main():
    print("=" * 60)
    print(" LAND-USE PIPELINE SMOKE TEST")
    print("=" * 60)
    keys_present = stage0_config()
    stage1_boundary()
    gee_ok = stage2_gee_auth(keys_present)
    stage3_gee_fetch(gee_ok)
    stage4_opentopo()
    stage5_osm()
    stage6_rasterize_vector()

    print("=" * 60)
    passed = sum(v == "PASS" for v in results.values())
    failed = sum(v == "FAIL" for v in results.values())
    skipped = sum(v == "SKIP" for v in results.values())
    print(f" SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
