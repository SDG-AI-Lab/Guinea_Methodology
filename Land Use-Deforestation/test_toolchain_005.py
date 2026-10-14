"""
Offline toolchain tests for feature 005 (T009) -- no network, no API keys.

specs/005-land-use-deforestation-views/contracts/toolchain.md §3-§4: the
zonal pass's land area and shares, the national summary's land area and
forest share, and the rate that is None, not 0.0, without a previous
year's forest (doc 70 K24, defect F4).

Run (the toolchain's dependencies live in the api image; the folder is
mounted at /landuse_tool):

    docker compose exec -T api sh -c \
        'cd /landuse_tool && pytest -q -p no:cacheprovider -p no:django test_toolchain_005.py'

stats.py is loaded by path, like test_units.py loads config.py, so the
package's network-facing modules are never imported.
"""

import importlib.util
import math
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(relative, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, relative))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stats = _load(os.path.join("landuse_tool", "deforestation_tool", "stats.py"), "defor_stats_005")


# --------------------------------------------------------------------------
# A synthetic grid: 4 x 6 pixels, projected (1 px = 1 ha), two zones
# --------------------------------------------------------------------------
SHAPE = (4, 6)


def _profile():
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    # 100 m pixels in UTM 28N: every pixel is exactly 1 ha.
    return {"transform": from_origin(500_000, 1_100_400, 100, 100), "crs": CRS.from_epsg(32628),
            "height": SHAPE[0], "width": SHAPE[1]}


def _zones():
    """West zone = columns 0-2, east zone = columns 3-5."""
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"shapeID": ["WEST", "EAST"]},
        geometry=[box(500_000, 1_100_000, 500_300, 1_100_400), box(500_300, 1_100_000, 500_600, 1_100_400)],
        crs="EPSG:32628",
    )


def _datamask():
    """Hansen datamask: 1 = land, 2 = water, 0 = no data."""
    mask = np.ones(SHAPE, dtype="uint8")
    mask[0, :3] = 2  # three water pixels in the west zone
    mask[3, 5] = 0   # one no-data pixel in the east zone
    return mask


# --------------------------------------------------------------------------
# §3: zone_land_ha, forest_pct, loss_pct
# --------------------------------------------------------------------------
def test_zone_land_counts_only_datamask_land():
    land = stats.zone_land_area(_datamask(), _zones(), _profile(), name_field="shapeID")
    assert land == {"WEST": 9.0, "EAST": 11.0}


def test_zonal_shares_follow_the_contract():
    per_zone = {
        "WEST": {2020: {"forest_ha": 6.0, "loss_ha": None}, 2021: {"forest_ha": 5.0, "loss_ha": 1.0},
                 2022: {"forest_ha": 3.0, "loss_ha": 2.0}},
        "EAST": {2020: {"forest_ha": 0.0, "loss_ha": 0.0}, 2021: {"forest_ha": 0.0, "loss_ha": 0.0}},
    }
    shares = stats.zonal_shares(per_zone, {"WEST": 9.0, "EAST": 11.0})
    west = shares["WEST"]
    assert west[2020]["zone_land_ha"] == 9.0
    assert west[2021]["forest_pct"] == pytest.approx(100 * 5.0 / 9.0)
    # loss of the year / forest of the previous year
    assert west[2021]["loss_pct"] == pytest.approx(100 * 1.0 / 6.0)
    assert west[2022]["loss_pct"] == pytest.approx(100 * 2.0 / 5.0)
    # No previous year: null, not 0.
    assert west[2020]["loss_pct"] is None
    # Previous year's forest is 0: null (the regression of toolchain.md §3).
    assert shares["EAST"][2021]["loss_pct"] is None
    assert shares["EAST"][2021]["forest_pct"] == 0.0


def test_a_missing_or_zero_denominator_gives_null():
    per_zone = {"A": {2020: {"forest_ha": 4.0, "loss_ha": 1.0}, 2021: {"forest_ha": None, "loss_ha": 1.0},
                      2022: {"forest_ha": 2.0, "loss_ha": 1.0}},
                "B": {2020: {"forest_ha": 4.0, "loss_ha": 1.0}}}
    shares = stats.zonal_shares(per_zone, {"A": 10.0, "B": 0.0})
    # A zone with loss but no forest metric: no forest share.
    assert shares["A"][2021]["forest_pct"] is None
    # The forest of the year before is missing: no loss share.
    assert shares["A"][2022]["loss_pct"] is None
    # A gap in the years is missing too, not the forest of an earlier year.
    gap = stats.zonal_shares({"A": {2020: {"forest_ha": 4.0, "loss_ha": 0.0},
                                    2022: {"forest_ha": 3.0, "loss_ha": 1.0}}}, {"A": 10.0})
    assert gap["A"][2022]["loss_pct"] is None
    # No land in the zone: no forest share.
    assert shares["B"][2020]["forest_pct"] is None
    # A zone without a land figure at all.
    assert stats.zonal_shares({"C": {2020: {"forest_ha": 1.0, "loss_ha": 0.0}}}, {})["C"][2020] == {
        "zone_land_ha": None, "forest_pct": None, "loss_pct": None}


# --------------------------------------------------------------------------
# §4: summary_table and deforestation_rate
# --------------------------------------------------------------------------
def _masks():
    forest_2020 = np.zeros(SHAPE, bool)
    forest_2020[1:, :4] = True  # 12 ha
    forest_2021 = forest_2020.copy()
    forest_2021[1, :2] = False  # 10 ha
    loss_2021 = forest_2020 & ~forest_2021
    return {2020: np.zeros(SHAPE, bool), 2021: loss_2021}, {2020: forest_2020, 2021: forest_2021}


def test_summary_table_adds_roi_land_and_forest_share():
    loss, forest = _masks()
    table = stats.summary_table(loss, forest, profile=_profile(), datamask=_datamask())
    records = {int(r["year"]): r for r in table.to_dict(orient="records")}
    assert records[2021]["roi_land_ha"] == 20.0
    assert records[2020]["forest_pct"] == pytest.approx(100 * 12.0 / 20.0)
    assert records[2021]["forest_pct"] == pytest.approx(100 * 10.0 / 20.0)
    assert records[2021]["rate_pct"] == pytest.approx(100 * 2.0 / 12.0, abs=1e-3)
    # The first year has no previous year: no rate.
    assert _missing(records[2020]["rate_pct"])


def test_summary_table_without_datamask_has_no_shares():
    loss, forest = _masks()
    records = stats.summary_table(loss, forest, profile=_profile()).to_dict(orient="records")
    assert all(_missing(r["roi_land_ha"]) and _missing(r["forest_pct"]) for r in records)


def test_deforestation_rate_is_none_without_previous_forest():
    """K24: None, not 0.0, when the previous year's forest is 0 or missing."""
    rates = stats.deforestation_rate({2019: 0.0, 2020: 10.0, 2021: 8.0, 2022: None, 2023: 5.0})
    assert rates[2020] is None  # previous forest 0
    assert rates[2021] == pytest.approx(20.0)
    assert rates[2022] is None  # this year's forest missing
    assert rates[2023] is None  # previous forest missing
    assert 2019 not in rates


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


# --------------------------------------------------------------------------
# §1: risk_zone_summary (T040) -- geographic grid, latitude-varying areas
# --------------------------------------------------------------------------
def _geo_profile():
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    # 3 rows x 6 columns of 1 degree from 12 N: each row has its own area.
    return {"transform": from_origin(-14.0, 12.0, 1.0, 1.0), "crs": CRS.from_epsg(4326), "height": 3, "width": 6}


def _row_area(row):
    """Hand-computed pixel area (ha) of a row: 1 deg x 1 deg at its centre latitude."""
    lat = 12.0 - (row + 0.5)
    return 111_320.0 * math.cos(math.radians(lat)) * 111_320.0 / 10_000.0


def _three_zones():
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame({"shapeID": ["A", "B", "C"]},
                            geometry=[box(-14, 9, -12, 12), box(-12, 9, -10, 12), box(-10, 9, -8, 12)],
                            crs="EPSG:4326")


def test_risk_zone_summary_on_three_zones():
    forest = np.zeros((3, 6), bool)
    forest[:, 0:2] = True      # A: all forest, all scored
    forest[:, 4:6] = True      # C: all forest, one column scored; B: no forest
    risk = np.full((3, 6), np.nan, dtype="float32")
    risk[:, 0] = [0.1, 0.5, 0.9]
    risk[:, 1] = [0.3, 0.3, 0.3]
    risk[:, 4] = [0.2, 0.4, 0.6]
    table = stats.risk_zone_summary(risk, forest, _three_zones(), _geo_profile())
    rows = {r["zone_id"]: r for r in table.to_dict(orient="records")}
    assert list(table.columns) == ["zone_id", "forest_ha", "scored_forest_ha", "mean_p", "scored_share"]

    areas = [_row_area(r) for r in range(3)]
    a_forest = 2 * sum(areas)
    assert rows["A"]["forest_ha"] == pytest.approx(a_forest, rel=1e-6)
    assert rows["A"]["scored_forest_ha"] == pytest.approx(a_forest, rel=1e-6)
    weighted = sum((risk[r, 0] + risk[r, 1]) * areas[r] for r in range(3)) / a_forest
    assert rows["A"]["mean_p"] == pytest.approx(weighted, rel=1e-5)
    # Area-weighted, not a plain pixel mean: the rows differ in area.
    assert rows["A"]["mean_p"] != pytest.approx(float(np.nanmean(risk[:, 0:2])), rel=1e-9)
    assert rows["A"]["scored_share"] == pytest.approx(1.0)

    # No forest: nothing scored, both NaN.
    assert rows["B"]["forest_ha"] == 0.0 and rows["B"]["scored_forest_ha"] == 0.0
    assert math.isnan(rows["B"]["mean_p"]) and math.isnan(rows["B"]["scored_share"])

    c_scored = sum(areas)
    assert rows["C"]["scored_forest_ha"] == pytest.approx(c_scored, rel=1e-6)
    assert rows["C"]["scored_share"] == pytest.approx(0.5, rel=1e-6)
    assert rows["C"]["mean_p"] == pytest.approx(sum(risk[r, 4] * areas[r] for r in range(3)) / c_scored, rel=1e-5)


def test_risk_zone_summary_forest_but_none_scored():
    forest = np.ones((3, 6), bool)
    risk = np.full((3, 6), np.nan, dtype="float32")
    rows = {r["zone_id"]: r for r in stats.risk_zone_summary(
        risk, forest, _three_zones(), _geo_profile()).to_dict(orient="records")}
    assert math.isnan(rows["A"]["mean_p"]) and rows["A"]["scored_forest_ha"] == 0.0
    assert rows["A"]["scored_share"] == 0.0


# --------------------------------------------------------------------------
# §2: pipeline result keys (T040)
# --------------------------------------------------------------------------
def _pipeline(monkeypatch, descriptive_sample_ok=True):
    import sys

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from landuse_tool.deforestation_tool import drivers as DR
    from landuse_tool.deforestation_tool import pipeline
    from landuse_tool.deforestation_tool import config as dcfg

    monkeypatch.setattr(dcfg, "WFP_PRICES_SOURCE", {"enabled": False}, raising=False)

    class Model:
        feature_importances_ = None

    calls = {"n": 0}

    def sample_training(loss, stable, stack, samples_per_class=0):
        calls["n"] += 1
        names = list(stack)
        # The second call samples the descriptive stack.
        if calls["n"] == 2 and not descriptive_sample_ok:
            return None, None, names
        return np.zeros((20, len(names))), np.array([0, 1] * 10), names

    monkeypatch.setattr(DR, "sample_training", sample_training)
    monkeypatch.setattr(DR, "fit_risk_model", lambda X, y, model_type=None: (Model(), {"n_samples": 20}))
    monkeypatch.setattr(DR, "driver_importance", lambda model, names: {name: 1.0 / len(names) for name in names})
    monkeypatch.setattr(DR, "predict_risk_map",
                        lambda model, stack, names, forest: np.where(forest, 0.5, np.nan).astype("float32"))
    treecover = np.full(SHAPE, 80.0)
    lossyear = np.zeros(SHAPE)
    lossyear[0, 0] = 21
    datamask = _datamask()
    return pipeline.run_from_arrays(
        treecover=treecover, lossyear=lossyear, profile=_profile(), years=[2020, 2021],
        shared_predictors={"DEM": np.ones(SHAPE, dtype="float32")}, canopy_threshold=30, datamask=datamask)


def test_the_pipeline_records_which_model_ranked_the_drivers(monkeypatch):
    results = _pipeline(monkeypatch)
    assert results["importance_model"] == "descriptive"
    assert "dist_forest_edge" in results["driver_importance"]
    np.testing.assert_array_equal(results["forest_now"], results["forest_by_year"][2021])
    np.testing.assert_array_equal(results["datamask"], _datamask())
    assert results["summary"]["roi_land_ha"].iloc[0] == 20.0


def test_the_pipeline_falls_back_to_the_risk_model(monkeypatch):
    results = _pipeline(monkeypatch, descriptive_sample_ok=False)
    assert results["importance_model"] == "risk"
    assert "dist_forest_edge" not in results["driver_importance"]


# --------------------------------------------------------------------------
# §5: class_areas (T062)
# --------------------------------------------------------------------------
def test_class_areas_on_a_synthetic_cog(tmp_path):
    import sys

    import rasterio
    from rasterio.transform import from_origin

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from landuse_tool.class_areas import class_areas

    values = np.array([[1, 2, 2, 10, 255, 5],
                       [2, 2, 1, 1, 5, 5],
                       [7, 255, 10, 2, 2, 2]], dtype="uint8")
    path = tmp_path / "lc.tif"
    with rasterio.open(path, "w", driver="GTiff", height=3, width=6, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(-14.0, 12.0, 1.0, 1.0), nodata=255,
                       tiled=True, blockxsize=16, blockysize=16) as dst:
        dst.write(values, 1)
    classes = {1: "Water", 2: "Trees", 5: "Crops", 7: "Built area", 11: "Rangeland"}
    areas = class_areas(str(path), classes)
    a = [_row_area(r) for r in range(3)]

    assert areas[1] == pytest.approx(a[0] + 2 * a[1], rel=1e-6)
    assert areas[2] == pytest.approx(2 * a[0] + 2 * a[1] + 3 * a[2], rel=1e-6)
    assert areas[5] == pytest.approx(a[0] + 2 * a[1], rel=1e-6)
    assert areas[7] == pytest.approx(a[2], rel=1e-6)
    assert areas[11] == 0.0  # a legend class with no pixel is 0, not absent
    # Clouds (10) are not a class: they are the no-data area. Nodata (255) is nowhere.
    assert 10 not in areas
    assert areas["no_data"] == pytest.approx(a[0] + a[2], rel=1e-6)
