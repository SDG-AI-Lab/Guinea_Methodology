"""
Offline toolchain tests for feature 006 -- no network, no API keys.

specs/006-plastic-waste-views/contracts/toolchain.md. Run in the api image
(the toolchain's dependencies, mapclassify included, live there; the folder is
mounted at /landuse_tool):

    docker compose exec -T api sh -c \\
        'cd /landuse_tool && pytest -q -p no:cacheprovider -p no:django plasticwaste_tool/test_toolchain_006.py'
"""

import importlib.util
import json
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(filename, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


breaks_script = _load("37_size_class_breaks.py", "size_class_breaks_37")


# --------------------------------------------------------------------------
# §2: 37_size_class_breaks.py (T017)
# --------------------------------------------------------------------------
def _groups():
    """Three clear groups of whole-pixel areas (400 m² pixels)."""
    small = [400, 400, 800, 400, 800, 400, 800, 400]
    medium = [4000, 4400, 4800, 4000, 4400]
    large = [20000, 22000, 24000]
    return small, medium, large


def test_size_class_breaks():
    small, medium, large = _groups()
    breaks = breaks_script.compute_breaks(small + medium + large, pixel_m2=400)
    # The upper bound of each lower group, whole pixels.
    assert breaks == [max(small), max(medium)]
    assert all(value % 400 == 0 for value in breaks)


def test_breaks_are_rounded_to_whole_pixels():
    breaks = breaks_script.compute_breaks([400, 410, 800, 3990, 4010, 4400, 19990, 20000, 21000], pixel_m2=400)
    assert all(value % 400 == 0 for value in breaks)


def test_a_collapsing_list_is_refused():
    with pytest.raises(breaks_script.BreaksRefused):
        breaks_script.compute_breaks([400, 400, 400, 800, 800], pixel_m2=400)  # two distinct areas, three classes
    with pytest.raises(breaks_script.BreaksRefused):
        # Distinct areas, but rounding puts both breaks on the same pixel count.
        breaks_script.compute_breaks([400, 401, 402, 403, 404, 405], pixel_m2=400)


def test_the_counts_must_be_the_methodology_s():
    by_region = {"conakry": [400.0] * 26, "fatala": [400.0] * 3}
    with pytest.raises(SystemExit, match="fatala: 3 patches, methodology 4"):
        breaks_script.check_counts(by_region, accept=False)
    assert breaks_script.check_counts(by_region, accept=True) == ["fatala: 3 patches, methodology 4"]


def test_the_config_block_is_written_between_the_markers(tmp_path):
    config = tmp_path / "config.py"
    config.write_text("from typing import Dict\nA = 1\n\n" + breaks_script.START + " (old) ---\n"
                      "SIZE_CLASS_CONFIG: Dict = {'breaks_m2': [1, 2]}\n" + breaks_script.END + "\nB = 2\n",
                      encoding="utf-8")
    block = breaks_script.config_block([800, 4000], n=148, computed_at="2026-09-26")
    breaks_script.write_config(block, config)
    namespace = {}
    exec(config.read_text(encoding="utf-8"), namespace)  # noqa: S102 - the file this test just wrote
    assert namespace["SIZE_CLASS_CONFIG"] == {
        "breaks_m2": [800, 4000], "method": "fisher-jenks", "k": 3,
        "reference": "national 2024-01, conakry+konkoure+fatala", "n": 148, "pixel_m2": 400,
        "computed_at": "2026-09-26"}
    assert namespace["A"] == 1 and namespace["B"] == 2


def test_the_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    features = [{"type": "Feature", "geometry": None, "properties": {"region": region, "area_m2": area}}
                for region, areas in (("conakry", [400] * 40 + [4000] * 30 + [20000] * 9),
                                      ("konkoure", [800] * 30 + [4400] * 30 + [22000] * 6),
                                      ("fatala", [400, 4000, 24000]))
                for area in areas]
    reference = tmp_path / "reference.geojson"
    reference.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
    config = tmp_path / "config.py"
    config.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(breaks_script, "CONFIG_PATH", config)
    assert breaks_script.main(["--dry-run", "--reference", str(reference)]) == 0
    assert "SIZE_CLASS_CONFIG" in capsys.readouterr().out
    assert config.read_text(encoding="utf-8") == "x = 1\n"


# --------------------------------------------------------------------------
# §3–§4: the Esri window's zoom and capture date (T049)
# --------------------------------------------------------------------------
IDENTIFY = os.path.join(_HERE, "test_data", "esri_identify_conakry.json")


class _Answer:
    def __init__(self, body=None, status=200, text=None):
        self._body, self.status_code = body, status
        self.text = text if text is not None else json.dumps(body)

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        if self._body is None:
            raise ValueError("not JSON")
        return self._body


def _identify(monkeypatch, answer=None, error=None):
    import requests

    calls = []

    def fake_get(url, params=None, timeout=None, **kwargs):
        calls.append({"url": url, "params": params, "timeout": timeout})
        if error is not None:
            raise error
        return answer

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


def test_esri_capture_date_at_the_fetched_zoom(monkeypatch):
    from plasticwaste_tool.dumpsite_tool import inference

    with open(IDENTIFY, encoding="utf-8") as handle:
        recorded = json.load(handle)
    calls = _identify(monkeypatch, _Answer(recorded))
    answer = inference.esri_capture_date(9.6, -13.6, 17)
    assert answer == {"status": "provided", "date": "2025-03-15", "source_info": "Vantor WV03, 0.31 m"}
    assert calls[0]["url"].endswith("/World_Imagery/MapServer/identify") and calls[0]["timeout"] == 60
    assert calls[0]["params"]["f"] == "json" and calls[0]["params"]["returnGeometry"] == "false"
    # The zoom 11 band says Null: no date is given, never a guessed one.
    assert inference.esri_capture_date(9.6, -13.6, 11) == {"status": "not_provided"}
    # No band covers zoom 20.
    assert inference.esri_capture_date(9.6, -13.6, 20) == {"status": "not_provided"}


def test_esri_capture_date_never_raises(monkeypatch):
    import requests

    from plasticwaste_tool.dumpsite_tool import inference

    _identify(monkeypatch, error=requests.Timeout("read timed out"))
    failed = inference.esri_capture_date(9.6, -13.6, 17)
    assert failed["status"] == "lookup_failed" and "timed out" in failed["error"]
    _identify(monkeypatch, _Answer(None, text="<html>busy</html>"))
    assert inference.esri_capture_date(9.6, -13.6, 17)["status"] == "lookup_failed"
    _identify(monkeypatch, _Answer({"results": [{"attributes": {"MinMapLevel": "x"}}]}))
    assert inference.esri_capture_date(9.6, -13.6, 17)["status"] in ("lookup_failed", "not_provided")
    _identify(monkeypatch, _Answer({"error": {"code": 500}}, status=500))
    assert inference.esri_capture_date(9.6, -13.6, 17)["status"] == "lookup_failed"


def test_fetch_tiles_esri_returns_the_zoom_used(monkeypatch, tmp_path):
    import sys
    import types

    import numpy as np
    import pandas as pd

    from plasticwaste_tool.dumpsite_tool import inference

    # contextily fetches the tiles; offline, a stand-in answers for it.
    cx = types.SimpleNamespace(providers=types.SimpleNamespace(Esri=types.SimpleNamespace(WorldImagery="esri")))
    monkeypatch.setitem(sys.modules, "contextily", cx)

    rng = np.random.default_rng(3)
    real = (rng.random((64, 64, 3)) * 255).astype("uint8")
    blank = np.full((64, 64, 3), 200, dtype="uint8")

    def bounds2img(minx, miny, maxx, maxy, zoom, source=None, ll=True):
        # No imagery at 17 for the first tile: the placeholder, so one zoom down.
        return (blank if (zoom == 17 and minx < 0.5) else real), None

    cx.bounds2img = bounds2img
    tiles = pd.DataFrame([{"tile_id": "r1c1", "minx": 0.0, "miny": 0.0, "maxx": 0.001, "maxy": 0.001},
                          {"tile_id": "r1c2", "minx": 1.0, "miny": 0.0, "maxx": 1.001, "maxy": 0.001}])
    paths, zoom_used = inference.fetch_tiles_esri(tiles, str(tmp_path), size_px=32, zoom=17)
    assert [os.path.basename(p) for p in paths] == ["r1c1.png", "r1c2.png"]
    assert zoom_used == {"r1c1": 16, "r1c2": 17}


# --------------------------------------------------------------------------
# §1: river coverage (T066; K36 regression)
# --------------------------------------------------------------------------
RIVER_RUN = os.path.join(_HERE, "manual_qa", "runs", "river_fdi", "river_fdi_2024-01-01_2024-02-01.json")
RIVER_AOIS = os.path.join(_HERE, "data", "rivers", "river_aois.json")


def test_river_coverage():
    """The January 2024 study run, with the tiles refused at the free preflight
    added as refused by the water guard, as river_fdi_workflow.ipynb Section 5
    and the methodology count them (117 corridor tiles; 49 valid; 3,295.3 of
    5,971.9 km)."""
    from plasticwaste_tool.fdi_tool.coverage import river_coverage

    with open(RIVER_RUN, encoding="utf-8") as handle:
        results = json.load(handle)["results"]
    with open(RIVER_AOIS, encoding="utf-8") as handle:
        corridors = json.load(handle)["aois"]
    ran = {row["aoi"] for row in results}
    results = results + [{"aoi": key, "assessed": False} for key in corridors if key not in ran]
    coverage = river_coverage(results, corridors)
    assert coverage["tiles_by_outcome"] == {"valid": 49, "below_water_floor": 65, "no_water_mask": 0,
                                            "implausible": 3}
    assert round(coverage["km_assessed"], 1) == 3295.3 and round(coverage["km_total"], 1) == 5971.9
    assert coverage["coverage_fraction"] == pytest.approx(coverage["km_assessed"] / coverage["km_total"])
    assert coverage["method"] == "river_coverage"
    assert len(coverage["outcome_by_tile"]) == 117


def test_river_coverage_outcomes_and_the_empty_case():
    from plasticwaste_tool.fdi_tool.coverage import river_coverage

    corridors = {name: {"length_km": 10.0, "main_riv": 1} for name in "abcde"}
    results = [
        {"aoi": "a", "assessed": True, "threshold_fdi": 0.02, "threshold_ndvi": 0.1},
        {"aoi": "b", "assessed": False, "threshold_fdi": 0.02, "threshold_ndvi": 0.1},
        {"aoi": "c", "assessed": True, "threshold_fdi": float("nan"), "threshold_ndvi": 0.1},
        {"aoi": "d", "assessed": True, "threshold_fdi": 0.02, "threshold_ndvi": 55.7},
        {"aoi": "e", "assessed": True, "threshold_fdi": 0.02, "threshold_ndvi": None},
        {"aoi": "a", "assessed": True, "threshold_fdi": 0.02, "threshold_ndvi": 0.1},  # a --resume duplicate
    ]
    coverage = river_coverage(results, corridors)
    assert coverage["outcome_by_tile"] == {"a": "valid", "b": "below_water_floor", "c": "no_water_mask",
                                           "d": "implausible", "e": "implausible"}
    assert coverage["km_assessed"] == 10.0 and coverage["km_total"] == 50.0
    empty = river_coverage([], {})
    assert empty["km_total"] == 0 and empty["coverage_fraction"] is None
