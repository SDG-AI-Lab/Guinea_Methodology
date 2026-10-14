"""
Offline unit tests -- no network, no API keys required.

Run:  pytest -q

These load config.py and aws_utils.py directly (they have no package-relative
imports), so the tests run even before the rest of the package is wired up.
"""

import os
import importlib.util

_PKG = os.path.join(os.path.dirname(__file__), "..", "landuse_tool")


def _load(name):
    path = os.path.join(_PKG, name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


config = _load("config")
azure_utils = _load("azure_utils")


# --------------------------------------------------------------------------
# Config catalog integrity
# --------------------------------------------------------------------------
def test_enabled_lists_nonempty():
    assert config.get_enabled_target_rasters(), "no targets enabled"
    assert config.get_enabled_predictor_rasters(), "no predictors enabled"


def test_every_enabled_gee_target_is_wellformed():
    for k in config.get_enabled_target_rasters():
        cfg = config.TARGET_RASTERS_CONFIG[k]
        if cfg.get("source") != "GEE":
            continue
        assert cfg.get("asset"), f"{k}: missing asset id"
        # Must know how to pick a single band per year.
        assert cfg.get("band") or cfg.get("band_per_year"), f"{k}: no band spec"
        assert cfg.get("reducer"), f"{k}: no reducer"
        assert cfg.get("temporal"), f"{k}: no temporal class"


def test_no_placeholder_urls_as_assets():
    # The old S2TSLULC bug stored a docs URL as the collection id.
    for cfg in config.TARGET_RASTERS_CONFIG.values():
        asset = cfg.get("asset", "")
        assert not asset.startswith("http"), f"asset looks like a URL: {asset}"


def test_glc_fcs30d_band_per_year_config():
    cfg = config.TARGET_RASTERS_CONFIG["GLC_FCS30D"]
    assert cfg["band_per_year"] is True
    assert cfg["band_base_year"] == 2000


def test_validate_required_keys_reports_when_missing(monkeypatch):
    # Clear the required env vars and confirm the validator flags them.
    for var in ("OPENTOPO_API_KEY", "GEE_SA_KEY", "GEE_PROJECT_ID"):
        monkeypatch.delenv(var, raising=False)
    # config caches gee_project_id at import; re-read from env for the test.
    config.PROCESSING_CONFIG["gee_project_id"] = os.environ.get("GEE_PROJECT_ID")
    problems = config.validate_required_keys()
    assert any("GEE_PROJECT_ID" in p or "GEE_SA_KEY" in p for p in problems)


# --------------------------------------------------------------------------
# Blob cache (local mode, no container => pure local behavior)
# --------------------------------------------------------------------------
def test_blobcache_local_roundtrip(tmp_path):
    cache = azure_utils.BlobCache(container=None, prefix="t", local_root=str(tmp_path))
    key = "targets/example_2020.tif"
    assert cache.exists(key) is False
    p = cache.local_path(key)
    with open(p, "w") as f:
        f.write("dummy")
    assert cache.exists(key) is True
    assert cache.upload(key, p) is False           # no container -> no-op
    assert key in cache.list_keys("targets")
