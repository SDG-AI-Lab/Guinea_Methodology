"""The improved method rebuilt with predictors left out or added (notebook 12).

The improved method (notebook 09, section 4): the deployed random forest with contagion features,
a logistic regression, and the copy of FLUS's network, rank-averaged with equal weight. Each part
is retrained here with the same predictor change, so the test measures what one predictor adds to
the whole method.
"""
import numpy as np

import benchmark_scoring as BS
import esri_methods as E
import flus_ann as FA
import nb_helpers as H


CANDIDATES = H.DATA / "candidates"
MINING_FILE = CANDIDATES / "mining_maus2022" / "global_mining_polygons_v2.gpkg"
FIRE_DIR = CANDIDATES / "fire_mcd64a1"


def mining_distance():
    """log(1 + km) to the nearest mining area mapped by Maus et al. (2022) from 2019 Sentinel-2
    images (large-scale and artisanal mines), on the ESRI grid. Static: the mines of 2019."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt
    out = CANDIDATES / "mining_log_km.tif"
    profile = E.grid()
    if not out.exists():
        t = profile["transform"]
        west, north = t.c, t.f
        east, south = west + t.a * profile["width"], north + t.e * profile["height"]
        mines = gpd.read_file(MINING_FILE, bbox=(west, south, east, north)).to_crs(profile["crs"])
        inside = rasterize([(geom, 1) for geom in mines.geometry], out_shape=(profile["height"], profile["width"]),
                           transform=t, all_touched=True, dtype="uint8").astype(bool)
        lat = np.deg2rad(north + t.e * (np.arange(profile["height"]) + 0.5))
        km_y, km_x = abs(t.e) * 111.32, t.a * 111.32 * np.cos(lat).mean()       # degrees to km at Guinea's latitude
        km = distance_transform_edt(~inside, sampling=(km_y, km_x))
        E.write_float(np.log1p(km).astype(np.float32), out, profile)
    return H.read_tif(out)[0]


def _earth_engine():
    import ee, json
    from dotenv import dotenv_values
    env = dotenv_values(H.LANDUSE_TOOL_PARENT / ".env")
    key = next(k for k in (H.LANDUSE_TOOL_PARENT / "landuse_tool" / "credentials").glob("*.json")
               if "client_email" in json.loads(k.read_text()))
    ee.Initialize(ee.ServiceAccountCredentials(json.loads(key.read_text())["client_email"], str(key)),
                  project=env.get("GEE_PROJECT_ID") or json.loads(key.read_text())["project_id"])
    return ee


def burned(year):
    """1 where MODIS MCD64A1 (collection 6.1, 500 m; Giglio et al. 2018) mapped a burn in `year`,
    on the ESRI grid (nearest neighbour). Downloaded once from Earth Engine."""
    import requests
    path = FIRE_DIR / f"burned_{year}.tif"
    if not path.exists():
        ee = _earth_engine()
        profile = E.grid()
        t = profile["transform"]
        image = (ee.ImageCollection("MODIS/061/MCD64A1").filterDate(f"{year}-01-01", f"{year + 1}-01-01")
                 .select("BurnDate").max().gt(0).unmask(0).toByte())
        url = image.getDownloadURL({"crs": "EPSG:4326", "crs_transform": [t.a, t.b, t.c, t.d, t.e, t.f],
                                    "dimensions": f"{profile['width']}x{profile['height']}", "format": "GEO_TIFF"})
        FIRE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(requests.get(url, timeout=600).content)
    return H.read_tif(path)[0].astype(np.float32)


def fire_layers(year, mask):
    """Burned in `year`, and the share of the five years to `year` burned: both known at `year`."""
    recent = np.mean([burned(y) for y in range(year - 4, year + 1)], axis=0)
    return np.stack([np.where(mask, burned(year), 0), np.where(mask, recent, 0)]).astype(np.float32)


def mining_layers(year, mask):
    return np.where(mask, mining_distance(), 0)[None].astype(np.float32)


def scale01(layer, valid):
    """0-1 inside `valid` (min-max), 0 outside, as FLUS's inputs (benchmark_export.flus_factors)."""
    v = layer[valid]
    lo, hi = float(v.min()), float(v.max())
    return np.where(valid, (layer - lo) / (hi - lo) if hi > lo else 0.0, 0.0).astype(np.float32)


def improved_variant(tag, phase, seed=E.SEED, drop=(), extra=None, log=print):
    """Train and score the improved method for one phase with the base predictors in `drop` left
    out, and with `extra` added: a callable(year, mask) -> (k, H, W) of layers known at `year`
    (the interval's start when training, the forecast start when predicting, as the contagion
    features). Parts are cached in RUNS/<tag>_rf, _lr, _flus; the average in RUNS/<tag>."""
    done = E.load_result(tag, phase)
    if done:
        return done
    profile = E.grid()
    guinea = H.guinea_mask(profile)
    _, t_start, _ = E.phase_years(phase)
    rf_feats = E.contagion_features if extra is None else \
        (lambda y, t, m: np.vstack([E.contagion_features(y, t, m), extra(y, m)]))
    lr_feats = None if extra is None else (lambda y, t, m: extra(y, m))
    for part, learner, feats in (("rf", "rf", rf_feats), ("lr", "logistic", lr_feats)):
        if not E.load_result(f"{tag}_{part}", phase):
            E.run_deployed(phase, "guinea", "dated", "start", tag=f"{tag}_{part}", seed=seed, learner=learner,
                           transition_features=feats, drop=drop, log=log)
    prob = E.RUNS / f"{tag}_flus" / phase / "Probability-of-occurrence.tif"
    if not prob.exists():
        valid = H.read_tif(BS.BENCH / "inputs" / f"landcover_{t_start}.tif")[0] > 0
        layers = [] if extra is None else [scale01(a, valid) for a in extra(t_start, E.deployed_mask(profile))]
        FA.probability_map(t_start, prob, seed=seed, drop=drop, extra=layers, log=log)
    transitions = [tuple(x) for x in E.load_result("R0c_guinea_dated", phase)["transitions"]]
    flus = BS.class_probability_atlas(BS.band_files(prob, prob.parent / "bands"), transitions,
                                      E.esri_cached(t_start), guinea, prob.parent, profile, "flus")
    rf_lr = E.average_atlases(E.atlas_from(E.RUNS / f"{tag}_rf" / phase / "surfaces", "expander"),
                              E.atlas_from(E.RUNS / f"{tag}_lr" / phase / "surfaces", "expander"),
                              0.5, E.RUNS / tag / phase / "surfaces_rf_lr", profile)
    return E.run_average(tag, phase, rf_lr, flus, 2 / 3, log=log)
