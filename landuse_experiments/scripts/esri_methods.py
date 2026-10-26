"""The deployed Land Use method on ESRI, rebuilt for controlled experiments.

Faithful to landuse_analysis_improved.ipynb (the run behind the methodology
document's 13.80%): per-transition Expander and Patcher Random Forests, trained on
the last interval of the training years, with the 12 predictors in the order the
notebook actually stacked them. Allocation and scoring use the deployed
landuse_tool functions unchanged.

Every departure from the deployed run is a named switch, so each experiment
changes exactly one thing:

  sample_domain   "grid"   : training pixels drawn from the whole bounding box (deployed)
                  "guinea" : training pixels inside Guinea only
  predictor_year  "latest" : one static stack, latest cached year (deployed: NDVI and
                             precipitation 2026, soil moisture 2025)
                  "dated"  : each interval's predictors from that interval's start year
  neighbourhood   "end"    : Expander density from the interval's end map (deployed)
                  "start"  : density from the interval's start map
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation, correlate
from sklearn.ensemble import RandomForestClassifier

import nb_helpers as H

ESRI_DIR = H.DATA / "esri"
PRED_DIR = H.DATA / "esri_predictors_src"
RUNS = H.OUTPUTS / "esri_runs"
MIN_PIXELS = 1000          # notebook's select_significant_transitions threshold
RADIUS = 5                 # notebook's run_expander_suitability(..., radius=5) call
SEED = 42
FIRST_YEAR = 2017          # first ESRI annual map; "new" change = class not held since then

# Order the deployed notebook ACTUALLY stacked (cell 3 output): rasters in fetch
# order, then the derived slope/aspect, then the rasterized vectors. Its label list
# assumed a different order; see notebook 06 section 1.
PREDICTORS = ["Elevation", "Population density", "Precipitation", "Soil moisture", "AGB",
              "NDVI", "Slope", "Aspect", "Roads (distance)", "Waterways (distance)",
              "POI (density)", "Protected areas"]
LATEST = {"NDVI": 2026, "PRECIPITATION": 2026, "SOIL_MOISTURE": 2025}
DATED_SOURCES = {"Precipitation": "PRECIPITATION", "Soil moisture": "SOIL_MOISTURE", "NDVI": "NDVI"}
STATIC_FILES = {"Elevation": "DEM_GEE.tif", "Population density": "POPULATION_DENSITY_2020.tif",
                "AGB": "AGB_NASA_2010.tif", "Slope": "SLOPE_derived.tif", "Aspect": "ASPECT_derived.tif",
                "Roads (distance)": "ROADS_rasterized.tif", "Waterways (distance)": "WATERWAYS_rasterized.tif",
                "POI (density)": "POI_rasterized.tif", "Protected areas": "PROTECTED_AREAS_rasterized.tif"}


# --------------------------------------------------------------------------
# Grid, masks, maps

def grid():
    with rasterio.open(ESRI_DIR / "ESRI_LULC_2017.tif") as src:
        return src.profile


def esri(year):
    with rasterio.open(ESRI_DIR / f"ESRI_LULC_{year}.tif") as src:
        return src.read(1)


def deployed_mask(profile):
    """The notebook's study mask: valid ESRI codes AND the ADM0 polygon with all_touched=True."""
    import json
    from rasterio.features import rasterize
    shapes = [(f["geometry"], 1) for f in json.loads(H.BOUNDARY.read_text())["features"]]
    poly = rasterize(shapes, out_shape=(profile["height"], profile["width"]),
                     transform=profile["transform"], fill=0, all_touched=True, dtype="uint8").astype(bool)
    first = esri(2017)
    return poly & ~np.isin(first, [0, 10, 254, 255])


def valid_classes(*maps):
    return ~np.logical_or.reduce([np.isin(m, H.ESRI_INVALID) for m in maps])


# --------------------------------------------------------------------------
# Predictors

def _aligned(path, profile):
    dst = np.zeros((profile["height"], profile["width"]), dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=profile["transform"], dst_crs=profile["crs"],
                  resampling=Resampling.nearest if src.dtypes[0] == "uint8" else Resampling.bilinear)
    return dst


def predictor_stack(profile, mask, year="latest", drop=()):
    """(12, H, W) float32, as the notebook built X_model_ready: aligned, NaN outside
    `mask`, then NaN -> 0. year="latest" reproduces the deployed static stack; an
    integer uses that year for the three time-varying layers (no later than it).
    drop: predictor names left out (the drop-one test, notebook 12)."""
    layers = []
    for name in (n for n in PREDICTORS if n not in drop):
        if name in DATED_SOURCES:
            key = DATED_SOURCES[name]
            y = LATEST[key] if year == "latest" else min(int(year), LATEST[key])
            path = PRED_DIR / f"{key}_{y}.tif"
        else:
            path = PRED_DIR / STATIC_FILES[name]
        layers.append(np.where(mask, _aligned(path, profile), np.nan))
    return np.nan_to_num(np.stack(layers).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def neighbourhood(land_cover, to_class, mask, radius=RADIUS):
    """Share of a circular window (radius in pixels) that is `to_class`, NaN outside
    `mask`. Same values as the notebook's generic_filter(np.mean) with its default
    'reflect' boundary, computed as a convolution (minutes faster per class)."""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (x ** 2 + y ** 2 <= radius ** 2).astype(np.float32)
    density = correlate((land_cover == to_class).astype(np.float32), kernel, mode="reflect") / kernel.sum()
    return np.where(mask, density, np.nan).astype(np.float32)


# --------------------------------------------------------------------------
# Transitions and models

def significant_transitions(start, end, domain, min_pixels=MIN_PIXELS):
    """(from, to) pairs with at least min_pixels changes inside `domain` (notebook rule)."""
    present = np.unique(np.concatenate([start[domain], end[domain]]))
    classes = [c for c in present if c not in H.ESRI_INVALID]
    out = []
    for f in classes:
        for t in classes:
            if f != t and int(((start == f) & (end == t) & domain).sum()) >= min_pixels:
                out.append((int(f), int(t)))
    return out


def _fit(X, idx_change, idx_persist, seed=SEED, learner="rf"):
    """learner "rf": the deployed Random Forest. "logistic": standardised logistic regression,
    the transition-potential model of MOLUSCE and CLUE-S (notebook 00). "mlp": a neural
    network with one hidden layer of 12 nodes, the size FLUS's manual uses (notebook 09)."""
    if learner in ("logistic", "mlp"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler
        clf = (LogisticRegression(max_iter=2000, class_weight="balanced") if learner == "logistic"
               else MLPClassifier((12,), max_iter=1000, early_stopping=True, random_state=seed))
        model = make_pipeline(FunctionTransformer(np.nan_to_num), StandardScaler(), clf)
    else:
        model = RandomForestClassifier(n_estimators=100, n_jobs=8, random_state=seed, class_weight="balanced")
    labels = np.concatenate([np.ones(len(idx_change)), np.zeros(len(idx_persist))])
    model.fit(X[np.concatenate([idx_change, idx_persist])], labels)
    return model


def train_growth_models(features, start, end, f, t, domain, nbh, rng, seed=SEED, learner="rf"):
    """Expander and Patcher models for f->t, as run_expander/patcher_suitability did.
    features: (n_features, H, W) base stack; nbh: (H, W) neighbourhood density for the
    Expander (appended as its last column). Returns {"expander": model|None, "patcher": ...}."""
    change = (start == f) & (end == t) & domain
    adjacent = binary_dilation(start == t, structure=np.ones((3, 3)))
    persist_idx = np.flatnonzero(((start == f) & (end == f) & domain).ravel())
    models = {}
    for mode, samples, X in (("expander", change & adjacent, np.vstack([features, nbh[None]])),
                             ("patcher", change & ~adjacent, features)):
        idx = np.flatnonzero(samples.ravel())
        n = min(len(idx), len(persist_idx))
        if len(idx) == 0 or n == 0:
            models[mode] = None
            continue
        persist = rng.choice(persist_idx, n, replace=False)
        models[mode] = (_fit(X.reshape(X.shape[0], -1).T, idx, persist, seed, learner), X.shape[0])
    return models


def predict_surface(model_info, features, nbh, start, f, where, batch=200_000):
    """Suitability for f->t on pixels of class f inside `where`; -1 elsewhere."""
    model, n_features = model_info
    X = features if n_features == features.shape[0] else np.vstack([features, nbh[None]])
    pick = np.flatnonzero(((start == f) & where).ravel())
    out = np.full(start.size, -1.0, dtype=np.float32)
    flat = X.reshape(X.shape[0], -1)
    for i in range(0, len(pick), batch):
        rows = pick[i:i + batch]
        out[rows] = model.predict_proba(flat[:, rows].T)[:, 1]
    return out.reshape(start.shape)


# lossless; uncompressed surfaces filled the disk (34 MB each, 9 MB compressed)
COMPRESS = dict(compress="deflate", tiled=True, blockxsize=512, blockysize=512)


def write_float(array, path, profile):
    p = dict(profile); p.update(dtype="float32", count=1, nodata=-1.0, **COMPRESS, predictor=3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array.astype(np.float32), 1)
    return str(path)


def write_classes(array, path, profile):
    p = dict(profile); p.update(dtype="uint8", count=1, nodata=255, **COMPRESS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(array.astype(np.uint8), 1)
    return str(path)


# --------------------------------------------------------------------------
# One full run of the deployed method for one phase

def phase_years(phase):
    """'validation': train 2022->2023, forecast 2023->2024. 'test': train 2023->2024,
    forecast 2024->2025. 'backtest': train 2021->2022, forecast 2022->2023, a second check
    never used to choose anything (notebook 11)."""
    return {"backtest": (2021, 2022, 2023), "validation": (2022, 2023, 2024), "test": (2023, 2024, 2025)}[phase]


def allocate_and_score(start_map, observed, surfaces_by_mode, guinea, out_dir, profile, counts=None):
    """Observed gross demand inside Guinea, Guinea-only eligibility, deployed allocator.
    counts: optional transition counts (DataFrame, from-class rows, to-class columns) to
    allocate instead of the observed ones, e.g. projected_counts (notebook 10)."""
    given = counts
    from landuse_tool import change_analysis as ca, prediction as pred
    import logging
    logging.getLogger("landuse_tool.prediction").setLevel(logging.ERROR)
    out_dir = Path(out_dir)
    b_path = write_classes(np.where(guinea, start_map, 255), out_dir / "start_guinea.tif", profile)
    o_path = write_classes(np.where(guinea, observed, 255), out_dir / "observed_guinea.tif", profile)
    _, counts, _ = ca.calculate_transition_matrix(b_path, o_path, nodata_value=255)
    keep = [c for c in counts.index if c not in H.ESRI_INVALID]
    counts = counts.loc[keep, keep] if given is None else given
    results, maps = {}, {}
    for mode, atlas in surfaces_by_mode.items():
        d = out_dir / mode.replace(" ", "_")
        d.mkdir(parents=True, exist_ok=True)
        path = pred.run_allocation_simulation(lc_end_file=b_path, transition_counts=counts,
                                              suitability_paths=atlas, temp_dir=str(d), stochastic=False)
        sim = H.read_tif(path)[0]
        maps[mode] = sim
        results[mode] = H.score(start_map, observed, sim, guinea)
    return results, maps


def run_deployed(phase, sample_domain="grid", predictor_year="latest", neighbourhood_from="end",
                 tag=None, transition_features=None, seed=SEED, learner="rf", drop=(), log=print):
    """Train the deployed method for one phase and score it inside Guinea.

    transition_features: optional callable(year, to_class, mask) -> (k, H, W) appended to
    the base predictors of every model for that transition, built from maps up to `year`
    (the interval's start year when training, the forecast start year when predicting).
    drop: base predictors left out (notebook 12).
    Returns scores per growth mode and the paths of the surfaces.
    """
    import time
    t_prev, t_start, t_fore = phase_years(phase)
    profile = grid()
    guinea = H.guinea_mask(profile)
    dmask = deployed_mask(profile)
    tag = tag or f"{sample_domain}_{predictor_year}_{neighbourhood_from}"
    out = RUNS / tag / phase
    a, b, c = esri(t_prev), esri(t_start), esri(t_fore)
    domain = valid_classes(a, b) & (np.ones_like(guinea) if sample_domain == "grid" else guinea)

    # predictors: the deployed static stack, or dated to each interval's start
    train_year = "latest" if predictor_year == "latest" else t_prev
    fore_year = "latest" if predictor_year == "latest" else t_start
    feats_train = predictor_stack(profile, dmask, train_year, drop)
    feats_fore = feats_train if fore_year == train_year else predictor_stack(profile, dmask, fore_year, drop)
    transitions = significant_transitions(a, b, domain)
    n_extra = 0 if transition_features is None else transition_features(t_prev, transitions[0][1], dmask).shape[0]
    log(f"[{tag}/{phase}] {len(transitions)} significant transitions; features {feats_train.shape[0] + n_extra}")
    rng = np.random.default_rng(seed)
    atlases = {"Expander-only": {}, "Patcher-only": {}}
    started = time.time()
    for f, t in transitions:
        ft, ff = feats_train, feats_fore
        if transition_features is not None:
            ft = np.vstack([feats_train, transition_features(t_prev, t, dmask)])
            ff = np.vstack([feats_fore, transition_features(t_start, t, dmask)])
        nbh_train = neighbourhood(b if neighbourhood_from == "end" else a, t, dmask)
        models = train_growth_models(ft, a, b, f, t, domain, nbh_train, rng, seed, learner)
        nbh_fore = neighbourhood(b, t, dmask)     # forecast from t_start: its map is the only one known
        for mode, key in (("expander", "Expander-only"), ("patcher", "Patcher-only")):
            if models[mode] is None:
                continue
            surface = predict_surface(models[mode], ff, nbh_fore, b, f, guinea)
            atlases[key][(f, t)] = write_float(surface, out / "surfaces" / f"{mode}_{f}_to_{t}.tif", profile)
        log(f"   {f}->{t} done ({time.time() - started:.0f} s)")
    growth = {(f, t, "expander"): p for (f, t), p in atlases["Expander-only"].items() if (f, t) in atlases["Patcher-only"]}
    growth.update({(f, t, "patcher"): p for (f, t), p in atlases["Patcher-only"].items() if (f, t) in atlases["Expander-only"]})
    atlases["Growth-mode (dilation)"] = growth
    scores, _ = allocate_and_score(b, c, atlases, guinea, out / "allocation", profile)
    result = {"tag": tag, "phase": phase, "years": [t_prev, t_start, t_fore], "sample_domain": sample_domain,
              "predictor_year": predictor_year, "neighbourhood": neighbourhood_from, "seed": seed,
              "n_features": int(feats_train.shape[0] + n_extra), "learner": learner, "dropped": list(drop),
              "transitions": transitions,
              "scores": scores, "minutes": round((time.time() - started) / 60, 1)}
    import json
    (out / "result.json").write_text(json.dumps(result, indent=2, default=float))
    result["atlases"] = atlases
    return result


def load_result(tag, phase):
    import json
    path = RUNS / tag / phase / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------
# R2: land-cover history features (from the annual ESRI series, up to `year`)

_MAPS = {}


def esri_cached(year):
    if year not in _MAPS:
        _MAPS[year] = esri(year)
    return _MAPS[year]


def _density(binary, radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (x ** 2 + y ** 2 <= radius ** 2).astype(np.float32)
    return correlate(binary.astype(np.float32), kernel, mode="reflect") / kernel.sum()


def _years_since_change(past):
    since = np.full(past[-1].shape, len(past), dtype=np.float32)   # no change seen in the window
    for k in range(1, len(past)):
        flip = (past[-k] != past[-k - 1]) & (since == len(past))
        since[flip] = k
    return since


HISTORY_NAMES = ["Changed in last year", "Years since last change", "Changes in last 5 years",
                 "Share of last 5 years as destination class", "Was destination class last year",
                 "Destination share, 3x3", "Destination share, radius 3", "Destination share, radius 10",
                 "Recent change share, radius 3", "Destination share trend, 3 years (radius 3)"]


def history_features(year, to_class, mask, window=5):
    """Ten layers describing each pixel's recent past, from the maps of year-window
    to year only (never later). Zero outside `mask`, like the base predictors."""
    cur, prev = esri_cached(year), esri_cached(year - 1)
    past = [esri_cached(y) for y in range(max(2017, year - window), year + 1)]
    changed = cur != prev
    n_changes = np.sum([past[i] != past[i - 1] for i in range(1, len(past))], axis=0)
    share_dest = np.mean([m == to_class for m in past[:-1]], axis=0)
    older = esri_cached(max(2017, year - 3))
    layers = [changed, _years_since_change(past), n_changes, share_dest, prev == to_class,
              _density(cur == to_class, 1), _density(cur == to_class, 3), _density(cur == to_class, 10),
              _density(changed, 3), _density(cur == to_class, 3) - _density(older == to_class, 3)]
    return np.stack([np.where(mask, layer, 0).astype(np.float32) for layer in layers])


# --------------------------------------------------------------------------
# R1: model-free rules

def rule_maps(t_prev, t_start):
    """Predicted maps from labels up to t_start only (the intern's rules, on ESRI)."""
    import sys
    if str(H.INTERN) not in sys.path:
        sys.path.insert(0, str(H.INTERN))
    from local_landuse.trivial_rules import neighbourhood_majority
    a, b = esri_cached(t_prev), esri_cached(t_start)
    classes = np.array([c for c in np.unique(b) if c not in H.ESRI_INVALID])
    valid = ~np.isin(b, H.ESRI_INVALID)
    nm8 = neighbourhood_majority(np.where(valid, b, 0), classes, include_self=False)
    return {"Eight-neighbour majority": np.where(valid, nm8, b),
            f"Revert to the {t_prev} class if changed {t_prev}->{t_start}":
                np.where((a != b) & ~np.isin(a, H.ESRI_INVALID), a, b),
            "No change": b.copy()}


def neighbourhood_only_atlas(t_start, transitions, out_dir, profile, guinea, radius=1):
    """Suitability = share of the destination class among a pixel's neighbours: the
    deployed allocator and demand, but no model at all."""
    b = esri_cached(t_start)
    atlas = {}
    for f, t in transitions:
        s = np.where((b == f) & guinea, _density(b == t, radius), -1.0)
        atlas[(f, t)] = write_float(s, Path(out_dir) / f"nbh_{f}_to_{t}.tif", profile)
    return atlas


# --------------------------------------------------------------------------
# R3: one multiclass model for all transitions

def class_agnostic_features(year, mask, rows):
    """Feature rows for the flat pixel indices `rows`: dated base predictors, current and
    last year's class (one-hot), class-independent history, and every class's
    neighbourhood share at radius 1 and 5. Uses maps up to `year` only."""
    profile = grid()
    base = predictor_stack(profile, mask, year)
    cur, prev = esri_cached(year), esri_cached(year - 1)
    past = [esri_cached(y) for y in range(max(2017, year - 5), year + 1)]
    n_changes = np.sum([past[i] != past[i - 1] for i in range(1, len(past))], axis=0)
    classes = sorted(H.ESRI_CLASSES)
    layers = list(base) + [cur == c for c in classes] + [prev == c for c in classes] + \
             [cur != prev, _years_since_change(past), n_changes, _density(cur != prev, 3)] + \
             [_density(cur == c, r) for c in classes for r in (1, 5)]
    names = PREDICTORS + [f"Class now: {H.ESRI_CLASSES[c]}" for c in classes] + \
            [f"Class last year: {H.ESRI_CLASSES[c]}" for c in classes] + \
            ["Changed in last year", "Years since last change", "Changes in last 5 years",
             "Recent change share, radius 3"] + \
            [f"{H.ESRI_CLASSES[c]} share, radius {r}" for c in classes for r in (1, 5)]
    X = np.stack([np.asarray(layer, dtype=np.float32).ravel()[rows] for layer in layers], axis=1)
    return X, names


def run_multiclass(phase, transitions=None, tag="R3_multiclass_xgb", n_sample=600_000, log=print):
    """One XGBoost model predicting the next class; suitability(f->t) = P(next = t) on
    pixels of class f. Same demand, allocator and transitions as the reference."""
    import json, time
    from xgboost import XGBClassifier
    t_prev, t_start, t_fore = phase_years(phase)
    profile = grid(); guinea = H.guinea_mask(profile); dmask = deployed_mask(profile)
    out = RUNS / tag / phase
    a, b, c = esri_cached(t_prev), esri_cached(t_start), esri_cached(t_fore)
    started = time.time()
    train_px = np.flatnonzero((guinea & valid_classes(a, b)).ravel())
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(train_px, min(n_sample, len(train_px)), replace=False))
    Xtr, names = class_agnostic_features(t_prev, dmask, pick)
    classes = sorted(H.ESRI_CLASSES)
    code = {cls: i for i, cls in enumerate(classes)}
    y = np.vectorize(code.get)(b.ravel()[pick])
    model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", n_jobs=8, random_state=SEED,
                          objective="multi:softprob", eval_metric="mlogloss")
    model.fit(Xtr, y)
    del Xtr
    log(f"[{tag}/{phase}] fitted on {len(pick):,} pixels, {len(names)} features ({time.time() - started:.0f} s)")
    fore_px = np.flatnonzero((guinea & ~np.isin(b, H.ESRI_INVALID)).ravel())
    Xf, _ = class_agnostic_features(t_start, dmask, fore_px)
    proba = np.vstack([model.predict_proba(Xf[i:i + 500_000]) for i in range(0, len(fore_px), 500_000)])
    del Xf
    transitions = transitions or significant_transitions(a, b, valid_classes(a, b) & guinea)
    atlas, start_flat = {}, b.ravel()[fore_px]
    for f, t in transitions:
        s = np.full(b.size, -1.0, dtype=np.float32)
        sel = start_flat == f
        s[fore_px[sel]] = proba[sel, code[t]]
        atlas[(f, t)] = write_float(s.reshape(b.shape), out / "surfaces" / f"multiclass_{f}_to_{t}.tif", profile)
    scores, _ = allocate_and_score(b, c, {"Multiclass XGBoost": atlas}, guinea, out / "allocation", profile)
    imp = sorted(zip(names, model.feature_importances_.tolist()), key=lambda r: -r[1])
    result = {"tag": tag, "phase": phase, "years": [t_prev, t_start, t_fore], "n_features": len(names),
              "transitions": transitions, "scores": scores, "top_features": imp[:15],
              "minutes": round((time.time() - started) / 60, 1)}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=float))
    result["atlases"] = {"Multiclass XGBoost": atlas}
    return result


# --------------------------------------------------------------------------
# R4: averaging two suitability sources

def _percentile_within(values, sel):
    """Rank-normalise suitability among the eligible pixels, so two models on different
    probability scales can be averaged."""
    from scipy.stats import rankdata
    out = np.full(values.shape, -1.0, dtype=np.float32)
    v = values[sel]
    out[sel] = (rankdata(v, method="average") / len(v)).astype(np.float32)
    return out


def average_atlases(atlas_a, atlas_b, weight_a, out_dir, profile):
    """For every transition both sources cover: weight_a * rank(a) + (1 - weight_a) * rank(b)."""
    atlas = {}
    for key in sorted(set(atlas_a) & set(atlas_b)):
        sa, sb = H.read_tif(atlas_a[key])[0], H.read_tif(atlas_b[key])[0]
        sel = (sa >= 0) & (sb >= 0)
        s = np.where(sel, weight_a * _percentile_within(sa, sel) + (1 - weight_a) * _percentile_within(sb, sel), -1.0)
        atlas[key] = write_float(s, Path(out_dir) / f"avg_{key[0]}_to_{key[1]}.tif", profile)
    return atlas


# --------------------------------------------------------------------------
# Second headline number: new change vs returns

def _past(start_year, window=None):
    """The maps before start_year: all of them since FIRST_YEAR (window=None, the
    whole series, as Pontius et al. 2017 define alternation), or the last `window`."""
    first = FIRST_YEAR if window is None else max(FIRST_YEAR, start_year - window)
    return np.stack([esri_cached(y) for y in range(first, start_year)])


def change_kind_scores(start_year, observed, simulated, mask, window=None):
    """Figure of Merit split by whether a change's destination is new to the pixel.

    A change (observed or predicted) is a 'return' when its destination class
    appears in the pixel's earlier maps (all since FIRST_YEAR, or the last `window`
    years), and 'new' otherwise. Both are known at forecast time, so a model can target
    either. Each kind gets its own FoM, built like the deployed figure_of_merit
    but restricted to that kind: hits of that kind divided by the pixels with an
    observed or a predicted change of that kind. Pixels follow H.score (inside
    `mask`, no invalid code in the start, observed or simulated map).
    """
    start = esri_cached(start_year)
    m = mask & ~(np.isin(start, H.ESRI_INVALID) | np.isin(observed, H.ESRI_INVALID)
                 | np.isin(simulated, H.ESRI_INVALID))
    past = _past(start_year, window)
    obs, sim = m & (observed != start), m & (simulated != start)
    hit = obs & sim & (simulated == observed)
    obs_new = obs & ~np.any(past == observed[None], axis=0)
    sim_new = sim & ~np.any(past == simulated[None], axis=0)
    out = {"FoM": float(100 * hit.sum() / (obs | sim).sum())}
    for kind, o, s in (("new", obs_new, sim_new), ("return", obs & ~obs_new, sim & ~sim_new)):
        h = hit & o
        out[kind] = {"FoM": float(100 * h.sum() / (o | s).sum()), "found_pct": float(100 * h.sum() / o.sum()),
                     "hits": int(h.sum()), "observed": int(o.sum()), "predicted": int(s.sum())}
    return out


# --------------------------------------------------------------------------
# R5: hybrid. Returns and new change allocated by different models

def atlas_from(folder, prefix):
    """{(f, t): path} from files named <prefix>_<f>_to_<t>.tif."""
    out = {}
    for p in Path(folder).glob(f"{prefix}_*_to_*.tif"):
        f, t = p.stem[len(prefix) + 1:].split("_to_")
        out[(int(f), int(t))] = str(p)
    return out


def held_recently(start_year, to_class, window=None):
    """True where `to_class` appears in the pixel's earlier maps (see _past)."""
    return np.any(_past(start_year, window) == to_class, axis=0)


def _quota_scores(group, surface, quota):
    """Score pixels of `group` so that its best `quota` pixels score >= 0.5 and the rest < 0.5,
    keeping the surface's order inside the group."""
    idx = np.flatnonzero(group.ravel())
    pos = np.empty(len(idx), dtype=np.float64)
    pos[np.argsort(-surface.ravel()[idx], kind="stable")] = np.arange(1, len(idx) + 1)
    return idx, (1 / (1 + pos / quota) if quota > 0 else 0.4 / (1 + pos / max(len(idx), 1))).astype(np.float32)


def run_hybrid(phase, return_atlas, new_atlas, tag="R5_hybrid", share_map=None, log=print):
    """Split each transition's observed demand into returns and new change using the share
    seen in the training interval (t_prev -> t_start, past maps only); allocate returns with
    `return_atlas` on pixels that held the destination class in an earlier map, and
    new change with `new_atlas` on all other pixels. One combined surface per transition, so
    the deployed allocator runs unchanged.

    share_map: optional predicted map for the same forecast (e.g. the deployed method's).
    The return share of each transition is then taken from that forecast instead of the
    training interval, so the new-change part keeps that model's own volume."""
    import json
    t_prev, t_start, t_fore = phase_years(phase)
    profile = grid()
    guinea = H.guinea_mask(profile)
    a, b, c = esri_cached(t_prev), esri_cached(t_start), esri_cached(t_fore)
    out = RUNS / tag / phase
    atlas, split = {}, []
    for f, t in sorted(set(return_atlas) | set(new_atlas)):
        s_ret = H.read_tif(return_atlas.get((f, t), new_atlas.get((f, t))))[0]
        s_new = H.read_tif(new_atlas.get((f, t), return_atlas.get((f, t))))[0]
        trained = guinea & (a == f) & (b == t)
        share = float(held_recently(t_prev, t)[trained].mean()) if trained.any() else 0.0
        if share_map is not None:
            chosen = guinea & (b == f) & (share_map == t)
            share = float(held_recently(t_start, t)[chosen].mean()) if chosen.any() else share
        demand = int((guinea & (b == f) & (c == t)).sum())     # observed, as in every other row
        q_ret = int(round(demand * share))
        held = held_recently(t_start, t)
        eligible = guinea & (b == f)
        score = np.full(b.size, -1.0, dtype=np.float32)
        for group, surface, quota in ((eligible & held, s_ret, q_ret), (eligible & ~held, s_new, demand - q_ret)):
            idx, val = _quota_scores(group, surface, quota)
            score[idx] = val
        atlas[(f, t)] = write_float(score.reshape(b.shape), out / "surfaces" / f"hybrid_{f}_to_{t}.tif", profile)
        observed_share = float(held[guinea & (b == f) & (c == t)].mean()) if demand else 0.0
        split.append(dict(transition=[f, t], demand=demand, return_share_used=round(share, 4),
                          return_share_observed_diagnostic=round(observed_share, 4)))
        log(f"   {f}->{t}: demand {demand:,}, return share {share:.2f} (observed later: {observed_share:.2f})")
    scores, maps = allocate_and_score(b, c, {"Hybrid": atlas}, guinea, out / "allocation", profile)
    scores["Hybrid"]["two_numbers"] = change_kind_scores(t_start, c, maps["Hybrid"], guinea)
    result = {"tag": tag, "phase": phase, "years": [t_prev, t_start, t_fore],
              "share_from": "training interval" if share_map is None else "share_map forecast",
              "split": split, "scores": scores}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=float))
    return result


# --------------------------------------------------------------------------
# The single score from notebook 06 on: new-change FoM, a random floor, coarser resolutions

def random_like(start, simulated, mask, seed=0):
    """Random placement with the forecast's own amount of change: the same number of each
    from->to transition as `simulated`, put on random pixels of the from-class inside `mask`
    (Pontius et al. 2004 compare a model against such a random model)."""
    rng = np.random.default_rng(seed)
    out = start.copy()
    moved = mask & (simulated != start)
    pairs, counts = np.unique(np.stack([start[moved], simulated[moved]]), axis=1, return_counts=True)
    free = mask.copy()
    for (f, t), n in zip(pairs.T, counts):
        idx = np.flatnonzero(free & (start == f))
        pick = rng.choice(idx, size=min(int(n), idx.size), replace=False)
        out.flat[pick] = t
        free.flat[pick] = False
    return out


def fom_by_resolution(start, observed, simulated, mask, blocks=(1, 4, 8, 20, 40)):
    """Figure of Merit (all change) when cells of k x k pixels are compared instead of single
    pixels: in each cell and destination class, agreement = min(observed, predicted) changes
    (the multiple-resolution comparison of Pontius et al. 2004). k = 1 is the pixel score; on
    the 250 m grid 4, 8, 20 and 40 px are 1, 2, 5 and 10 km."""
    m = mask & valid_classes(start, observed, simulated)
    obs, sim = m & (observed != start), m & (simulated != start)
    out = {}
    for k in blocks:
        hh, ww = (start.shape[0] // k) * k, (start.shape[1] // k) * k
        blk = lambda a: a[:hh, :ww].reshape(hh // k, k, ww // k, k).sum(axis=(1, 3))
        agree = n_obs = n_sim = 0
        for c in H.ESRI_COLOURS:
            o, p = blk(obs & (observed == c)), blk(sim & (simulated == c))
            agree += int(np.minimum(o, p).sum()); n_obs += int(o.sum()); n_sim += int(p.sum())
        out[k] = 100 * agree / (n_obs + n_sim - agree)
    return out


def run_average(tag, phase, atlas_a, atlas_b, weight_a=0.5, log=print):
    """Rank-average two models' suitability surfaces (average_atlases), allocate the observed
    demand with the deployed allocator and score; cached in RUNS/tag/phase/result.json."""
    import json
    _, t_start, t_fore = phase_years(phase)
    profile = grid()
    guinea = H.guinea_mask(profile)
    out = RUNS / tag / phase
    atlas = average_atlases(atlas_a, atlas_b, weight_a, out / "surfaces", profile)
    scores, _ = allocate_and_score(esri_cached(t_start), esri_cached(t_fore), {"Expander-only": atlas},
                                   guinea, out / "allocation", profile)
    result = {"tag": tag, "phase": phase, "weight_a": weight_a, "scores": scores}
    (out / "result.json").write_text(json.dumps(result, indent=2, default=float))
    log(f"[{tag}/{phase}] FoM {scores['Expander-only']['figure_of_merit_pct']:.2f}")
    return result


def projected_counts(t_prev, t_start, guinea):
    """Transition counts for the step after t_start projected by a first-order Markov chain, the
    usual way to set the demand of CA-Markov, FLUS and PLUS (Liu et al. 2017; Liang et al. 2021):
    the share of each class that went to each other class in t_prev->t_start, times the class's
    pixels in t_start. Rows: from-class; columns: to-class; inside Guinea, valid classes only."""
    import pandas as pd
    a, b = esri_cached(t_prev), esri_cached(t_start)
    m = guinea & valid_classes(a, b)
    classes = sorted(H.ESRI_COLOURS)
    rates = pd.DataFrame([[((a == f) & (b == t) & m).sum() for t in classes] for f in classes], index=classes, columns=classes)
    rates = rates.div(rates.sum(axis=1).clip(lower=1), axis=0)
    now = pd.Series([((b == f) & guinea & valid_classes(b)).sum() for f in classes], index=classes)
    return rates.mul(now, axis=0).round().astype(int)


def occurrence_atlas(phase, tag, transitions, hidden=12, sample_share=0.02, seed=SEED, log=print):
    """FLUS's suitability model (Liu et al. 2017): one neural network, with a hidden layer of
    `hidden` nodes, learns from the predictors (scaled 0-1) which class each pixel of the start
    map has, trained on a random `sample_share` of Guinea's pixels (FLUS's sampling rate 20 per
    thousand). Predictors dated to the start year, as FLUS was run on 2024. For f->t the surface
    is the probability of class t on class-f pixels, as FLUS's own maps were scored (notebook 00).
    Writes RUNS/tag/phase/surfaces/occ_f_to_t.tif; returns the atlas."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
    _, t_start, _ = phase_years(phase)
    profile = grid()
    guinea = H.guinea_mask(profile)
    b = esri_cached(t_start)
    idx = np.flatnonzero((guinea & valid_classes(b)).ravel())
    X = predictor_stack(profile, deployed_mask(profile), t_start)
    X = X.reshape(X.shape[0], -1)[:, idx].T
    sample = np.random.default_rng(seed).choice(idx.size, int(sample_share * idx.size), replace=False)
    model = make_pipeline(FunctionTransformer(np.nan_to_num), MinMaxScaler(),
                          MLPClassifier((hidden,), max_iter=1000, random_state=seed))
    model.fit(X[sample], b.ravel()[idx][sample])
    P = model.predict_proba(X)
    col = {int(c): k for k, c in enumerate(model.classes_)}
    log(f"[{tag}/{phase}] trained on {sample.size:,} pixels; training accuracy {model.score(X[sample], b.ravel()[idx][sample]):.3f}")
    atlas = {}
    for f, t in transitions:
        s = np.full(b.size, -1.0, np.float32)
        on_f = b.ravel()[idx] == f
        s[idx[on_f]] = P[on_f, col[t]]
        atlas[(f, t)] = write_float(s.reshape(b.shape), RUNS / tag / phase / "surfaces" / f"occ_{f}_to_{t}.tif", profile)
    return atlas


def new_change_fom_by_resolution(start_year, observed, simulated, mask, blocks=(1, 4, 8, 20, 40)):
    """New-change FoM when cells of k x k pixels are compared instead of single pixels: in each
    cell and destination class, agreement = min(observed, predicted) new changes (the
    multiple-resolution comparison of Pontius et al. 2004). k = 1 is the pixel score; on the
    250 m grid 4, 8, 20 and 40 px are 1, 2, 5 and 10 km."""
    start = esri_cached(start_year)
    m = mask & valid_classes(start, observed, simulated)
    past = _past(start_year)
    obs = m & (observed != start) & ~np.any(past == observed[None], axis=0)
    sim = m & (simulated != start) & ~np.any(past == simulated[None], axis=0)
    out = {}
    for k in blocks:
        hh, ww = (start.shape[0] // k) * k, (start.shape[1] // k) * k
        blk = lambda a: a[:hh, :ww].reshape(hh // k, k, ww // k, k).sum(axis=(1, 3))
        agree = n_obs = n_sim = 0
        for c in H.ESRI_COLOURS:
            o, p = blk(obs & (observed == c)), blk(sim & (simulated == c))
            agree += int(np.minimum(o, p).sum()); n_obs += int(o.sum()); n_sim += int(p.sum())
        out[k] = 100 * agree / (n_obs + n_sim - agree)
    return out


# --------------------------------------------------------------------------
# Contagion: is new change happening nearby? (notebook 08)

CONTAGION_NAMES = ["New change into destination nearby, last year (750 m)",
                   "New change into destination nearby, last year (2.5 km)",
                   "New change into destination nearby, last year (5 km)",
                   "Any new change nearby, last year (2.5 km)",
                   "Distance to last year's new change into destination (px)",
                   "New change into destination nearby, last 3 years (2.5 km)"]


_NEW_ANY = {}


def new_change_map(year, to_class=None):
    """Pixels that changed in year-1 -> year into a class they had not held in any map
    before year-1 (since FIRST_YEAR): last year's genuinely new change, into `to_class`
    or into any class."""
    if year not in _NEW_ANY:
        prev, cur = esri_cached(year - 1), esri_cached(year)
        new = valid_classes(prev, cur) & (cur != prev)
        if year - 1 > FIRST_YEAR:
            new &= ~np.any(_past(year - 1) == cur[None], axis=0)
        _NEW_ANY[year] = new
    new = _NEW_ANY[year]
    return new if to_class is None else new & (esri_cached(year) == to_class)


def _density_fft(binary, radius):
    """_density computed by FFT: same 'reflect' edges and disc kernel, much faster for big radii."""
    from scipy.signal import fftconvolve
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (x ** 2 + y ** 2 <= radius ** 2).astype(np.float64)
    padded = np.pad(binary.astype(np.float64), radius, mode="symmetric")   # = scipy.ndimage 'reflect'
    return (fftconvolve(padded, kernel, mode="valid") / kernel.sum()).astype(np.float32)


def contagion_features(year, to_class, mask):
    """Six layers from maps up to `year` only: how much genuinely new change happened
    near each pixel recently (Rosa et al. 2013: deforestation spreads from recent clearings)."""
    from scipy.ndimage import distance_transform_edt
    into = new_change_map(year, to_class)
    recent3 = into | new_change_map(year - 1, to_class) | new_change_map(year - 2, to_class)
    dist = distance_transform_edt(~into) if into.any() else np.full(into.shape, 1e3)
    layers = [_density_fft(into, 3), _density_fft(into, 10), _density_fft(into, 20),
              _density_fft(new_change_map(year), 10), np.minimum(dist, 200), _density_fft(recent3, 10)]
    return np.stack([np.where(mask, layer, 0).astype(np.float32) for layer in layers])
