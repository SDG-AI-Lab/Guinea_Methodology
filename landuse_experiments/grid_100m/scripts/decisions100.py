"""Notebook 05: the methodology document's method with more seeds, and two ways of adding what makes
MOLUSCE's logistic regression place change well at 100 m (notebook 07), chosen on the validation window.

  document      deployed100.run_document, seeds 42, 1 and 2 on the three windows
  document3x3   the same, with each class's share of the 3 x 3 window around a cell (start map of the
                interval) as 7 more predictors: the neighbourhood MOLUSCE's regression reads (ns = 1)
  lr            a copy of MOLUSCE 5.1's logistic regression (algorithms/models/lr): one multinomial
                regression whose inputs are the 3 x 3 window of classes (dummy coded, the last class as
                base) and of the 12 predictors (each standardised on its own map), whose outcome is the
                from -> to class of the interval, fitted on 50,000 random cells with at most 100
                iterations and no penalty. MOLUSCE fits with its own BFGS code; this copy uses
                scikit-learn's L-BFGS. The potential of f -> t is the probability of that outcome; cells
                whose window is incomplete get 0, as MOLUSCE leaves them out
  blend         the document's map and the regression's map of each transition averaged as ranks among
                the cells of the start class (equal weights; Clemen 1989); a transition with one map
                keeps that map's ranks

All are allocated with main's run_allocation_simulation (deployed100.allocate), as the document's method.
"""

import json
import shutil
import warnings
from functools import partial

import numpy as np
import pandas as pd

import deployed100 as D

P, G, C, P100 = D.P, D.G, D.C, D.P100
PHASES, SEEDS, INVALID = D.PHASES, D.SEEDS, D.INVALID
SAMPLES = 50_000                                                    # MOLUSCE's default, as run at 250 m
MAX_ITER = 100                                                      # MOLUSCE LR's default
ROWS = 128                                                          # rows per block when predicting
NEW = ("document3x3", "lr", "blend")


def _classes(year):
    return [int(k) for k in np.unique(C.class_100(year)) if k not in INVALID]


# --------------------------------------------------------------------------
# document3x3

def share3_files(year, to_class=None):
    """Share of each class in the 3 x 3 window, on `year`'s map (NaN outside Guinea); the folder is per
    year so that the names are the same in training and prediction."""
    import rasterio
    from scipy.ndimage import uniform_filter
    folder = P.INPUTS / "neighbourhood" / f"share3_{year}"
    paths = [folder / f"SHARE3_CLASS_{k}.tif" for k in _classes(year)]
    if not all(p.exists() for p in paths):
        folder.mkdir(parents=True, exist_ok=True)
        cls = G.to_grid(C.class_100(year), fill=0, dtype="uint8")
        inside = G.inside_mask()
        for k, path in zip(_classes(year), paths):
            share = uniform_filter((cls == k).astype(np.float32), size=3, mode="reflect")
            share[~inside] = np.nan
            with rasterio.open(path, "w", **G.profile("float32", np.nan)) as dst:
                dst.write(share, 1)
    return [str(p) for p in paths]


def run_document3x3(phase, seed, log=print):
    return D.run_document(phase, seed, log=log, method="document3x3", extra=share3_files)


# --------------------------------------------------------------------------
# lr: MOLUSCE's logistic regression

def _factors(year):
    """(12, rows + 2, cols + 2) predictors standardised on their own map (MOLUSCE: normalize('mean')),
    padded by one cell with NaN."""
    import rasterio
    paths = P.write_predictors(year)
    g = G.grid()
    out = np.full((len(paths), g["height"] + 2, g["width"] + 2), np.nan, dtype=np.float32)
    for i, path in enumerate(paths):
        with rasterio.open(path) as src:
            v = src.read(1).astype(np.float32)
        out[i, 1:-1, 1:-1] = (v - np.nanmean(v)) / np.nanstd(v)
    return out


def _state(year):
    return np.pad(G.to_grid(C.class_100(year), fill=0, dtype="uint8"), 1, constant_values=0)


def _complete(state, factors):
    """Cells whose whole 3 x 3 window has a valid class and all 12 predictors (unpadded grid)."""
    from scipy.ndimage import minimum_filter
    ok = ~np.isin(state, INVALID) & np.isfinite(factors).all(axis=0)
    return minimum_filter(ok, size=3, mode="constant", cval=False)[1:-1, 1:-1]


def _inputs(state, factors, rows, cols, cats):
    """MOLUSCE's Sampler.get_inputs for cells (rows, cols) of the unpadded grid: the 3 x 3 window of
    dummy-coded classes, then of each predictor."""
    parts = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            c = state[rows + 1 + di, cols + 1 + dj]
            parts.append(np.stack([c == k for k in cats[:-1]], axis=1).astype(np.float32))
    parts = np.concatenate(parts, axis=1)
    f = np.stack([factors[:, rows + 1 + di, cols + 1 + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)], axis=1)
    return np.concatenate([parts, f.transpose(2, 0, 1).reshape(rows.size, -1)], axis=1)


def run_lr(phase, seed, log=print):
    """Fit on the first interval (a -> b, predictors of a), predict from b (predictors of b); writes
    suitability_<f>_to_<t>.tif (0 outside class f) and the projected demand of the document's run."""
    import rasterio
    from sklearn.linear_model import LogisticRegression
    a, b, c = PHASES[phase]
    out = P100.run_dir(D.tag(phase, seed, "lr"))
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    with P100.Peak() as peak:
        state, factors = _state(a), _factors(a)
        end = G.to_grid(C.class_100(b), fill=0, dtype="uint8")
        cats = _classes(a)
        ok = _complete(state, factors) & ~np.isin(end, INVALID)
        pool = np.flatnonzero(ok)
        pick = np.sort(np.random.default_rng(seed).choice(pool, SAMPLES, replace=False))
        r, q = np.unravel_index(pick, ok.shape)
        X = _inputs(state, factors, r, q, cats)
        y = state[r + 1, q + 1].astype(np.int32) * 100 + end[r, q]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                         # 100 iterations do not converge, as in MOLUSCE
            model = LogisticRegression(penalty=None, max_iter=MAX_ITER).fit(X, y)
        del state, factors, end, ok, pool, X
        log(f"[{out.name}] fitted, {len(model.classes_)} outcomes")

        state, factors = _state(b), _factors(b)
        base = state[1:-1, 1:-1]
        ok = _complete(state, factors)
        keys = [(int(k // 100), int(k % 100)) for k in model.classes_ if k // 100 != k % 100]
        froms = sorted({f for f, _ in keys})
        flat = {f: np.flatnonzero(base == f) for f in froms}     # potentials kept on the class-f cells only
        cols = {f: [list(model.classes_).index(f * 100 + t) for g, t in keys if g == f] for f in froms}
        pot = {f: np.zeros((flat[f].size, len(cols[f])), dtype=np.float32) for f in froms}
        width = base.shape[1]
        for r0 in range(0, base.shape[0], ROWS):
            r, q = np.nonzero(ok[r0:r0 + ROWS])
            if not r.size:
                continue
            r = r + r0
            proba = model.predict_proba(_inputs(state, factors, r, q, cats))
            here = base[r, q]
            for f in froms:
                m = here == f
                pot[f][np.searchsorted(flat[f], r[m] * width + q[m])] = proba[m][:, cols[f]]
        del factors, ok
        for f in froms:
            for i, t in enumerate(t for g, t in keys if g == f):
                v = np.zeros(base.size, dtype=np.float32)
                v[flat[f]] = pot[f][:, i]
                with rasterio.open(out / f"suitability_{f}_to_{t}.tif", "w", **G.profile("float32", None)) as dst:
                    dst.write(v.reshape(base.shape), 1)
    _copy_demand(phase, out)
    result = {"tag": out.name, "train": [a, b], "base": b, "target": c, "seed": seed, "method": "lr",
              "outcomes": [int(k) for k in model.classes_], "models": [list(k) for k in keys],
              "stages": [{"stage": "Fit and predict (MOLUSCE's regression)", "seconds": peak.seconds, "peak_GB": peak.peak / 1e9}]}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


def _copy_demand(phase, out):
    """The platform's projected demand does not depend on the method or the seed."""
    shutil.copy(P100.run_dir(D.tag(phase, 42, "document")) / "demand.csv", out / "demand.csv")


# --------------------------------------------------------------------------
# blend

def _ranks(path, cells):
    """Ranks (0-1] of a suitability map on `cells` (Guinea cell indices)."""
    import rasterio
    from scipy.stats import rankdata
    with rasterio.open(path) as src:
        v = np.nan_to_num(G.from_grid(src.read(1))[cells], nan=0.0)
    return rankdata(v) / v.size


def run_blend(phase, seed, log=print, members=("document", "lr"), method="blend", weights=None):
    """Each transition's maps of `members` as ranks among the cells of the start class, averaged with
    equal weights (or `weights`, one per member: notebook 06's check); a transition with fewer maps
    keeps the mean of those it has."""
    import rasterio
    a, b, c = PHASES[phase]
    out = P100.run_dir(D.tag(phase, seed, method))
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    for m in members:
        RUN[m](phase, seed, log)
    out.mkdir(parents=True, exist_ok=True)
    atlases = {m: D.surfaces(D.tag(phase, seed, m)) for m in members}
    base = C.class_100(b)
    with P100.Peak() as peak:
        for f, t in sorted(set().union(*atlases.values())):
            cells = np.flatnonzero(base == f)
            have = [i for i, atl in enumerate(atlases.values()) if (f, t) in atl]
            maps = [list(atlases.values())[i][(f, t)] for i in have]
            w = None if weights is None or len(have) < len(members) else [weights[i] for i in have]
            v = np.zeros(base.size, dtype=np.float32)
            v[cells] = np.average([_ranks(m, cells) for m in maps], axis=0, weights=w)
            with rasterio.open(out / f"suitability_{f}_to_{t}.tif", "w", **G.profile("float32", None)) as dst:
                dst.write(G.to_grid(v, fill=0.0, dtype="float32"), 1)
    _copy_demand(phase, out)
    result = {"tag": out.name, "base": b, "target": c, "seed": seed, "method": method, "members": list(members),
              **{f"from_{m}": len(atl) for m, atl in atlases.items()},
              "stages": [{"stage": "Rank average", "seconds": peak.seconds, "peak_GB": peak.peak / 1e9}]}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


# --------------------------------------------------------------------------
# contagion: new change nearby (250 m notebook 08; Rosa et al. 2013)

RADII_KM = (0.75, 2.5, 5.0)                                         # the 250 m layers' 3, 10 and 20 pixels
ANY_KM, CAP_KM = 2.5, 50.0                                          # any new change nearby; distance cap (200 pixels)
CONTAGION_DIR = P100.RUNS.parent / "inputs_contagion"              # on E:


def _cells(km):
    return int(round(km * 1000 / G.SCALE_M))


def _new_change(year, to_class=None):
    """Cells that changed in year-1 -> year into a class they had not held in any map since 2017 before
    year-1: last year's genuinely new change (250 m esri_methods.new_change_map), as a grid."""
    prev, cur = C.class_100(year - 1), C.class_100(year)
    new = ~np.isin(prev, INVALID) & ~np.isin(cur, INVALID) & (cur != prev)
    for y in range(C.FIRST_YEAR, year - 1):
        new &= C.class_100(y) != cur
    if to_class is not None:
        new &= cur == to_class
    return G.to_grid(new, fill=False, dtype=bool)


def _density(binary, radius):
    """Share of the cells within a disc of `radius` cells that are True ('reflect' edges, as 250 m
    _density_fft), by overlap-add FFT."""
    from scipy.signal import oaconvolve
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (x ** 2 + y ** 2 <= radius ** 2).astype(np.float32)
    padded = np.pad(binary.astype(np.float32), radius, mode="symmetric")
    return np.maximum(oaconvolve(padded, kernel, mode="valid") / kernel.sum(), 0).astype(np.float32)   # FFT rounding


def contagion_files(year, to_class):
    """The six contagion layers of the 250 m method from maps up to `year` only, at the same distances
    in km (NaN outside Guinea); written once per year and destination, the same names in every folder."""
    import rasterio
    from scipy.ndimage import distance_transform_edt
    folder = CONTAGION_DIR / str(year) / str(to_class)
    paths = [folder / f"CONTAGION_{k}.tif" for k in range(1, 7)]
    if not all(p.exists() for p in paths):
        folder.mkdir(parents=True, exist_ok=True)
        inside = G.inside_mask()
        into = _new_change(year, to_class)
        cap = _cells(CAP_KM)

        def distance():
            if not into.any():
                return np.full(into.shape, cap, np.float32)
            return np.minimum(distance_transform_edt(~into), cap).astype(np.float32)

        def recent3():
            return _density(into | _new_change(year - 1, to_class) | _new_change(year - 2, to_class), _cells(RADII_KM[1]))

        layers = [lambda: _density(into, _cells(RADII_KM[0])), lambda: _density(into, _cells(RADII_KM[1])),
                  lambda: _density(into, _cells(RADII_KM[2])), lambda: _density(_new_change(year), _cells(ANY_KM)),
                  distance, recent3]
        for path, make in zip(paths, layers):
            v = make()
            v[~inside] = np.nan
            with rasterio.open(path, "w", **G.profile("float32", np.nan)) as dst:
                dst.write(v, 1)
            del v
    return [str(p) for p in paths]


def run_contagion(phase, seed, log=print):
    return D.run_document(phase, seed, log=log, method="contagion", extra=contagion_files)


# --------------------------------------------------------------------------
# flusnet: FLUS's network (250 m notebook 09, kept in the improved method of 250 m notebook 11)

def run_flusnet(phase, seed, log=print):
    """The copy of FLUS's network (scripts/flus_ann.py) trained as FLUS trains it, on the base-year map
    and FLUS's 0-1 predictors of the base year (2 % of each class's cells); the suitability of f -> t
    is the probability of class t on the class-f cells."""
    import rasterio
    import tools100 as T
    import flus_ann as FA
    a, b, c = PHASES[phase]
    out = P100.run_dir(D.tag(phase, seed, "flusnet"))
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    codes = {int(e): int(k) for k, e in T.TO_ESRI.items()}          # Esri class -> tool code 1..7
    with P100.Peak() as peak:
        land = C.class_100(b)
        valid = np.isin(land, list(codes))
        X = np.empty((int(valid.sum()), len(P.NAMES)), dtype=np.float32)
        for i, name in enumerate(P.NAMES):                          # as tools_export100.factors: min-max, NaN -> 0
            v = P.predictor(name, b)
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
            X[:, i] = np.nan_to_num((v - lo) / (hi - lo) if hi > lo else np.zeros_like(v))[valid]
        lut = np.zeros(256, dtype=int)
        for e, k in codes.items():
            lut[e] = k - 1
        y = lut[land[valid]]
        sample = FA.proportional_sample(y, 0.02, seed)
        net = FA.OccurrenceNetwork(seed=seed).fit(X[sample], y[sample])
        log(f"[{out.name}] trained on {sample.size:,} cells")
        proba = net.predict_proba(X).astype(np.float32)
        del X
        keys = []
        for f in codes:
            on_f = land[valid] == f
            for t in codes:
                if f != t:
                    v = np.zeros(land.size, dtype=np.float32)
                    v[np.flatnonzero(valid)[on_f]] = proba[on_f, codes[t] - 1]
                    with rasterio.open(out / f"suitability_{f}_to_{t}.tif", "w", **G.profile("float32", None)) as dst:
                        dst.write(G.to_grid(v, fill=0.0, dtype="float32"), 1)
                    keys.append([f, t])
    _copy_demand(phase, out)
    result = {"tag": out.name, "base": b, "target": c, "seed": seed, "method": "flusnet", "models": keys,
              "stages": [{"stage": "FLUS's network", "seconds": peak.seconds, "peak_GB": peak.peak / 1e9}]}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


def run_document(phase, seed, log=print):
    return D.run_document(phase, seed, log=log)


RUN = {"document": run_document, "document3x3": run_document3x3, "lr": run_lr, "blend": run_blend,
       "contagion": run_contagion, "flusnet": run_flusnet,
       "blendc": partial(run_blend, members=("contagion", "lr"), method="blendc"),
       "blend3": partial(run_blend, members=("document", "lr", "flusnet"), method="blend3"),
       "blend3c": partial(run_blend, members=("contagion", "lr", "flusnet"), method="blend3c"),
       "blendw25": partial(run_blend, method="blendw25", weights=(0.25, 0.75)),
       "blendw75": partial(run_blend, method="blendw75", weights=(0.75, 0.25))}
WEIGHTS = {"blendw25": 0.25, "blend": 0.5, "blendw75": 0.75}            # the document's weight (notebook 06)
SECOND = ("blendc", "blend3", "blend3c")                             # the second round, against the blend

# The order of notebook 05's runs: the document's seeds first, then the candidates on validation.
# The candidates go to test and backtest only if one beats the document's method on validation by more
# than the seeds' spread (run05.py --winner <method>). The lr run on test checks the copy against
# MOLUSCE's own code (notebook 07: 19.34-19.72 with the same amount); it is not used to choose.
JOBS = ([("document", ph, s) for ph in ("validation", "test", "backtest") for s in SEEDS]
        + [(m, "validation", s) for m in NEW for s in SEEDS]
        + [("lr", "test", 42)])


# --------------------------------------------------------------------------
# Notebook 05

METHODS = {"document": "Document's method", "document3x3": "Document + 3 × 3 class shares",
           "lr": "MOLUSCE's regression (copy)", "blend": "Blend: document + regression",
           "contagion": "Document + contagion", "flusnet": "FLUS's network (copy)",
           "blendc": "Blend with contagion", "blend3": "Blend + FLUS's network",
           "blend3c": "Blend with contagion + FLUS's network",
           "blendw25": "Blend, document weight 0.25", "blendw75": "Blend, document weight 0.75"}


def scores():
    """Every finished score: one row per method, window, seed and amount."""
    rows = []
    for method in RUN:
        for phase in PHASES:
            for seed in SEEDS:
                for demand in ("observed", "projected"):
                    f = D._alloc_dir(D.tag(phase, seed, method), "as it runs", demand) / "score.json"
                    if f.exists():
                        rows.append({"method": method, "phase": phase, "seed": seed, "demand": demand,
                                     **json.loads(f.read_text())})
    return pd.DataFrame(rows)


def costs():
    """Minutes and peak memory (GB) of each method's own stage, per run (allocation is main's in all)."""
    rows = []
    for method in RUN:
        for phase in PHASES:
            for seed in SEEDS:
                f = P100.run_dir(D.tag(phase, seed, method)) / "result.json"
                if f.exists():
                    for s in json.loads(f.read_text())["stages"]:
                        rows.append({"method": method, "phase": phase, "seed": seed, "stage": s["stage"],
                                     "minutes": s["seconds"] / 60, "peak_GB": s["peak_GB"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The amount of change: the platform's trend against the recent observed intervals

AMOUNTS = {"trend": 0, "last interval": 1, "mean of 2 intervals": 2, "mean of 3 intervals": 3}


def amount(phase, kind):
    """Transition counts (from x to, Esri codes, clouds left out) to place in the forecast year:
    the platform's trend projection (change_analysis, as the document's runs), or the mean of the last
    k observed intervals up to the base year (k = 1: a one-step Markov projection, as MOLUSCE and the
    250 m runs; Mas et al. 2014)."""
    a, b, c = PHASES[phase]
    k = AMOUNTS[kind]
    if k == 0:
        t = D.projected_demand(D.tag(phase, 42, "document"))
    else:
        t = sum(D.observed_demand(y - 1, y) for y in range(b - k + 1, b + 1)).fillna(0) / k
    keep = [x for x in t.index if x not in INVALID]
    t = t.loc[keep, [x for x in t.columns if x not in INVALID]].round().astype(int)
    for x in t.index:
        if x in t.columns:
            t.loc[x, x] = 0
    return t


def allocate_amount(phase, seed, kind, method="blend"):
    """A method's maps placed with one amount by main's allocator; returns the score (cached)."""
    run_tag = D.tag(phase, seed, method)
    if kind == "trend":                                             # the run as it runs
        path = D.allocate(run_tag, phase, "as it runs", "projected")
    else:
        P100.tool()
        from landuse_tool import prediction
        out = P100.run_dir(run_tag) / "alloc" / f"amount_{kind.replace(' ', '_')}"
        path = out / "predicted_land_cover.tif"
        if not path.exists():
            out.mkdir(parents=True, exist_ok=True)
            atlas, _ = D.variant_inputs(run_tag, phase, "as it runs", "observed")
            report = {}
            prediction.run_allocation_simulation(P.write_esri(PHASES[phase][1]), amount(phase, kind), atlas, str(out),
                                                 stochastic=False, allocation_report=report)
            (out / "allocation.json").write_text(json.dumps(report, indent=1, default=float))
    cache = path.parent / "score.json"
    if not cache.exists():
        cache.write_text(json.dumps(D.score(phase, path), default=float))
    return json.loads(cache.read_text())


def amount_scores(method="blend"):
    rows = []
    for phase in PHASES:
        for seed in SEEDS:
            for kind in AMOUNTS:
                f = (D._alloc_dir(D.tag(phase, seed, method), "as it runs", "projected") if kind == "trend" else
                     P100.run_dir(D.tag(phase, seed, method)) / "alloc" / f"amount_{kind.replace(' ', '_')}") / "score.json"
                if f.exists():
                    rows.append({"method": method, "phase": phase, "seed": seed, "amount": kind, **json.loads(f.read_text())})
    return pd.DataFrame(rows)
