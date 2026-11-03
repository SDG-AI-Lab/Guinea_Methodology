"""The risk model on the 100 m grid, built and scored in pieces, and the rule for a lost cell.

Everything is kept as values over the cells with any forest in 2000 (12.6 million of the
25.1 million cells inside Guinea). Only distances and the local loss rate are computed on the
whole grid, one at a time. A window's predictors are built, used and dropped before the next:
the calibration stack gives the training samples, then the test stack gives the scores, in
chunks of a million cells.

Training calls the deployed fit_risk_model unchanged, with the improved model's predictors
(improved_model.json; notebook 05). Sampling and scoring are the
deployed sample_training and predict_risk_map, applied to vectors instead of 2D grids.

Which cells are forest and which are lost, from the 30 m shares of each cell:

  half      the platform's (HansenShares.display_arrays): forest while at least half the cell is
            forest; lost in the year it falls below half
  any_half  forest while at least half the cell is; lost when any of its forest is lost
  any       forest while any of the cell is; lost when any of its forest is lost
  weighted  forest while any of the cell is; training cells drawn in proportion to the forest
            lost and kept in them (the same as sampling 30 m pixels), so the model gives the
            chance that a 30 m forest pixel of the cell is lost

Distances to prior and recent loss use the rule's lost cells ("half" for half, "any" for the
others). The local loss rate is the share of the forest lost around the cell, from the shares.

Every rule is scored the same way, over the 30 m forest pixels at the start of the test window:
each cell counts with the forest it lost and kept in the test window (AUC over pixels).
"""

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]                      # deforestation_experiments/
sys.path.insert(0, str(HERE / "scripts"))
import predictors as P                                          # noqa: E402  (puts data_100m_tools on the path)
import defor_methods as M                                       # noqa: E402

G = P.G
DR = M.DR
SPLITS = M.SPLITS
SAMPLES = M.SAMPLES
RUNS = HERE / "outputs" / "runs"
WINDOW_CELLS = 51                                               # 5 km at 100 m

RULES = {"half": "Half the cell (platform)", "any_half": "Any loss, half-forest cells",
         "any": "Any loss, any forest", "weighted": "Weighted by the shares"}
IMPROVED = json.loads((HERE / "scripts" / "improved_model.json").read_text())
DEPLOYED = {"names": M.DEPLOYED_PREDICTORS, "predictor_year": "latest", "extras": []}   # yearly layers at M.LATEST


def features(config):
    return config["names"] + config.get("extras", []) + ["Distance to prior loss"]


FEATURES = features(IMPROVED)
CHANGING = {"Population density", "Distance to built-up area", "Protected area",
            "Fire history (years burned of 5)", *P.COARSE_ANNUAL}


# --------------------------------------------------------------------------
# Hansen shares over the cells with forest in 2000

@lru_cache(maxsize=1)
def hansen():
    """{"idx": positions among the cells inside Guinea, "forest2000": %, "loss": % (year x cell,
    2001 first), "area": ha} over the cells with any forest in 2000."""
    forest2000 = G.read_layer("hansen_shares", ["forest2000"])["forest2000"]
    keep = np.flatnonzero(forest2000 > 0)
    years = sorted(int(c[5:]) for c in G.layers()["hansen_shares"]["columns"] if c.startswith("loss_"))
    loss = np.empty((len(years), keep.size), dtype="uint8")
    for i, y in enumerate(years):                               # a column at a time: less memory
        loss[i] = G.read_layer("hansen_shares", [f"loss_{y}"])[f"loss_{y}"][keep]
    return {"idx": keep, "forest2000": forest2000[keep], "loss": loss, "first": years[0],
            "area": G.cell_area_ha(G.cells()[keep])}


def forest_pct(year):
    """Share of each cell still forest at the end of `year` (whole %)."""
    h = hansen()
    lost = h["loss"][: year - h["first"] + 1].sum(axis=0, dtype="int16")
    return np.maximum(h["forest2000"].astype("int16") - lost, 0)


def loss_pct(a, b):
    """Share of each cell lost in the years a+1..b (whole %)."""
    h = hansen()
    return h["loss"][a + 1 - h["first"]: b + 1 - h["first"]].sum(axis=0, dtype="int16")


def lost(family, a, b):
    """Cells counted as lost in the years a+1..b: "half" when they fall below half forest,
    "any" when any of their forest is lost."""
    if family == "half":
        return (forest_pct(a) >= 50) & (forest_pct(b) < 50)
    return loss_pct(a, b) > 0


def _family(rule):
    return "half" if rule == "half" else "any"


# --------------------------------------------------------------------------
# Predictors over the cells with forest in 2000

def _on_grid(values, fill):
    inside = np.full(G.cells().size, fill, dtype=np.asarray(values).dtype)
    inside[hansen()["idx"]] = values
    return G.to_grid(inside, fill=fill)


def distance_to(mask):
    """Distance (m) to the nearest cell of `mask`, as drivers.distance_to (the platform's)."""
    return P._distance_m(_on_grid(mask, False))[hansen()["idx"]]


def local_loss_rate(year, span=3, size=WINDOW_CELLS):
    """Share of the forest of `year - span` lost in the `span` years up to `year`, within
    `size` x `size` cells (5 km), from the 30 m shares."""
    from scipy.ndimage import uniform_filter
    forest = uniform_filter(_on_grid(forest_pct(year - span).astype("float32"), 0.0), size)
    rate = uniform_filter(_on_grid(loss_pct(year - span, year).astype("float32"), 0.0), size)
    rate /= np.maximum(forest, 1e-6)
    del forest
    return G.from_grid(rate)[hansen()["idx"]]


_STATIC = {}


def _shared(name, year):
    if name in CHANGING:
        return P.at_year(name, year)[hansen()["idx"]]
    if name not in _STATIC:
        _STATIC[name] = P.static(name)[hansen()["idx"]]
        P.static.cache_clear()                                  # keep only the forest cells
    return _STATIC[name]


def stack(year, config=IMPROVED, families=("half", "any")):
    """{family: {predictor: float32 over the cells with forest in 2000}} at `year` (a window's start).
    The families share every predictor but the distances to prior and recent loss. With
    predictor_year "latest" (the deployed model), yearly layers are read at their latest year."""
    import warnings
    shared = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name in config["names"] + config.get("extras", []):
            if name == "Recent loss rate around the cell (5 km)":
                shared[name] = local_loss_rate(year)
            elif name != "Distance to recent loss":
                latest = config.get("predictor_year") == "latest" and name in M.LATEST
                shared[name] = _shared(name, M.LATEST[name] if latest else year)
    out = {}
    for family in families:
        out[family] = {**shared, "Distance to prior loss": distance_to(lost(family, hansen()["first"] - 1, year))}
        if "Distance to recent loss" in config.get("extras", []):
            out[family]["Distance to recent loss"] = distance_to(lost(family, year - 3, year))
    return out


# --------------------------------------------------------------------------
# Training, scoring, AUC over pixels

def training_cells(rule, t1, t2):
    """(lost cells, kept cells, their weights or None) of the calibration window, by the rule."""
    f1, lw = forest_pct(t1), loss_pct(t1, t2)
    if rule == "half":
        return np.flatnonzero(lost("half", t1, t2)), np.flatnonzero(forest_pct(t2) >= 50), None
    forest = f1 >= 50 if rule == "any_half" else f1 > 0
    if rule == "weighted":
        kept = np.clip(f1 - lw, 0, None)
        pos, neg = np.flatnonzero(forest & (lw > 0)), np.flatnonzero(forest & (kept > 0))
        area = hansen()["area"]
        return pos, neg, (lw[pos] * area[pos], kept[neg] * area[neg])
    return np.flatnonzero(forest & (lw > 0)), np.flatnonzero(forest & (lw == 0)), None


def _pick(cells, n, rng, weights=None):
    if weights is None:
        return cells[rng.choice(cells.size, n, replace=False)]
    keys = rng.exponential(size=cells.size) / weights          # weighted, without replacement
    return cells[np.argpartition(keys, n)[:n]]


def sample(rule, t1, t2, s, seed, names=FEATURES):
    """Balanced (X, y) as drivers.sample_training: cells with every predictor, up to SAMPLES each."""
    finite = np.all([np.isfinite(s[n]) for n in names], axis=0)
    pos, neg, weights = training_cells(rule, t1, t2)
    keep_pos, keep_neg = finite[pos], finite[neg]
    pos, neg = pos[keep_pos], neg[keep_neg]
    w_pos, w_neg = (None, None) if weights is None else (weights[0][keep_pos], weights[1][keep_neg])
    rng = np.random.default_rng(seed)
    n = min(SAMPLES, pos.size, neg.size)
    cells = np.concatenate([_pick(pos, n, rng, w_pos), _pick(neg, n, rng, w_neg)])
    X = np.column_stack([s[name][cells] for name in names])
    return X, np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])


def score(model, s, rows, chunk=1_000_000, names=FEATURES):
    """Risk of the given cells, a million at a time (NaN where a predictor has no value)."""
    risk = np.full(rows.size, np.nan, dtype="float32")
    for i in range(0, rows.size, chunk):
        X = np.column_stack([s[name][rows[i:i + chunk]] for name in names])
        ok = np.isfinite(X).all(axis=1)
        part = risk[i:i + chunk]
        part[ok] = model.predict_proba(X[ok])[:, 1]
    return risk


def weighted_auc(score_, w_pos, w_neg):
    """AUC where each cell counts with weight w_pos as lost and w_neg as kept (ties count half)."""
    ok = np.isfinite(score_) & ((w_pos > 0) | (w_neg > 0))
    _, inverse = np.unique(score_[ok], return_inverse=True)
    pos = np.bincount(inverse, weights=w_pos[ok].astype("float64"))
    neg = np.bincount(inverse, weights=w_neg[ok].astype("float64"))
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


def _peak_gb():
    import psutil
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", getattr(info, "rss", 0)) / 1e9


def run_split(split, seeds=(42, 1, 2), log=print):
    """Every rule on one split: training cells, AUC over pixels (all forest and the rule's own
    forest cells), the cell AUC each rule gives itself, importances, time and peak memory."""
    t1, t2, t3 = split
    start = time.perf_counter()
    area = hansen()["area"]
    out = {"split": M.split_label(split), "rules": {}}

    # Calibration window: build, sample, drop
    cal = stack(t1)
    log(f"{out['split']}: calibration stack {time.perf_counter() - start:.0f} s")
    lw = loss_pct(t1, t2).astype("float32") / 100
    models = {}
    for rule in RULES:
        pos, neg, _ = training_cells(rule, t1, t2)
        out["rules"][rule] = {"lost_cells": int(pos.size), "kept_cells": int(neg.size),
                              "lost_cells_ha": float(area[pos].sum()),
                              "loss_in_lost_cells_ha": float((lw[pos] * area[pos]).sum()),
                              "aucs_all": [], "aucs_scored": [], "aucs_own": []}
        for seed in seeds:
            X, y = sample(rule, t1, t2, cal[_family(rule)], seed)
            models[rule, seed] = DR.fit_risk_model(X, y, seed=seed)
    out["calibration_loss_ha"] = float((lw * area).sum())
    del cal, lw
    log(f"{out['split']}: models fitted {time.perf_counter() - start:.0f} s")

    # Test window: build, score, drop
    test = stack(t2)
    f2, lt = forest_pct(t2), loss_pct(t2, t3)
    rows = np.flatnonzero(f2 > 0)
    w_pos = lt[rows].astype("float32") / 100 * area[rows]
    w_neg = np.clip(f2[rows] - lt[rows], 0, None).astype("float32") / 100 * area[rows]
    half = f2[rows] >= 50
    own = {"half": lost("half", t2, t3)[rows], "any": lt[rows] > 0}
    out["coverage"] = {"forest_ha": float((f2[rows] / 100 * area[rows]).sum()), "test_loss_ha": float(w_pos.sum()),
                       "forest_in_half_cells": float((f2[rows] / 100 * area[rows])[half].sum()),
                       "test_loss_in_half_cells": float(w_pos[half].sum()),
                       "half_cells": int(half.sum()), "any_cells": int(rows.size)}
    for (rule, seed), (model, cv) in models.items():
        risk = score(model, test[_family(rule)], rows)
        scored = half if rule in ("half", "any_half") else np.ones(rows.size, dtype=bool)
        label = own[_family(rule)]
        r = out["rules"][rule]
        r["aucs_all"].append(weighted_auc(risk, w_pos, w_neg))
        r["aucs_scored"].append(weighted_auc(risk[scored], w_pos[scored], w_neg[scored]))
        r["aucs_own"].append(weighted_auc(risk[scored], label[scored].astype("float32"), (~label[scored]).astype("float32")))
        if seed == seeds[0]:
            r["importance"] = DR.driver_importance(model, FEATURES)
            r["cv_accuracy"] = cv["cv_accuracy_mean"]
        log(f"{out['split']}: {rule} seed {seed} scored {time.perf_counter() - start:.0f} s")
    del test
    out["seconds"] = time.perf_counter() - start
    out["peak_gb"] = _peak_gb()
    return out


def cached(name, function, rerun=False):
    """Run `function()` once and keep its JSON result as outputs/runs/<name>.json."""
    path = RUNS / f"{name}.json"
    if path.exists() and not rerun:
        return json.loads(path.read_text(encoding="utf-8"))
    result = function()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    return result


def split_name(split):
    return "training_rules_{}_{}_{}".format(*split)


if __name__ == "__main__":
    for split in SPLITS:
        cached(split_name(split), lambda: run_split(split), rerun="--rerun" in sys.argv)
        print("SAVED", split_name(split), flush=True)
