"""Notebooks 05 and 06: one change at a time in the improved model.

Starting from the improved model trained on the shares (steps 4-5), each variant changes one thing:

  latest      yearly layers (population, land-surface temperature, soil moisture) at their latest
              year instead of the window's start (the deployed hold-out's reading, notebook 03)
  drop        each predictor of the improved model left out in turn (notebooks 02-03: kept
              because leaving it out costs AUC)
  add         each predictor tested and not kept added in turn: aspect, yearly rainfall
              instead of its 1991-2020 normal, protected area, distance to mines, roads (GRIP4),
              ecoregion, night lights, distance to forest edge at the window's start
  ecoregion   one model per ecoregion, put back on one scale with the ecoregion's calibration
              loss rate (prior correction), against the national model

To make 25 variants affordable, the training cells are drawn once per seed (in proportion to the
forest lost and kept, notebook 04) and every variant fits on its columns of those cells; the test
scores a fixed random sample of 3 million forest cells, the same for every variant, by AUC over
the 30 m pixels. The model is the deployed random forest (drivers.fit_risk_model's settings)
without its cross-validation, which is not used here.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as MM                                              # noqa: E402

G, M, P = MM.G, MM.M, MM.P
BASE = MM.FEATURES                                              # the improved model (16 predictors)
LATEST = ["Population density", "Land-surface temperature", "Soil moisture"]
ADDED = ["Aspect", "Precipitation", "Protected area", "Distance to mines (2019)", "Distance to roads (GRIP4)",
         "Ecoregion", "Night lights", "Distance to forest edge"]
TEST_CELLS = 3_000_000


def variants():
    """{name: predictor list}; "(latest year)" marks a yearly layer read at its latest year."""
    out = {"Improved model": BASE,
           "Yearly layers at their latest year": [f"{n} (latest year)" if n in LATEST else n for n in BASE]}
    out.update({f"Without {n}": [m for m in BASE if m != n] for n in BASE})
    for n in ADDED:
        if n == "Precipitation":
            out["Yearly rainfall instead of the normal"] = [n if m == "Precipitation (1991–2020 normal)" else m for m in BASE]
        else:
            out[f"With {n}"] = BASE + [n]
    return out


def _forest_edge_distance(year):
    """Distance (m) to the edge of the forest at `year`: cells at least half forest next to one that is not."""
    edge = M.DR.forest_edge(MM._on_grid(MM.forest_pct(year) >= 50, False))
    return P._distance_m(edge)[MM.hansen()["idx"]]


def build(year, names, rows):
    """{predictor: values at `rows`} at `year`, one predictor at a time (only `rows` are kept)."""
    import warnings
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for name in names:
            if name.endswith(" (latest year)"):
                base = name[: -len(" (latest year)")]
                v = MM._shared(base, M.LATEST[base])
            elif name == "Recent loss rate around the cell (5 km)":
                v = MM.local_loss_rate(year)
            elif name == "Distance to recent loss":
                v = MM.distance_to(MM.lost("any", year - 3, year))
            elif name == "Distance to prior loss":
                v = MM.distance_to(MM.lost("any", MM.hansen()["first"] - 1, year))
            elif name == "Distance to forest edge":
                v = _forest_edge_distance(year)
            else:
                v = MM._shared(name, year)
            out[name] = np.asarray(v)[rows]
    return out


def ecoregions():
    """Ecoregion code of each cell with forest in 2000; cells just outside the polygons take the nearest one."""
    from scipy.ndimage import distance_transform_edt
    eco = G.read_layer("ecoregions", ["ecoregion"], as_grid=True, fill=0)["ecoregion"].astype("int16")
    r, c = distance_transform_edt(eco == 0, return_distances=False, return_indices=True)
    return G.from_grid(eco[r, c])[MM.hansen()["idx"]]


def _fit(X, y, seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=5, n_jobs=-1,
                                  random_state=seed, class_weight="balanced").fit(X, y)


def _draw(t1, t2, seed, within=None):
    """Lost and kept cells drawn in proportion to their shares (notebook 04), optionally within a mask."""
    pos, neg, (w_pos, w_neg) = MM.training_cells("weighted", t1, t2)
    if within is not None:
        kp, kn = within[pos], within[neg]
        pos, neg, w_pos, w_neg = pos[kp], neg[kn], w_pos[kp], w_neg[kn]
    rng = np.random.default_rng(seed)
    n = min(MM.SAMPLES, pos.size, neg.size)
    cells = np.concatenate([MM._pick(pos, n, rng, w_pos), MM._pick(neg, n, rng, w_neg)])
    return cells, np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])


def _xy(values, cells, names):
    X = np.column_stack([values[n][cells] for n in names])
    ok = np.isfinite(X).all(axis=1)
    return X, ok


def run_split(split, seeds=(42, 1, 2), log=print):
    t1, t2, t3 = split
    start = time.perf_counter()
    area = MM.hansen()["area"]
    eco = ecoregions()
    VARIANTS = variants()
    all_names = sorted({n for names in VARIANTS.values() for n in names})

    # Training cells: national, and per forest ecoregion, for each seed
    f1, lw = MM.forest_pct(t1), MM.loss_pct(t1, t2)
    codes = [int(c) for c in np.unique(eco[f1 > 0]) if (lw[eco == c] > 0).sum() >= 1000]
    draws = {(None, s): _draw(t1, t2, s) for s in seeds}
    draws.update({(c, s): _draw(t1, t2, s, eco == c) for c in codes for s in seeds})
    rate = {c: float((lw * area)[eco == c].sum() / (f1 * area)[eco == c].sum()) for c in codes}
    cal_rows = np.unique(np.concatenate([cells for cells, _ in draws.values()]))
    cal = build(t1, all_names, cal_rows)
    log(f"{M.split_label(split)}: calibration predictors {time.perf_counter() - start:.0f} s")

    # Test cells: a fixed random sample of the forest at t2, weighted by forest lost and kept
    f2, lt = MM.forest_pct(t2), MM.loss_pct(t2, t3)
    forest2 = np.flatnonzero(f2 > 0)
    test_rows = np.sort(np.random.default_rng(0).choice(forest2, min(TEST_CELLS, forest2.size), replace=False))
    w_pos = lt[test_rows] / 100 * area[test_rows]
    w_neg = np.clip(f2[test_rows] - lt[test_rows], 0, None) / 100 * area[test_rows]
    test = build(t2, all_names, test_rows)
    log(f"{M.split_label(split)}: test predictors {time.perf_counter() - start:.0f} s")

    def fit_score(names, key, seed, rows=None):
        """Fit on the drawn cells of `key` and score the test cells (or those at positions `rows`)."""
        cells, y = draws[key, seed]
        X, ok = _xy(cal, np.searchsorted(cal_rows, cells), names)
        model = _fit(X[ok], y[ok], seed)
        rows = np.arange(test_rows.size) if rows is None else rows
        Xt, okt = _xy(test, rows, names)
        risk = np.full(rows.size, np.nan, dtype="float32")
        risk[okt] = model.predict_proba(Xt[okt])[:, 1]
        return risk

    out = {"split": M.split_label(split), "test_cells": int(test_rows.size), "variants": {}}
    for name, names in VARIANTS.items():
        out["variants"][name] = [MM.weighted_auc(fit_score(names, None, s), w_pos, w_neg) for s in seeds]
        log(f"{out['split']}: {name} {np.mean(out['variants'][name]):.4f} {time.perf_counter() - start:.0f} s")

    # One model per ecoregion, against the national model (same predictors, same test cells)
    eco_test = eco[test_rows]
    names_json = {int(k): v for k, v in json.loads((M.DD.DATA / "ecoregions.json").read_text()).items()}
    eco_out = {"national": [], "per_ecoregion_raw": [], "per_ecoregion_corrected": [],
               "inside": {names_json.get(c, str(c)): {"national": [], "corrected": []} for c in codes},
               "calibration_loss_rate": {names_json.get(c, str(c)): rate[c] for c in codes}}
    for s in seeds:
        national = fit_score(BASE, None, s)
        raw = np.full(test_rows.size, np.nan, dtype="float32")
        corrected = raw.copy()
        for c in codes:
            inside = eco_test == c
            p = fit_score(BASE, c, s, np.flatnonzero(inside))
            raw[inside] = p
            odds = np.clip(p, 1e-4, 1 - 1e-4)
            odds = odds / (1 - odds) * rate[c] / (1 - rate[c])
            corrected[inside] = odds / (1 + odds)
        eco_out["national"].append(MM.weighted_auc(national, w_pos, w_neg))
        eco_out["per_ecoregion_raw"].append(MM.weighted_auc(raw, w_pos, w_neg))
        eco_out["per_ecoregion_corrected"].append(MM.weighted_auc(corrected, w_pos, w_neg))
        for c in codes:
            inside = eco_test == c
            n = names_json.get(c, str(c))
            eco_out["inside"][n]["national"].append(MM.weighted_auc(national[inside], w_pos[inside], w_neg[inside]))
            eco_out["inside"][n]["corrected"].append(MM.weighted_auc(corrected[inside], w_pos[inside], w_neg[inside]))
        log(f"{out['split']}: ecoregions seed {s} {time.perf_counter() - start:.0f} s")
    out["ecoregions"] = eco_out
    out["seconds"] = time.perf_counter() - start
    out["peak_gb"] = MM._peak_gb()
    return out


def split_name(split):
    return "model_variants_{}_{}_{}".format(*split)


if __name__ == "__main__":
    for split in MM.SPLITS:
        MM.cached(split_name(split), lambda: run_split(split), rerun="--rerun" in sys.argv)
        print("SAVED", split_name(split), flush=True)
