"""Notebook 08: existing tools on the 100 m grid, on the 30 m shares.

  riskmapjnr     its moving-window method, written here on the shares (riskmapjnr 1.3.2's steps,
                 line by line): the local annual loss rate in a window, as forest lost / forest at
                 the start; zero risk beyond the distance from non-forest holding 99.5 % of the loss;
                 30 classes (equal area or equal interval); each class's observed rate; the window
                 and slicing with the lowest wRMSE in the validation (test) window, as makemap
                 chooses. On the shares the window sums are those of the 30 m pixels it would count
                 at its native resolution. The package needs whole cells lost or kept, which at
                 100 m means one of the rules notebook 04 rejected. Checked against the package on a
                 block of the 100 m grid (check_against_package).
  forestatrisk   its binomial iCAR model (Vieilledent et al. 2022), run with its own code in
                 .venv_tools (tools_forestatrisk.py) on the improved model's predictors, with
                 training cells drawn in proportion to their shares (= sampling 30 m pixels) and
                 10 km cells for the spatial random effects
  weights of evidence  Dinamica EGO's method (Bonham-Carter 1994), with the loss and the kept
                 forest of each range in hectares from the shares
  rank average   the improved model and riskmapjnr's local rate, each cell's mean percentile rank

Scored as every model at 100 m: AUC over the 30 m forest pixels, Verra's tests with forest and loss in
hectares (verra.py).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_variants as D                                     # noqa: E402
import model as MM                                              # noqa: E402
import verra as V                                              # noqa: E402

G, M = MM.G, MM.M
SCRIPTS = Path(__file__).resolve().parent
RMJ_WINDOWS = (13, 27, 51)            # cells: 1.3, 2.7 and 5.1 km
RMJ_SQUARE = 100                      # riskmapjnr's validation squares: 10 km
ICAR_CELL = 100                       # forestatrisk's spatial cells: 10 km
WORK = MM.RUNS / "tools_work"


# --------------------------------------------------------------------------
# riskmapjnr

def rmj_rescale(value, min_val=2, max_val=10000):
    """riskmapjnr.misc.rescale: rates in [0, 1] -> integers in [min_val, max_val]."""
    value = np.asarray(value, dtype=float).copy()
    value[value < 1e-06] = 1e-06
    return np.rint(((value * 1e6 - 1) * (max_val - min_val) / 999999.0) + min_val).astype(int)


def rmj_local_rate_grid(defor, forest, win_size, years):
    """riskmapjnr.local_defor_rate on 2D arrays: sums in a win_size window (zeros beyond the edge),
    annual rate 1 - (1 - defor / forest) ^ (1 / years) where forest > 0 (NaN elsewhere)."""
    from scipy.ndimage import uniform_filter
    n = win_size ** 2
    win_defor = uniform_filter(defor.astype("float64"), win_size, mode="constant", cval=0) * n
    win_for = uniform_filter(forest.astype("float64"), win_size, mode="constant", cval=0) * n
    out = np.full(defor.shape, np.nan)
    w = forest > 0
    out[w] = 1 - (1 - np.clip(win_defor[w] / win_for[w], 0, 1)) ** (1 / years)
    return out


def rmj_local_rate(year_from, year_to, win_size):
    """Local annual loss rate over the cells with forest in 2000, from the shares."""
    defor = MM._on_grid(MM.loss_pct(year_from, year_to).astype("float32"), 0.0)
    forest = MM._on_grid(MM.forest_pct(year_from).astype("float32"), 0.0)
    return G.from_grid(rmj_local_rate_grid(defor, forest, win_size, year_to - year_from))[MM.hansen()["idx"]]


def rmj_distance(year):
    """Distance (m) from non-forest: cells under half forest; those with some forest sit at half a cell."""
    f = MM.forest_pct(year)
    d = V.distance_from_nonforest(f >= 50)
    return np.where((f > 0) & (f < 50), V.pixel_size_m() / 2, d)


def rmj_threshold(distance, loss_ha):
    """riskmapjnr.dist_edge_threshold: the distance holding 99.5 % of the loss (bins of one cell)."""
    width = V.pixel_size_m()
    edges = np.arange(0, distance.max() + 2 * width, width)
    cumulative = np.cumsum(np.histogram(distance, bins=edges, weights=loss_ha)[0]) / loss_ha.sum()
    return float(edges[int(np.argmax(cumulative >= 0.995)) + 1])


def rmj_bins(values, weights, method, ncat=30):
    """riskmapjnr.defor_cat's class limits on the rescaled values (1-10000), the histogram weighted by
    each cell's forest (its 30 m pixels)."""
    if method == "ei":
        size = (10000 - 1) / ncat
        bins = np.rint([0] + [1 + i * size for i in range(ncat)] + [10000]).astype(int).tolist()
    else:
        counts = np.bincount(values, weights=weights, minlength=10001)[1:10001]
        cum = np.cumsum(counts / counts.sum())
        cum[-1] = 1.0
        bins = [0, 1]
        for q in [i / ncat for i in range(1, ncat + 1)]:
            k = int(np.sum(cum <= q))
            if k > 1:
                bins.append(k)
        bins = sorted(set(bins))
    bins[-1] = 10001
    return bins


def rmj_classes(values, bins):
    """pd.cut(values, bins, right=False, labels=False) + 1; 0 where there is no value."""
    out = np.zeros(values.shape, dtype="int32")
    ok = values > 0
    out[ok] = np.digitize(values[ok], bins[1:-1], right=False) + 1
    return out


def riskmapjnr(split, log=print, same_timing=False):
    """riskmapjnr's method. As designed, its calibration map is the local rate of the calibration
    window's own loss (t1-t2), and the same map at t2 predicts the test window. same_timing=True
    gives it the timing of every other model here (and of VT0007's benchmark, distance at t1): the
    calibration map from the loss of the years before t1 (t1-3 to t1), the test map unchanged."""
    t1, t2, t3 = split
    area = MM.hansen()["area"]
    f1, f2 = MM.forest_pct(t1), MM.forest_pct(t2)
    forest1, forest2 = f1 / 100 * area, f2 / 100 * area
    loss_cal, loss_val = MM.loss_pct(t1, t2) / 100 * area, MM.loss_pct(t2, t3) / 100 * area
    d1, d2 = rmj_distance(t1), rmj_distance(t2)
    thresh = rmj_threshold(d1[f1 > 0], loss_cal[f1 > 0])
    row, col = G.row_col(G.cells()[MM.hansen()["idx"]])
    square = (row // RMJ_SQUARE) * (G.grid()["width"] // RMJ_SQUARE + 1) + col // RMJ_SQUARE
    has = np.bincount(square, weights=forest2) > 0
    rows, best = [], None
    for ws in RMJ_WINDOWS:
        theta = rmj_local_rate(t1, t2, ws)
        theta_fit = rmj_local_rate(t1 - (t2 - t1), t1, ws) if same_timing else theta
        ldefz = np.where(f1 > 0, rmj_rescale(np.nan_to_num(theta_fit)), 0)
        ldefz[(f1 > 0) & (d1 >= thresh)] = 1
        ldefz_v = np.where(f2 > 0, rmj_rescale(np.nan_to_num(theta)), 0)
        ldefz_v[(f2 > 0) & (d2 >= thresh)] = 1
        for method in ("ei", "ea"):
            bins = rmj_bins(ldefz[f1 > 0], forest1[f1 > 0], method)
            cat_fit, cat_val = rmj_classes(ldefz, bins), rmj_classes(ldefz_v, bins)
            k = cat_fit.max() + 1
            nfor = np.bincount(cat_fit, weights=forest1, minlength=k)
            ndefor = np.bincount(cat_fit, weights=loss_cal, minlength=k)
            share = np.clip(np.divide(ndefor, nfor, out=np.zeros(k), where=nfor > 0), 0, 1)   # rounding of the shares
            rate = 1 - (1 - share) ** (1 / (t2 - t1))
            pred = forest2 * (1 - (1 - rate[np.minimum(cat_val, k - 1)]) ** (t3 - t2))
            pred[cat_val == 0] = 0
            p = np.bincount(square, weights=pred)[has]
            o = np.bincount(square, weights=loss_val)[has]
            w = np.bincount(square, weights=forest2)[has]
            p *= o.sum() / p.sum()                                   # no_quantity_error=True
            wrmse = float(np.sqrt(np.sum((p - o) ** 2 * w / w.sum())))
            rows.append({"ws": ws, "m": method, "wRMSE": round(wrmse, 1)})
            log(f"{M.split_label(split)}: riskmapjnr {rows[-1]}")
            if np.isfinite(wrmse) and (best is None or wrmse < best[0]):
                best = (wrmse, ws, method, cat_fit, cat_val, ldefz, ldefz_v)
    _, ws, method, cat_fit, cat_val, ldefz, ldefz_v = best
    choice = {"dist_thresh_m": thresh, "ws_hat": ws, "m_hat": method, "squares": int(has.sum()), "models": rows}
    risk_fit = np.where(f1 > 0, ldefz, np.nan).astype("float32")     # its continuous map: the rescaled local rate
    risk_test = np.where(f2 > 0, ldefz_v, np.nan).astype("float32")
    return (cat_fit, cat_val), (risk_fit, risk_test), choice


CHECK_CENTRE = (-10.1, 9.2)                                      # forest-savanna around Kissidougou
CHECK_CELLS = 1000                                               # 1,000 x 1,000 cells of the 100 m grid


def check_against_package(windows=(5, 11, 21), split=(2013, 2016), size=CHECK_CELLS):
    """Our local rate against riskmapjnr's own (local_defor_rate, run in .venv_tools) on the same
    forest-change raster: a block of the 100 m grid, 1 = cell with loss in the window, 2 = forest kept,
    0 = no forest at its start. Per window size: share of forest cells with the same value."""
    import rasterio
    from rasterio.transform import Affine
    from defor_tools import TOOLS_PYTHON, TOOLS_ENV
    t1, t2 = split
    g = G.grid()
    t = g["transform"]
    r0 = int((t[5] - CHECK_CENTRE[1]) / -t[4]) - size // 2
    c0 = int((CHECK_CENTRE[0] - t[2]) / t[0]) - size // 2
    forest = MM._on_grid(MM.forest_pct(t1), 0)[r0:r0 + size, c0:c0 + size] > 0
    lost = MM._on_grid(MM.loss_pct(t1, t2), 0)[r0:r0 + size, c0:c0 + size] > 0
    fcc = np.where(forest, np.where(lost, 1, 2), 0).astype("uint8")
    folder = MM.RUNS / "riskmapjnr_check"
    folder.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": size, "width": size, "count": 1, "dtype": "uint8", "nodata": 0, "crs": "EPSG:4326",
               "transform": Affine(t[0], 0, t[2] + c0 * t[0], 0, t[4], t[5] + r0 * t[4])}
    with rasterio.open(folder / "fcc123.tif", "w", **profile) as dst:
        dst.write(fcc, 1)
    done = subprocess.run([str(TOOLS_PYTHON), "-W", "ignore", str(SCRIPTS / "tools_riskmapjnr_rate.py"), str(folder / "fcc123.tif"),
                           str(folder), str(t2 - t1), *map(str, windows)], env=TOOLS_ENV, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-2000:])
    out = {"block": {"cells": size * size, "forest_cells": int(forest.sum()), "cells_with_loss": int((fcc == 1).sum())}}
    for ws in windows:
        with rasterio.open(folder / f"ldefrate_ws{ws}.tif") as src:
            package = src.read(1)
        ours = np.where(forest, rmj_rescale(np.nan_to_num(rmj_local_rate_grid(fcc == 1, fcc > 0, ws, t2 - t1))), 0)
        out[ws] = {"identical": float((ours[forest] == package[forest]).mean()),
                   "largest_difference": int(np.abs(ours[forest].astype(int) - package[forest].astype(int)).max())}
    return out


# --------------------------------------------------------------------------
# Weights of evidence

def weights_of_evidence(cal, test, rows1, rows2, t1, t2, bins=10):
    """Each predictor in `bins` quantile ranges; weight = ln(P(range | lost) / P(range | kept)) with the
    lost and kept forest in hectares; risk = prior log-odds + the weights of the cell's ranges."""
    area = MM.hansen()["area"]
    f1, lw = MM.forest_pct(t1)[rows1], MM.loss_pct(t1, t2)[rows1]
    lost_ha = lw / 100 * area[rows1]
    kept_ha = np.clip(f1 - lw, 0, None) / 100 * area[rows1]
    valid = np.all([np.isfinite(cal[n]) for n in MM.FEATURES], axis=0)
    prior = lost_ha[valid].sum() / (lost_ha[valid].sum() + kept_ha[valid].sum())
    logit_fit = np.full(rows1.size, np.log(prior / (1 - prior)))
    logit_test = np.full(rows2.size, np.log(prior / (1 - prior)))
    for name in MM.FEATURES:
        values = cal[name][valid]
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
        k = np.digitize(values, edges)
        p_d = np.bincount(k, weights=lost_ha[valid], minlength=edges.size + 1) + 0.5
        p_n = np.bincount(k, weights=kept_ha[valid], minlength=edges.size + 1) + 0.5
        weight = np.log((p_d / p_d.sum()) / (p_n / p_n.sum()))
        for logit, values_all in ((logit_fit, cal[name]), (logit_test, test[name])):
            logit += np.where(np.isfinite(values_all), weight[np.digitize(np.nan_to_num(values_all), edges)], 0)
    return (1 / (1 + np.exp(-logit_fit))).astype("float32"), (1 / (1 + np.exp(-logit_test))).astype("float32")


# --------------------------------------------------------------------------
# forestatrisk

def _icar_cells(rows):
    r, c = G.row_col(G.cells()[MM.hansen()["idx"][rows]])
    return ((r // ICAR_CELL) * ((G.grid()["width"] + ICAR_CELL - 1) // ICAR_CELL) + c // ICAR_CELL).astype("int32")


def _write_matrix(path, values, rows_count):
    """Write the predictor matrix one column at a time (no second copy in memory)."""
    out = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(rows_count, len(MM.FEATURES)))
    for j, name in enumerate(MM.FEATURES):
        out[:, j] = values[name]
    out.flush()
    del out


def run_forestatrisk(log=print):
    from defor_tools import TOOLS_PYTHON, TOOLS_ENV
    done = subprocess.run([str(TOOLS_PYTHON), "-W", "ignore", str(SCRIPTS / "tools_forestatrisk.py"), str(WORK)],
                          env=TOOLS_ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        raise RuntimeError(f"forestatrisk failed:\n{done.stderr[-3000:]}")
    log(done.stdout[-500:])


# --------------------------------------------------------------------------
# One split, every tool

def _rank(values, forest):
    """Percentile rank over the forest cells with a value (NaN elsewhere)."""
    from scipy.stats import rankdata
    valid = forest & np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype="float32")
    out[valid] = rankdata(values[valid]) / valid.sum()
    return out


def run_split(split, seeds=(42, 1, 2), log=print):
    t1, t2, t3 = split
    start = time.perf_counter()
    n = MM.hansen()["idx"].size
    f1, f2 = MM.forest_pct(t1), MM.forest_pct(t2)
    rows1, rows2 = np.flatnonzero(f1 > 0), np.flatnonzero(f2 > 0)
    out = {"split": M.split_label(split), "tools": {}}

    def score(name, fit, test, classes=None, aucs=None):
        tests = V.vt0007_tests(split, fit, test, classes=classes)
        out["tools"][name] = {"aucs": aucs if aucs is not None else [V.pixel_auc(test, t2, t3)],
                              "fit_medae_ha": tests["fit_medae_ha"], "prediction_medae_ha": tests["prediction_medae_ha"],
                              "loss_without_class": tests["loss_without_class"]}
        log(f"{out['split']}: {name} AUC {np.mean(out['tools'][name]['aucs']):.4f} fit {tests['fit_medae_ha']:.0f} "
            f"prediction {tests['prediction_medae_ha']:.0f} ({time.perf_counter() - start:.0f} s)")

    # 1 riskmapjnr
    classes, (rmj_fit, rmj_test), out["riskmapjnr_choice"] = riskmapjnr(split, log)
    score("riskmapjnr (moving window)", rmj_fit, rmj_test, classes=classes)

    # Calibration predictors: improved model, weights of evidence, forestatrisk inputs
    WORK.mkdir(parents=True, exist_ok=True)
    cal = D.build(t1, MM.FEATURES, rows1)
    models = {}
    for s in seeds:
        cells, y = D._draw(t1, t2, s)
        X = np.column_stack([cal[m][np.searchsorted(rows1, cells)] for m in MM.FEATURES])
        ok = np.isfinite(X).all(axis=1)
        models[s] = D._fit(X[ok], y[ok], s)
        if s == seeds[0]:
            np.savez(WORK / "training.npz", names=np.array(MM.FEATURES), X=X[ok], y=y[ok],
                     cell=_icar_cells(cells)[ok],
                     ncell_x=(G.grid()["width"] + ICAR_CELL - 1) // ICAR_CELL,
                     ncell_y=(G.grid()["height"] + ICAR_CELL - 1) // ICAR_CELL)
    improved_fit = np.full(n, np.nan, dtype="float32")
    improved_fit[rows1] = _score_rows(models[seeds[0]], cal, rows1.size)
    _write_matrix(WORK / "X_fit.npy", cal, rows1.size)
    np.save(WORK / "cell_fit.npy", _icar_cells(rows1))
    test = D.build(t2, MM.FEATURES, rows2)        # built before the weights, which need both windows
    woe_fit, woe_test = weights_of_evidence(cal, test, rows1, rows2, t1, t2)
    del cal
    log(f"{out['split']}: calibration done ({time.perf_counter() - start:.0f} s)")

    # Test predictors
    improved_aucs, improved_test = [], None
    for s in seeds:
        risk = np.full(n, np.nan, dtype="float32")
        risk[rows2] = _score_rows(models[s], test, rows2.size)
        improved_aucs.append(V.pixel_auc(risk, t2, t3))
        if s == seeds[0]:
            improved_test = risk
    _write_matrix(WORK / "X_test.npy", test, rows2.size)
    np.save(WORK / "cell_test.npy", _icar_cells(rows2))
    del test, models

    # 2 weights of evidence
    full_fit, full_test = np.full(n, np.nan, "float32"), np.full(n, np.nan, "float32")
    full_fit[rows1], full_test[rows2] = woe_fit, woe_test
    score("Weights of evidence (Dinamica EGO's method)", full_fit, full_test)

    # 3 forestatrisk
    run_forestatrisk(log)
    far_fit, far_test = np.full(n, np.nan, "float32"), np.full(n, np.nan, "float32")
    far_fit[rows1], far_test[rows2] = np.load(WORK / "far_fit.npy"), np.load(WORK / "far_test.npy")
    score("forestatrisk (iCAR)", far_fit, far_test)
    for f in WORK.glob("*"):
        f.unlink()

    # 4 the improved model, and its rank average with riskmapjnr's local rate
    score("Improved model", improved_fit, improved_test, aucs=improved_aucs)
    score("Rank average: improved model + riskmapjnr",
          (_rank(improved_fit, f1 > 0) + _rank(rmj_fit, f1 > 0)) / 2,
          (_rank(improved_test, f2 > 0) + _rank(rmj_test, f2 > 0)) / 2)
    out["seconds"] = time.perf_counter() - start
    out["peak_gb"] = MM._peak_gb()
    return out


def _score_rows(model, values, count, chunk=1_000_000):
    risk = np.full(count, np.nan, dtype="float32")
    for i in range(0, count, chunk):
        X = np.column_stack([values[m][i:i + chunk] for m in MM.FEATURES])
        ok = np.isfinite(X).all(axis=1)
        part = risk[i:i + chunk]
        part[ok] = model.predict_proba(X[ok])[:, 1]
    return risk


def split_name(split):
    return "tools_{}_{}_{}".format(*split)


if __name__ == "__main__":
    print("riskmapjnr check against the package:", MM.cached("riskmapjnr_check", check_against_package), flush=True)
    for split in MM.SPLITS:
        MM.cached(split_name(split), lambda: run_split(split), rerun="--rerun" in sys.argv)
        print("SAVED", split_name(split), flush=True)


def run_same_timing(split, log=print):
    """riskmapjnr with the same timing as the other models, and its rank average with the improved
    model (the saved first-seed maps of improve100.save_maps, identical to this step's)."""
    import improve as I
    t1, t2, t3 = split
    f1, f2 = MM.forest_pct(t1), MM.forest_pct(t2)
    classes, (rmj_fit, rmj_test), choice = riskmapjnr(split, log, same_timing=True)
    out = {"split": M.split_label(split), "riskmapjnr_choice": choice, "tools": {}}
    for name, fit, test, cls in (
            ("riskmapjnr (moving window)", rmj_fit, rmj_test, classes),
            ("Rank average: improved model + riskmapjnr",
             (_rank(I.load_map("improved_fit", split), f1 > 0) + _rank(rmj_fit, f1 > 0)) / 2,
             (_rank(I.load_map("improved_test", split), f2 > 0) + _rank(rmj_test, f2 > 0)) / 2, None)):
        tests = V.vt0007_tests(split, fit, test, classes=cls)
        out["tools"][name] = {"aucs": [V.pixel_auc(test, t2, t3)], "fit_medae_ha": tests["fit_medae_ha"],
                              "prediction_medae_ha": tests["prediction_medae_ha"],
                              "loss_without_class": tests["loss_without_class"]}
        log(f"{out['split']}: {name} (same timing) AUC {out['tools'][name]['aucs'][0]:.4f} "
            f"fit {tests['fit_medae_ha']:.0f} prediction {tests['prediction_medae_ha']:.0f}")
    return out
