"""Verra VT0007 (v1.0, 2024) on the 100 m grid, and the baseline runs (notebooks 03 and 07).

Each cell's forest and loss are in hectares from its 30 m shares (forest share x cell area), so a
cell that is partly forest is counted for its forest:

  benchmark    distance of each forest cell from the cells under half forest (or outside Guinea);
               cells under half forest are at distance 0. The negligible-risk threshold holds
               99.5 % of the calibration loss (ha) within it; 30 classes as Table 2
  alternative  the model's risk in 30 geometric classes (defor_methods.alternative_classes);
               forest the risk map leaves out (no risk value) gets no class and no predicted loss
  density      loss (ha) / forest (ha) of each modelling region (class x admin region) in the
               calibration window, times the forest (ha) of each cell
  tests        fit and prediction MedAE (ha) over the ~100,000 ha blocks wholly in Guinea, with
               the prediction scaled to the observed total and capped at each cell's forest

Models (all with the platform's grid and its predictors, notebook 05):

  deployed / half      the deployed predictors, yearly layers at their latest year, trained on
                       the platform's half-cell masks, risk only for cells at least half forest
  improved / half      the improved model, same masks
  improved / weighted  the improved model trained in proportion to the shares (notebook 04), risk for
                       every cell with forest
"""

import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as MM                                              # noqa: E402

G, M = MM.G, MM.M
N_CLASSES = M.N_CLASSES
COARSE_CELL_HA = M.COARSE_CELL_HA

MODELS = {"deployed_half": (MM.DEPLOYED, "half"), "improved_half": (MM.IMPROVED, "half"),
          "improved_weighted": (MM.IMPROVED, "weighted")}
LABELS = {"benchmark": "VT0007 benchmark (distance to edge)", "deployed_half": "Deployed model, half-cell masks (platform)",
          "improved_half": "Improved model, half-cell masks", "improved_weighted": "Improved model, weighted by the shares"}


def pixel_size_m():
    return G.grid()["cell_deg"] * 111_320.0                     # as drivers.distance_to


@lru_cache(maxsize=1)
def admin_regions():
    adm = G.read_layer("gaul_admin1", ["adm1"])["adm1"][MM.hansen()["idx"]].astype("int64")
    _, codes = np.unique(adm, return_inverse=True)
    return codes.astype("int32") + 1


@lru_cache(maxsize=1)
def coarse_cells():
    """VT0007 5.5: square blocks of ~100,000 ha wholly in Guinea -> (block of each forest cell or -1,
    ids of all such blocks, forest or not)."""
    row, col = G.row_col()
    side = int(round(np.sqrt(COARSE_CELL_HA / G.cell_area_ha().mean())))
    ids = (row // side) * (G.grid()["width"] // side + 1) + col // side
    full = np.bincount(ids) == side * side
    blocks = np.flatnonzero(full)
    return np.where(full[ids], ids, -1)[MM.hansen()["idx"]], blocks


def medae(predicted_ha, observed_ha):
    cells, blocks = coarse_cells()
    keep = cells >= 0
    n = blocks.max() + 1
    p = np.bincount(cells[keep], weights=predicted_ha[keep], minlength=n)[blocks]
    o = np.bincount(cells[keep], weights=observed_ha[keep], minlength=n)[blocks]
    return float(np.median(np.abs(p - o)))


def distance_from_nonforest(half_forest):
    """Distance (m) of each cell from the nearest cell under half forest or outside Guinea."""
    from scipy.ndimage import distance_transform_edt
    d = G.from_grid(distance_transform_edt(MM._on_grid(half_forest, False)))[MM.hansen()["idx"]]
    return (d * pixel_size_m()).astype("float32")


def negligible_risk_threshold(distance, loss_ha):
    width = pixel_size_m()
    edges = np.arange(0, distance.max() + 2 * width, width)
    cumulative = np.cumsum(np.histogram(distance, bins=edges, weights=loss_ha)[0]) / loss_ha.sum()
    i = int(np.argmax(cumulative >= 0.995))
    return (edges[i] + edges[i + 1]) / 2


def benchmark_classes(distance, forest, nrt):
    r = (pixel_size_m() / nrt) ** (1 / (N_CLASSES - 1))
    classes = np.ones(distance.shape, dtype="uint8")
    for k in range(2, N_CLASSES + 1):
        classes[distance < nrt * r ** (k - 2)] = k
    classes[~forest] = 0
    return classes


def _regions(classes):
    return classes.astype("int64") * 1000 + admin_regions()


def frequencies(classes, loss_ha, forest_ha):
    sel = classes > 0
    ids, inverse = np.unique(_regions(classes)[sel], return_inverse=True)
    return dict(zip(ids, np.bincount(inverse, weights=loss_ha[sel]) / np.bincount(inverse, weights=forest_ha[sel])))


def density(classes, forest_ha, freq):
    out = np.zeros(classes.shape, dtype="float64")
    sel = classes > 0
    keys, inverse = np.unique(_regions(classes)[sel], return_inverse=True)
    out[sel] = np.array([freq.get(k, 0.0) for k in keys])[inverse] * forest_ha[sel]
    return out


def vt0007_tests(split, risk_fit=None, risk_test=None, keep_maps=False, classes=None):
    """Fit and prediction MedAE (ha); without risk maps, the benchmark. Also the share of the
    confirmation loss in forest without a class (left out of the risk map), and with keep_maps
    the fitted and predicted loss of each cell (ha). classes: (fit, prediction) class maps made by
    another tool, any codes >= 1 on forest."""
    t1, t2, t3 = split
    area = MM.hansen()["area"]
    f1, f2 = MM.forest_pct(t1), MM.forest_pct(t2)
    forest1, forest2 = f1 / 100 * area, f2 / 100 * area
    loss_cal = MM.loss_pct(t1, t2) / 100 * area
    loss_conf = MM.loss_pct(t2, t3) / 100 * area
    if classes is not None:
        classes_fit = np.where(f1 > 0, classes[0], 0)
        classes_pred = np.where(f2 > 0, classes[1], 0)
    elif risk_fit is None:
        d1, d2 = distance_from_nonforest(f1 >= 50), distance_from_nonforest(f2 >= 50)
        nrt = negligible_risk_threshold(d1[f1 > 0], loss_cal[f1 > 0])
        classes_fit, classes_pred = benchmark_classes(d1, f1 > 0, nrt), benchmark_classes(d2, f2 > 0, nrt)
    else:
        classes_fit = M.alternative_classes(risk_fit, f1 > 0)
        classes_pred = M.alternative_classes(risk_test, f2 > 0)
    freq = frequencies(classes_fit, loss_cal, forest1)
    fitted = density(classes_fit, forest1, freq)
    predicted = density(classes_pred, forest2, freq)
    for _ in range(50):                                         # VT0007 5.3.2.3, capped at the cell's forest
        ratio = loss_conf.sum() / predicted.sum()
        predicted = np.minimum(predicted * ratio, forest2)
        if ratio <= 1.00001:
            break
    out = {"fit_medae_ha": medae(fitted, loss_cal), "prediction_medae_ha": medae(predicted, loss_conf),
           "loss_without_class": float(loss_conf[classes_pred == 0].sum() / loss_conf.sum())}
    if keep_maps:
        out["fitted"], out["predicted"] = fitted, predicted
    if risk_fit is None and classes is None:                    # its risk map: allocated loss per ha of forest
        out["nrt_m"] = float(nrt)
        score = np.where(classes_pred > 0, predicted / np.maximum(forest2, 1e-9), np.nan).astype("float32")
        out["aucs"] = [pixel_auc(score, t2, t3)]
    return out


def pixel_auc(risk, t2, t3):
    """AUC over the 30 m forest pixels at t2 (notebook 04); forest without a risk value counts as lowest."""
    area = MM.hansen()["area"]
    f2, lt = MM.forest_pct(t2), MM.loss_pct(t2, t3)
    rows = np.flatnonzero(f2 > 0)
    w_pos = lt[rows] / 100 * area[rows]
    w_neg = np.clip(f2[rows] - lt[rows], 0, None) / 100 * area[rows]
    return MM.weighted_auc(np.where(np.isfinite(risk[rows]), risk[rows], -1.0), w_pos, w_neg)


def run_split(split, seeds=(42, 1, 2), log=print):
    """Benchmark and the three models on one split: pixel AUC per seed, VT0007 tests (first seed),
    importances, time and peak memory."""
    t1, t2, t3 = split
    start = time.perf_counter()
    n = MM.hansen()["idx"].size
    out = {"split": M.split_label(split), "models": {"benchmark": vt0007_tests(split)}}
    log(f"{out['split']}: benchmark {time.perf_counter() - start:.0f} s")
    for config, rules in ((MM.DEPLOYED, ["half"]), (MM.IMPROVED, ["half", "weighted"])):
        names = MM.features(config)
        families = sorted({MM._family(r) for r in rules})
        keys = [k for k, (c, r) in MODELS.items() if c is config and r in rules]
        cal = MM.stack(t1, config, families)
        models = {}
        for key in keys:
            rule = MODELS[key][1]
            for seed in seeds:
                X, y = MM.sample(rule, t1, t2, cal[MM._family(rule)], seed, names)
                models[key, seed] = M.DR.fit_risk_model(X, y, seed=seed)
        fit_maps = {}
        f1 = MM.forest_pct(t1)
        for key in keys:
            rule = MODELS[key][1]
            rows = np.flatnonzero(f1 >= 50 if rule == "half" else f1 > 0)
            risk = np.full(n, np.nan, dtype="float32")
            risk[rows] = MM.score(models[key, seeds[0]][0], cal[MM._family(rule)], rows, names=names)
            fit_maps[key] = risk
        del cal
        log(f"{out['split']}: {', '.join(keys)} fitted {time.perf_counter() - start:.0f} s")
        test = MM.stack(t2, config, families)
        f2 = MM.forest_pct(t2)
        for key in keys:
            rule = MODELS[key][1]
            rows = np.flatnonzero(f2 >= 50 if rule == "half" else f2 > 0)
            r = out["models"][key] = {"aucs": []}
            for seed in seeds:
                risk = np.full(n, np.nan, dtype="float32")
                risk[rows] = MM.score(models[key, seed][0], test[MM._family(rule)], rows, names=names)
                r["aucs"].append(pixel_auc(risk, t2, t3))
                if seed == seeds[0]:
                    r.update(vt0007_tests(split, fit_maps[key], risk))
                    r["importance"] = M.DR.driver_importance(models[key, seed][0], names)
            log(f"{out['split']}: {key} scored {time.perf_counter() - start:.0f} s")
        del test, fit_maps, models
    out["seconds"] = time.perf_counter() - start
    out["peak_gb"] = MM._peak_gb()
    return out


def split_name(split):
    return "baseline_{}_{}_{}".format(*split)


if __name__ == "__main__":
    for split in MM.SPLITS:
        MM.cached(split_name(split), lambda: run_split(split), rerun="--rerun" in sys.argv)
        print("SAVED", split_name(split), flush=True)
