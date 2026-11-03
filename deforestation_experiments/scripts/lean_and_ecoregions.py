"""Notebooks 05 and 06: the predictors in groups, and one model per ecoregion under Verra's tests.

Both start from the improved model trained on the shares (notebook 04) and score full risk maps:
AUC over the 30 m forest pixels (three seeds) and Verra's fit and prediction tests (first seed,
verra.vt0007_tests).

  lean        nested sets of predictors, grown by theme in a fixed order (not chosen by score):
              loss history -> + terrain -> + access -> + climate -> the improved model
  ecoregion   one model per forest ecoregion, rescaled with its calibration loss rate
              (decisions100), against the national model: Verra's tests nationally, the loss
              allocated to each ecoregion against the loss it had, and MedAE over the blocks
              mostly inside each ecoregion

Training cells are drawn as in decisions100 (once per seed, in proportion to the forest lost and
kept). Predictors are built for every cell with forest at the window's start, one window at a time.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_variants as D                                     # noqa: E402
import model as MM                                              # noqa: E402
import verra as V                                              # noqa: E402

M = MM.M
LOSS_HISTORY = ["Recent loss rate around the cell (5 km)", "Distance to recent loss", "Distance to prior loss"]
TERRAIN = ["Elevation", "Slope"]
ACCESS = ["Distance to roads", "Distance to built-up area", "Travel time to cities (2015)", "Population density"]
CLIMATE = ["Land-surface temperature", "Soil moisture", "Precipitation (1991–2020 normal)"]
LEAN = {"Loss history (3)": LOSS_HISTORY,
        "+ terrain (5)": LOSS_HISTORY + TERRAIN,
        "+ access (9)": LOSS_HISTORY + TERRAIN + ACCESS,
        "+ climate (12)": LOSS_HISTORY + TERRAIN + ACCESS + CLIMATE,
        "Improved model (16)": MM.FEATURES}


def _score(model, values, names, rows):
    """Risk over the cells at positions `rows` of `values` (a million at a time)."""
    risk = np.full(rows.size, np.nan, dtype="float32")
    for i in range(0, rows.size, 1_000_000):
        X = np.column_stack([values[n][rows[i:i + 1_000_000]] for n in names])
        ok = np.isfinite(X).all(axis=1)
        part = risk[i:i + 1_000_000]
        part[ok] = model.predict_proba(X[ok])[:, 1]
    return risk


def _rescale(p, rate):
    odds = np.clip(p, 1e-4, 1 - 1e-4)
    odds = odds / (1 - odds) * rate / (1 - rate)
    return odds / (1 + odds)


def _blocks_by_ecoregion(eco):
    """Ecoregion holding most of the forest cells of each ~100,000 ha block."""
    cells, blocks = V.coarse_cells()
    keep = cells >= 0
    pairs = cells[keep].astype("int64") * 100 + eco[keep]
    ids, counts = np.unique(pairs, return_counts=True)
    order = np.lexsort((-counts, ids // 100))
    first = np.unique(ids[order] // 100, return_index=True)[1]
    winner = dict(zip(ids[order][first] // 100, ids[order][first] % 100))
    return np.array([winner.get(b, 0) for b in blocks])


def _medae_by_ecoregion(predicted, observed, block_eco, codes):
    cells, blocks = V.coarse_cells()
    keep = cells >= 0
    n = blocks.max() + 1
    p = np.bincount(cells[keep], weights=predicted[keep], minlength=n)[blocks]
    o = np.bincount(cells[keep], weights=observed[keep], minlength=n)[blocks]
    return {c: (float(np.median(np.abs(p - o)[block_eco == c])) if (block_eco == c).any() else None,
                int((block_eco == c).sum())) for c in codes}


def run_split(split, seeds=(42, 1, 2), log=print):
    t1, t2, t3 = split
    start = time.perf_counter()
    n = MM.hansen()["idx"].size
    area = MM.hansen()["area"]
    eco = D.ecoregions()
    names_json = {int(k): v for k, v in json.loads((M.DD.DATA / "ecoregions.json").read_text()).items()}
    f1, lw = MM.forest_pct(t1), MM.loss_pct(t1, t2)
    codes = [int(c) for c in np.unique(eco[f1 > 0]) if (lw[eco == c] > 0).sum() >= 1000]
    rate = {c: float((lw * area)[eco == c].sum() / (f1 * area)[eco == c].sum()) for c in codes}
    log(f"{M.split_label(split)}: own models for ecoregions {[names_json[c] for c in codes]}")
    draws = {(None, s): D._draw(t1, t2, s) for s in seeds}
    draws.update({(c, s): D._draw(t1, t2, s, eco == c) for c in codes for s in seeds})

    # Calibration window: fit every model, score the fit maps (first seed)
    rows1 = np.flatnonzero(f1 > 0)
    cal = D.build(t1, MM.FEATURES, rows1)
    log(f"{M.split_label(split)}: calibration predictors {time.perf_counter() - start:.0f} s")
    models = {}
    for key, s in draws:
        cells, y = draws[key, s]
        sets = LEAN.items() if key is None else [("Improved model (16)", MM.FEATURES)]
        for name, names in sets:
            X = np.column_stack([cal[m][np.searchsorted(rows1, cells)] for m in names])
            ok = np.isfinite(X).all(axis=1)
            models[name, key, s] = D._fit(X[ok], y[ok], s)
    fit_maps = {}
    for name, names in LEAN.items():
        fit_maps[name] = np.full(n, np.nan, dtype="float32")
        fit_maps[name][rows1] = _score(models[name, None, seeds[0]], cal, names, np.arange(rows1.size))
    eco_fit = np.full(n, np.nan, dtype="float32")
    for c in codes:
        inside = np.flatnonzero(eco[rows1] == c)
        eco_fit[rows1[inside]] = _rescale(_score(models["Improved model (16)", c, seeds[0]], cal, MM.FEATURES, inside), rate[c])
    del cal
    log(f"{M.split_label(split)}: models fitted {time.perf_counter() - start:.0f} s")

    # Test window: score every model, AUC per seed, Verra's tests on the first seed
    f2 = MM.forest_pct(t2)
    rows2 = np.flatnonzero(f2 > 0)
    test = D.build(t2, MM.FEATURES, rows2)
    log(f"{M.split_label(split)}: test predictors {time.perf_counter() - start:.0f} s")
    out = {"split": M.split_label(split), "lean": {}, "ecoregion": {}}
    for name, names in LEAN.items():
        r = out["lean"][name] = {"aucs": [], "n": len(names)}
        for s in seeds:
            risk = np.full(n, np.nan, dtype="float32")
            risk[rows2] = _score(models[name, None, s], test, names, np.arange(rows2.size))
            r["aucs"].append(V.pixel_auc(risk, t2, t3))
            if s == seeds[0]:
                r.update(V.vt0007_tests(split, fit_maps[name], risk))
                if name == "Improved model (16)":
                    national_test = risk
        log(f"{out['split']}: {name} {np.mean(r['aucs']):.4f} {time.perf_counter() - start:.0f} s")

    eco_aucs = []
    for s in seeds:
        risk = np.full(n, np.nan, dtype="float32")
        for c in codes:
            inside = np.flatnonzero(eco[rows2] == c)
            risk[rows2[inside]] = _rescale(_score(models["Improved model (16)", c, s], test, MM.FEATURES, inside), rate[c])
        eco_aucs.append(V.pixel_auc(risk, t2, t3))
        if s == seeds[0]:
            eco_test = risk
    del test

    # Verra's tests, nationally and by ecoregion, for the national and the per-ecoregion model
    loss_conf = MM.loss_pct(t2, t3) / 100 * area
    loss_cal = lw / 100 * area
    block_eco = _blocks_by_ecoregion(eco)
    for label, fit, pred, aucs in (("national", fit_maps["Improved model (16)"], national_test, out["lean"]["Improved model (16)"]["aucs"]),
                                   ("per_ecoregion", eco_fit, eco_test, eco_aucs)):
        tests = V.vt0007_tests(split, fit, pred, keep_maps=True)
        r = out["ecoregion"][label] = {"aucs": aucs, "fit_medae_ha": tests["fit_medae_ha"],
                                       "prediction_medae_ha": tests["prediction_medae_ha"]}
        r["loss_by_ecoregion"] = {names_json[c]: (float(tests["predicted"][eco == c].sum()), float(loss_conf[eco == c].sum()))
                                  for c in codes}
        fit_blocks = _medae_by_ecoregion(tests["fitted"], loss_cal, block_eco, codes)
        pred_blocks = _medae_by_ecoregion(tests["predicted"], loss_conf, block_eco, codes)
        r["medae_by_ecoregion"] = {names_json[c]: {"blocks": pred_blocks[c][1], "fit": fit_blocks[c][0], "prediction": pred_blocks[c][0]}
                                   for c in codes}
    out["ecoregion"]["calibration_loss_rate"] = {names_json[c]: rate[c] for c in codes}
    out["seconds"] = time.perf_counter() - start
    out["peak_gb"] = MM._peak_gb()
    return out


def split_name(split):
    return "lean_and_ecoregions_{}_{}_{}".format(*split)


if __name__ == "__main__":
    for split in MM.SPLITS:
        MM.cached(split_name(split), lambda: run_split(split), rerun="--rerun" in sys.argv)
        print("SAVED", split_name(split), flush=True)
