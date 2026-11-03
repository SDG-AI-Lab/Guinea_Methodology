"""Notebook 10: how Verra's calibration map is built, and whether the answer depends on the score.

Verra's tests turn each model's risk map into hectares: 30 classes, each with the loss rate it had in
the calibration window (its "calibration map"), then the loss is allocated by those rates. Two timings:

  A  same timing     every model's calibration map uses what was known at the start of the
                     calibration window (VT0007 v1.0; UDef-ARP: "covariates appropriate to T1")
  B  as designed     riskmapjnr's calibration map is the local rate of the calibration window's own
                     loss (Verra's 2021 draft JNR tool)

For each model, each window and three block sizes, this keeps the fitted, predicted and observed loss
(ha) of every block wholly in Guinea, so that any score can be computed: Verra's MedAE (v1.0), the
2021 draft's forest-weighted RMSE, mean absolute error, R^2, at ~25,000, ~100,000 and ~400,000 ha.
The maps are the first-seed maps of the improved model and riskmapjnr's (improve.save_maps), so the
100,000 ha MedAE equals notebooks 08-09.

Models:
  benchmark                 VT0007's distance to the forest edge
  improved_A                the improved model (notebook 07), predictors at t1 / t2
  rmj_A                     riskmapjnr, calibration map from the loss of the three years before t1
  rank_A                    rank average of the two
  rmj_B                     riskmapjnr as designed
  rmj_split_B               riskmapjnr's classes as designed, each pair split in two by our risk (notebook 09)
  rank_B                    rank average with riskmapjnr's calibration map as designed
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import improve as I                                              # noqa: E402
import model as MM                                               # noqa: E402
import tool_comparison as T                                      # noqa: E402
import verra as V                                                # noqa: E402

G, M = MM.G, MM.M
SIZES_HA = {"25,000 ha": 25_000, "100,000 ha": 100_000, "400,000 ha": 400_000}
LABELS = {"benchmark": "VT0007 benchmark", "improved_A": "Improved model (A)", "rmj_A": "riskmapjnr, same timing (A)",
          "rank_A": "Rank average (A)", "rmj_B": "riskmapjnr as designed (B)",
          "rmj_split_B": "riskmapjnr's classes split by our risk (B)", "rank_B": "Rank average, riskmapjnr as designed (B)"}


def _blocks(size_ha):
    """Block of each forest cell (-1 outside) and the ids of the blocks wholly in Guinea, as verra.coarse_cells."""
    row, col = G.row_col()
    side = int(round(np.sqrt(size_ha / G.cell_area_ha().mean())))
    ids = (row // side) * (G.grid()["width"] // side + 1) + col // side
    full = np.bincount(ids) == side * side
    return np.where(full[ids], ids, -1)[MM.hansen()["idx"]], np.flatnonzero(full)


def _sum(values, cells, blocks):
    keep = cells >= 0
    return np.bincount(cells[keep], weights=values[keep], minlength=blocks.max() + 1)[blocks]


def run_split(split, seed=42, log=print):
    t1, t2, t3 = split
    area = MM.hansen()["area"]
    f1, f2 = MM.forest_pct(t1), MM.forest_pct(t2)
    observed = {"fit": MM.loss_pct(t1, t2) / 100 * area, "prediction": MM.loss_pct(t2, t3) / 100 * area}
    forest = {"fit": f1 / 100 * area, "prediction": f2 / 100 * area}
    fit, test = I.load_map("improved_fit", split, seed), I.load_map("improved_test", split, seed)
    rfit_b, rtest = I.load_map("rmj_fit", split, seed), I.load_map("rmj_test", split, seed)
    classes_b = (I.load_map("rmj_classes_fit", split, seed), I.load_map("rmj_classes_test", split, seed))
    classes_a, (rfit_a, _), _ = T.riskmapjnr(split, log=lambda *a: None, same_timing=True)
    rank = lambda a, b, f: (T._rank(a, f) + T._rank(b, f)) / 2
    models = {"benchmark": {},
              "improved_A": {"risk_fit": fit, "risk_test": test},
              "rmj_A": {"classes": classes_a},
              "rank_A": {"risk_fit": rank(fit, rfit_a, f1 > 0), "risk_test": rank(test, rtest, f2 > 0)},
              "rmj_B": {"classes": classes_b},
              "rmj_split_B": {"classes": (I_split(classes_b[0], fit, forest["fit"]), I_split(classes_b[1], test, forest["prediction"]))},
              "rank_B": {"risk_fit": rank(fit, rfit_b, f1 > 0), "risk_test": rank(test, rtest, f2 > 0)}}
    out = {"split": M.split_label(split), "blocks": {}, "models": {}}
    grids = {label: _blocks(size) for label, size in SIZES_HA.items()}
    for label, (cells, blocks) in grids.items():
        out["blocks"][label] = {f"observed_{t}": _sum(observed[t], cells, blocks).round(2).tolist() for t in observed}
        out["blocks"][label].update({f"forest_{t}": _sum(forest[t], cells, blocks).round(2).tolist() for t in forest})
    for key, kwargs in models.items():
        r = V.vt0007_tests(split, keep_maps=True, **kwargs)
        out["models"][key] = {"fit_medae_ha": r["fit_medae_ha"], "prediction_medae_ha": r["prediction_medae_ha"],
                              **{label: {"fit": _sum(r["fitted"], *grids[label]).round(2).tolist(),
                                         "prediction": _sum(r["predicted"], *grids[label]).round(2).tolist()}
                                 for label in SIZES_HA}}
        log(f"{out['split']}: {key} prediction MedAE {r['prediction_medae_ha']:.0f} ha")
    return out


def I_split(rmj_classes, risk, forest_ha):
    """riskmapjnr's classes in pairs, each pair split in two by our risk (improve.adjustments)."""
    bucket = np.where(rmj_classes > 0, (rmj_classes.astype(int) - 1) // 2, -1)
    c = np.zeros(risk.shape, dtype="uint8")
    for b in np.unique(bucket[bucket >= 0]):
        inside = (bucket == b) & np.isfinite(risk) & (forest_ha > 0)
        sub = I.equal_area_classes(np.where(inside, risk, np.nan), np.where(inside, forest_ha, 0), n=2)
        c[inside] = b * 2 + sub[inside]
    missing = (bucket >= 0) & (c == 0) & (forest_ha > 0)
    c[missing] = bucket[missing] * 2 + 1
    return c


# --------------------------------------------------------------------------
# Scores of one test from the block sums

def scores(predicted, observed, forest):
    """Verra's MedAE (v1.0); the 2021 draft's RMSE weighted by forest (riskmapjnr's wRMSE); the mean
    absolute error; R^2 (squared correlation, as deforisk reports); the misplaced share."""
    p, o, w = map(np.asarray, (predicted, observed, forest))
    e = p - o
    return {"MedAE (ha)": float(np.median(np.abs(e))), "Mean absolute error (ha)": float(np.mean(np.abs(e))),
            "wRMSE (ha)": float(np.sqrt(np.sum(w * e ** 2) / np.sum(w))), "R²": float(np.corrcoef(p, o)[0, 1] ** 2),
            "Misplaced (% of the loss)": float(50 * np.sum(np.abs(e)) / np.sum(o))}


def split_name(split):
    return "calibration_timing_{}_{}_{}".format(*split)


if __name__ == "__main__":
    for split in MM.SPLITS:
        MM.cached(split_name(split), lambda: run_split(split), rerun="--rerun" in sys.argv)
        print("SAVED", split_name(split), flush=True)
