"""Tests of the fixes for the problems found in the platform's calculation.

predictor_timing   The deployed model (its predictors, the half-cell rule) with its yearly
                   predictors read at their latest year, as the platform's hold-out does, against
                   read at each window's start: does the latest year inflate the score?
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as MM                                              # noqa: E402

TIMINGS = {"latest": "Latest year (platform)", "start": "Window start"}


def predictor_timing(split, seeds=(42, 1, 2), log=print):
    """AUC of the deployed model on the test window with each predictor timing: over the 30 m
    pixels of all forest, of the half-forest cells it scores, and on its own terms (half cells)."""
    t1, t2, t3 = split
    start = time.perf_counter()
    area = MM.hansen()["area"]
    f2, lt = MM.forest_pct(t2), MM.loss_pct(t2, t3)
    rows = np.flatnonzero(f2 > 0)
    w_pos = lt[rows].astype("float32") / 100 * area[rows]
    w_neg = np.clip(f2[rows] - lt[rows], 0, None).astype("float32") / 100 * area[rows]
    half = f2[rows] >= 50
    own = MM.lost("half", t2, t3)[rows]
    out = {"split": MM.M.split_label(split), "timings": {}}
    for timing in TIMINGS:
        config = {**MM.DEPLOYED, "predictor_year": timing}
        names = MM.features(config)
        cal = MM.stack(t1, config, families=("half",))["half"]
        models = [MM.DR.fit_risk_model(*MM.sample("half", t1, t2, cal, seed, names), seed=seed)[0] for seed in seeds]
        del cal
        test = MM.stack(t2, config, families=("half",))["half"]
        r = {"aucs_all": [], "aucs_scored": [], "aucs_own": []}
        for model in models:
            risk = MM.score(model, test, rows, names=names)
            r["aucs_all"].append(MM.weighted_auc(risk, w_pos, w_neg))
            r["aucs_scored"].append(MM.weighted_auc(risk[half], w_pos[half], w_neg[half]))
            r["aucs_own"].append(MM.weighted_auc(risk[half], own[half].astype("float32"), (~own[half]).astype("float32")))
        del test
        out["timings"][timing] = r
        log(f"{out['split']}: {timing} done {time.perf_counter() - start:.0f} s", flush=True)
    out["peak_gb"] = MM._peak_gb()
    return out


def split_name(split, prefix="holdout_timing"):
    return "{}_{}_{}_{}".format(prefix, *split)


if __name__ == "__main__":
    for s in MM.SPLITS:
        MM.cached(split_name(s), lambda s=s: predictor_timing(s))
