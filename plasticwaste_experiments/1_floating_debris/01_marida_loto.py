"""
Floating-debris benchmark on MARIDA, leave-one-tile-out.

The published evaluation trains on MARIDA's train split and tests on its test
split, where 91 % of patches share a Sentinel-2 tile with training; only two
tiles are unseen. Here every tile holding debris is held out in turn, so every
debris pixel is scored by a model that never saw its location. Every candidate
gets the same pixels, the same scoring and the same scene-level thresholds.

Rules are thresholds per scene, mean + 3 sigma of the scene's water pixels
(Digital Earth Africa). Trained models see only the deployed twelve features.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import MARIDA, ROOT  # noqa: E402

from plasticwaste_tool.marida_tool import baseline as B  # noqa: E402
from plasticwaste_tool.marida_tool import data as D  # noqa: E402
from plasticwaste_tool.marida_tool import domain as DOM  # noqa: E402
from plasticwaste_tool.marida_tool import features as F  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402

OUT = ROOT / "1_floating_debris" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
TREES, LEAF, SEED, P_MIN = 200, 2, 20240119, 0.5


def load():
    cache = OUT / "marida_pixels.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], z["pid"]
    parts = [D.load_pixels(s, root=MARIDA) for s in ("train", "val", "test")]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    pid = np.concatenate([p[2] for p in parts])
    np.savez_compressed(cache, X=X, y=y, pid=pid)
    return X, y, pid


def scene_rule(score, is_water, scored, scene):
    """score > mean + 3 sigma of the scene's water pixels."""
    pred = np.zeros(len(score), bool)
    for s in np.unique(scene):
        m = scene == s
        t = B.de_africa_threshold(score[m], is_water[m])
        if np.isfinite(t):
            pred[m] = scored[m] & (score[m] > t)
    return pred


def scene_otsu(score, scored, scene):
    """Otsu on the scene's scored pixels, as Kikaki et al. (2023) use FDI."""
    from skimage.filters import threshold_otsu
    pred = np.zeros(len(score), bool)
    for s in np.unique(scene):
        m = (scene == s) & scored & np.isfinite(score)
        if m.sum() > 16 and np.ptp(score[m]) > 0:
            pred[m] = score[m] > threshold_otsu(score[m])
    return pred


def main():
    X, y, pid = load()
    pid = np.array([D.normalise_id(p) for p in pid])
    tile = np.array([p.split("_")[2] for p in pid])
    scene = np.array([p.rsplit("_", 1)[0] for p in pid])
    print(f"{len(y):,} labelled pixels, {(y == D.TARGET).sum():,} debris, "
          f"{len(set(scene))} scenes, {len(set(tile))} tiles", flush=True)

    fb = F.build(X)
    cols = list(DOM.PIPELINE_FEATURES)
    Fx = fb[cols].to_numpy("float32")
    is_water = np.isin(y, D.WATER_CLASSES)
    scored = ~np.isin(y, D.NON_WATER)
    t = y == D.TARGET
    fdi, ndvi, fai, pi = (fb[c].to_numpy() for c in ("FDI", "NDVI", "FAI", "PI"))

    rules = {
        "FDI mean+3sd": scene_rule(fdi, is_water, scored, scene),
        "FDI Otsu": scene_otsu(fdi, scored, scene),
        "FAI mean+3sd": scene_rule(fai, is_water, scored, scene),
        "PI mean+3sd": scene_rule(pi, is_water, scored, scene),
    }
    rules["FDI and NDVI mean+3sd (rule deployed)"] = rules["FDI mean+3sd"] & scene_rule(
        ndvi, is_water, scored, scene)

    debris_tiles = sorted(set(tile[t]))
    print("tiles with debris:", debris_tiles, flush=True)
    p_rf = np.full(len(y), np.nan, "float32")
    p_nb = np.full(len(y), np.nan, "float32")
    t0 = time.time()
    for i, tl in enumerate(debris_tiles):
        te = tile == tl
        tr = ~te & scored
        rf = RandomForestClassifier(n_estimators=TREES, min_samples_leaf=LEAF,
                                    class_weight="balanced_subsample", n_jobs=-1,
                                    random_state=SEED)
        rf.fit(Fx[tr], t[tr].astype(int))
        p_rf[te] = rf.predict_proba(Fx[te])[:, 1]
        nb = GaussianNB().fit(np.c_[ndvi[tr], fdi[tr]], t[tr])
        p_nb[te] = nb.predict_proba(np.c_[ndvi[te], fdi[te]])[:, 1]
        print(f"{i + 1}/{len(debris_tiles)} {tl}: {int(t[te].sum())} debris, "
              f"{te.sum():,} px, {time.time() - t0:.0f}s", flush=True)

    held = np.isfinite(p_rf) & scored
    rf_kept = p_rf >= P_MIN
    preds = dict(rules)
    preds["Random forest alone"] = rf_kept
    preds["Naive Bayes on NDVI-FDI (Biermann 2020)"] = p_nb >= P_MIN
    preds["FDI rule, then forest"] = rules["FDI mean+3sd"] & rf_kept
    preds["Deployed: FDI and NDVI rule, then forest"] = (
        rules["FDI and NDVI mean+3sd (rule deployed)"] & rf_kept)

    def row(name, pred, m):
        tp, fp = int((pred & t & m).sum()), int((pred & ~t & m).sum())
        fn = int((~pred & t & m).sum())
        return dict(method=name, tp=tp, fp=fp, fn=fn,
                    precision=tp / (tp + fp) if tp + fp else np.nan,
                    recall=tp / (tp + fn) if tp + fn else np.nan)

    res = pd.DataFrame([row(k, v, held) for k, v in preds.items()])
    res["f1"] = 2 * res.precision * res.recall / (res.precision + res.recall)

    # 90 % intervals, resampling tiles: pixels of one tile are not independent.
    rng = np.random.default_rng(SEED)
    tiles = np.array(debris_tiles)
    idx = {tl: np.flatnonzero(held & (tile == tl)) for tl in tiles}
    ci = {k: [] for k in preds}
    for _ in range(2000):
        draw = rng.choice(tiles, len(tiles))
        ii = np.concatenate([idx[tl] for tl in draw])
        for k, v in preds.items():
            tp, fp, fn = (v[ii] & t[ii]).sum(), (v[ii] & ~t[ii]).sum(), (~v[ii] & t[ii]).sum()
            ci[k].append((tp / (tp + fp) if tp + fp else np.nan,
                          tp / (tp + fn) if tp + fn else np.nan,
                          2 * tp / (2 * tp + fp + fn) if tp else 0.0))
    for j, n in enumerate(("precision", "recall", "f1")):
        res[f"{n}_p05"] = [np.nanquantile([c[j] for c in ci[k]], 0.05) for k in res.method]
        res[f"{n}_p95"] = [np.nanquantile([c[j] for c in ci[k]], 0.95) for k in res.method]

    rank = {"forest_ap": average_precision_score(t[held], p_rf[held]),
            "forest_auc": roc_auc_score(t[held], p_rf[held]),
            "fdi_ap": average_precision_score(t[held], np.nan_to_num(fdi[held])),
            "fdi_auc": roc_auc_score(t[held], np.nan_to_num(fdi[held])),
            "n_held_px": int(held.sum()), "n_debris": int(t[held].sum()),
            "n_tiles": len(debris_tiles), "n_scenes_all": len(set(scene)),
            "base_rate": float(t[held].mean())}
    per = []
    for k, v in preds.items():
        for tl in tiles:
            r = row(k, v, held & (tile == tl))
            r["tile"] = tl
            per.append(r)
    pd.DataFrame(per).to_csv(OUT / "01_loto_by_tile.csv", index=False)
    res.sort_values("f1", ascending=False).to_csv(OUT / "01_loto_scores.csv", index=False)
    (OUT / "01_loto_rank.json").write_text(json.dumps(rank, indent=1), encoding="utf-8")
    np.savez_compressed(OUT / "01_loto_probs.npz", p_rf=p_rf, y=y, tile=tile, scene=scene)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(res.sort_values("f1", ascending=False).round(3).to_string())
    print(rank)


if __name__ == "__main__":
    main()
