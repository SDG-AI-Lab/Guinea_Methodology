"""
Dumpsite benchmark on the two exhaustively labelled blind grids, free Esri imagery.

Every representation gets the same logistic probe: standardised, balanced, C chosen
by inner cross-validation on the training set (blocks of ground, never the test
set). Three training sets:
  points         the 651 point-centred crops, as deployed
  other town     the other town's blind grid: same task, local negatives
  both           points + other town
The frozen deployed head is scored too, to check the reproduction. Intervals
resample 400 m blocks of ground.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, TOOL  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = ROOT / "2_dumpsites" / "outputs"
REPS = {"colour": "Colour and texture summary (control)",
        "resnet50": "ImageNet ResNet-50 (deployed)",
        "aerialwaste": "AerialWaste ResNet-50 FPN (waste-trained)",
        "convnext": "ConvNeXt-Base (ImageNet)",
        "vit": "ViT-B/16 SWAG"}
CS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SEED = 3


def load(key, rep):
    return np.load(OUT / f"feat_{key}_{rep}.npy")


def meta(key):
    d = pd.read_csv(OUT / f"set_{key}.csv")
    d = d.rename(columns={"long": "lon"}) if "long" in d and "lon" not in d else d
    if key == "points":
        oof = np.load(TOOL / "manual_qa" / "_oof_pt80_esri.npz", allow_pickle=True)
        d["cell"] = d.sid.astype(str).map(dict(zip(oof["sid"].astype(str), oof["groups"].astype(str))))
    if "cell" not in d:
        d["cell"] = (np.floor(d.lat / 0.0036).astype(int).astype(str) + "_"
                     + np.floor(d.lon / 0.0036).astype(int).astype(str))
    return d


def pipe(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))


def choose_c(X, y, groups):
    best, score = CS[0], -1
    k = min(5, len(set(groups)))
    for c in CS:
        s = []
        for tr, te in GroupKFold(k).split(X, y, groups):
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            s.append(roc_auc_score(y[te], pipe(c).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
        if s and np.mean(s) > score:
            best, score = c, float(np.mean(s))
    return best


def ci(y, s, groups, n=2000):
    rng = np.random.default_rng(SEED)
    blocks = np.unique(groups)
    idx = {b: np.flatnonzero(groups == b) for b in blocks}
    a = []
    for _ in range(n):
        ii = np.concatenate([idx[b] for b in rng.choice(blocks, len(blocks))])
        if 0 < y[ii].sum() < len(ii):
            a.append(roc_auc_score(y[ii], s[ii]))
    return float(np.quantile(a, .05)), float(np.quantile(a, .95))


def main():
    towns = ("conakry", "boke")
    M = {k: meta(k) for k in ("points", *towns)}
    head, _ = (joblib.load(TOOL / "dumpsite_tool" / "models" / "probe_esri.joblib"), None)
    rows, scores = [], {}
    for t in towns:
        s = head.predict_proba(load(t, "resnet50"))[:, 1]
        y = M[t].y.to_numpy()
        scores[(t, "deployed head")] = s
        rows.append({"test": t, "train": "frozen deployed head", "rep": "ImageNet ResNet-50 (deployed)",
                     "n_test": len(y), "auc": roc_auc_score(y, s), "ap": average_precision_score(y, s),
                     "auc_p05": ci(y, s, M[t].cell.to_numpy())[0], "auc_p95": ci(y, s, M[t].cell.to_numpy())[1],
                     "base": y.mean(), "C": None})
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)

    for rep, name in REPS.items():
        P = load("points", rep)
        Xs = {t: load(t, rep) for t in towns}
        for t in towns:
            o = towns[1 - towns.index(t)]
            sets = {"points": (P, M["points"].y.to_numpy(), M["points"].cell.to_numpy()),
                    "other town": (Xs[o], M[o].y.to_numpy(), M[o].cell.to_numpy()),
                    "both": (np.r_[P, Xs[o]], np.r_[M["points"].y.to_numpy(), M[o].y.to_numpy()],
                             np.r_[M["points"].cell.to_numpy(), M[o].cell.to_numpy()])}
            for tr, (X, y, g) in sets.items():
                c = choose_c(X, y, g)
                s = pipe(c).fit(X, y).predict_proba(Xs[t])[:, 1]
                yt = M[t].y.to_numpy()
                lo, hi = ci(yt, s, M[t].cell.to_numpy())
                rows.append({"test": t, "train": tr, "rep": name, "n_test": len(yt),
                             "auc": roc_auc_score(yt, s), "ap": average_precision_score(yt, s),
                             "auc_p05": lo, "auc_p95": hi, "base": yt.mean(), "C": c})
                scores[(t, f"{rep}|{tr}")] = s
                print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "02_benchmark.csv", index=False)
    r["mean_key"] = r.rep + " | " + r.train
    wide = r.pivot_table(index="mean_key", columns="test", values=["auc", "ap"])
    wide["mean_auc"] = wide["auc"].mean(axis=1)
    wide = wide.sort_values("mean_auc", ascending=False)
    wide.to_csv(OUT / "02_benchmark_wide.csv")
    pd.set_option("display.width", 200, "display.max_rows", 40, "display.max_colwidth", 70)
    print(wide.round(3).to_string())
    np.savez_compressed(OUT / "02_scores.npz", **{f"{a}__{b}": v for (a, b), v in scores.items()})


if __name__ == "__main__":
    main()
