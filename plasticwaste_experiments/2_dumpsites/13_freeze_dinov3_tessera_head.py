"""
The adopted waste-site model, frozen for the platform: DINOv3 web at 150 m with TESSERA's
150 m mean and spread, one logistic step (section 8 of the notebook).

Fitted on every labelled point of the pool (2,348; unsure labels left out), as the
recommendation says ("refit it on all labels"). Same pipeline as the benchmark:
StandardScaler + LogisticRegression(class_weight="balanced"), C chosen by average precision
over 5 size-balanced group folds (07's folds), seeds 0, 1, 2. TESSERA year: 2025, the latest
released; points without an embedding take the column means, which are stored with the head
so the platform fills the same way.

Tiers. The platform shows three tiers. Their thresholds are set on the out-of-fold scores of
12_geotessera_benchmark.py (2025) so each tier keeps the recall of the initial head's tier on
the labelled points (high 0.30, medium 0.51, all 0.64), and their precision is reported.

Must run with scikit-learn 1.4.2, the platform's version:
    "E:/SDG AI Lab/Guinea/national_scan_2026/venv_gpu/Scripts/python" 2_dumpsites/13_freeze_dinov3_tessera_head.py

Output: outputs/geotessera/probe_dinov3_tessera.joblib and probe_dinov3_tessera_meta.json,
to be copied to the platform's dumpsite_tool/models/.
"""
import torch  # noqa: F401  -- before scikit-learn on Windows (DLL order)
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bench", HERE / "07_linear_probe_benchmark.py")
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
OUT = HERE / "outputs" / "geotessera"
YEAR = 2025
NAME = "probe_dinov3_tessera"
TIER_RECALL = {"high": 0.30, "medium": 0.51, "all": 0.64}


def main():
    assert sklearn.__version__ == "1.4.2", sklearn.__version__
    pool = pd.read_parquet(B.WORK / "pool.parquet")
    keep = (pool.label >= 0).to_numpy()
    df = pool[keep].reset_index(drop=True).assign(y=pool.label[keep].to_numpy())
    y, g = df.y.to_numpy(), df.group.to_numpy()
    t = np.load(OUT / f"tessera_{YEAR}.npz")
    assert (t["uid"] == pool.uid.to_numpy().astype(str)).all()
    ctx = t["ctx150"][keep].astype(np.float64)
    ctx = np.where(np.isfinite(ctx), ctx, np.nan)
    fill = np.nanmean(ctx, 0)
    ctx = np.where(np.isnan(ctx), fill, ctx)
    X = np.hstack([np.load(B.WORK / "feat_dinov3_web_150m.npy")[keep].astype(np.float64), ctx])
    print("X", X.shape, "positives", int(y.sum()), flush=True)

    ap = {}
    for C in B.CS:
        s = []
        for seed in B.SEEDS:
            oof = np.zeros(len(y))
            for tr, te in B.group_folds(y, g, 5, seed):
                oof[te] = B.fit(X[tr], y[tr], C).predict_proba(X[te])[:, 1]
            s.append(AP(y, oof))
        ap[C] = float(np.mean(s))
        print(f"C={C:g}: AP {ap[C]:.3f}", flush=True)
    C = max(ap, key=ap.get)
    head = B.fit(X, y, C)

    oof = np.load(OUT / f"oof_{YEAR}_dinov3_web_150+ctx150.npy")
    tiers, points = {}, []
    for name, r in TIER_RECALL.items():
        order = np.sort(oof[y == 1])[::-1]
        thr = float(order[int(np.ceil(r * len(order))) - 1])
        flag = oof >= thr
        tiers[name] = round(thr, 4)
        points.append({"tier": name, "threshold": round(thr, 4), "recall": round(float(flag[y == 1].mean()), 3),
                       "precision_eval": round(float(y[flag].mean()), 3), "fpr": round(float(flag[y == 0].mean()), 3)})
    # Each tier on the two blind grids (windows labelled one by one, Conakry and Boke): the
    # precision and recall the platform quotes for area checks. One stage, so recall is end to end.
    blind = {}
    for town, key in (("conakry", "blind_conakry"), ("boke", "blind_boke")):
        m = (df["set"] == key).to_numpy()
        yb, sb = y[m], oof[m]
        blind[town] = {"n": int(m.sum()), "base_rate": round(float(yb.mean()), 3), "by_tier": {}}
        for name, thr in tiers.items():
            f = sb >= thr
            blind[town]["by_tier"][name] = {"precision": round(float(yb[f].mean()), 3) if f.any() else None,
                                            "recall": round(float(f[yb == 1].mean()), 3),
                                            "flagged": int(f.sum())}
    res = json.loads((OUT / f"results_{YEAR}.json").read_text())["dinov3_web_150+ctx150"]
    meta = {
        "backbone": "dinov3_vitl16_lvd1689m_cls_meanpatch2048+tessera_ctx150_256",
        "dino_repo": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "dino_input_px": 224, "crop_px": 128, "tile_m": 150.0, "source": "esri", "zoom": 17,
        "tessera_store": "https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate",
        "tessera_year": YEAR, "tessera_grid": {"step_m": 30.0, "half": 2},
        "tessera_fill": [round(float(v), 6) for v in fill],
        "n_dino": 2048, "n_tessera": 256,
        "C": C, "C_ap": ap, "n_train": int(len(y)), "n_pos": int(y.sum()),
        "sklearn": sklearn.__version__,
        "tiers": tiers, "default_tier": "high", "operating_points": points,
        "held_out_metrics": {"protocol": "nested spatial-block CV, 5 folds x 3 seeds (07)", **res},
        "blind_grid": blind,
        "model_version": f"dinov3-web-tessera{YEAR}-150m-esri",
    }
    joblib.dump(head, OUT / f"{NAME}.joblib")
    (OUT / f"{NAME}_meta.json").write_text(json.dumps(meta, indent=2))
    print("C", C, "tiers", tiers)
    for p in points:
        print(p)
    print(json.dumps(blind, indent=1))
    print("in-sample AUC", round(AUC(y, head.predict_proba(X)[:, 1]), 3), "(not a test)")


if __name__ == "__main__":
    main()
