"""
GeoTessera test, step 2: score TESSERA features with the benchmark's own protocol.

Same labels, folds, seeds, inner choice of C and test sets as 07_linear_probe_benchmark.py
(main run: unsure labels left out), so the numbers sit directly beside Table 4.

Feature sets:
  tessera_pixel          the 10 m pixel (128)
  tessera_ctx150         mean and SD over about 150 m (256)
  dinov3_web_150         the adopted model, rerun here as the reference
  dinov3_web_150+ctx150  both, to see whether TESSERA adds to DINOv3

Town AP gain is bootstrapped against the adopted model (dinov3_web, 150 m), not the
initial one, because the question is whether TESSERA improves on what will be deployed.

    "E:/SDG AI Lab/Guinea/national_scan_2026/venv_gpu/Scripts/python" 2_dumpsites/12_geotessera_benchmark.py 2024

Output: outputs/geotessera/results_<year>.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bench", HERE / "07_linear_probe_benchmark.py")
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
OUT = HERE / "outputs" / "geotessera"


def main(year):
    pool = pd.read_parquet(B.WORK / "pool.parquet")
    keep = (pool.label >= 0).to_numpy()
    df = pool[keep].reset_index(drop=True).assign(y=pool.label[keep].to_numpy())
    g, y = df.group.to_numpy(), df.y.to_numpy()

    t = np.load(OUT / f"tessera_{year}.npz")
    assert (t["uid"] == pool.uid.to_numpy().astype(str)).all(), "point order differs from pool.parquet"
    ok = (t["status"] == "valid")[keep]
    print(f"{year}: {ok.sum()} of {keep.sum()} labeled points have a valid embedding", flush=True)
    dino = np.load(B.WORK / "feat_dinov3_web_150m.npy")[keep]
    sets = {"tessera_pixel": t["pixel"][keep], "tessera_ctx150": t["ctx150"][keep],
            "dinov3_web_150": dino, "dinov3_web_150+ctx150": np.hstack([dino, t["ctx150"][keep]])}

    # points without an embedding (water, no data) get the column means, so every method
    # is scored on the same rows
    for k, X in sets.items():
        X = np.where(np.isfinite(X), X, np.nan)
        sets[k] = np.where(np.isnan(X), np.nanmean(X, 0), X)

    results, scores = {"year": year, "valid_points": int(ok.sum()), "labeled_points": int(keep.sum())}, {}
    for name, X in sets.items():
        runs = B.nested_oof_all(X, y, g)
        s = np.mean([r[0] for r in runs], 0); scores[name] = s
        results[name] = {**B.metrics(df, s), "town_ap_per_seed": [B.metrics(df, r[0])["town"]["ap"] for r in runs]}
        r = results[name]
        print(f"{name:24s} curated AP {r['curated']['ap']:.3f} AUC {r['curated']['auc']:.3f} | "
              f"Boke AUC {r['blind_boke']['auc']:.3f} Conakry AUC {r['blind_conakry']['auc']:.3f} | "
              f"town AP {r['town']['ap']:.3f} AUC {r['town']['auc']:.3f} top5 {r['town']['top5']:.3f}", flush=True)
        np.save(OUT / f"oof_{year}_{name}.npy", s)

    # Document-style rows, as 10_document_table.py: 90 % paired block bootstrap of the town-AP
    # gain, the published mark (town AUC >= 0.80 and gain interval above zero), and the
    # four-test count, both against the initial model and against the adopted DINOv3 model.
    initial = np.mean([np.load(f) for f in [B.WORK / "oof_main_imagenet_80m.npy"]], 0)
    refs = {"initial": initial, "dinov3_web_150": scores["dinov3_web_150"]}
    m = (df.set == "round1").to_numpy(); d = df[m].reset_index(drop=True)
    yt, wt, gt_ = d.y.to_numpy(), B.town_weights(d), d.group.to_numpy()
    ug = np.unique(gt_); idx = {u: np.where(gt_ == u)[0] for u in ug}
    for name, s in scores.items():
        k = B.metrics(df, s)
        for rname, ref in refs.items():
            if rname == name:
                continue
            r0 = B.metrics(df, ref)
            rng = np.random.default_rng(0); gains, aucs = [], []
            for _ in range(2000):
                j = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
                if yt[j].min() == yt[j].max():
                    continue
                gains.append(B.AP(yt[j], s[m][j], sample_weight=wt[j]) - B.AP(yt[j], ref[m][j], sample_weight=wt[j]))
                aucs.append(B.AUC(yt[j], s[m][j], sample_weight=wt[j]))
            lo, hi = np.percentile(gains, [5, 95])
            tests = [(k["curated"]["ap"], r0["curated"]["ap"]), (k["blind_boke"]["auc"], r0["blind_boke"]["auc"]),
                     (k["blind_conakry"]["auc"], r0["blind_conakry"]["auc"]), (k["town"]["ap"], r0["town"]["ap"])]
            row = {"gain_ap": round(k["town"]["ap"] - r0["town"]["ap"], 3), "gain_90": [round(lo, 3), round(hi, 3)],
                   "town_auc_90": [round(x, 2) for x in np.percentile(aucs, [5, 95])],
                   "beats_on": int(sum(a > b for a, b in tests)),
                   "passes_mark": bool(k["town"]["auc"] >= 0.80 and lo > 0)}
            results[name][f"vs_{rname}"] = row
            print(f"{name:24s} vs {rname:15s} {row}", flush=True)
    json.dump(results, open(OUT / f"results_{year}.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024)
