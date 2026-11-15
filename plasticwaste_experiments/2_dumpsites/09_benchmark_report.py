"""Benchmark step 5: one table for all methods, from the saved out-of-fold scores of 07 and 08.

Same metrics and the same paired block bootstrap as 07, so linear probes and trained
networks are compared on identical test windows. Reference: ImageNet ResNet-50 at 80 m
(the current model's network on a true 80 m crop). Trained networks use one seed (seed-0
folds), linear probes the mean of three seeds.

Usage: 09_benchmark_report.py [main|sensitivity]
Output: outputs/benchmark/benchmark_table_<run>.csv
"""
import importlib.util, sys
from pathlib import Path

import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ev", HERE / "07_linear_probe_benchmark.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
WORK = ev.WORK
TRAINED = ("finetune", "aw_protocol")
NAMES = {"colour": "Colour/texture (baseline)", "imagenet": "ImageNet ResNet-50 (current network)",
         "aw_italy": "AerialWaste ResNet-50+FPN, frozen", "aw_guat": "Guatemala checkpoint, frozen",
         "dinov3_sat": "DINOv3 ViT-L, satellite, frozen", "dinov3_web": "DINOv3 ViT-L, web, frozen",
         "finetune": "ResNet-50 fully fine-tuned", "aw_protocol": "AerialWaste, Guatemala protocol"}


def main(run="main"):
    pool = pd.read_parquet(WORK / "pool.parquet")
    lab = pool.label.replace(-1, 1) if run == "sensitivity" else pool.label
    keep = (lab >= 0).to_numpy()
    df = pool[keep].reset_index(drop=True).assign(y=lab[keep].to_numpy())
    scores = {}
    for f in sorted(WORK.glob(f"oof_{run}_*_*m.npy")):
        name, w = f.stem[len(f"oof_{run}_"):].rsplit("_", 1)
        s = np.load(f)
        if len(s) == len(pool):
            s = s[keep]
        if np.isnan(s).any():
            continue
        scores[(name, int(w[:-1]))] = s
    ref = scores[("imagenet", 80)]
    rows = []
    for (name, w), s in sorted(scores.items(), key=lambda k: (k[0][0] in TRAINED, k[0][0], k[0][1])):
        m = ev.metrics(df, s)
        b = ev.boot_town(df, ref, s) if (name, w) != ("imagenet", 80) else {"d_ap": 0.0, "ci95": [0.0, 0.0]}
        rows.append({"method": NAMES.get(name, name), "key": name, "window_m": w,
                     "curated_ap": m["curated"]["ap"], "curated_auc": m["curated"]["auc"],
                     "curated_f1_at_guat_prev": m["curated"]["f1_at33"],
                     "blind_boke_auc": m["blind_boke"]["auc"], "blind_conakry_auc": m["blind_conakry"]["auc"],
                     "town_ap": m["town"]["ap"], "town_auc": m["town"]["auc"], "town_top5": m["town"]["top5"],
                     "town_d_ap_vs_ref": b["d_ap"], "ci_lo": b["ci95"][0], "ci_hi": b["ci95"][1]})
    t = pd.DataFrame(rows)
    t["beats_ref"] = t.ci_lo > 0
    t.to_csv(WORK / f"benchmark_table_{run}.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.drop(columns=["key"]).to_string(index=False))
    print("town base rate:", ev.metrics(df, ref)["town"]["prev"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
