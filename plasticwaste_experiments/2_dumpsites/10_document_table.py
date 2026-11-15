"""Benchmark step 6: the numbers for the methodology document's waste-site table.

The comparison follows the land-use and deforestation modules. A method qualifies if it beats
the initial method on all four independent tests: curated AP, Boke AUC, Conakry AUC and town AP.
Among the qualifiers, the one with the highest town AP (the realistic test) is adopted.
Each method is also given the document's published mark: town AUC of at least 0.80
(Hosmer et al. 2013), with a 90 % paired bootstrap interval of the town-AP gain above zero
(Efron & Tibshirani 1993; Schenker & Gentleman 2001).

The fine-tuned ResNet-50 is averaged over its three seeds (08 with seed 0, 1, 2), as 07
averages the linear probes, so every row rests on the same number of splits.

Output: outputs/benchmark/document_table.json
"""
import importlib.util, json
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ev", HERE / "07_linear_probe_benchmark.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
W = ev.WORK
ROWS = [("imagenet", 80), ("colour", 80), ("aw_italy", 150), ("aw_guat", 80), ("aw_protocol", 150),
        ("finetune", 80), ("dinov3_sat", 80), ("dinov3_web", 80), ("dinov3_web", 150)]


def load(name, w, keep, n_pool):
    fs = [W / f"oof_main_{name}_{w}m.npy"] + sorted(W.glob(f"oof_main_{name}_{w}m_seed*.npy"))
    arr = []
    for f in fs:
        if f.exists():
            s = np.load(f); arr.append(s[keep] if len(s) == n_pool else s)
    return np.mean(arr, 0), len(arr)


def main():
    pool = pd.read_parquet(W / "pool.parquet"); keep = (pool.label >= 0).to_numpy()
    df = pool[keep].reset_index(drop=True).assign(y=pool.label[keep].to_numpy())
    m = (df.set == "round1").to_numpy(); d = df[m].reset_index(drop=True)
    y = d.y.to_numpy(); wt = ev.town_weights(d); g = d.group.to_numpy()
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}
    ref, _ = load("imagenet", 80, keep, len(pool))
    r0 = ev.metrics(df, ref)
    out = []
    for name, w in ROWS:
        s, n_seed = load(name, w, keep, len(pool))
        k = ev.metrics(df, s)
        rng = np.random.default_rng(0); gains, aucs = [], []
        for _ in range(2000):
            j = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
            if y[j].min() == y[j].max():
                continue
            gains.append(AP(y[j], s[m][j], sample_weight=wt[j]) - AP(y[j], ref[m][j], sample_weight=wt[j]))
            aucs.append(AUC(y[j], s[m][j], sample_weight=wt[j]))
        tests = {"curated_ap": (k["curated"]["ap"], r0["curated"]["ap"]),
                 "boke_auc": (k["blind_boke"]["auc"], r0["blind_boke"]["auc"]),
                 "conakry_auc": (k["blind_conakry"]["auc"], r0["blind_conakry"]["auc"]),
                 "town_ap": (k["town"]["ap"], r0["town"]["ap"])}
        beats = sum(v > b for v, b in tests.values()) if (name, w) != ("imagenet", 80) else None
        lo, hi = np.percentile(gains, [5, 95])
        out.append({"key": f"{name}_{w}m", "seeds": n_seed, **{t: round(v, 3) for t, (v, _) in tests.items()},
                    "town_auc": round(k["town"]["auc"], 3), "town_auc_90": [round(x, 2) for x in np.percentile(aucs, [5, 95])],
                    "town_top5": round(k["town"]["top5"], 3),
                    "gain_ap": round(k["town"]["ap"] - r0["town"]["ap"], 3), "gain_90": [round(lo, 3), round(hi, 3)],
                    "beats_initial_on": beats,
                    "passes_mark": bool(k["town"]["auc"] >= 0.80 and lo > 0)})
    q = [r for r in out if r["beats_initial_on"] == 4]
    res = {"rows": out, "qualifiers": [r["key"] for r in q],
           "adopted": max(q, key=lambda r: r["town_ap"])["key"] if q else None,
           "base_rates": {"curated": round(r0["curated"]["prev"], 3), "boke": round(r0["blind_boke"]["prev"], 3),
                          "conakry": round(r0["blind_conakry"]["prev"], 3), "town": round(r0["town"]["prev"], 3)},
           "town_sites": int(y.sum())}
    json.dump(res, open(W / "document_table.json", "w"), indent=1)
    for r in out:
        print(r)
    print({k_: v for k_, v in res.items() if k_ != "rows"})


if __name__ == "__main__":
    main()
