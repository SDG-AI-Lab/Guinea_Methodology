"""Benchmark step 3: nested spatial-block CV of linear probes on frozen features.

Protocol (identical for every backbone x window):
  * outer: 5 size-balanced group folds on leakage-safe groups (05), repeated with 3 seeds;
  * inner: 4 size-balanced group folds inside the training folds pick C for
    StandardScaler + LogisticRegression(class_weight="balanced") by average precision.
    The test fold is never used to choose anything (Cawley & Talbot 2010; Varma & Simon 2006);
  * main run: unsure (-1) labels left out; sensitivity run: unsure counted as a waste site.

Reported on out-of-fold scores (seed-averaged), separately per test set:
  curated  natsample (651)        AP, AUC, and P/R/F1 at t=0.5 restated at Guatemala's 33 % prevalence
  blind    Boke / Conakry grids   AP, AUC
  town     round1 (400 random)    AP, AUC, top-5 %/10 % precision, weighted to the national
                                  windows by deployed-score band (as the November test)
Paired block bootstrap (2000) of town AP against the reference (imagenet, 80 m).
The deployed model's own fixed scores are reported on round1 as "deployed today".

Output: outputs/benchmark/results_<run>.json and a printed table.
"""
import json, sys
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT as EXP, PROJECT, TOOL  # noqa: E402
NATSCAN = Path(r"E:\SDG AI Lab\Guinea\national_scan_2026")  # Esri tile driver + tile cache of the Nov 2026 national scan
WORK = EXP / "2_dumpsites" / "outputs" / "benchmark"; WORK.mkdir(parents=True, exist_ok=True)
RES = WORK
CS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SEEDS = (0, 1, 2)
REF = ("imagenet", 80)
# national windows per deployed-score band (manual_qa/national_labels_2026/README.md)
N_BAND = {0.9: 35958, 0.7: 71415, 0.5: 62088, 0.3: 60572, 0.0: 75020}
GUAT_PREV = 68 / 208


def group_folds(y, g, k, seed):
    """k group folds of near-equal size, positives balanced as a tie-break.

    Each blind grid is one group (its windows touch), so a whole town lands in one fold.
    StratifiedGroupKFold balances class shares only and, with two groups of 400-560 rows,
    gave test folds of 38 to 984 rows. Here groups are shuffled (seed), then placed largest
    first into the fold with fewest rows, as GroupKFold does.
    """
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(g, return_inverse=True)
    size = np.bincount(inv); pos = np.bincount(inv, weights=y)
    order = rng.permutation(len(ug)); order = order[np.argsort(-size[order], kind="stable")]
    rows, npos, fold = np.zeros(k), np.zeros(k), np.empty(len(ug), int)
    for u in order:
        f = int(np.lexsort((npos, rows))[0])
        fold[u] = f; rows[f] += size[u]; npos[f] += pos[u]
    f_row = fold[inv]
    return [(np.where(f_row != j)[0], np.where(f_row == j)[0]) for j in range(k)]


def fit(X, y, C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000)).fit(X, y)


def _fold(X, y, g, tr, te, seed):
    inner = group_folds(y[tr], g[tr], 4, seed + 100)
    best, bestC = -1, None
    for C in CS:
        s = np.zeros(len(tr))
        for a, b in inner:
            s[b] = fit(X[tr][a], y[tr][a], C).predict_proba(X[tr][b])[:, 1]
        v = AP(y[tr], s)
        if v > best:
            best, bestC = v, C
    return te, fit(X[tr], y[tr], bestC).predict_proba(X[te])[:, 1], bestC


def nested_oof_all(X, y, g):
    """All seeds x outer folds in parallel; returns [(oof, chosen C list)] per seed."""
    from joblib import Parallel, delayed
    tasks = [(s, tr, te) for s in SEEDS
             for tr, te in group_folds(y, g, 5, s)]
    out = Parallel(n_jobs=-1)(delayed(_fold)(X, y, g, tr, te, s) for s, tr, te in tasks)
    runs = []
    for s in SEEDS:
        oof = np.full(len(y), np.nan); ch = []
        for (s_, _, _), (te, p, C) in zip(tasks, out):
            if s_ == s:
                oof[te] = p; ch.append(C)
        runs.append((oof, ch))
    return runs


def town_weights(d):
    n = d.groupby("stratum").size()
    return d["stratum"].map(lambda s: N_BAND[s] / n[s]).to_numpy()


def top_prec(y, s, w, q):
    o = np.argsort(-s); cw = np.cumsum(w[o]) / w.sum(); top = o[cw <= q]
    return float((y[top] * w[top]).sum() / max(w[top].sum(), 1e-9))


def metrics(df, s):
    out = {}
    c = df["set"] == "natsample"; y = df.y[c].to_numpy(); sc = s[c.to_numpy()]
    p = sc >= 0.5; tpr = (p & (y == 1)).sum() / (y == 1).sum(); fpr = (p & (y == 0)).sum() / (y == 0).sum()
    prec = tpr * GUAT_PREV / max(tpr * GUAT_PREV + fpr * (1 - GUAT_PREV), 1e-9)
    out["curated"] = {"ap": AP(y, sc), "auc": AUC(y, sc), "prec_at33": prec, "rec": tpr,
                      "f1_at33": 2 * prec * tpr / max(prec + tpr, 1e-9), "prev": y.mean()}
    for b in ("blind_boke", "blind_conakry"):
        m = (df["set"] == b).to_numpy(); out[b] = {"ap": AP(df.y[m], s[m]), "auc": AUC(df.y[m], s[m]), "prev": df.y[m].mean()}
    m = (df["set"] == "round1").to_numpy(); d = df[m]; y = d.y.to_numpy(); w = town_weights(d)
    out["town"] = {"ap": AP(y, s[m], sample_weight=w), "auc": AUC(y, s[m], sample_weight=w),
                   "top5": top_prec(y, s[m], w, 0.05), "top10": top_prec(y, s[m], w, 0.10),
                   "prev": float((y * w).sum() / w.sum())}
    return {k: {a: round(float(b), 3) for a, b in v.items()} for k, v in out.items()}


def boot_town(df, s_a, s_b, n=2000, seed=0):
    m = (df["set"] == "round1").to_numpy(); d = df[m].reset_index(drop=True)
    y = d.y.to_numpy(); w = town_weights(d); a, b = s_a[m], s_b[m]; g = d.group.to_numpy()
    ug = np.unique(g); idx = {u: np.where(g == u)[0] for u in ug}; rng = np.random.default_rng(seed); diffs = []
    for _ in range(n):
        k = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if y[k].min() == y[k].max():
            continue
        diffs.append(AP(y[k], b[k], sample_weight=w[k]) - AP(y[k], a[k], sample_weight=w[k]))
    d_ = np.array(diffs)
    return {"d_ap": round(float(AP(y, b, sample_weight=w) - AP(y, a, sample_weight=w)), 3),
            "ci95": [round(float(np.percentile(d_, 2.5)), 3), round(float(np.percentile(d_, 97.5)), 3)]}


def main(run="main"):
    pool = pd.read_parquet(WORK / "pool.parquet")
    lab = pool.label.copy()
    if run == "sensitivity":
        lab = lab.replace(-1, 1)
    keep = (lab >= 0).to_numpy()
    df = pool[keep].reset_index(drop=True).assign(y=lab[keep].to_numpy())
    g = df.group.to_numpy(); y = df.y.to_numpy()
    feats = sorted({(p.stem.split("_", 1)[1].rsplit("_", 1)[0], int(p.stem.rsplit("_", 1)[1][:-1]))
                    for p in WORK.glob("feat_*_*m.npy")})
    results, scores = {}, {}
    for name, w in feats:
        X = np.load(WORK / f"feat_{name}_{w}m.npy")[keep]
        runs = nested_oof_all(X, y, g)
        s = np.mean([r[0] for r in runs], 0); scores[(name, w)] = s
        per_seed = [metrics(df, r[0])["town"]["ap"] for r in runs]
        results[f"{name}_{w}m"] = {**metrics(df, s), "town_ap_per_seed": per_seed,
                                   "C_chosen": sorted({c for r in runs for c in r[1]})}
        print(f"{name:10s} {w:3d}m  curated AP {results[f'{name}_{w}m']['curated']['ap']:.3f} "
              f"AUC {results[f'{name}_{w}m']['curated']['auc']:.3f} | town AP {results[f'{name}_{w}m']['town']['ap']:.3f} "
              f"AUC {results[f'{name}_{w}m']['town']['auc']:.3f} top5 {results[f'{name}_{w}m']['town']['top5']:.3f}", flush=True)
        np.save(RES / f"oof_{run}_{name}_{w}m.npy", s)
    if REF in scores:
        for k, s in scores.items():
            if k != REF:
                results[f"{k[0]}_{k[1]}m"]["vs_ref_town"] = boot_town(df, scores[REF], s)
    m = (df["set"] == "round1").to_numpy(); d = df[m]
    results["deployed_today"] = {"town": {"ap": round(float(AP(d.y, d.score_deployed, sample_weight=town_weights(d))), 3),
                                          "auc": round(float(AUC(d.y, d.score_deployed, sample_weight=town_weights(d))), 3)}}
    json.dump(results, open(RES / f"results_{run}.json", "w"), indent=1)
    print("deployed today, town:", results["deployed_today"]["town"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
