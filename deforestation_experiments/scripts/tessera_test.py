"""Notebook 11: does a foundation model's picture of each place improve the risk model?

TESSERA (Feng et al. 2026, CVPR; embeddings CC0) summarises a year of Sentinel-1 and Sentinel-2
over each 10 m pixel in 128 numbers. Averaged over a 100 m cell, the vector of the year before a
window describes the cell as it was when the window started: a predictor measured at the
window's start, as every predictor here is (notebook 05).

The national embeddings are 300-370 GB a year, so the test runs on the 16 forest blocks of
data_100m_tools/tessera_blocks.py (0.1 degree tiles, drawn by ecoregion in proportion to loss).
TESSERA starts in 2017 and Hansen in the store ends in 2024, so two windows have an embedding
strictly before the calibration loss:

  calibration 2018-2020 (predictors at 2017), test 2021-2023 (predictors at 2020)
  calibration 2019-2021 (predictors at 2018), test 2022-2024 (predictors at 2021)

Variants, all fitted on the same cells drawn in the blocks (notebook 04's weighted draw, the
improved model's random forest, seeds 42, 1, 2) and scored on every forest cell of the blocks
at the test window's start (AUC over the 30 m pixels, as notebook 04):

  Improved model               the 16 predictors
  Improved + TESSERA           and the 128 numbers of the cell
  Improved + TESSERA (16 PCs)  and the first 16 principal components (fitted on calibration cells)
  TESSERA alone                the 128 numbers

The gain over the improved model gets a 90 % interval from a paired bootstrap over the blocks
(2000 draws), seed-averaged risk; noise between seeds is reported as in notebook 06.

Output: outputs/runs/11_tessera_<t1>_<t2>_<t3>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as MM                                               # noqa: E402
import model_variants as D                                       # noqa: E402

G, M = MM.G, MM.M
TES = G.STORE / "tessera"
SPLITS = [(2017, 2020, 2023), (2018, 2021, 2024)]
SEEDS = (42, 1, 2)
N_PC = 16
TNAMES = [f"tessera_{k:03d}" for k in range(128)]


def blocks():
    """Forest blocks: name, positions among the hansen() rows of their forest cells."""
    meta = json.loads((TES / "blocks.json").read_text())["blocks"]
    cells = np.load(TES / "blocks_cells.npz")
    idx = MM.hansen()["idx"]
    out = []
    for b in meta:
        if b["kind"] != "forest":
            continue
        pos = cells[f"{b['name']}_pos"]
        r = np.searchsorted(idx, pos)
        ok = (r < idx.size) & (idx[np.minimum(r, idx.size - 1)] == pos)
        out.append((b["name"], r[ok], np.flatnonzero(ok)))
    return out


def embedding(year, blk):
    """(hansen rows, 128) embeddings of the forest cells of the blocks at `year`."""
    n = MM.hansen()["idx"].size
    E = np.full((n, 128), np.nan, np.float32)
    for name, rows, k in blk:
        E[rows] = np.load(TES / str(year) / f"{name}.npy").astype(np.float32)[k]
    return E


def run_split(split, log=print):
    t1, t2, t3 = split
    start = time.perf_counter()
    blk = blocks()
    n = MM.hansen()["idx"].size
    in_blocks = np.zeros(n, bool); block_id = np.full(n, -1)
    for i, (_, rows, _) in enumerate(blk):
        in_blocks[rows] = True; block_id[rows] = i
    area = MM.hansen()["area"]

    draws = {s: D._draw(t1, t2, s, within=in_blocks) for s in SEEDS}
    cal_rows = np.unique(np.concatenate([c for c, _ in draws.values()]))
    f2, lt = MM.forest_pct(t2), MM.loss_pct(t2, t3)
    test_rows = np.flatnonzero(in_blocks & (f2 > 0))
    w_pos = lt[test_rows] / 100 * area[test_rows]
    w_neg = np.clip(f2[test_rows] - lt[test_rows], 0, None) / 100 * area[test_rows]

    cal = D.build(t1, D.BASE, cal_rows)
    test = D.build(t2, D.BASE, test_rows)
    Ec, Et = embedding(t1, blk)[cal_rows], embedding(t2, blk)[test_rows]
    for j, nm in enumerate(TNAMES):
        cal[nm], test[nm] = Ec[:, j], Et[:, j]
    from sklearn.decomposition import PCA
    okc = np.isfinite(Ec).all(1)
    pca = PCA(N_PC, random_state=0).fit(Ec[okc])
    PC = [f"tessera_pc{k:02d}" for k in range(N_PC)]
    for X, d in ((Ec, cal), (Et, test)):
        ok = np.isfinite(X).all(1); Z = np.full((X.shape[0], N_PC), np.nan, np.float32)
        Z[ok] = pca.transform(X[ok])
        for k, nm in enumerate(PC):
            d[nm] = Z[:, k]
    log(f"{M.split_label(split)}: {len(blk)} blocks, {cal_rows.size} drawn cells, {test_rows.size} test cells, "
        f"predictors {time.perf_counter() - start:.0f} s", flush=True)

    variants = {"Improved model": D.BASE, "Improved + TESSERA": D.BASE + TNAMES,
                "Improved + TESSERA (16 PCs)": D.BASE + PC, "TESSERA alone": TNAMES}
    risks, out = {}, {"split": M.split_label(split), "blocks": [b[0] for b in blk], "drawn_cells": int(cal_rows.size),
                      "test_cells": int(test_rows.size), "lost_ha": float(w_pos.sum()), "pca_variance_16": float(pca.explained_variance_ratio_.sum()),
                      "auc_per_seed": {}, "auc_seed_mean_risk": {}, "gain_vs_improved": {}}
    for label, names in variants.items():
        rs = []
        for s in SEEDS:
            cells, y = draws[s]
            X, ok = D._xy(cal, np.searchsorted(cal_rows, cells), names)
            model = D._fit(X[ok], y[ok], s)
            Xt, okt = D._xy(test, np.arange(test_rows.size), names)
            r = np.full(test_rows.size, np.nan, np.float32); r[okt] = model.predict_proba(Xt[okt])[:, 1]
            rs.append(r)
        risks[label] = np.nanmean(rs, 0)
        out["auc_per_seed"][label] = [MM.weighted_auc(r, w_pos, w_neg) for r in rs]
        out["auc_seed_mean_risk"][label] = MM.weighted_auc(risks[label], w_pos, w_neg)
        log(f"{out['split']}: {label:28s} AUC {np.mean(out['auc_per_seed'][label]):.4f} "
            f"(seeds {', '.join(f'{a:.4f}' for a in out['auc_per_seed'][label])}) {time.perf_counter() - start:.0f} s", flush=True)

    b_test = block_id[test_rows]; ub = np.unique(b_test); idx = {u: np.flatnonzero(b_test == u) for u in ub}
    rng = np.random.default_rng(0)
    draws_b = [np.concatenate([idx[u] for u in rng.choice(ub, ub.size)]) for _ in range(2000)]
    base = risks["Improved model"]
    for label in variants:
        if label == "Improved model":
            continue
        d = [MM.weighted_auc(risks[label][j], w_pos[j], w_neg[j]) - MM.weighted_auc(base[j], w_pos[j], w_neg[j]) for j in draws_b]
        lo, hi = np.percentile(d, [5, 95])
        out["gain_vs_improved"][label] = {"gain": out["auc_seed_mean_risk"][label] - out["auc_seed_mean_risk"]["Improved model"],
                                          "ci90": [float(lo), float(hi)]}
        log(f"{out['split']}: {label} gain {out['gain_vs_improved'][label]['gain']:+.4f} "
            f"(90 % {lo:+.4f} to {hi:+.4f})", flush=True)
    out["seconds"] = time.perf_counter() - start
    return out


CHECK_VARIANTS = ("Improved model", "Improved + TESSERA", "TESSERA alone")


def checks(split, log=print):
    """Two checks of a large gain.

    gap         Hansen often dates a clearing to the year after it happened, so the year of the
                embedding may already show a loss counted in the window's first year. The first
                year of both windows is left out: calibration loss t1+2..t2, test loss t2+2..t3.
    blocks      the same windows, but each block is mapped by models fitted in the other blocks
                (four folds), so a model cannot learn a place and find it again.
    """
    t1, t2, t3 = split
    blk = blocks()
    n = MM.hansen()["idx"].size
    in_blocks = np.zeros(n, bool); block_id = np.full(n, -1)
    for i, (_, rows, _) in enumerate(blk):
        in_blocks[rows] = True; block_id[rows] = i
    area = MM.hansen()["area"]
    f2 = MM.forest_pct(t2)
    test_rows = np.flatnonzero(in_blocks & (f2 > 0))
    out = {"split": M.split_label(split)}
    for check in ("gap", "blocks"):
        a1, a2 = (t1 + 1, t2 + 1) if check == "gap" else (t1, t2)      # loss years a+1..b
        lt = MM.loss_pct(a2, t3)
        f_at = MM.forest_pct(a2)
        w_pos = lt[test_rows] / 100 * area[test_rows]
        w_neg = np.clip(f_at[test_rows] - lt[test_rows], 0, None) / 100 * area[test_rows]
        fold = np.random.default_rng(0).permutation(len(blk)) % 4
        groups = [None] if check == "gap" else list(range(4))
        draws = {(g, s): D._draw(a1, t2, s, within=in_blocks if g is None else in_blocks & (fold[block_id] != g))
                 for g in groups for s in SEEDS}
        cal_rows = np.unique(np.concatenate([c for c, _ in draws.values()]))
        cal, test = D.build(t1, D.BASE, cal_rows), D.build(t2, D.BASE, test_rows)
        Ec, Et = embedding(t1, blk)[cal_rows], embedding(t2, blk)[test_rows]
        for j, nm in enumerate(TNAMES):
            cal[nm], test[nm] = Ec[:, j], Et[:, j]
        variants = {"Improved model": D.BASE, "Improved + TESSERA": D.BASE + TNAMES, "TESSERA alone": TNAMES}
        res = {}
        for label, names in variants.items():
            rs = []
            for s in SEEDS:
                r = np.full(test_rows.size, np.nan, np.float32)
                for g in groups:
                    cells, y = draws[g, s]
                    X, ok = D._xy(cal, np.searchsorted(cal_rows, cells), names)
                    model = D._fit(X[ok], y[ok], s)
                    rows = np.arange(test_rows.size) if g is None else np.flatnonzero(fold[block_id[test_rows]] == g)
                    Xt, okt = D._xy(test, rows, names)
                    p = np.full(rows.size, np.nan, np.float32); p[okt] = model.predict_proba(Xt[okt])[:, 1]
                    r[rows] = p
                rs.append(MM.weighted_auc(r, w_pos, w_neg))
            res[label] = rs
            log(f"{out['split']} [{check}]: {label:20s} AUC {np.mean(rs):.4f} (seeds {', '.join(f'{a:.4f}' for a in rs)})", flush=True)
        out[check] = res
    return out


ESRI_SHARES = ["water", "trees", "flooded_vegetation", "crops", "built_area", "bare_ground", "rangeland"]


def esri_check(split, log=print):
    """Is TESSERA only the land cover at the window's start? The improved model with Esri's class
    shares of the cell at t1 (test: t2), the national yearly map the platform already reads, on the
    same cells and draws as run_split."""
    t1, t2, t3 = split
    blk = blocks()
    n = MM.hansen()["idx"].size
    in_blocks = np.zeros(n, bool)
    for _, rows, _ in blk:
        in_blocks[rows] = True
    area = MM.hansen()["area"]
    draws = {s: D._draw(t1, t2, s, within=in_blocks) for s in SEEDS}
    cal_rows = np.unique(np.concatenate([c for c, _ in draws.values()]))
    f2, lt = MM.forest_pct(t2), MM.loss_pct(t2, t3)
    test_rows = np.flatnonzero(in_blocks & (f2 > 0))
    w_pos = lt[test_rows] / 100 * area[test_rows]
    w_neg = np.clip(f2[test_rows] - lt[test_rows], 0, None) / 100 * area[test_rows]
    cal, test = D.build(t1, D.BASE, cal_rows), D.build(t2, D.BASE, test_rows)
    idx = MM.hansen()["idx"]
    for d, rows, yr in ((cal, cal_rows, t1), (test, test_rows, t2)):
        e = G.read_layer(f"esri_{yr}", ESRI_SHARES, scaled=True)
        for c in ESRI_SHARES:
            d[f"esri_{c}"] = e[c][idx][rows]
    Ec, Et = embedding(t1, blk)[cal_rows], embedding(t2, blk)[test_rows]
    for j, nm in enumerate(TNAMES):
        cal[nm], test[nm] = Ec[:, j], Et[:, j]
    EN = [f"esri_{c}" for c in ESRI_SHARES]
    variants = {"Improved model": D.BASE, "Improved + Esri shares": D.BASE + EN,
                "Improved + TESSERA": D.BASE + TNAMES, "Improved + Esri + TESSERA": D.BASE + EN + TNAMES}
    out = {"split": M.split_label(split)}
    for label, names in variants.items():
        rs = []
        for s in SEEDS:
            cells, y = draws[s]
            X, ok = D._xy(cal, np.searchsorted(cal_rows, cells), names)
            model = D._fit(X[ok], y[ok], s)
            Xt, okt = D._xy(test, np.arange(test_rows.size), names)
            r = np.full(test_rows.size, np.nan, np.float32); r[okt] = model.predict_proba(Xt[okt])[:, 1]
            rs.append(MM.weighted_auc(r, w_pos, w_neg))
        out[label] = rs
        log(f"{out['split']} [esri]: {label:26s} AUC {np.mean(rs):.4f} (seeds {', '.join(f'{a:.4f}' for a in rs)})", flush=True)
    return out


if __name__ == "__main__":
    if "--esri" in sys.argv:
        for split in SPLITS:
            res = esri_check(split)
            (MM.RUNS / "11_tessera_esri_{}_{}_{}.json".format(*split)).write_text(json.dumps(res, indent=1))
    elif "--checks" in sys.argv:
        for split in SPLITS:
            res = checks(split)
            (MM.RUNS / "11_tessera_checks_{}_{}_{}.json".format(*split)).write_text(json.dumps(res, indent=1))
    else:
        for split in SPLITS:
            res = run_split(split)
            (MM.RUNS / "11_tessera_{}_{}_{}.json".format(*split)).write_text(json.dumps(res, indent=1))
