"""Notebook 08: could a foundation model give the land-use module a steadier yearly map than Esri?

The module learns change from Esri's yearly 10 m land cover on the 100 m grid. Esri changes
10-15 % of Guinea a year and a third of that change flips back the next year (notebook 01;
studies/dataset_checks 01), mostly trees <-> rangeland in the forest-savanna mosaic. TESSERA
(Feng et al. 2026, CVPR; embeddings CC0) summarises a year of Sentinel-1 and Sentinel-2 over each
10 m pixel in 128 numbers; a land-cover map read from it uses the whole year at once.

Test, on the 24 sample blocks of data_100m_tools/tessera_blocks.py (0.1 degree tiles):

  Classifier   a linear probe (standardised embeddings, multinomial logistic regression,
               balanced classes), the evaluation TESSERA's authors and the waste-site benchmark use.
               It is taught Esri's own classes on stable cells: one class in every year 2017-2025
               with at least 90 % of the cell in it, so it learns Esri's legend where Esri does not
               flicker. Maps of the blocks come from four block folds: a block is mapped by a probe
               that never saw it.
  Accuracy     at the 503 GLanCE points of notebook 02 (independent interpreters, label year
               2017-2020), with notebook 02's legend and crosswalk: TESSERA at the point's 10 m
               pixel and over its 100 m cell, against Esri at 10 m (notebook 02) and Esri's class of
               the 100 m cell (the module's input). A second probe is taught on the GLanCE points
               themselves (five folds of 0.5 degree squares) to show what independent labels give.
  Stability    yearly change and flips back (notebook 01's definitions) of the TESSERA maps and of
               Esri on the same cells.
  Real change  cells that lost at least half their area of forest in a year (Hansen): the share
               that each map moves out of trees between the year before and the year after.

Output: grid_100m/outputs/08_tessera_*.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]                        # grid_100m/
ROOT = HERE.parents[1]                                            # d:/Guinea_Methodology
sys.path.insert(0, str(ROOT / "data_100m_tools"))
import grid100 as G                                               # noqa: E402

TES = G.STORE / "tessera"
OUT = HERE / "outputs"
YEARS = list(range(2017, 2026))
CLASSES = [1, 2, 4, 5, 7, 8, 11]                                  # Esri codes (no clouds, no data)
NAMES = {1: "Water", 2: "Trees", 4: "Flooded vegetation", 5: "Crops", 7: "Built area", 8: "Bare ground", 11: "Rangeland"}
COMMON = {1: "Tree cover", 2: "Shrub & grassland", 3: "Cropland", 4: "Built-up", 5: "Bare", 6: "Water", 7: "Herbaceous wetland"}
ESRI_TO_COMMON = {1: 6, 2: 1, 4: 7, 5: 3, 7: 4, 8: 5, 11: 2}       # notebook 02
SHARE_COL = {1: "water", 2: "trees", 4: "flooded_vegetation", 5: "crops", 7: "built_area", 8: "bare_ground", 11: "rangeland"}
PER_CLASS = 4000


def probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))


def load_blocks():
    meta = json.loads((TES / "blocks.json").read_text())["blocks"]
    cells = np.load(TES / "blocks_cells.npz")
    pos = np.concatenate([cells[f"{b['name']}_pos"] for b in meta])
    block = np.concatenate([np.full(cells[f"{b['name']}_pos"].size, i) for i, b in enumerate(meta)])
    E = {y: np.concatenate([np.load(TES / str(y) / f"{b['name']}.npy") for b in meta]).astype(np.float32) for y in YEARS}
    esri = {y: G.read_layer(f"esri_{y}")["class"][pos].astype(int) for y in YEARS}
    share = {y: G.read_layer(f"esri_{y}", list(SHARE_COL.values())) for y in YEARS}
    share = {y: {c: share[y][SHARE_COL[c]][pos] for c in CLASSES} for y in YEARS}
    return meta, pos, block, E, esri, share


def stable_cells(esri, share):
    first = esri[YEARS[0]]
    same = np.all([esri[y] == first for y in YEARS], 0) & np.isin(first, CLASSES)
    pure = np.ones_like(same)
    for y in YEARS:
        s = np.zeros(first.size)
        for c in CLASSES:
            s = np.where(first == c, share[y][c], s)
        pure &= s >= 90
    return same & pure, first


def train_rows(mask, label, E, rng):
    """Cell-years of stable cells: up to PER_CLASS per class, all years pooled."""
    X, y = [], []
    for c in CLASSES:
        idx = np.flatnonzero(mask & (label == c))
        if idx.size == 0:
            continue
        cy = [(i, yr) for i in idx for yr in YEARS]
        pick = rng.choice(len(cy), min(PER_CLASS, len(cy)), replace=False)
        for k in pick:
            i, yr = cy[k]
            if np.isfinite(E[yr][i]).all():
                X.append(E[yr][i]); y.append(c)
    return np.array(X), np.array(y)


def change_rates(maps, area, valid):
    """notebook 01: % of area changing class between consecutive years, and of it the % back the year after."""
    rows = []
    for i in range(len(YEARS) - 1):
        a, b = maps[YEARS[i]], maps[YEARS[i + 1]]
        ok = valid[YEARS[i]] & valid[YEARS[i + 1]]
        ch = ok & (a != b)
        row = {"from": YEARS[i], "to": YEARS[i + 1], "changed (% of area)": 100 * area[ch].sum() / area[ok].sum()}
        if i + 2 < len(YEARS):
            row["of it, back the next year (%)"] = 100 * area[ch & (maps[YEARS[i + 2]] == a)].sum() / max(area[ch].sum(), 1e-9)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(0)
    meta, pos, block, E, esri, share = load_blocks()
    area = G.cell_area_ha(G.cells()[pos])
    stable, label = stable_cells(esri, share)
    print(f"{len(meta)} blocks, {pos.size:,} cells, {stable.sum():,} stable cells "
          f"({pd.Series(label[stable]).map(NAMES).value_counts().to_dict()})", flush=True)

    # Maps of the blocks, four block folds
    folds = rng.permutation(len(meta)) % 4
    tess = {y: np.zeros(pos.size, int) for y in YEARS}
    for f in range(4):
        test = folds[block] == f
        X, y = train_rows(stable & ~test, label, E, rng)
        m = probe().fit(X, y)
        for yr in YEARS:
            ok = test & np.isfinite(E[yr]).all(1)
            tess[yr][ok] = m.predict(E[yr][ok])
        print(f"fold {f}: trained on {len(y):,} cell-years", flush=True)

    # Stability, on cells where both maps have a class in both years
    valid = {y: np.isin(esri[y], CLASSES) & (tess[y] > 0) for y in YEARS}
    rates = {"Esri (module input)": change_rates(esri, area, valid), "TESSERA probe": change_rates(tess, area, valid)}
    st = pd.concat(rates, names=["map"]).reset_index(level=0)
    st.to_csv(OUT / "08_tessera_stability.csv", index=False)
    summary = st.groupby("map")[["changed (% of area)", "of it, back the next year (%)"]].mean()
    print(summary.round(1), flush=True)

    # Change that is real: cells that lost >= half their area of forest in a year
    h = G.read_layer("hansen_shares", [f"loss_{y}" for y in range(2018, 2025)], scaled=True)
    real = []
    for y in range(2018, 2025):
        lost = h[f"loss_{y}"][pos] >= 0.5
        for nm, mp in (("Esri (module input)", esri), ("TESSERA probe", tess)):
            was = lost & (mp[y - 1] == 2) & valid[y - 1] & valid[y + 1] if y + 1 in mp else None
            if was is None or was.sum() == 0:
                continue
            real.append({"loss year": y, "map": nm, "cells": int(was.sum()),
                         "out of trees the year after (%)": 100 * float((mp[y + 1][was] != 2).mean())})
    real = pd.DataFrame(real)
    real.to_csv(OUT / "08_tessera_real_change.csv", index=False)
    print(real.groupby("map")[["cells", "out of trees the year after (%)"]].agg({"cells": "sum", "out of trees the year after (%)": "mean"}).round(1), flush=True)

    # Accuracy at the GLanCE points
    gl = pd.read_parquet(ROOT / "studies" / "dataset_checks" / "output" / "02_glance_points.parquet")
    prod = pd.read_parquet(ROOT / "studies" / "dataset_checks" / "output" / "02_glance_products.parquet").set_index("pid")

    def common_class(row):
        l1, l2, mod = row["Glance_Class_ID_level1"], row.get("Glance_Class_ID_level2"), str(row.get("Veg_Modifier"))
        if l1 == 7:
            return 3 if (l2 == 12 or "Cropland" in mod) else (7 if "Wetland" in mod else 2)
        return {1: 6, 3: 4, 4: 5, 5: 1, 6: 2}.get(l1, 0)
    gl["reference"] = gl.apply(common_class, axis=1); gl["pid"] = np.arange(len(gl))
    pts = np.load(TES / "glance_points.npz")
    d = gl.set_index("pid").loc[pts["pid"]].reset_index()
    d["year"] = pts["year"]
    keep = (d.reference > 0).to_numpy()
    d, Pp, Pc = d[keep].reset_index(drop=True), pts["pixel"][keep], pts["cell"][keep]

    X, y = train_rows(stable, label, E, rng)
    m_all = probe().fit(X, y)
    to_c = np.vectorize(lambda c: ESRI_TO_COMMON.get(int(c), 0))
    cell_id = np.floor((d.Lat - G.grid()["transform"][5]) / G.grid()["transform"][4]).astype(int) * G.grid()["width"] + \
        np.floor((d.Lon - G.grid()["transform"][2]) / G.grid()["transform"][0]).astype(int)
    where = np.searchsorted(G.cells(), cell_id)
    where = np.minimum(where, G.cells().size - 1)
    inside = G.cells()[where] == cell_id
    esri100 = np.zeros(len(d), int)
    for yr in d.year.unique():
        k = (d.year == yr).to_numpy() & inside
        esri100[k] = G.read_layer(f"esri_{yr}")["class"][where[k]]
    preds = {"Esri, 10 m pixel (notebook 02)": to_c([prod.loc[p, f"esri_{yr}"] for p, yr in zip(d.pid, d.year)]),
             "Esri, 100 m cell (module input)": to_c(esri100),
             "TESSERA probe taught by Esri, 10 m pixel": to_c(m_all.predict(Pp)),
             "TESSERA probe taught by Esri, 100 m cell": to_c(m_all.predict(Pc))}

    # Probe taught on GLanCE itself: five folds of 0.5 degree squares
    sq = (np.floor(d.Lon / 0.5) * 100 + np.floor(d.Lat / 0.5)).to_numpy()
    usq = np.unique(sq); fold_of = dict(zip(usq, rng.permutation(usq.size) % 5)); f5 = np.array([fold_of[s] for s in sq])
    for nm, P in (("TESSERA probe taught by GLanCE (5 folds), 10 m pixel", Pp), ("TESSERA probe taught by GLanCE (5 folds), 100 m cell", Pc)):
        out = np.zeros(len(d), int)
        for f in range(5):
            tr = f5 != f
            out[~tr] = probe().fit(P[tr], d.reference[tr]).predict(P[~tr])
        preds[nm] = out
    rows = []
    ref = d.reference.to_numpy()
    for nm, p in preds.items():
        k, n = int((p == ref).sum()), len(ref)
        z = 1.96; c = (k / n + z**2 / (2 * n)) / (1 + z**2 / n); hw = z * np.sqrt(k / n * (1 - k / n) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        rows.append({"Map": nm, "Points": n, "Agreement %": round(100 * k / n, 1), "95 % CI": f"{100 * (c - hw):.0f}-{100 * (c + hw):.0f}"})
        pd.crosstab(pd.Series(ref).map(COMMON).rename("GLanCE"), pd.Series(p).map(COMMON).rename(nm)).to_csv(
            OUT / f"08_tessera_confusion_{len(rows)}.csv")
    acc = pd.DataFrame(rows)
    acc.to_csv(OUT / "08_tessera_accuracy.csv", index=False)
    print(acc.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
