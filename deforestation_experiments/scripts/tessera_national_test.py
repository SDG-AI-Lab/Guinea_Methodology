"""Notebook 11, national stage: does TESSERA still improve the improved model over all of Guinea?

The block test (tessera_test.py) found +0.11 to +0.15 AUC in 16 forest blocks. Verra's tests
(VT0007 fit and prediction MedAE on ~100,000 ha blocks) need the whole country, so this reads the
national embeddings of data_100m_tools/tessera_national.py: for every cell with forest in 2000,
the first 16 principal components of the cell's mean embedding (fixed rotation, fitted once on
the 24 sample blocks; 85 % of the variance; the blocks found 16 PCs score as well as all 128).

Window: calibration 2019-2021 (predictors and embedding at 2018), test 2022-2024 (predictors and
embedding at 2021), the third window of the README tables (split 2018, 2021, 2024).

Variants, fitted on the same cells (notebook 04's weighted draw over the whole country, the
improved model's random forest, seeds 42, 1, 2) and scored on every forest cell, as improve.py:

  Improved model              the 16 predictors
  Improved + TESSERA          and the 16 TESSERA components
  Rank average (A), each      the cell's mean percentile rank of the model and riskmapjnr's local
                              rate with the same timing (tool_comparison.run_same_timing), the
                              risk map the README adopts

Scores: AUC over the 30 m pixels (verra.pixel_auc), VT0007 fit and prediction MedAE (ha).
The 799 cells without TESSERA data (0.006 %) take the components' mean (0).

Decision rule, written before the run. Stage 2 (the 2018-2020 window, about 18 hours of reading)
goes ahead only if, for both the model alone and the rank average, adding TESSERA
  - raises the AUC in every seed by more than the noise between runs (0.003), and
  - lowers the mean prediction MedAE by more than the seed range (max - min) of the variant
    without TESSERA.
Otherwise TESSERA is recorded as tested nationally and not adopted for deforestation.

  python tessera_national_test.py --check   rows of the national file agree with the block files
  python tessera_national_test.py           the test, outputs/runs/11_tessera_national_2018_2021_2024.json
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as MM                                               # noqa: E402
import model_variants as D                                       # noqa: E402
import verra as V                                                # noqa: E402

G, M = MM.G, MM.M
NAT = G.STORE / "tessera" / "national"
SPLIT = (2018, 2021, 2024)
SEEDS = (42, 1, 2)
PC = [f"TESSERA component {k + 1}" for k in range(16)]
NOISE_AUC = 0.003
RANK = {"Improved model": "Rank average (A): improved model + riskmapjnr",
        "Improved + TESSERA": "Rank average (A): improved + TESSERA + riskmapjnr"}


def components(year, rows):
    """{name: values at rows} of the 16 national components at `year` (hansen() row order)."""
    a = np.load(NAT / f"pc16_{year}.npy", mmap_mode="r")
    Z = np.nan_to_num(np.asarray(a[rows], dtype=np.float32), nan=0.0)
    return {nm: Z[:, k].copy() for k, nm in enumerate(PC)}


def check(log=print):
    """The national rows against the block embeddings projected with the same rotation."""
    import json
    tes = G.STORE / "tessera"
    meta = json.loads((tes / "blocks.json").read_text())["blocks"]
    cells = np.load(tes / "blocks_cells.npz")
    pca = np.load(NAT / "pca16.npz")
    idx = MM.hansen()["idx"]
    out = {}
    for year in (2018, 2021):
        nat = np.load(NAT / f"pc16_{year}.npy", mmap_mode="r")
        cors, diffs = [], []
        for b in meta:
            pos = cells[f"{b['name']}_pos"]
            r = np.searchsorted(idx, pos)
            ok = (r < idx.size) & (idx[np.minimum(r, idx.size - 1)] == pos)
            if not ok.any():
                continue
            E = np.load(tes / str(year) / f"{b['name']}.npy").astype(np.float32)[ok]
            fin = np.isfinite(E).all(1)
            z_blk = (E[fin] - pca["mean"]) @ pca["components"].T
            z_nat = np.asarray(nat[r[ok][fin]], dtype=np.float32)
            good = np.isfinite(z_nat).all(1)
            cors.append(np.corrcoef(z_blk[good, 0], z_nat[good, 0])[0, 1])
            diffs.append(float(np.abs(z_blk[good] - z_nat[good]).max()))
        out[year] = {"blocks": len(cors), "min_corr_pc1": float(np.min(cors)), "max_abs_diff": float(np.max(diffs))}
        log(f"{year}: {out[year]}")
    return out


def run(log=print):
    t1, t2, t3 = SPLIT
    import tool_comparison as T
    start = time.perf_counter()
    n = MM.hansen()["idx"].size
    f1, f2 = MM.forest_pct(t1), MM.forest_pct(t2)
    rows1, rows2 = np.flatnonzero(f1 > 0), np.flatnonzero(f2 > 0)
    variants = {"Improved model": D.BASE, "Improved + TESSERA": D.BASE + PC}
    rank_label = {nm: RANK[nm] for nm in variants}

    _, (rmj_fit, rmj_test), choice = T.riskmapjnr(SPLIT, log=lambda *a: None, same_timing=True)
    log(f"riskmapjnr (same timing) {choice['ws_hat']} {choice['m_hat']}, {time.perf_counter() - start:.0f} s", flush=True)

    cal = D.build(t1, D.BASE, rows1); cal.update(components(t1, rows1))
    log(f"calibration predictors {time.perf_counter() - start:.0f} s, peak {MM._peak_gb():.1f} GB", flush=True)
    models, fits = {}, {}
    for s in SEEDS:
        cells, y = D._draw(t1, t2, s)
        at = np.searchsorted(rows1, cells)
        for name, names in variants.items():
            X, ok = D._xy(cal, at, names)
            models[name, s] = D._fit(X[ok], y[ok], s)
            risk = np.full(n, np.nan, dtype="float32")
            risk[rows1] = _predict(models[name, s], cal, names, rows1.size)
            fits[name, s] = risk
        log(f"seed {s} fitted, {time.perf_counter() - start:.0f} s", flush=True)
    del cal

    test = D.build(t2, D.BASE, rows2); test.update(components(t2, rows2))
    out = {"split": M.split_label(SPLIT), "riskmapjnr_choice": {k: choice[k] for k in ("ws_hat", "m_hat", "dist_thresh_m")},
           "variants": {}}
    for name, names in variants.items():
        for label in (name, rank_label[name]):
            out["variants"][label] = {"aucs": [], "fit": [], "prediction": []}
        for s in SEEDS:
            risk = np.full(n, np.nan, dtype="float32")
            risk[rows2] = _predict(models[name, s], test, names, rows2.size)
            fit = fits.pop((name, s))
            rank_fit = (T._rank(fit, f1 > 0) + T._rank(rmj_fit, f1 > 0)) / 2
            rank_test = (T._rank(risk, f2 > 0) + T._rank(rmj_test, f2 > 0)) / 2
            for label, a, b in ((name, fit, risk), (rank_label[name], rank_fit, rank_test)):
                r = out["variants"][label]
                tests = V.vt0007_tests(SPLIT, a, b)
                r["aucs"].append(V.pixel_auc(b, t2, t3))
                r["fit"].append(tests["fit_medae_ha"])
                r["prediction"].append(tests["prediction_medae_ha"])
                log(f"{label} seed {s}: AUC {r['aucs'][-1]:.4f} fit {r['fit'][-1]:.0f} prediction {r['prediction'][-1]:.0f} "
                    f"({time.perf_counter() - start:.0f} s)", flush=True)
        if name == "Improved + TESSERA":
            out["importance"] = M.DR.driver_importance(models[name, SEEDS[0]], names)
    del test

    pairs = (("Improved model", "Improved + TESSERA"), (RANK["Improved model"], RANK["Improved + TESSERA"]))
    out["decision"] = {}
    for a, b in pairs:
        A, B = out["variants"][a], out["variants"][b]
        auc_gain = np.subtract(B["aucs"], A["aucs"])
        pred_gain = np.mean(A["prediction"]) - np.mean(B["prediction"])
        seed_range = np.ptp(A["prediction"])
        out["decision"][b] = {"auc_gain_per_seed": auc_gain.tolist(), "auc_clear": bool((auc_gain > NOISE_AUC).all()),
                              "prediction_gain_ha": float(pred_gain), "seed_range_ha": float(seed_range),
                              "prediction_clear": bool(pred_gain > seed_range)}
        log(f"{b}: AUC gain {', '.join(f'{g:+.4f}' for g in auc_gain)}; prediction {pred_gain:+.0f} ha (seed range {seed_range:.0f})", flush=True)
    out["stage_2"] = all(d["auc_clear"] and d["prediction_clear"] for d in out["decision"].values())
    out["seconds"] = time.perf_counter() - start
    out["peak_gb"] = MM._peak_gb()
    log(f"stage 2: {out['stage_2']}; {out['seconds'] / 60:.0f} min, peak {out['peak_gb']:.1f} GB", flush=True)
    return out


def _predict(model, values, names, count, chunk=1_000_000):
    risk = np.full(count, np.nan, dtype="float32")
    for i in range(0, count, chunk):
        X = np.column_stack([values[m][i:i + chunk] for m in names])
        ok = np.isfinite(X).all(axis=1)
        part = risk[i:i + chunk]
        part[ok] = model.predict_proba(X[ok])[:, 1]
    return risk


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        MM.cached("11_tessera_national_{}_{}_{}".format(*SPLIT), run, rerun="--rerun" in sys.argv)
        print("SAVED", flush=True)
