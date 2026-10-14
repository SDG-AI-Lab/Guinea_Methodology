#!/usr/bin/env python
"""
Can machine learning improve the FDI layer, and what is the FDI rule really
detecting?

    python 32_run_marida.py                 # full analysis
    python 32_run_marida.py --no-guinea     # skip the Guinea transfer check
    python 32_run_marida.py --trees 200     # faster, slightly noisier

Writes to manual_qa/runs/marida/. Costs nothing after the one-off 1.16 GB
MARIDA download: no Earth Engine call is made, and the Guinea transfer check
reads the Conakry composite already in the raster cache.

THE TWO QUESTIONS, IN ORDER OF IMPORTANCE

1. WHAT DOES THE DEPLOYED RULE FIRE ON? Guinea's mean+3sigma FDI rule has
   never been evaluated against labels, because no labelled marine debris
   exists for Guinea. MARIDA supplies labels, including separate classes for
   Sargassum, foam, ships, wakes and turbid water -- the things an anomaly
   rule is at risk of confusing with debris. This question is answerable
   regardless of whether any model is ever deployed, and its answer changes
   how every existing patch count must be described.

2. DOES A CLASSIFIER BEAT THE RULE? Compared at MATCHED RECALL, never at
   arbitrary operating points, and evaluated on geographically held-out
   locations rather than MARIDA's official split alone -- 91% of that split's
   test patches sit on MGRS tiles also present in training, so its headline
   numbers are optimistic for a model deployed somewhere new.

And one question that governs whether the answers are usable:

3. DOES ANY OF IT TRANSFER TO GUINEA? Measured, not assumed. See domain.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from plasticwaste_tool.marida_tool import baseline as B      # noqa: E402
from plasticwaste_tool.marida_tool import data as D          # noqa: E402
from plasticwaste_tool.marida_tool import domain as DOM      # noqa: E402
from plasticwaste_tool.marida_tool import evaluate as E      # noqa: E402
from plasticwaste_tool.marida_tool import features as F      # noqa: E402
from plasticwaste_tool.marida_tool import model as M         # noqa: E402

import os                                                  # noqa: E402
from plasticwaste_tool import config as pwcfg              # noqa: E402

OUT = HERE / "manual_qa" / "runs" / "marida"
# The raster cache the FDI pipeline writes to: <APP_SCRATCH_DIR>/cache/rasters
# (api_connector._cache). Read from config rather than hard-coded, so the same
# script finds it on a workstation and in the backend container.
# PW_RASTER_CACHE overrides it, as in scripts 34 and 36.
CACHE = Path(os.environ.get("PW_RASTER_CACHE")
             or Path(pwcfg.PROCESSING_CONFIG["cache_dir"]) / "rasters")


def _conakry_stem():
    """Cache name of the Conakry January 2024 composite, built exactly as
    pipeline.run_from_gee names it (AOI + FDI formula token + window), so it
    always refers to the composite of the current FDI definition."""
    from plasticwaste_tool import api_connector as api
    return f"conakry_{api.aoi_cache_token('conakry')}_2024-01-01_2024-02-01_composite"


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _fit(Ftr, ytr, cols, trees, multiclass=False):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=trees, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=M.SEED)
    clf.fit(Ftr[cols].values, ytr if multiclass else (ytr == D.TARGET).astype(int))
    return clf


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--no-guinea", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"generated": str(HERE / "32_run_marida.py"), "trees": args.trees}

    # ---------------- load ----------------
    print("loading MARIDA ...")
    Xtr, ytr, ptr = D.load_pixels("train")
    Xte, yte, pte = D.load_pixels("test")
    tile_tr = {D.normalise_id(p).split("_")[2] for p in ptr}
    tile_te = np.array([D.normalise_id(p).split("_")[2] for p in pte])
    scene_te = np.array([p.rsplit("_", 1)[0] for p in pte])
    unseen = ~np.isin(tile_te, list(tile_tr))

    print(f"  train {len(ytr):,} px ({int((ytr == 1).sum()):,} debris), "
          f"test {len(yte):,} px ({int((yte == 1).sum()):,} debris)")
    print(f"  test pixels on GEOGRAPHICALLY UNSEEN tiles: "
          f"{int(unseen.sum()):,} ({unseen.mean():.1%})")
    D.class_counts(ytr).to_csv(OUT / "class_balance_train.csv", index=False)

    summary["data"] = {
        "n_train_px": len(ytr), "n_test_px": len(yte),
        "n_train_debris": int((ytr == 1).sum()),
        "n_test_debris": int((yte == 1).sum()),
        "test_px_unseen_tiles": int(unseen.sum()),
        "unseen_tiles": sorted(set(tile_te[unseen])),
        "train_tiles": sorted(tile_tr),
    }

    # ---------------- Q1: what does the rule fire on ----------------
    print("\n=== Q1. What the deployed mean+3sigma FDI rule fires on ===")
    r = B.predict(Xte, yte, group=scene_te)
    fired = B.what_does_it_fire_on(yte, r["pred"])
    fired.to_csv(OUT / "rule_confusion.csv", index=False)
    top = fired[fired.flagged > 0].head(6)
    for row in top.itertuples():
        print(f"  {row._2:26s} {row.share_of_all_detections:6.1%} of detections "
              f"({row.flagged:,} px, {row.role})")
    scores = B.binary_scores(yte, r["pred"], r["scored"])
    print(f"  -> precision {scores['precision']:.3f}, recall "
          f"{scores['recall']:.3f}, lift {scores['lift']:.0f}x over base rate")
    summary["rule"] = {"scores": scores,
                       "detection_composition": fired.to_dict("records")}

    # ---------------- Q2: does a classifier beat it ----------------
    print("\n=== Q2. Random forest vs the rule, at MATCHED RECALL ===")
    t0 = time.time()
    Ftr, Fte = F.build(Xtr), F.build(Xte)
    all_cols = list(Ftr.columns)

    results = {}
    for label, cols in [("full", all_cols), ("pipeline", DOM.PIPELINE_FEATURES)]:
        clf = _fit(Ftr, ytr, cols, args.trees)
        s = clf.predict_proba(Fte[cols].values)[:, list(clf.classes_).index(1)]
        row = {}
        for tag, mask in [("all_test", r["scored"]),
                          ("unseen_locations", r["scored"] & unseen)]:
            row[tag] = E.compare_at_matched_recall((yte == 1), r["pred"], s, mask)
        results[label] = row
        results[label]["_score"] = s
        results[label]["_model"] = clf
        results[label]["_cols"] = cols
        a = row["all_test"]; u = row["unseen_locations"]
        print(f"  {label:9s} ({len(cols):2d} features)  "
              f"all-test AUC {a['model_roc_auc']:.4f} | "
              f"unseen AUC {u['model_roc_auc']:.4f} | "
              f"precision {u['rule_precision']:.3f} -> {u['model_precision']:.3f} "
              f"({u['precision_ratio']:.2f}x)")
    print(f"  ({time.time() - t0:.0f}s)")

    M.importances(results["full"]["_model"], all_cols).to_csv(
        OUT / "feature_importances.csv", index=False)

    print("\n  The 'pipeline' model uses ONLY the seven bands the Guinea fetch\n"
          "  already downloads, so adopting it would cost no extra imagery.")

    # bootstrap the held-out-location comparison: it rests on few debris pixels
    print(f"\n  bootstrapping unseen-location result ({args.bootstrap} draws, "
          f"resampling PATCHES not pixels) ...")
    boot = _bootstrap(yte, r, results["pipeline"]["_score"], pte,
                      r["scored"] & unseen, args.bootstrap)
    print(f"    ROC-AUC        {boot['auc_median']:.4f} "
          f"[{boot['auc_p05']:.4f}, {boot['auc_p95']:.4f}]")
    print(f"    precision gain {boot['ratio_median']:.2f}x "
          f"[{boot['ratio_p05']:.2f}, {boot['ratio_p95']:.2f}], "
          f"favours model in {boot['frac_gt_1']:.0%} of draws")

    summary["comparison"] = {
        k: {t: v for t, v in row.items() if not t.startswith("_")}
        for k, row in results.items()}
    summary["bootstrap_unseen"] = boot

    # multiclass: name the confusions
    mc = _fit(Ftr, ytr, DOM.PIPELINE_FEATURES, args.trees, multiclass=True)
    sel = r["pred"] & r["scored"]
    named = _name_predictions(mc.predict(Fte[DOM.PIPELINE_FEATURES].values)[sel])
    named.to_csv(OUT / "rule_detections_named.csv", index=False)

    # ---------------- Q3: does it transfer ----------------
    if not args.no_guinea:
        print("\n=== Q3. Does any of this transfer to Guinea? ===")
        try:
            summary["transfer"] = _guinea_check(Xtr, ytr, ptr, mc,
                                                results["pipeline"], OUT)
        except FileNotFoundError as exc:
            print(f"  skipped: {exc}")
            summary["transfer"] = {"error": str(exc)}

    sites = D.site_coverage()
    sites.to_csv(OUT / "marida_sites.csv", index=False)
    summary["geographic_caveat"] = E.transfer_warning(sites)
    print("\n" + summary["geographic_caveat"])

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json), encoding="utf-8")
    print(f"\nwrote {len(list(OUT.glob('*')))} files to {OUT}")
    return 0


def _bootstrap(yte, r, score, pte, mask, n_draws):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(M.SEED)
    t = (yte == 1)[mask]; s = score[mask]; rp = r["pred"][mask]; grp = pte[mask]
    patches = np.unique(grp)
    where = {p: np.flatnonzero(grp == p) for p in patches}
    aucs, ratios = [], []
    for _ in range(n_draws):
        idx = np.concatenate([where[p] for p in
                              rng.choice(patches, len(patches), replace=True)])
        tt, ss, rr = t[idx], s[idx], rp[idx]
        if tt.sum() == 0 or tt.all():
            continue
        aucs.append(roc_auc_score(tt, ss))
        rtp, rfp = int((tt & rr).sum()), int((~tt & rr).sum())
        if not (rtp + rfp):
            continue
        rprec, rrec = rtp / (rtp + rfp), rtp / tt.sum()
        mp = ss >= M.threshold_for_recall(tt, ss, rrec)
        mtp, mfp = int((tt & mp).sum()), int((~tt & mp).sum())
        if rprec > 0 and (mtp + mfp):
            ratios.append((mtp / (mtp + mfp)) / rprec)
    q = lambda a, p: float(np.quantile(a, p)) if a else float("nan")  # noqa: E731
    return {"n_patches": len(patches), "n_debris_px": int(t.sum()),
            "auc_median": q(aucs, .5), "auc_p05": q(aucs, .05),
            "auc_p95": q(aucs, .95), "ratio_median": q(ratios, .5),
            "ratio_p05": q(ratios, .05), "ratio_p95": q(ratios, .95),
            "frac_gt_1": float(np.mean(np.array(ratios) > 1)) if ratios else np.nan}


def _name_predictions(pred):
    import pandas as pd
    v, c = np.unique(pred, return_counts=True)
    df = pd.DataFrame({"code": v,
                       "predicted_class": [D.CLASSES[int(i)] for i in v],
                       "pixels": c})
    df["share"] = df["pixels"] / df["pixels"].sum()
    return df.sort_values("pixels", ascending=False)


def _guinea_check(Xtr, ytr, ptr, mc_model, pipeline_result, out: Path):
    """
    Apply the rule and the model to Conakry, and measure the distribution
    shift that decides whether the result means anything.
    """
    import rasterio

    stem = _conakry_stem()
    comp = CACHE / f"{stem}.tif"
    wat = CACHE / f"{stem}_WATER.tif"
    if not comp.exists():
        raise FileNotFoundError(
            f"{comp} not in the raster cache; run the Conakry FDI first.")

    with rasterio.open(comp) as s:
        arr = s.read().astype("float32"); names = list(s.descriptions)
    with rasterio.open(wat) as s:
        water = np.nan_to_num(s.read(1), nan=0.0) > 0

    band = {n: arr[i] for i, n in enumerate(names)}
    fdi = band["FDI"]
    valid = water & np.isfinite(fdi) & np.isfinite(band["B8"])
    thr = float(fdi[valid].mean() + 3 * fdi[valid].std())
    flagged = valid & (fdi > thr)
    print(f"  Conakry: {int(valid.sum()):,} water px, threshold {thr:.5f}, "
          f"{int(flagged.sum()):,} flagged")

    X = np.zeros((int(flagged.sum()), len(D.BANDS)), dtype="float32")
    for i, b in enumerate(D.BANDS):
        for comp_name, mar in DOM.COMPOSITE_TO_MARIDA.items():
            if mar == b and comp_name in band:
                X[:, i] = band[comp_name][flagged]

    bands_df = DOM.compare_water_populations(Xtr, ytr, band, valid)
    feats_df = DOM.feature_shift(Xtr, ytr, X)
    # The decisive comparison: flagged pixels against flagged pixels. Measuring
    # flagged pixels against the WATER population reports a 100% out-of-range
    # rate on FDI by construction, since flagged pixels ARE the water tail.
    scn = np.array([D.normalise_id(q).rsplit("_", 1)[0] for q in ptr])
    flag_df = DOM.flagged_shift(Xtr, ytr, scn, X)
    bands_df.to_csv(out / "domain_shift_bands.csv", index=False)
    feats_df.to_csv(out / "domain_shift_features.csv", index=False)
    flag_df.to_csv(out / "domain_shift_flagged.csv", index=False)

    print("\n  water-pixel band shift (MARIDA -> Conakry):")
    for row in bands_df.itertuples():
        mark = "  <== SHIFTED" if row.shift_flag == "SHIFTED" else ""
        print(f"    {row.band:4s} median {row.marida_median:.4f} -> "
              f"{row.deploy_median:.4f}  ratio {row.ratio:5.2f}{mark}")

    cols = pipeline_result["_cols"]
    p_deb = pipeline_result["_model"].predict_proba(
        F.build(X)[cols].values)[:, list(pipeline_result["_model"].classes_).index(1)]
    named = _name_predictions(mc_model.predict(F.build(X)[cols].values))
    named.to_csv(out / "conakry_detections_named.csv", index=False)

    print(f"\n  P(debris) on Conakry's FDI detections: median "
          f"{np.median(p_deb):.3f}, mean {p_deb.mean():.3f}")
    print("  model's names for them:")
    for row in named.head(5).itertuples():
        print(f"    {row.predicted_class:24s} {row.share:6.1%}")

    print("\n  like-for-like (flagged vs flagged), in training spreads:")
    for row in flag_df.head(4).itertuples():
        print(f"    {row.feature:5s} shift {row.shift_in_training_sd:+.2f} sd"
              f"   ({row.severity})")
    print(f"    {1 - flag_df.attrs['frac_outside_any']:.0%} of Conakry's "
          f"flagged pixels lie inside MARIDA's flagged range on every feature")

    v = DOM.verdict(bands_df, flag_df, DOM.COMPOSITE_CAVEAT,
                    frac_outside=flag_df.attrs["frac_outside_any"])
    print("\n  " + v.replace(". ", ".\n  "))
    return {"threshold": thr, "n_flagged": int(flagged.sum()),
            "p_debris_median": float(np.median(p_deb)),
            "p_debris_mean": float(p_deb.mean()),
            "named": named.to_dict("records"),
            "band_shift": bands_df.to_dict("records"),
            "feature_shift": feats_df.to_dict("records"),
            "flagged_shift": flag_df.to_dict("records"),
            "frac_inside_training_support":
                1 - flag_df.attrs["frac_outside_any"],
            "verdict": v}


if __name__ == "__main__":
    raise SystemExit(main())
