#!/usr/bin/env python
"""
How much of the FDI detection set is a water-mask EDGE artefact?

    python 36_edge_distance.py
    python 36_edge_distance.py --products composite single

Writes manual_qa/runs/marida/edge_distance.json.

WHY THIS EXISTS

Section 7.1.14.5 reports a human annotation of 150 Conakry detections, and
about half were called shoreline. That is one person's judgement on a sample,
and it inherits every weakness a small annotated sample has.

This measures the same thing without a human and without a sample. The water
mask is ESA WorldCover class 80; a Euclidean distance transform gives, for
every water pixel, how far it lies from the nearest non-water pixel. The
question is then arithmetic: are flagged pixels closer to that boundary than
water pixels in general?

The mechanism is not in doubt -- a 20 m pixel straddling the land/water
boundary contains land reflectance, land is bright in the near infrared, and
the Floating Debris Index is built on near-infrared excess over a water
baseline, so boundary pixels must sit in the FDI tail. What is in doubt is
the MAGNITUDE, and that is what decides whether this is a footnote or the
dominant failure mode of the layer.

WHAT IT DOES NOT SHOW

A detection near the boundary is not automatically wrong. Floating material
accumulates against shorelines, and in a tidal estuary it accumulates there
preferentially. Enrichment at the edge is consistent both with mixed-pixel
contamination and with genuine strandline debris, and the distance transform
cannot separate them. What it establishes is that the two are confounded in
the present product, which is why the recommendation in section 9.3.0 is to
FLAG boundary-adjacent detections rather than to drop them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from plasticwaste_tool import config as pwcfg              # noqa: E402

# The raster cache the FDI pipeline writes to: <APP_SCRATCH_DIR>/cache/rasters
# (api_connector._cache), from config rather than a hard-coded drive, so the
# script runs on a workstation and in the backend container alike.
CACHE = Path(os.environ.get("PW_RASTER_CACHE")
             or Path(pwcfg.PROCESSING_CONFIG["cache_dir"]) / "rasters")
MQ = HERE / "manual_qa"
OUT = MQ / "runs" / "marida"

# label -> the window each product covers. Both rasters carry a WATER band.
PRODUCTS = {
    "composite": ("2024-01-01", "2024-02-01"),   # monthly, January 2024
    "single": ("2024-01-11", "2024-01-12"),      # one acquisition (script 33)
}


def _stem(label):
    """Cache name of a product, built exactly as the pipeline and script 33
    name it (AOI + FDI formula token + window), so it always refers to the
    raster of the current FDI definition."""
    from plasticwaste_tool import api_connector as api
    start, end = PRODUCTS[label]
    return f"conakry_{api.aoi_cache_token('conakry')}_{start}_{end}_composite"

RINGS = (1, 2, 3, 5)


def _read(stem: str):
    """Read FDI, WATER and B8, preferring the per-band files and falling back
    to the multiband stack. The single-date fetch writes both; the operational
    composite writes only the stack, so neither path can be assumed."""
    import rasterio

    out = {}
    for band in ("FDI", "WATER", "B8"):
        p = CACHE / f"{stem}_{band}.tif"
        if p.exists():
            with rasterio.open(p) as s:
                out[band] = s.read(1).astype("float64")
                out["transform"] = s.transform
    missing = [b for b in ("FDI", "WATER", "B8") if b not in out]
    if missing:
        p = CACHE / f"{stem}.tif"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (needed for {missing})")
        with rasterio.open(p) as s:
            names = list(s.descriptions)
            for b in missing:
                if b not in names:
                    raise KeyError(f"{p.name} has no {b} band: {names}")
                out[b] = s.read(names.index(b) + 1).astype("float64")
            out["transform"] = s.transform
    return out


def analyse(label: str, stem: str) -> dict:
    from scipy import ndimage as ndi

    d = _read(stem)
    tr = d["transform"]
    px_m = abs(tr.a) * 111320.0        # degrees -> metres at the equator

    water = np.nan_to_num(d["WATER"], nan=0.0) > 0
    fdi, b8 = d["FDI"], d["B8"]
    valid = water & np.isfinite(fdi) & np.isfinite(b8)
    thr = float(fdi[valid].mean() + 3 * fdi[valid].std())
    flagged = valid & (fdi > thr)

    # Distance from each water pixel to the nearest non-water pixel. Computed
    # on the mask itself rather than on `valid`, so that cloud or nodata holes
    # inside the water body are not mistaken for coastline.
    dist = ndi.distance_transform_edt(water)

    print(f"\n=== {label}: {stem}")
    print(f"  {int(valid.sum()):,} water px at {px_m:.0f} m, "
          f"threshold {thr:.5f}, {int(flagged.sum()):,} flagged")
    if not flagged.any():
        return {"label": label, "stem": stem, "n_flagged": 0}

    rings = {}
    print(f"  {'within':>14s} {'flagged':>9s} {'of water':>9s} "
          f"{'enrichment':>11s}")
    for r in RINGS:
        f_in = float((flagged & (dist <= r)).sum()) / flagged.sum()
        w_in = float((valid & (dist <= r)).sum()) / valid.sum()
        enr = f_in / w_in if w_in else float("nan")
        rings[str(r)] = {"metres": r * px_m, "flagged_share": f_in,
                         "water_share": w_in, "enrichment": enr}
        print(f"  {r:2d} px ({r * px_m:3.0f} m) {f_in:8.1%} {w_in:8.1%} "
              f"{enr:9.0f}x")

    interior = flagged & (dist > 3)
    return {
        "label": label, "stem": stem, "pixel_m": px_m,
        "threshold": thr, "n_water": int(valid.sum()),
        "n_flagged": int(flagged.sum()),
        "rings": rings,
        "n_interior_gt3px": int(interior.sum()),
        "interior_share": float(interior.sum() / flagged.sum()),
    }


def classify_by_ring(stem: str, rings=(2, 3)) -> dict:
    """
    Does the classifier read boundary-adjacent detections differently from
    interior ones?

    This is what decides whether flagging them is worth anything. Geometry
    alone shows only that edge pixels are over-represented; it cannot say
    whether they are worse. If the interior set carries a materially higher
    debris probability, a distance-to-boundary field lets a user recover it.
    If both sets read the same, the field is bookkeeping and should be sold
    as nothing more.

    The model is the seven-band pipeline classifier from section 7.1.14.2,
    retrained here rather than loaded so that this script does not depend on
    the order in which the notebook's cells were run.
    """
    from scipy import ndimage as ndi
    from sklearn.ensemble import RandomForestClassifier

    from plasticwaste_tool.marida_tool import data as D
    from plasticwaste_tool.marida_tool import domain as DOM
    from plasticwaste_tool.marida_tool import features as F
    from plasticwaste_tool.marida_tool import model as M

    print("\n=== classifier reading, by distance to the mask boundary")
    print("  training the seven-band pipeline model on MARIDA ...", flush=True)
    Xtr, ytr, _ = D.load_pixels("train")
    cols = DOM.PIPELINE_FEATURES
    Ftr = F.build(Xtr)[cols].values

    def _rf(target):
        c = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=M.SEED)
        c.fit(Ftr, target)
        return c

    clf = _rf((ytr == D.TARGET).astype(int))
    mc = _rf(ytr)

    import rasterio
    with rasterio.open(CACHE / f"{stem}.tif") as s:
        names = list(s.descriptions)
        arr = s.read().astype("float64")
    band = {n: arr[i] for i, n in enumerate(names)}

    water = np.nan_to_num(band["WATER"], nan=0.0) > 0
    fdi = band["FDI"]
    valid = water & np.isfinite(fdi) & np.isfinite(band["B8"])
    thr = float(fdi[valid].mean() + 3 * fdi[valid].std())
    flagged = valid & (fdi > thr)
    dist = ndi.distance_transform_edt(water)

    idx = np.flatnonzero(flagged.ravel())
    X = np.zeros((idx.size, len(D.BANDS)), dtype="float32")
    for i, b in enumerate(D.BANDS):
        for comp, mar in DOM.COMPOSITE_TO_MARIDA.items():
            if mar == b and comp in band:
                X[:, i] = band[comp].ravel()[idx]
    feats = F.build(X)[cols].values
    p = clf.predict_proba(feats)[:, list(clf.classes_).index(1)]
    named = mc.predict(feats)
    d_flat = dist.ravel()[idx]

    out = {}
    for r in rings:
        near = d_flat <= r
        for tag, sel in (("edge", near), ("interior", ~near)):
            if not sel.any():
                continue
            v, c = np.unique(named[sel], return_counts=True)
            top = sorted(zip(v, c), key=lambda kv: -kv[1])[:3]
            out[f"{tag}_le{r}px" if tag == "edge" else f"{tag}_gt{r}px"] = {
                "n": int(sel.sum()),
                "p_debris_mean": float(p[sel].mean()),
                "p_debris_median": float(np.median(p[sel])),
                "share_above_0.5": float((p[sel] >= 0.5).mean()),
                "top_classes": [[D.CLASSES[int(i)], int(n) / int(sel.sum())]
                                for i, n in top],
            }
            print(f"  {tag:9s} (d {'<=' if tag == 'edge' else '>'} {r} px): "
                  f"n={int(sel.sum()):5,}  mean P(debris)={p[sel].mean():.4f}  "
                  f"top: " + ", ".join(
                      f"{D.CLASSES[int(i)]} {n / sel.sum():.0%}"
                      for i, n in top))
    return out


def check_annotation() -> dict:
    """
    Does the human `shoreline` label agree with the geometry?

    Two things are checked, and they answer different questions. Whether the
    label tracks distance to the boundary is a check on the ANNOTATION.
    Whether the label tracks longitude is a check on the QUEUE: script 34
    emits patches in raster-scan order, so presentation order and geography
    are perfectly confounded, and a label that separates on position cannot be
    distinguished from a label that drifted during the session.
    """
    import pandas as pd
    from scipy import ndimage as ndi
    from scipy import stats
    from sklearn.metrics import roc_auc_score

    csv = MQ / "water_annotations.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    # Only labels made for the current queue (see 35_score_water_annotation).
    qpath = MQ / "water_queue.json"
    if not qpath.exists():
        return {"note": f"{csv.name} predates queue tracking; not compared"}
    q = json.loads(qpath.read_text(encoding="utf-8"))
    if not q["items"]:
        return {"note": f"current queue ({q['source']}) has no published "
                        f"detections; nothing to compare"}
    pos = {i["patch_id"]: (round(i["lat"], 5), round(i["lon"], 5)) for i in q["items"]}
    if any(pos.get(r.patch_id) != (round(r.lat, 5), round(r.lon, 5))
           for r in df.itertuples()):
        return {"note": f"{csv.name} was labelled for an earlier queue than "
                        f"{q['stamp']}; not compared"}
    df = df[df["category"].notna()]
    if df["category"].nunique() < 2:
        return {"note": "only one category used; nothing to compare"}

    d = _read(_stem("single"))
    tr = d["transform"]
    water = np.nan_to_num(d["WATER"], nan=0.0) > 0
    dist = ndi.distance_transform_edt(water)
    inv = ~tr

    vals = []
    for _, r in df.iterrows():
        # Script 34 writes the pixel CENTRE, transform * (cx + .5, cy + .5).
        # Inverting without removing that half pixel lands on a neighbour and
        # silently randomises every distance below. Verified by confirming
        # that all single-pixel patches recover their recorded FDI exactly.
        c, rr = inv * (r["lon"], r["lat"])
        ri, ci = int(round(rr - 0.5)), int(round(c - 0.5))
        vals.append(float(dist[ri, ci])
                    if 0 <= ri < dist.shape[0] and 0 <= ci < dist.shape[1]
                    else float("nan"))
    df["edge_dist_px"] = vals
    df["queue_idx"] = df["patch_id"].str.extract(r"(\d+)").astype(int)

    y = (df["category"] == "shoreline").astype(int)
    if y.nunique() < 2:
        return {"note": "no shoreline label present"}
    ok = df["edge_dist_px"].notna()
    auc_d = roc_auc_score(y[ok], -df.loc[ok, "edge_dist_px"])
    auc_lon = roc_auc_score(y, df["lon"])
    rho = stats.spearmanr(df["queue_idx"], df["lat"])

    print("\n=== does the human label agree with the geometry?")
    print(df.groupby("category")["edge_dist_px"]
          .describe()[["count", "mean", "50%", "max"]].to_string())
    print(f"\n  edge distance reproducing the label : AUC {auc_d:.3f}")
    print(f"  longitude reproducing the label     : AUC {auc_lon:.3f}")
    print(f"  queue order vs latitude             : rho {rho.statistic:+.5f}")
    if abs(rho.statistic) > 0.95:
        print("  -> presentation order IS geography. A label that separates "
              "on position is equally consistent with a real spatial split "
              "and with drift during the session; this queue cannot tell them "
              "apart. Shuffle the order before re-annotating.")

    return {
        "auc_edge_distance_vs_label": float(auc_d),
        "auc_longitude_vs_label": float(auc_lon),
        "spearman_queue_vs_latitude": float(rho.statistic),
        "median_edge_dist_px": {k: float(v) for k, v in
                                df.groupby("category")["edge_dist_px"]
                                .median().items()},
        "max_edge_dist_px": {k: float(v) for k, v in
                             df.groupby("category")["edge_dist_px"]
                             .max().items()},
        "ordering_confound": bool(abs(rho.statistic) > 0.95),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", nargs="*", default=list(PRODUCTS),
                    choices=list(PRODUCTS))
    ap.add_argument("--classify", action="store_true",
                    help="also score edge against interior detections with "
                         "the MARIDA classifier (trains a model; ~2 min)")
    args = ap.parse_args()

    res = {}
    for label in args.products:
        try:
            res[label] = analyse(label, _stem(label))
        except (FileNotFoundError, KeyError) as exc:
            print(f"{label}: skipped -- {exc}")

    rings = {}
    if args.classify:
        try:
            rings = classify_by_ring(_stem("composite"))
        except Exception as exc:                          # noqa: BLE001
            print(f"classification skipped -- {type(exc).__name__}: {exc}")

    ann = check_annotation()
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "products": res, "annotation_geometry": ann,
        "classifier_by_ring": rings,
        "note": ("Edge enrichment does not by itself make a detection wrong: "
                 "floating material accumulates against shorelines. It shows "
                 "the two are confounded in this product, which is why the "
                 "recommendation is to flag boundary-adjacent detections "
                 "rather than to drop them."),
    }
    p = OUT / "edge_distance.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
