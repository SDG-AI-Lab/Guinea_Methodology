#!/usr/bin/env python
"""
Does scoring a SINGLE ACQUISITION change what the classifier says?

    python 33_single_date_marida.py --list
    python 33_single_date_marida.py --dates 2024-01-11 2024-01-26
    python 33_single_date_marida.py --dates 2024-01-11 --aoi conakry

Writes to manual_qa/runs/marida/single_date.json. Fetches imagery, so unlike
scripts 31 and 32 this one is not free; each date is one AOI-sized composite
and is cached afterwards.

WHY THIS EXISTS

Section 8.13 identifies temporal compositing as the largest remaining limit
on transferring the MARIDA classifier to Guinea, and it is a limit of a
particular kind: it biases the answer in a KNOWN DIRECTION.

Guinea's operational layer composites a monthly median. MARIDA annotates
single acquisitions. A median over a month keeps whatever is present on most
dates and discards whatever is present on few -- so persistent features
(turbid plumes, shallow bathymetry, moored vessels, permanent wave patterns)
survive compositing, while transient floating material does not. Debris is
transient. The composite therefore under-represents exactly the class we care
about and over-represents the confusers, by construction rather than by
accident.

That predicts something testable: scoring a single date should raise the
debris share relative to the composite. If it does, compositing is confirmed
as a real cost and the operational recommendation follows. If it does not,
the low debris share at Conakry is a property of the water rather than of
the processing, and the recommendation in section 9.3.0 should be dropped.

Either outcome is informative, which is why this is worth a fetch.

WHAT IT CANNOT SETTLE

Nothing here validates the classifier on Guinean data -- there are still no
Guinean labels. This compares two PROCESSING CHOICES under one model. A
difference tells us the composite is costing us something; it does not tell
us the single-date answer is correct.
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

OUT = HERE / "manual_qa" / "runs" / "marida"


def _load_env() -> None:
    """The scripts are normally launched from the notebook, which has already
    loaded .env. Running this one directly must not fail for want of a
    project id, so the same file is read here if the variables are absent."""
    env = HERE.parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def list_dates(aoi_key: str, start: str, end: str):
    import collections
    import datetime

    import ee

    from plasticwaste_tool import api_connector as api
    api.initialize_gee()
    aoi = api.get_aoi_geometry(aoi_key)
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(aoi).filterDate(start, end))
    dt = col.aggregate_array("system:time_start").getInfo()
    cl = col.aggregate_array("CLOUDY_PIXEL_PERCENTAGE").getInfo()
    agg = collections.defaultdict(list)
    for t, c in zip(dt, cl):
        agg[datetime.datetime.utcfromtimestamp(t / 1000)
            .strftime("%Y-%m-%d")].append(c)
    print(f"{aoi_key}: {len(agg)} acquisition dates in {start}..{end}")
    print("date          cloud%")
    for d in sorted(agg):
        mark = "  <- usable" if min(agg[d]) < 10 else ""
        print(f"  {d}   {min(agg[d]):5.1f}{mark}")
    return {d: min(v) for d, v in agg.items()}


def score_one(aoi_key: str, date: str, clf, mc_clf, cols):
    """
    Fetch one acquisition, apply the deployed rule, score what it flags.

    A one-day window is used rather than a single asset id so that the same
    api_connector path serves both this and the operational run: whatever
    masking and scaling the pipeline applies to a composite is applied here
    too, and the only thing that differs is the length of the window.
    """
    import datetime

    from plasticwaste_tool import api_connector as api
    from plasticwaste_tool import data_loader as DL
    from plasticwaste_tool.marida_tool import data as D
    from plasticwaste_tool.marida_tool import domain as DOM
    from plasticwaste_tool.marida_tool import features as F

    nxt = (datetime.date.fromisoformat(date)
           + datetime.timedelta(days=1)).isoformat()

    api.initialize_gee()
    aoi = api.get_aoi_geometry(aoi_key)
    gee = api.build_fdi_pipeline(date, nxt, aoi)
    prefix = f"{aoi_key}_{api.aoi_cache_token(aoi_key)}_{date}_{nxt}"
    path = api.download_composite_bands(gee, aoi, prefix)
    bands, profile = DL.read_composite(path)

    fdi = bands.get("FDI")
    water = bands.get("WATER")
    if fdi is None:
        from plasticwaste_tool.fdi_tool import spectral as SP
        fdi = SP.compute_fdi(bands)
    is_water = np.nan_to_num(water, nan=0.0) > 0 if water is not None \
        else np.ones_like(fdi, bool)
    valid = is_water & np.isfinite(fdi) & np.isfinite(bands["B8"])
    if valid.sum() < 1000:
        return {"date": date, "error": f"only {int(valid.sum())} water px"}

    thr = float(fdi[valid].mean() + 3 * fdi[valid].std())
    flagged = valid & (fdi > thr)
    if not flagged.any():
        return {"date": date, "threshold": thr, "n_water": int(valid.sum()),
                "n_flagged": 0}

    idx = np.flatnonzero(flagged.ravel())
    X = np.zeros((idx.size, len(D.BANDS)), dtype="float32")
    for i, b in enumerate(D.BANDS):
        for comp, mar in DOM.COMPOSITE_TO_MARIDA.items():
            if mar == b and comp in bands:
                X[:, i] = np.asarray(bands[comp]).ravel()[idx]

    feats = F.build(X)[cols].values
    p = clf.predict_proba(feats)[:, list(clf.classes_).index(1)]
    named = mc_clf.predict(feats)
    v, c = np.unique(named, return_counts=True)
    comp = {D.CLASSES[int(i)]: int(n) for i, n in zip(v, c)}

    return {
        "date": date, "aoi": aoi_key, "threshold": thr,
        "n_water": int(valid.sum()), "n_flagged": int(flagged.sum()),
        "p_debris_median": float(np.median(p)),
        "p_debris_mean": float(p.mean()),
        "debris_share_at_cut": float((p >= 0.5).mean()),
        "named": comp,
        "named_share": {k: n / sum(comp.values()) for k, n in comp.items()},
        "raster": str(path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aoi", default="conakry")
    ap.add_argument("--dates", nargs="*", default=[])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--start", default="2023-11-01")
    ap.add_argument("--end", default="2024-03-01")
    ap.add_argument("--trees", type=int, default=300)
    args = ap.parse_args()

    _load_env()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.list or not args.dates:
        list_dates(args.aoi, args.start, args.end)
        if not args.dates:
            print("\nPass --dates YYYY-MM-DD [...] to score them.")
            return 0

    from sklearn.ensemble import RandomForestClassifier

    from plasticwaste_tool.marida_tool import data as D
    from plasticwaste_tool.marida_tool import domain as DOM
    from plasticwaste_tool.marida_tool import features as F
    from plasticwaste_tool.marida_tool import model as M

    print("training on MARIDA (pipeline bands only) ...")
    Xtr, ytr, _ = D.load_pixels("train")
    cols = DOM.PIPELINE_FEATURES
    Ftr = F.build(Xtr)[cols].values

    def _rf(target):
        c = RandomForestClassifier(
            n_estimators=args.trees, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=M.SEED)
        c.fit(Ftr, target)
        return c

    clf = _rf((ytr == D.TARGET).astype(int))
    mc = _rf(ytr)

    rows = []
    for date in args.dates:
        print(f"\n--- {args.aoi} {date} ---", flush=True)
        try:
            r = score_one(args.aoi, date, clf, mc, cols)
        except Exception as exc:                      # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            rows.append({"date": date, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append(r)
        if "error" in r:
            print(f"  {r['error']}")
            continue
        print(f"  {r['n_water']:,} water px, threshold {r['threshold']:.5f}, "
              f"{r['n_flagged']:,} flagged")
        print(f"  P(debris) median {r['p_debris_median']:.3f}, "
              f"share above 0.5: {r['debris_share_at_cut']:.1%}")
        for k, v in sorted(r["named_share"].items(),
                           key=lambda kv: -kv[1])[:5]:
            print(f"    {k:24s} {v:6.1%}")

    path = OUT / "single_date.json"
    payload = {"aoi": args.aoi, "trees": args.trees, "results": rows}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")

    ok = [r for r in rows if "error" not in r and r.get("n_flagged")]
    if len(ok) >= 1:
        print("\nCompare against the MONTHLY COMPOSITE for the same window "
              "(runs/marida/summary.json, transfer.p_debris_median). If the "
              "single-date debris share is materially higher, compositing is "
              "costing detections and section 9.3.0's recommendation stands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
