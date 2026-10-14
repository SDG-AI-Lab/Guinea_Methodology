"""Copied from 08_point_model.py and renamed so it can be imported.

The numbered original stays in place for the notebooks; this copy is
what the package and the backend import. Runtime importlib loads of
sibling scripts have been rewritten as relative imports.
Regenerate with scratchpad/build_package.py -- do not edit by hand
until the originals are retired.
"""

"""
08 — Point-centred dumpsite model: build samples, fetch imagery, evaluate.

WHY THIS SCRIPT EXISTS (the result that forced it)
--------------------------------------------------
Scoring 250 m grid cells produced a model that did not detect waste at all. It
detected settlement. Measured on 600 randomly-sampled labelled cells, the mean
out-of-fold score for *residential negatives* was 0.626 while *real dumpsites*
scored 0.618 — ordinary built-up land outranked actual dumps. The model reached
52% precision only because dumps happen to sit in settlements.

The cause is target dilution: a dump spanning tens of metres occupies ~1-2% of
a 250 m frame, so there is nothing for the classifier to latch onto except
context. The fix is to crop a SMALL window CENTRED ON THE DUMP rather than a
large cell centred on nothing. That is possible because the labeller recorded
explicit lat/lon pins for every positive — and often several per cell, since one
250 m cell can contain several distinct dumps.

Measured effect (Esri z17, ImageNet features, nested CV, spatial blocks):

    footprint   ROC-AUC   avg prec   dumps   residential   verdict
    250 m cell    0.843      0.410   0.618         0.626   settlement detector
     80 m point   0.887      0.619   0.669         0.576   WASTE detector
    150 m point   0.881      0.605   0.660         0.555   WASTE detector

The ordering inverts at 80 m: that is the whole point of the change.

A SEPARATE QUESTION: RESOLUTION IS NOT FOOTPRINT
------------------------------------------------
An earlier controlled ablation degraded Conakry imagery from 0.29 to 1.18 m/px
and found no effect (dAUC -0.011, 95% CI [-0.053, +0.028], P=0.70). Resolution
does not matter; footprint does. The two were repeatedly conflated, including by
this project, and they are not the same thing.

What resolution DOES constrain is how few pixels a small footprint contains.
Outside Conakry, Esri caps at z17 (~1.17 m/px), so an 80 m window is only ~68
real pixels upsampled to the model's 800 px input. `--source google` exists to
test whether more pixels inside that same 80 m window helps; Google serves the
same Maxar/Vantor imagery at deeper zoom. Use `--budget` to cap billable calls.

READING THE NUMBERS HONESTLY
----------------------------
The evaluation set's prevalence (~17%) is a SAMPLING ARTEFACT: negatives are
drawn one per 250 m cell, positives one per dump point. A real sliding scan at
80 m has roughly 2% prevalence, and precision falls accordingly — `evaluate`
prints both, because quoting the evaluation-set precision as a deployment
figure would overstate the model by a factor of ~4.

Usage
-----
    python 08_point_model.py build-samples
    python 08_point_model.py fetch --tile-m 80 --source esri
    python 08_point_model.py fetch --tile-m 80 --source google --zoom 19 --budget 700
    python 08_point_model.py evaluate --tile-m 80 --source esri
    python 08_point_model.py sheets   --tile-m 80 --source esri
"""

import argparse
import importlib.util
import json
import logging
import math
import re
from pathlib import Path

import torch  # noqa: F401  — must precede sklearn/GDAL on Windows (WinError 1114)
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
# This file exists twice: as the numbered original in plasticwaste_tool/ and as
# the importable copy in plasticwaste_tool/dumpsite_tool/. Anchor everything on
# the tool root so both copies read the same labels and write the same models
# directory -- otherwise the copy freezes a model into a path nothing loads.
TOOLROOT = HERE.parent if HERE.name == "dumpsite_tool" else HERE
MODELS_DIR = TOOLROOT / "dumpsite_tool" / "models"
MQ = TOOLROOT / "manual_qa"
SAMPLE_GEOJSON = MQ / "regions" / "national_random_sample.geojson"
SAMPLES_CSV = MQ / "_point_samples.csv"
CS = [0.001, 0.01, 0.1, 1.0, 10.0]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiles_dir(tile_m, source):
    suffix = "" if source == "esri" else "_" + source
    return MQ / f"tiles_pt{int(tile_m)}{suffix}"


# ---------------------------------------------------------------------------
def build_samples():
    """One row per dump point (positive) and per negative cell centre.

    Positives are expanded per PIN, not per cell: the labeller recorded several
    coordinates in cells containing several dumps, so 55 positive cells yield
    111 positive samples. Those extra positives cost no additional labelling and
    are the main reason this approach is affordable.
    """
    feats = json.loads(SAMPLE_GEOJSON.read_text(encoding="utf-8"))["features"]
    rows = []
    for f in feats:
        p = f["properties"]
        lab = (p.get("label") or "").strip()
        # A GeoJSON polygon ring is CLOSED: its last point repeats the first.
        # Averaging all of them double-counts that corner and pulls the centre
        # ~35m diagonally for a 250m cell, which silently mis-centres every
        # negative crop. Drop the closing point before averaging.
        ring = f["geometry"]["coordinates"][0]
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        clat = sum(q[1] for q in ring) / len(ring)
        clon = sum(q[0] for q in ring) / len(ring)
        if lab == "1":
            la = re.findall(r"-?\d+\.\d+", str(p.get("lat") or ""))
            lo = re.findall(r"-?\d+\.\d+", str(p.get("long") or ""))
            pts = list(zip(la, lo)) or [(clat, clon)]
            for j, (a, b) in enumerate(pts):
                rows.append({"sid": f"{p['tile_id']}_p{j}", "lat": float(a),
                             "lon": float(b), "label": 1,
                             "region": p["region"], "cell": p["tile_id"],
                             "note": (p.get("notes") or "").lower()})
        elif lab == "0":
            rows.append({"sid": p["tile_id"], "lat": clat, "lon": clon,
                         "label": 0, "region": p["region"],
                         "cell": p["tile_id"],
                         "note": (p.get("notes") or "").lower()})

    df = pd.DataFrame(rows)
    df.to_csv(SAMPLES_CSV, index=False)
    logger.info("%d samples: %d positive (from %d cells), %d negative -> %s",
                len(df), int(df.label.sum()),
                df[df.label == 1].cell.nunique(),
                int((df.label == 0).sum()), SAMPLES_CSV.name)
    return df


def _bounds(df, tile_m):
    out = []
    for r in df.itertuples():
        hlat = (tile_m / 2) / 110_574.0
        hlon = (tile_m / 2) / max(111_320.0 * math.cos(math.radians(r.lat)), 1.0)
        out.append({"tile_id": r.sid,
                    "minx": r.lon - hlon, "miny": r.lat - hlat,
                    "maxx": r.lon + hlon, "maxy": r.lat + hlat})
    return pd.DataFrame(out)


def fetch(tile_m, source, zoom, budget):
    from . import inference as gi
    df = pd.read_csv(SAMPLES_CSV)
    out = tiles_dir(tile_m, source)
    bounds = _bounds(df, tile_m)

    if source == "esri":
        lat = float(df.lat.median())
        mpp = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
        logger.info("Esri z%d ~%.2f m/px -> a %.0fm window is ~%d real pixels",
                    zoom, mpp, tile_m, int(tile_m / mpp))
        gi.fetch_tiles_esri(bounds, out, size_px=800, zoom=zoom)
    else:
        n_new = sum(1 for t in bounds.tile_id if not (out / (t + ".png")).exists())
        logger.info("Google Static Maps z%d: %d tiles, %d not cached, "
                    "budget %s billable requests", zoom, len(bounds), n_new,
                    budget if budget is not None else "uncapped")
        gi.fetch_tiles_google(bounds, out, size_px=800, zoom=zoom, budget=budget)
    logger.info("images in %s: %d", out.name, len(list(out.glob("*.png"))))


# ---------------------------------------------------------------------------
def _nested_cv(X, y, groups):
    """Outer folds score, inner folds choose C.

    C must be tuned per representation: a single fixed value penalises 2048
    dimensions far harder than 19, which once made a colour histogram appear to
    beat ImageNet. Tuning on the reported folds would instead flatter whichever
    representation has the most capacity, so the choice is made strictly inside
    the training part of each outer split.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros(len(y))
    k = min(5, len(np.unique(groups)), int(min(np.bincount(y))))
    for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=True,
                                       random_state=0).split(X, y, groups):
        best_c, best_s = CS[0], -1.0
        ik = min(4, len(np.unique(groups[tr])), int(min(np.bincount(y[tr]))))
        for C in CS:
            s = np.zeros(len(tr))
            for a, b in StratifiedGroupKFold(
                    n_splits=ik, shuffle=True,
                    random_state=1).split(X[tr], y[tr], groups[tr]):
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000,
                                                     class_weight="balanced",
                                                     C=C))
                m.fit(X[tr][a], y[tr][a])
                s[b] = m.predict_proba(X[tr][b])[:, 1]
            v = average_precision_score(y[tr], s)
            if v > best_s:
                best_s, best_c = v, C
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000,
                                             class_weight="balanced", C=best_c))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def evaluate(tile_m, source, zoom):
    from sklearn.metrics import roc_auc_score, average_precision_score
    from . import feature_probe as fp
    from . import features as bm

    df = pd.read_csv(SAMPLES_CSV)
    d = tiles_dir(tile_m, source)
    df["path"] = [str(d / (s + ".png")) for s in df.sid]
    have = df[[Path(p).exists() for p in df.path]].reset_index(drop=True)
    if len(have) < len(df):
        logger.warning("%d/%d images present; evaluating on those only",
                       len(have), len(df))

    y = have.label.to_numpy()
    groups, _ = fp.spatial_groups(have.lat, have.lon, 1000.0)
    X = bm.imagenet_features(have.path.tolist())
    oof = _nested_cv(X, y, groups)

    np.savez(MQ / ("_oof_pt%d_%s.npz" % (int(tile_m), source)),
             y=y, groups=groups, score=oof, sid=have.sid.to_numpy())

    ev_prev = float(y.mean())
    wpc = (250.0 / tile_m) ** 2                 # windows per 250 m cell
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    dep_prev = n_pos / (n_neg * wpc + n_pos)

    res = (y == 0) & have.note.fillna("").str.contains("residential").to_numpy()
    md = oof[y == 1].mean()
    mr = oof[res].mean() if res.any() else float("nan")

    print("\n" + "=" * 78)
    print("%.0fm point-centred crops from %s (z%d) - n=%d, %d positive"
          % (tile_m, source, zoom, len(y), n_pos))
    print("=" * 78)
    print("  ROC-AUC          %.3f" % roc_auc_score(y, oof))
    print("  avg precision    %.3f  (no-skill = %.3f)"
          % (average_precision_score(y, oof), ev_prev))
    print("\n  IS IT DETECTING WASTE, OR JUST SETTLEMENT?")
    print("    mean score, real dumpsites        %.3f" % md)
    print("    mean score, residential negatives %.3f" % mr)
    print("    -> %s" % ("WASTE detector" if md > mr else "SETTLEMENT detector"))

    print("\n  evaluation prevalence %.3f is a SAMPLING ARTEFACT (negatives per "
          "cell,\n  positives per point). A sliding %.0fm scan has ~%.3f "
          "prevalence." % (ev_prev, tile_m, dep_prev))
    print("\n  %7s %7s %7s %10s %12s"
          % ("thresh", "recall", "FPR", "prec@eval", "prec@deploy"))
    print("  " + "-" * 48)
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        pred = oof >= t
        tp = int((pred & (y == 1)).sum())
        fpv = int((pred & (y == 0)).sum())
        tpr, fpr = tp / max(n_pos, 1), fpv / max(n_neg, 1)
        pe = tp / max(tp + fpv, 1)
        pdep = (dep_prev * tpr) / max(dep_prev * tpr + (1 - dep_prev) * fpr, 1e-12)
        print("  %7.2f %7.3f %7.3f %10.3f %12.3f" % (t, tpr, fpr, pe, pdep))
    print("\n  'prec@deploy' is the number to quote. Quoting 'prec@eval' would "
          "overstate\n  the model roughly fourfold.")

    # Persist the run so an archived result can be compared later without
    # re-running anything. Per-SITE precision is recorded alongside per-window
    # because a cell with several dump pins occupies several top-40 slots, and
    # the per-site figure is what a reviewer actually experiences.
    cells = have.cell.to_numpy()
    order = np.argsort(-oof)
    top = order[:40]
    seen, per_site = set(), []
    for i in order:
        if cells[i] not in seen:
            seen.add(cells[i])
            per_site.append(int(y[i]))
    rec = {
        "source": source, "tile_m": tile_m, "zoom": zoom,
        "n": int(len(y)), "n_pos": n_pos, "n_neg": n_neg,
        "eval_prevalence": ev_prev, "deploy_prevalence": float(dep_prev),
        "roc_auc": float(roc_auc_score(y, oof)),
        "avg_precision": float(average_precision_score(y, oof)),
        "mean_score_dumps": float(md), "mean_score_residential": float(mr),
        "detector": "waste" if md > mr else "settlement",
        "top40_precision_per_window": float((y[top] == 1).mean()),
        "top40_precision_per_site": float(np.mean(per_site[:40])),
        "operating_points": [
            {"threshold": t,
             "recall": float((oof >= t)[y == 1].mean()),
             "fpr": float((oof >= t)[y == 0].mean()),
             "precision_eval": (float(y[oof >= t].mean())
                                if (oof >= t).any() else 0.0)}
            for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)],
    }
    mp = MQ / ("_metrics_pt%d_%s.json" % (int(tile_m), source))
    mp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("  metrics -> %s" % mp.name)


def sheets(tile_m, source):
    """Contact sheets of top-scoring and lowest-scoring-positive crops.

    Metrics say how well the model ranks; only the images say WHAT it confuses,
    which is how the 250 m settlement-proxy failure was caught in the first place.
    """
    from PIL import Image, ImageDraw
    z = np.load(MQ / ("_oof_pt%d_%s.npz" % (int(tile_m), source)),
                allow_pickle=True)
    y, s, sid = z["y"], z["score"], z["sid"]
    df = pd.read_csv(SAMPLES_CSV).set_index("sid")
    d = tiles_dir(tile_m, source)
    TH, COLS = 128, 8

    def sheet(idx, title, dest):
        rows = (len(idx) + COLS - 1) // COLS
        out = Image.new("RGB", (COLS * TH, rows * (TH + 18) + 22), (18, 18, 18))
        ImageDraw.Draw(out).text((6, 5), title, fill=(255, 255, 255))
        for k, i in enumerate(idx):
            p = d / (str(sid[i]) + ".png")
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB").resize((TH, TH), Image.LANCZOS)
            x, yy = (k % COLS) * TH, 22 + (k // COLS) * (TH + 18)
            out.paste(im, (x, yy))
            ok = y[i] == 1
            dr = ImageDraw.Draw(out)
            dr.rectangle([x, yy, x + TH - 1, yy + TH - 1],
                         outline=(60, 200, 90) if ok else (230, 70, 70), width=3)
            dr.rectangle([x, yy + TH, x + TH, yy + TH + 17], fill=(0, 0, 0))
            reg = str(df.region.get(sid[i], ""))[:6]
            dr.text((x + 3, yy + TH + 3),
                    "%.2f %s %s" % (s[i], "DUMP" if ok else "not", reg),
                    fill=(150, 255, 170) if ok else (255, 160, 160))
        out.save(dest)
        return dest

    top = np.argsort(-s)[:40]
    prec = float((y[top] == 1).mean())
    a = sheet(top, "TOP 40 - %.0fm %s - green=real, red=false (precision %.0f%%)"
              % (tile_m, source, prec * 100),
              MQ / "regions" / ("results_pt%d_%s_flagged.png"
                                % (int(tile_m), source)))
    pos = np.flatnonzero(y == 1)
    b = sheet(pos[np.argsort(s[pos])][:24],
              "MISSED - lowest-scoring real dumpsites - %.0fm %s"
              % (tile_m, source),
              MQ / "regions" / ("results_pt%d_%s_missed.png"
                                % (int(tile_m), source)))
    print("top-40 precision %.0f%%\n  %s\n  %s" % (prec * 100, a, b))



def archive(tile_m, source, tag=None):
    """Snapshot a completed run so its results stay traceable.

    Esri and Google are carried as parallel tracks: if Google does not
    justify its per-image cost and licensing questions, the Esri result must
    still be citable in the methodology without re-running it. Files are
    COPIED, not moved, so the live pipeline keeps working. Imagery is
    deliberately not archived -- it is large and reproducible from the
    samples CSV via the `fetch` subcommand.
    """
    import shutil
    from datetime import datetime

    stamp = tag or datetime.now().strftime("%Y%m%d")
    dest = MQ / "runs" / ("%s_%dm_%s" % (source, int(tile_m), stamp))
    dest.mkdir(parents=True, exist_ok=True)

    wanted = [
        MQ / ("_oof_pt%d_%s.npz" % (int(tile_m), source)),
        MQ / ("_metrics_pt%d_%s.json" % (int(tile_m), source)),
        SAMPLES_CSV,
        SAMPLE_GEOJSON,
        MQ / "regions" / ("results_pt%d_%s_flagged.png"
                          % (int(tile_m), source)),
        MQ / "regions" / ("results_pt%d_%s_missed.png"
                          % (int(tile_m), source)),
    ]
    copied = []
    for f in wanted:
        if f.exists():
            shutil.copy2(f, dest / f.name)
            copied.append(f.name)
        else:
            logger.warning("not archived (missing): %s", f.name)

    mp = MQ / ("_metrics_pt%d_%s.json" % (int(tile_m), source))
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        pct = lambda v: "%.0f%%" % (v * 100)
        md_lines = [
            "# Run: %s, %.0fm point-centred crops (z%s)"
            % (m["source"], m["tile_m"], m["zoom"]),
            "",
            "Archived %s. Imagery not copied; re-fetch with" % stamp,
            "`python 08_point_model.py fetch --tile-m %.0f --source %s`"
            % (m["tile_m"], m["source"]),
            "",
            "| metric | value |",
            "|---|---|",
            "| ROC-AUC | %.3f |" % m["roc_auc"],
            "| average precision | %.3f (no-skill %.3f) |"
            % (m["avg_precision"], m["eval_prevalence"]),
            "| mean score, dumps | %.3f |" % m["mean_score_dumps"],
            "| mean score, residential | %.3f |"
            % m["mean_score_residential"],
            "| verdict | **%s detector** |" % m["detector"],
            "| top-40 precision, per window | %s |"
            % pct(m["top40_precision_per_window"]),
            "| top-40 precision, per site | %s |"
            % pct(m["top40_precision_per_site"]),
            "| deployment prevalence | %.3f |" % m["deploy_prevalence"],
            "",
            "Files archived: %s" % ", ".join(copied),
            "",
        ]
        (dest / "README.md").write_text(chr(10).join(md_lines),
                                        encoding="utf-8")
    logger.info("archived %d files -> %s", len(copied), dest)
    return dest


# Confidence tiers published on the platform. Deliberately three, not one.
#
# Global Forest Watch publishes GLAD/RADD alerts as low confidence until
# corroborated and lets users tick a box to see only high-confidence ones;
# Global Plastic Watch runs three sensitivity regimes and attaches a confidence
# score to every merged polygon. Neither picks a single cut-off, because the
# right one depends on what the viewer intends to do. So the threshold is a
# DISPLAY FILTER, not a data decision: every polygon keeps its raw score, the
# tiers can be re-tuned later without re-scanning, and "high" is the default
# because a government-facing map should not be mostly wrong on first load.
TIERS = {"high": 0.90, "medium": 0.70, "all": 0.50}
DEFAULT_TIER = "high"


def freeze(tile_m, source, zoom):
    """Fit the deployable head on ALL samples and save it.

    Everything reported so far is cross-validation: fit on 4/5, score the
    held-out 1/5, discard the model. That measures honestly but leaves nothing
    to ship. This fits one model on every sample and writes it to disk.

    C is chosen by the same blocked inner CV used during evaluation rather than
    hardcoded, so the deployed model is regularised the way the measured one
    was. The metrics written into the metadata come from the *evaluation* run,
    not from this fit -- scoring a model on its own training data would report
    an accuracy the deployed system will never achieve.
    """
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from . import feature_probe as fp
    from . import features as bm

    df = pd.read_csv(SAMPLES_CSV)
    d = tiles_dir(tile_m, source)
    df["path"] = [str(d / (s + ".png")) for s in df.sid]
    have = df[[Path(p).exists() for p in df.path]].reset_index(drop=True)
    y = have.label.to_numpy()
    groups, _ = fp.spatial_groups(have.lat, have.lon, 1000.0)
    X = bm.imagenet_features(have.path.tolist())

    best_c, best_s = CS[0], -1.0
    k = min(5, len(np.unique(groups)), int(min(np.bincount(y))))
    for C in CS:
        oof = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=True,
                                           random_state=1).split(X, y, groups):
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=5000,
                                                 class_weight="balanced", C=C))
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        s = average_precision_score(y, oof)
        if s > best_s:
            best_s, best_c = s, C
    logger.info("C=%s chosen by blocked inner CV (avg precision %.3f)",
                best_c, best_s)

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000,
                                             class_weight="balanced", C=best_c))
    model.fit(X, y)

    out = MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "probe.joblib")

    ev = MQ / ("_metrics_pt%d_%s.json" % (int(tile_m), source))
    meta = {
        "backbone": "imagenet_resnet50_gap2048",
        "img_size": fp.IMG_SIZE,
        "tile_m": tile_m, "source": source, "zoom": zoom,
        "C": best_c, "n_train": int(len(y)), "n_pos": int(y.sum()),
        "tiers": TIERS, "default_tier": DEFAULT_TIER,
        "held_out_metrics": (json.loads(ev.read_text(encoding="utf-8"))
                             if ev.exists() else None),
        "note": ("held_out_metrics come from the nested-CV evaluation run, not "
                 "from this fit. Do not score this model on its training data."),
    }
    (out / "probe_meta.json").write_text(json.dumps(meta, indent=2),
                                         encoding="utf-8")
    logger.info("frozen head -> %s", out / "probe.joblib")
    print("\nfrozen on %d samples (%d positive), C=%s" % (len(y), y.sum(), best_c))
    print("tiers: " + ", ".join("%s>=%.2f" % (k_, v) for k_, v in TIERS.items()))
    print("  %s" % (out / "probe.joblib"))
    print("  %s" % (out / "probe_meta.json"))


def freeze_screen():
    """Fit the STAGE-1 screen: 250m cells, free Esri imagery.

    The cascade needs two heads, not one. Stage 2 (80m, Google) is what
    actually detects waste, but running it over the whole national footprint is
    872,000 billable requests. Stage 1 is a deliberately cheap, deliberately
    generous filter on FREE imagery whose only job is to discard the ~two
    thirds of the country where nothing is happening.

    It is trained on the CELL labels -- one row per 250m cell -- not on the dump
    points, because that is the unit it will score at deployment. Its own
    accuracy is poor (it is the settlement detector described in the package
    README) and that is acceptable here: at a threshold of 0.30 it keeps 31.6%
    of cells and 85.5% of true sites, and stage 2 removes the false positives it
    lets through. What it must not do is discard real sites, which is why the
    threshold is low and the model is scored on recall rather than precision.
    """
    import joblib
    import geopandas as gpd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from . import feature_probe as fp
    from . import features as bm

    g = gpd.read_file(SAMPLE_GEOJSON)
    g = g[g.label.notna() & (g.label.astype(str).str.strip() != "")].copy()
    g["label"] = g.label.astype(str).str.strip().astype(int)
    c = g.geometry.centroid
    g["path"] = [str(MQ / "tiles_national" / (t + ".png")) for t in g.tile_id]
    have = g[[Path(p).exists() for p in g.path]].reset_index(drop=True)
    if len(have) < len(g):
        logger.warning("%d/%d screen tiles present", len(have), len(g))

    y = have.label.to_numpy()
    cc = have.geometry.centroid
    groups, _ = fp.spatial_groups(cc.y, cc.x, 1000.0)
    X = bm.imagenet_features(have.path.tolist())

    k = min(5, len(np.unique(groups)), int(min(np.bincount(y))))
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits=k, shuffle=True,
                                       random_state=0).split(X, y, groups):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000,
                                             class_weight="balanced", C=0.001))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    keep = float((oof >= 0.30).mean())
    rec = float((oof[y == 1] >= 0.30).mean())

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000,
                                             class_weight="balanced", C=0.001))
    model.fit(X, y)

    out = MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "probe_screen.joblib")
    (out / "probe_screen_meta.json").write_text(json.dumps({
        "role": "stage-1 screen",
        "backbone": "imagenet_resnet50_gap2048",
        "img_size": fp.IMG_SIZE, "tile_m": 250.0, "source": "esri", "zoom": 17,
        "C": 0.001, "n_train": int(len(y)), "n_pos": int(y.sum()),
        "screen_threshold": 0.30,
        "held_out_roc_auc": round(float(auc), 4),
        "cells_kept_at_threshold": round(keep, 4),
        "site_recall_at_threshold": round(rec, 4),
        "note": ("Deliberately generous. Its job is to discard empty country "
                 "cheaply on free imagery, not to be accurate. Stage 2 removes "
                 "the false positives it passes through."),
    }, indent=2), encoding="utf-8")

    print("\nstage-1 screen frozen on %d cells (%d positive)" % (len(y), y.sum()))
    print("  blocked-CV ROC-AUC      %.3f" % auc)
    print("  at threshold 0.30: keeps %.0f%% of cells, retains %.0f%% of sites"
          % (keep * 100, rec * 100))
    print("  %s" % (out / "probe_screen.joblib"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-samples")
    sub.add_parser("freeze-screen")
    for name in ("fetch", "evaluate", "sheets", "archive", "freeze"):
        p = sub.add_parser(name)
        p.add_argument("--tile-m", type=float, default=80.0)
        p.add_argument("--source", choices=["esri", "google"], default="esri")
        if name == "archive":
            p.add_argument("--tag", default=None,
                           help="folder suffix; defaults to today")
        if name not in ("sheets", "archive"):
            p.add_argument("--zoom", type=int, default=None,
                           help="default 17 for esri (its ceiling outside "
                                "Conakry), 19 for google")
        if name == "freeze":
            p.set_defaults(source="google")
        if name == "fetch":
            p.add_argument("--budget", type=int, default=None,
                           help="cap on BILLABLE requests; cached tiles are "
                                "free. Always set this for --source google")
    a = ap.parse_args()

    if a.cmd == "build-samples":
        build_samples()
        return
    if a.cmd == "freeze-screen":
        freeze_screen()
        return
    zoom = getattr(a, "zoom", None)
    if zoom is None:
        zoom = 17 if a.source == "esri" else 19
    if a.cmd == "fetch":
        if not SAMPLES_CSV.exists():
            build_samples()
        fetch(a.tile_m, a.source, zoom, a.budget)
    elif a.cmd == "evaluate":
        evaluate(a.tile_m, a.source, zoom)
    elif a.cmd == "archive":
        archive(a.tile_m, a.source, a.tag)
    elif a.cmd == "freeze":
        freeze(a.tile_m, a.source, zoom)
    else:
        sheets(a.tile_m, a.source)


if __name__ == "__main__":
    main()
