"""
38 -- Blind-grid accuracy of the FREE cascade: Esri stage 2.

The national scan can confirm on free Esri z17 imagery instead of licensed
Google z19 (dumpsite_tool.scan, confirm_source="esri"). Its accuracy is NOT the
Google accuracy: the stage-2 head is a different fit on different pixels, so
the figures published with an Esri scan have to be measured on Esri. This
script measures them on the two hand-labelled blind grids that exist, with
the same labels and the same metrics as 26_blind_grid_protocol.py, and prints
the Google figures beside them.

Free: Esri tiles are not metered. Tiles are cached under
manual_qa/blindgrid_cache_{town}/confirm_esri/, so a re-run fetches nothing.
First run scores ~1,000 windows on CPU (roughly 15-20 minutes).

    python 38_esri_cascade_evaluation.py              # both towns
    python 38_esri_cascade_evaluation.py --town boke

Writes manual_qa/runs/blindgrid_esri/evaluation.json and one
{town}_windows_scored.csv per town.
"""

# torch before anything that loads GDAL (Windows DLL order); see
# backend/plasticwaste_api/tasks.py.
try:
    import torch  # noqa: F401
except Exception:  # noqa: BLE001 -- reported by the scoring step if missing
    pass

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from plasticwaste_tool.dumpsite_tool import inference as INF   # noqa: E402
from plasticwaste_tool.dumpsite_tool import scan as SCAN       # noqa: E402

MQ = HERE / "manual_qa"
REGIONS = MQ / "regions"
OUT = MQ / "runs" / "blindgrid_esri"
THRESHOLDS = (0.90, 0.70, 0.50, 0.30)
TOWNS = ("conakry", "boke")


def _labels(town):
    """Window labels, with the Google-head score where one was recorded."""
    feats = json.loads((REGIONS / f"{town}_blindgrid_review.geojson")
                       .read_text(encoding="utf-8"))["features"]
    df = pd.DataFrame([f["properties"] for f in feats])
    df["y"] = df["label"].astype(str).str.strip().astype(int)
    if "model_score" in df:
        df["score_google"] = pd.to_numeric(df["model_score"], errors="coerce")
    else:
        sc = pd.read_csv(MQ / "runs" / "cascade_demo_conakry" / "windows_scored.csv")
        df = df.merge(sc[["tile_id", "score"]].rename(columns={"score": "score_google"}),
                      on="tile_id", how="left")
    return df[["tile_id", "y", "score_google"]]


def _windows(tile_ids, tile_m=SCAN.CONFIRM_TILE_M):
    """Rebuild each window's footprint from its global-lattice tile id."""
    dlat = tile_m / 110_574.0
    rows = []
    for t in tile_ids:
        j, i = int(t[1:8]), int(t[9:16])
        rows.append(SCAN._window(j, i, dlat, tile_m))
    return pd.DataFrame(rows)


def _metrics(y, s, screen_retention):
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    base = float(y.mean())
    ops = []
    for t in THRESHOLDS:
        m = s >= t
        tp, flagged = int((y[m] == 1).sum()), int(m.sum())
        prec = tp / flagged if flagged else 0.0
        rec = tp / max(int(y.sum()), 1)
        ops.append({"threshold": t, "flagged": flagged, "true": tp,
                    "precision": round(prec, 4), "recall": round(rec, 4),
                    "lift": round(prec / base, 3) if base else 0.0,
                    "end_to_end_recall": round(rec * screen_retention, 4)})
    return {"n": int(len(y)), "n_pos": int(y.sum()), "base_rate": round(base, 4),
            "unexamined": int((~ok).sum()),
            "roc_auc": round(float(roc_auc_score(y, s)), 4),
            "avg_precision": round(float(average_precision_score(y, s)), 4),
            "operating_points": ops}


def evaluate(town, model, zoom, screen_retention):
    lab = _labels(town)
    win = _windows(lab.tile_id)
    cache = MQ / f"blindgrid_cache_{town}" / "confirm_esri"
    print(f"\n[{town}] {len(win)} windows: fetching Esri z{zoom} (free, cached) ...")
    paths, _ = INF.fetch_tiles_esri(win, str(cache), size_px=800, zoom=zoom)
    print(f"[{town}] scoring with probe_esri.joblib ...")
    lab["score_esri"] = SCAN._score([str(p) if p else None for p in paths], model)
    OUT.mkdir(parents=True, exist_ok=True)
    lab.to_csv(OUT / f"{town}_windows_scored.csv", index=False)

    y = lab.y.to_numpy()
    res = {"esri": _metrics(y, lab.score_esri.to_numpy(float), screen_retention)}
    if lab.score_google.notna().any():
        res["google"] = _metrics(y, lab.score_google.to_numpy(float), screen_retention)
    return res


def _print(town, res):
    print(f"\nBLIND GRID -- {town}")
    for src in ("esri", "google"):
        if src not in res:
            continue
        m = res[src]
        print(f"  {src:<6} n {m['n']}  pos {m['n_pos']}  base {m['base_rate']:.3f}  "
              f"ROC-AUC {m['roc_auc']:.3f}  AP {m['avg_precision']:.3f}"
              + (f"  ({m['unexamined']} windows not fetched)" if m["unexamined"] else ""))
        print(f"    {'thr':>5}{'flagged':>9}{'true':>6}{'precision':>11}"
              f"{'recall':>8}{'lift':>7}{'end-to-end':>12}")
        for o in m["operating_points"]:
            print(f"    {o['threshold']:>5.2f}{o['flagged']:>9}{o['true']:>6}"
                  f"{o['precision']:>11.3f}{o['recall']:>8.3f}{o['lift']:>6.2f}x"
                  f"{o['end_to_end_recall']:>11.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", choices=TOWNS)
    args = ap.parse_args()

    model, meta = SCAN.load_head(SCAN.confirm_head_for("esri"))
    zoom = int(meta.get("zoom", SCAN.SCREEN_ZOOM))
    screen = json.loads((SCAN.MODELS / "probe_screen_meta.json").read_text(encoding="utf-8"))
    retention = float(screen["site_recall_at_threshold"])

    out_p = OUT / "evaluation.json"
    report = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for town in ([args.town] if args.town else TOWNS):
        report[town] = evaluate(town, model, zoom, retention)
        _print(town, report[town])

    report["_meta"] = {
        "stage2_head": "probe_esri.joblib", "stage2_zoom": zoom,
        "screen_retention": retention,
        "note": ("end_to_end_recall = stage-1 site retention x stage-2 blind-grid "
                 "recall. Precision is cumulative: every window at or above the "
                 "threshold. Unfetched windows are excluded, never scored 0."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWritten to {out_p}")


if __name__ == "__main__":
    main()
