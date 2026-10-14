"""
What is the dumpsite model actually responding to?

WHY THIS EXISTS
---------------
`26_blind_grid_protocol.py evaluate` answers one narrow question: does the model
clear the pre-declared criterion on a blind grid? At Boke it returned ROC-AUC
0.738 and missed the criterion on lift alone (1.48x against 1.5x).

Neither number says WHY. An AUC of 0.738 is consistent with a model that
detects waste imperfectly, and equally consistent with a model that detects
something correlated with waste and never sees waste at all. Those two
possibilities imply opposite decisions about deployment, so they have to be
separated before anything is published.

The Boke review set can separate them, because the annotator did more than tick
a box. Every positive tile carries hand-digitised POINT COORDINATES for the
waste features visible in it -- 755 points across 243 tiles, up to nine in one
window, semicolon-separated in the `lat`/`long` fields. That converts the grid
from a binary label set into a quantity-and-position record, and quantity and
position are exactly what a detector should be sensitive to.

THE TEST
--------
If the model detects waste, then among positive tiles its score must rise with
the amount of waste present. This is a dose-response test, and it is strictly
internal to the positive class -- it never compares against negatives, so it is
immune to every question about where the positive/negative line was drawn or
how confident the annotator was.

A second test asks whether a target near the centre of the window scores higher
than one at the edge, which is the point-centred effect that the 0.911
inventory figure depends on.

A third stratifies scores by the land cover the annotator recorded in `notes`.
The notes are DESCRIPTIVE ONLY: they group rows for reporting and enter no
metric. Every figure of record comes from `label` and `model_score` alone.

WHAT IT FOUND AT BOKE
---------------------
No dose-response (rho = -0.020, P = 0.76) and no centrality effect (P = 0.37),
while forest and water score ~0.01, settlement without waste scores ~0.50, and
construction sites -- bare, disturbed, no waste -- score ~0.90, higher than any
waste tile.

The model discriminates disturbed anthropogenic ground from natural cover. Waste
tiles outscore forest because waste sits on disturbed ground, not because the
waste is visible to the model. The AUC is real and it is not waste detection.

A NOTE ON WHAT THE LABELS MEAN
------------------------------
Nearly every positive is annotated "could be dumpsite" at medium confidence.
That is not hedging to be discounted: it is an accurate statement of what
overhead imagery can support at this scale, and it is the limit of the
instrument rather than of the annotator. The protocol's own instruction,
"1 = waste visible", asked for a certainty the imagery cannot deliver; that was
a flaw in the protocol. The labels are treated here exactly as given.

Usage:
    python 27_blindgrid_diagnostics.py --town boke
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
REGIONS = HERE / "manual_qa" / "regions"
RUNS = HERE / "manual_qa" / "runs"

TIERS = {"high": 0.9, "medium": 0.7, "all": 0.5}

# A point further than this from its own tile is a data-entry slip, not a
# location. Boke has three, all in Nunavut -- a paste from another window.
# Excluded from geometry, never from the label: the tile is still a positive.
MAX_POINT_OFFSET_M = 200.0

# Free-text land cover, typed by hand, so the same category arrives spelled
# several ways. Used for GROUPING IN THE REPORT ONLY.
NOTE_CANON = [
    ("could be dumpsite but", "possible_waste"),
    ("could be but", "possible_waste"),
    ("coud be dumpsite", "possible_waste"),
    ("could be dumpsite", "possible_waste"),
    ("waste on water", "waste_on_water"),
    ("built-up", "settlement_no_waste"),
    ("forest", "forest"),
    ("water body", "water"),
    ("empty area", "bare_or_sparse"),
    ("construction", "construction_site"),
    ("plantation", "plantation"),
    ("quarry", "quarry_borrow_pit"),
]


def canon_note(s):
    t = (s or "").strip().lower()
    for frag, name in NOTE_CANON:
        if frag in t:
            return name
    return "other" if t else "none"


def parse_points(props, cx, cy):
    """
    Pull the digitised waste points out of the `lat`/`long` fields.

    Both fields hold semicolon-separated lists when the annotator marked more
    than one feature in a window. A negative tile keeps the default single
    coordinate at the window centre, which is not an annotation and is dropped
    by the centre test below rather than counted as a waste point.
    """
    lats = [t.strip() for t in str(props.get("lat", "")).split(";") if t.strip()]
    lons = [t.strip() for t in str(props.get("long", "")).split(";") if t.strip()]
    pts, suspect = [], 0
    for la, lo in zip(lats, lons):
        try:
            la, lo = float(la), float(lo)
        except ValueError:
            suspect += 1
            continue
        dx = (lo - cx) * 111320.0 * math.cos(math.radians(cy))
        dy = (la - cy) * 110574.0
        dist = math.hypot(dx, dy)
        if dist > MAX_POINT_OFFSET_M:
            suspect += 1
            continue
        pts.append((la, lo, dist))
    return pts, suspect, abs(len(lats) - len(lons))


def load(town):
    path = REGIONS / f"{town}_blindgrid_review.geojson"
    feats = json.loads(path.read_text(encoding="utf-8"))["features"]
    rows = []
    for f in feats:
        p = f["properties"]
        ring = f["geometry"]["coordinates"][0]
        cx = (min(x for x, _ in ring) + max(x for x, _ in ring)) / 2.0
        cy = (min(y for _, y in ring) + max(y for _, y in ring)) / 2.0
        if str(p.get("label", "")).strip() == "":
            raise SystemExit(f"{path.name}: unlabelled windows remain.")
        y = int(p["label"])
        pts, suspect, mismatch = parse_points(p, cx, cy)
        rows.append({
            "tile_id": p.get("tile_id", ""),
            "y": y,
            "score": float(p["model_score"]),
            "note": canon_note(p.get("notes", "")),
            "confidence": str(p.get("confidence", "")).strip().lower(),
            # Only a positive tile's coordinates are annotations; a negative
            # keeps the untouched default and contributes no points.
            "n_points": len(pts) if y == 1 else 0,
            "d_nearest": min((d for _, _, d in pts), default=None) if y == 1 else None,
            "suspect_points": suspect,
            "coord_count_mismatch": mismatch,
        })
    return path, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", required=True)
    args = ap.parse_args()
    town = args.town.lower()

    path, rows = load(town)
    y = np.array([r["y"] for r in rows])
    s = np.array([r["score"] for r in rows])
    npts = np.array([r["n_points"] for r in rows])
    note = np.array([r["note"] for r in rows])
    dn = np.array([np.nan if r["d_nearest"] is None else r["d_nearest"]
                   for r in rows], dtype=float)

    out = {
        "town": town, "source": str(path), "n_windows": int(len(y)),
        "n_positive": int(y.sum()),
        "n_digitised_points": int(npts.sum()),
        "suspect_point_entries": int(sum(r["suspect_points"] for r in rows)),
        "roc_auc_all": float(roc_auc_score(y, s)),
        "average_precision_all": float(average_precision_score(y, s)),
        "base_rate": float(y.mean()),
    }

    # ---- TEST 1. Dose-response, positives only ----
    pm = y == 1
    rho, p_rho = spearmanr(npts[pm], s[pm])
    bands = [(1, 1, "1 point"), (2, 3, "2-3 points"),
             (4, 6, "4-6 points"), (7, 99, "7+ points")]
    dose = []
    for lo, hi, lab in bands:
        m = pm & (npts >= lo) & (npts <= hi)
        if not m.sum():
            continue
        dose.append({"band": lab, "n": int(m.sum()),
                     "mean_score": float(s[m].mean()),
                     "median_score": float(np.median(s[m])),
                     "frac_high_tier": float((s[m] >= TIERS["high"]).mean())})
    out["dose_response"] = {
        "spearman_rho": float(rho), "p_value": float(p_rho),
        "n_positive": int(pm.sum()), "bands": dose,
        "detects_waste": bool(p_rho < 0.05 and rho > 0),
        "interpretation": (
            "Score does not rise with the amount of waste present, so the "
            "ranking is not driven by waste." if not (p_rho < 0.05 and rho > 0)
            else "Score rises with waste quantity, consistent with detection."),
    }

    # ---- TEST 2. Centrality ----
    ok = pm & np.isfinite(dn)
    near, far = ok & (dn < 20.0), ok & (dn >= 20.0)
    cent = {"n_with_geometry": int(ok.sum()),
            "median_distance_m": float(np.median(dn[ok])) if ok.sum() else None}
    if near.sum() and far.sum():
        u, p_u = mannwhitneyu(s[near], s[far])
        cent.update({
            "n_near": int(near.sum()), "mean_score_near": float(s[near].mean()),
            "n_far": int(far.sum()), "mean_score_far": float(s[far].mean()),
            "p_value": float(p_u),
            "centrality_effect": bool(p_u < 0.05 and s[near].mean() > s[far].mean()),
        })
    out["centrality"] = cent

    # ---- TEST 3. What the score tracks ----
    strata = {}
    for nv in sorted(set(note.tolist())):
        m = note == nv
        strata[nv] = {"n": int(m.sum()),
                      "labelled_positive": int(y[m].sum()),
                      "mean_score": float(s[m].mean()),
                      "median_score": float(np.median(s[m])),
                      "frac_high_tier": float((s[m] >= TIERS["high"]).mean())}
    out["score_by_land_cover"] = strata
    out["_note"] = ("`notes` group rows for reporting only. Every metric above "
                    "is computed from `label` and `model_score` alone.")

    nat = np.isin(note, ["forest", "water"])
    dist = np.isin(note, ["settlement_no_waste", "construction_site",
                          "quarry_borrow_pit"])
    if nat.sum() and dist.sum() and pm.sum():
        out["what_it_separates"] = {
            "natural_cover_mean": float(s[nat].mean()),
            "disturbed_no_waste_mean": float(s[dist].mean()),
            "waste_mean": float(s[pm].mean()),
            "gap_natural_to_disturbed": float(s[dist].mean() - s[nat].mean()),
            "gap_disturbed_to_waste": float(s[pm].mean() - s[dist].mean()),
        }

    verdict = (
        "LAND-COVER PROXY. The model separates disturbed anthropogenic ground "
        "from natural cover and does not respond to waste itself: score is "
        "independent of how much waste a window contains. A blind-grid AUC "
        "above 0.5 is therefore not evidence of waste detection, and a "
        "national waste layer is not supportable on it."
        if not out["dose_response"]["detects_waste"] else
        "Score tracks waste quantity; detection is supported on this grid.")
    out["verdict"] = verdict

    out_dir = RUNS / f"blindgrid_{town}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "diagnostics.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---------------- report ----------------
    print(f"BLIND GRID DIAGNOSTICS - {town}")
    print(f"  {out['n_windows']} windows, {out['n_positive']} positive, "
          f"{out['n_digitised_points']} digitised waste points")
    print(f"  ROC-AUC {out['roc_auc_all']:.3f}  AP {out['average_precision_all']:.3f}"
          f"  base rate {out['base_rate']:.3f}")
    if out["suspect_point_entries"]:
        print(f"  ! {out['suspect_point_entries']} point entries lie >"
              f"{MAX_POINT_OFFSET_M:.0f} m from their own tile (data-entry "
              f"slips). Excluded from geometry; labels untouched.")
    print()
    print("  TEST 1 - does the score rise with the amount of waste?")
    for b in dose:
        print(f"    {b['band']:12s} n={b['n']:4d} mean={b['mean_score']:.3f} "
              f"median={b['median_score']:.3f} >=high={b['frac_high_tier']:.0%}")
    print(f"    Spearman rho={rho:+.3f} P={p_rho:.3f}  ->  "
          f"{'DOSE-RESPONSE' if out['dose_response']['detects_waste'] else 'NO DOSE-RESPONSE'}")
    print()
    if "p_value" in cent:
        print("  TEST 2 - does a central target score higher?")
        print(f"    <20 m  n={cent['n_near']:4d} mean={cent['mean_score_near']:.3f}")
        print(f"    >=20 m n={cent['n_far']:4d} mean={cent['mean_score_far']:.3f}")
        print(f"    P={cent['p_value']:.3f}  ->  "
              f"{'CENTRALITY EFFECT' if cent['centrality_effect'] else 'NO CENTRALITY EFFECT'}")
        print()
    print("  TEST 3 - what the score tracks (notes = grouping only):")
    for nv, r in sorted(strata.items(), key=lambda kv: kv[1]["mean_score"]):
        print(f"    {nv:22s} n={r['n']:4d} pos={r['labelled_positive']:4d} "
              f"mean={r['mean_score']:.3f} >=high={r['frac_high_tier']:5.1%}")
    print()
    print(f"  VERDICT: {verdict}")
    print(f"\n  Written to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
