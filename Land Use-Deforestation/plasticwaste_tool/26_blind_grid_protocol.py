"""
26 — The second blind grid: build, label, evaluate, decide.

WHAT THIS IS FOR
----------------
The dumpsite layer's deployment-realistic figures come from ONE city, in an
area chosen because it contained a cluster of known dumps. Its 26.1% base rate
is therefore enriched relative to a randomly chosen urban area, and precision
elsewhere would be lower. Methodology section 9.3 names a second blind grid in
a town outside Conakry as "the single highest-value outstanding action", and it
is the gate between "screening aid, evaluated in one place" and "layer we are
willing to publish nationally".

This script runs that protocol end to end. It does not decide anything for you:
step 2 is a human labelling every window by eye, and step 4 prints the numbers
against a pre-declared pass criterion rather than a verdict.

WHY A NEW TOWN, AND WHY BLIND
-----------------------------
Blocked cross-validation on point-centred crops gave ROC-AUC 0.911. The same
frozen model on a blind grid gave 0.682. The gap is covariate shift: 43% of
training positives are Conakry against 9% of negatives, so "dense urban"
correlates with the positive label, and that correlation does not survive
contact with a city where every window is dense urban. Spatial blocking cannot
detect this -- every fold shared the same mostly-rural negative distribution.
Only a blind grid in ground the model has never seen can.

"Blind" means: choose the area BEFORE looking at model scores, label every
window in it, and do not drop windows that turn out to be awkward. Choosing an
area because you know it has dumps is what produced the enriched base rate this
exercise exists to correct.

    STEP 1   python 26_blind_grid_protocol.py build --town boke \
                     --bounds -14.30 10.93 -14.28 10.945 --budget 200
    STEP 2   python manual_qa/regions/label_tool.py boke_blindgrid_review.geojson
    STEP 3   python 26_blind_grid_protocol.py check --town boke
    STEP 4   python 26_blind_grid_protocol.py evaluate --town boke

ALWAYS run `estimate` first. A 2.2 x 1.7 km AOI is about 560 windows, not the
250 an earlier version of this docstring guessed; starting with too small a
budget spends the money and yields a ragged strip rather than a grid. Cost is
roughly US$1 per 500 windows. The real cost is labelling time: budget 2-4
hours for ~400 windows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Make `plasticwaste_tool` importable however this script is invoked -- from
# inside the package directory (the natural thing, since the script lives
# there) or from the project root. Without this, running it from
# plasticwaste_tool/ fails with ModuleNotFoundError because the package's own
# parent is not on the path.
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

MQ = HERE / "manual_qa"
REGIONS = MQ / "regions"
RUNS = MQ / "runs"

# Pre-declared pass criterion. Written here, before the second grid exists, so
# the bar cannot drift to meet whatever the result turns out to be.
#
# The test is NOT "does the new town beat Conakry" -- a different town can
# legitimately be harder. It is "does the Conakry figure GENERALISE", which
# means the new town must stay usefully above its own base rate. A detector
# that only matches chance in a second town is not a national screening layer,
# whatever it scored in the first.
PASS_CRITERION = {
    "min_lift_over_base_rate": 1.5,
    "min_precision_at_high": 0.35,
    "min_windows": 250,
    "min_positives": 20,
    "note": (
        "Lift is precision at the default tier divided by the area's own base "
        "rate. Conakry measured 0.584 / 0.261 = 2.24x. A second town at or "
        "above 1.5x supports a national screening layer with the tier and "
        "recall caveats already documented; below that it should stay a "
        "per-city tool."),
    "known_defect": (
        "Lift is capped at 1/base_rate, so a grid over a dirtier town is "
        "scored against a lower ceiling for reasons that have nothing to do "
        "with the model: 3.83x at 26% prevalence, 2.29x at 44%. ROC-AUC would "
        "have been the base-rate-invariant choice. This is left as declared "
        "because it was fixed before any second grid existed and changing it "
        "afterwards is the exact freedom pre-declaration removes -- but "
        "`evaluate` now prints the invariant comparison beside the verdict, "
        "and a FAIL here must not be read as 'discrimination got worse'."),
}


def _windows_from_cache(cache_dir, tile_m=80.0):
    """
    Rebuild window geometry from the tile ids already on disk.

    `scan.py` does NOT write a per-window score table. An earlier version of
    this script assumed it did, inferring that from an artefact in the Conakry
    demo folder which something else had produced. It does not need to: a tile
    id is `r{row:07d}c{col:07d}` on a global lattice anchored at (-180, -90), so
    the id alone reconstructs the window exactly. That also means an
    interrupted fetch loses nothing that was already paid for -- the cached
    tiles can always be re-scored for free.
    """
    import re

    from plasticwaste_tool.dumpsite_tool import scan as ds

    dlat = tile_m / 110_574.0
    rows = []
    for png in sorted(Path(cache_dir).glob("*.png")):
        m = re.match(r"r(\d{7})c(\d{7})$", png.stem)
        if not m:
            continue
        j, i = int(m.group(1)), int(m.group(2))
        win = ds._window(j, i, dlat, tile_m)
        win["path"] = str(png)
        win["row"], win["col"] = j, i
        rows.append(win)
    return pd.DataFrame(rows)


def _estimate_windows(bounds, tile_m=80.0):
    """How many 80 m windows an AOI contains, before any imagery is bought."""
    from plasticwaste_tool.dumpsite_tool import scan as ds
    return len(ds._windows(bounds, tile_m))


def cmd_estimate(args):
    """Cost a candidate AOI without spending anything."""
    n = _estimate_windows(tuple(args.bounds))
    # Stage 1 discards empty country, but inside a town it discards very
    # little -- the Boke run kept 62% of cells. Quote the range, not one number.
    lo, hi = int(n * 0.55), n
    print(f"AOI {tuple(args.bounds)}")
    print(f"  80 m windows in the AOI       : {n}")
    print(f"  stage 2 will fetch about      : {lo}-{hi}")
    print(f"  approximate cost              : US${hi * 0.002:.2f} at $2/1000")
    print(f"  labelling at ~20 s/window     : {hi * 20 / 3600:.1f} hours")
    print()
    if n < PASS_CRITERION["min_windows"]:
        print(f"  ! Below the {PASS_CRITERION['min_windows']}-window minimum. "
              f"Widen the AOI.")
    elif n > 900:
        print(f"  ! {n} windows is a lot to label by hand. "
              f"{PASS_CRITERION['min_windows']}-500 is the useful range.")
    else:
        print("  Sensible size. Run:")
        print(f"    python 26_blind_grid_protocol.py build --town {args.town} "
              f"--bounds {' '.join(str(b) for b in args.bounds)} --budget {hi}")
    return 0


# ============================================================================
# STEP 1 — build
# ============================================================================
def cmd_build(args):
    """
    Run the cascade over the chosen AOI and emit an unlabelled review set.

    Scores are computed and stored, but the labelling tool does not display
    them. The point of a blind grid is that the human verdict is formed without
    the model's opinion in view.
    """
    import torch  # noqa: F401  -- must precede the GEE/GDAL stack on Windows

    from plasticwaste_tool.dumpsite_tool import scan as ds

    town = args.town.lower()
    out_dir = RUNS / f"blindgrid_{town}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = MQ / f"blindgrid_cache_{town}"
    confirm = cache / "confirm"

    bounds = tuple(args.bounds)
    needed = _estimate_windows(bounds)
    already = len(list(confirm.glob("*.png"))) if confirm.exists() else 0

    # NEVER silently overwrite labelling work. Building writes the review file
    # from scratch, and a grid of 400 windows is several hours of human
    # judgement that exists nowhere else. Losing it to a re-run of an earlier
    # command would be the single most expensive mistake this script could
    # make, so it refuses rather than trusting the operator to remember.
    review_path = REGIONS / f"{town}_blindgrid_review.geojson"
    if review_path.exists() and not args.force:
        existing = json.loads(review_path.read_bytes().decode("utf-8"))
        done = sum(1 for f in existing.get("features", [])
                   if str(f["properties"].get("label", "")).strip() != "")
        if done:
            raise SystemExit(
                "\nREFUSING TO OVERWRITE LABELLED WORK.\n"
                f"  {review_path.name} already holds {done} labelled windows.\n"
                "  Rebuilding writes the file from scratch and would discard "
                "them.\n\n"
                "  If you are extending the grid, back the file up first:\n"
                f'    copy "{review_path}" "{review_path}.bak"\n'
                "  then re-run with --force and merge the labels back by "
                "tile_id.\n\n"
                "  If the labels are not worth keeping, pass --force.")
        print(f"  note: overwriting {review_path.name} "
              f"(exists, but nothing labelled yet)")

    print(f"Building blind grid for {town}")
    print(f"  bounds        : {bounds}")
    print(f"  windows in AOI: {needed}")
    print(f"  already cached: {already}")
    print(f"  budget        : {args.budget} billable requests")
    print()

    # PRE-FLIGHT. A budget that cannot cover the AOI produces a RAGGED,
    # TRUNCATED grid: stage 2 stops mid-fetch, so what you end up with is the
    # first N windows in iteration order -- not a census of the area, and not a
    # random sample of it either. Refusing before the money is spent is much
    # better than discovering it afterwards.
    shortfall = needed - already - args.budget
    if shortfall > 0 and not args.allow_partial:
        raise SystemExit(
            "\nREFUSING TO START.\n"
            f"  The AOI needs about {needed} windows; {already} are cached and "
            f"the budget allows {args.budget} more, leaving {shortfall} "
            f"unfetched.\n"
            "  Stage 2 would stop partway and produce a ragged strip rather "
            "than a grid over the area you chose.\n\n"
            "  Either raise the budget:\n"
            f"    --budget {needed - already}    "
            f"(about US${(needed - already) * 0.002:.2f} more)\n"
            "  or shrink the AOI:\n"
            f"    python 26_blind_grid_protocol.py estimate --town {town} "
            "--bounds <smaller box>\n\n"
            "  Pass --allow-partial to proceed anyway and grid only what is "
            "fetched. Do that ONLY if you accept that the footprint will be "
            "whatever stage 2 reached, not the box you asked for.")

    gdf = ds.scan(bounds, str(cache), budget=args.budget, imagery_date=None)
    print(f"\nscan() returned {len(gdf)} candidate sites at its own tier.")

    # Rebuild the per-window table from the cached tiles and score them here.
    # scan() reports CANDIDATES, not every window it looked at, and a blind
    # grid needs the negatives just as much as the positives.
    df = _windows_from_cache(confirm)
    if df.empty:
        raise SystemExit(
            f"No tiles in {confirm}. Stage 2 fetched nothing -- check the "
            f"budget, and that the AOI is over land with Google coverage.")

    print(f"Scoring {len(df)} cached windows (free, no further requests)...")
    model, _meta = ds.load_head("probe.joblib")
    df["score"] = ds._score(list(df["path"]), model)
    df = df[df["score"].notna()].reset_index(drop=True)

    covered = len(df) / max(needed, 1)
    print(f"{len(df)} windows scored -- {covered:.0%} of the AOI.")
    if len(df) < PASS_CRITERION["min_windows"]:
        print(f"  ! Below the {PASS_CRITERION['min_windows']}-window minimum; "
              f"the precision interval will be too wide to decide anything.")
    if covered < 0.95:
        print(f"  ! Only {covered:.0%} of the AOI was fetched, so this grid is "
              f"a ragged subset rather than a census of the box. Quote the "
              f"ACTUAL extent below as the area, never the box requested.")

    features = []
    mid_lon = (float(df.minx.min()) + float(df.maxx.max())) / 2.0
    for r in df.itertuples():
        lon = (r.minx + r.maxx) / 2.0
        lat = (r.miny + r.maxy) / 2.0
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [r.minx, r.miny], [r.maxx, r.miny],
                [r.maxx, r.maxy], [r.minx, r.maxy], [r.minx, r.miny]]]},
            "properties": {
                "tile_id": r.tile_id,
                "region": town,
                "lat": f"{lat:.6f}",
                "long": f"{lon:.6f}",
                "label": "",
                "confidence": "",
                "notes": "",
                "annotator": "",
                # Carried so `evaluate` can pair score with label. The
                # labelling tool does not show it.
                "model_score": float(r.score),
                "half": "east" if lon >= mid_lon else "west",
            },
        })

    review = review_path
    review.write_bytes(json.dumps(
        {"type": "FeatureCollection", "features": features},
        indent=1).encode("utf-8"))

    actual = [float(df.minx.min()), float(df.miny.min()),
              float(df.maxx.max()), float(df.maxy.max())]
    (out_dir / "build_meta.json").write_bytes(json.dumps({
        "town": town,
        "requested_bounds": list(bounds),
        "actual_extent": actual,
        "windows_in_aoi": needed,
        "windows_scored": len(df),
        "coverage_fraction": round(covered, 4),
        "budget": args.budget,
        "review_geojson": str(review),
        "cache_dir": str(cache),
        "candidates_at_scan_tier": int(len(gdf)),
    }, indent=2).encode("utf-8"))

    print(f"\nWrote {review}")
    print(f"  actual extent: {actual}")
    print("\nSTEP 2 -- label every window, without looking at the scores:")
    print(f"    python manual_qa/regions/label_tool.py {review.name}")
    print("\n  label: 1 = waste visible, 0 = no waste visible.")
    print("  Label EVERY window. Dropping the awkward ones is what makes a base")
    print("  rate unrepresentative, which is the flaw this exercise corrects.")
    return 0


# ============================================================================
# STEP 3 — check
# ============================================================================
def cmd_check(args):
    """Report labelling progress and warn about patterns that invalidate the test."""
    town = args.town.lower()
    review = REGIONS / f"{town}_blindgrid_review.geojson"
    if not review.exists():
        raise SystemExit(f"{review} not found. Run `build` first.")

    props = [f["properties"] for f in
             json.loads(review.read_text(encoding="utf-8"))["features"]]
    total = len(props)
    labelled = [p for p in props if str(p.get("label", "")).strip() != ""]
    pos = [p for p in labelled if str(p["label"]).strip() == "1"]

    print(f"{town}: {len(labelled)}/{total} labelled "
          f"({len(labelled)/max(total,1):.0%}), {len(pos)} positive")

    if len(labelled) < total:
        print(f"\n  {total - len(labelled)} windows still unlabelled. Finish "
              f"before evaluating -- a partially labelled grid is not blind, "
              f"because what you stopped at correlates with what you found.")
        return 1

    base = len(pos) / total
    print(f"  base rate: {base:.1%}")
    if len(pos) < PASS_CRITERION["min_positives"]:
        print(f"\n  ! Only {len(pos)} positives. Below "
              f"{PASS_CRITERION['min_positives']} the precision estimate has "
              f"an interval too wide to decide anything. Extend the grid.")
    if base > 0.35:
        print(f"\n  ! Base rate {base:.0%} is high -- higher than Conakry's "
              f"26.1%. If the area was chosen because it had known dumps, this "
              f"grid is not blind and repeats the flaw being corrected.")

    # A labeller who never uses '0' has not been labelling blind.
    scores = np.array([float(p["model_score"]) for p in labelled])
    labels = np.array([int(str(p["label"]).strip()) for p in labelled])
    if labels.sum() == 0 or labels.sum() == len(labels):
        print("\n  ! All windows carry the same label. Nothing can be measured.")
        return 1

    # If labels track scores suspiciously closely, the labeller may have seen
    # them. Not proof, but worth surfacing before the numbers are believed.
    order = np.argsort(-scores)
    top_decile = labels[order[:max(len(labels)//10, 1)]].mean()
    if top_decile > 0.95:
        print(f"\n  ! Every window in the model's top decile is labelled "
              f"positive ({top_decile:.0%}). Confirm the labeller could not "
              f"see model_score; if they could, this is not a blind test.")
    print("\n  Ready. Run `evaluate`.")
    return 0


# ============================================================================
# STEP 4 — evaluate
# ============================================================================
def cmd_evaluate(args):
    """Blind-grid metrics for the new town, against Conakry and the criterion."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    town = args.town.lower()
    review = REGIONS / f"{town}_blindgrid_review.geojson"
    if not review.exists():
        raise SystemExit(f"{review} not found.")

    props = [f["properties"] for f in
             json.loads(review.read_text(encoding="utf-8"))["features"]]
    labelled = [p for p in props if str(p.get("label", "")).strip() != ""]
    if len(labelled) < len(props):
        raise SystemExit(
            f"{len(props) - len(labelled)} windows unlabelled. "
            f"Run `check` and finish labelling first.")

    y = np.array([int(str(p["label"]).strip()) for p in labelled])
    s = np.array([float(p["model_score"]) for p in labelled])
    base = float(y.mean())

    rows = []
    for t in (0.90, 0.70, 0.50):
        m = s >= t
        tp = int((y[m] == 1).sum())
        flagged = int(m.sum())
        rows.append({
            "threshold": t, "flagged": flagged, "true": tp,
            "precision": tp / flagged if flagged else 0.0,
            "recall": tp / max(int(y.sum()), 1),
            "lift": (tp / flagged / base) if (flagged and base) else 0.0,
        })

    auc = float(roc_auc_score(y, s))
    ap = float(average_precision_score(y, s))
    high = rows[0]

    print(f"\nBLIND GRID — {town}")
    print(f"  windows {len(y)}   positive {int(y.sum())}   base rate {base:.3f}")
    print(f"  ROC-AUC {auc:.3f}   average precision {ap:.3f} "
          f"(no-skill baseline = base rate, {base:.3f})")
    print(f"\n  {'tier':<8}{'flagged':>9}{'true':>7}{'precision':>11}"
          f"{'recall':>9}{'lift':>8}")
    for r, name in zip(rows, ("high", "medium", "all")):
        print(f"  {name:<8}{r['flagged']:>9}{r['true']:>7}"
              f"{r['precision']:>11.3f}{r['recall']:>9.3f}{r['lift']:>8.2f}x")

    print("\n  Conakry, for comparison:")
    print("  windows 422   positive 110   base rate 0.261")
    print("  ROC-AUC 0.682   average precision 0.477")
    print("  high        77     45      0.584    0.409   2.24x")

    # End-to-end recall folds in the stage-1 screen. Quoting stage 2's recall
    # alone omits the 14.5% of sites the free screen discards before stage 2
    # ever sees them.
    screen_recall = 0.855
    e2e = screen_recall * high["recall"]
    print(f"\n  End-to-end recall at 'high': {screen_recall:.3f} x "
          f"{high['recall']:.3f} = {e2e:.1%} of sites present.")

    c = PASS_CRITERION
    checks = [
        ("windows >= %d" % c["min_windows"], len(y) >= c["min_windows"]),
        ("positives >= %d" % c["min_positives"], int(y.sum()) >= c["min_positives"]),
        ("lift >= %.1fx" % c["min_lift_over_base_rate"],
         high["lift"] >= c["min_lift_over_base_rate"]),
        ("precision@high >= %.2f" % c["min_precision_at_high"],
         high["precision"] >= c["min_precision_at_high"]),
    ]
    print("\n  PRE-DECLARED CRITERION (fixed before this grid existed):")
    for label, ok in checks:
        print(f"    [{'pass' if ok else 'FAIL'}] {label}")

    passed = all(ok for _, ok in checks)
    print()
    if passed:
        print("  Criterion met. The layer is supportable as a national screening")
        print("  aid, with the tier and recall caveats already in the")
        print("  methodology. It is still not an inventory.")
    else:
        print("  Criterion NOT met. Do not publish a national layer.")

    # A FAIL here does NOT by itself mean discrimination got worse, and an
    # earlier version of this script wrongly said so ("the Conakry figure did
    # not generalise"). Lift is precision divided by base rate, so it is capped
    # at 1/base_rate: 3.83x where prevalence is 26%, but only 2.29x where it is
    # 44%. A grid over a dirtier town is scored against a lower ceiling for
    # arithmetic reasons that have nothing to do with the model. That is a flaw
    # in this criterion, which was fixed before any second grid existed and is
    # therefore reported as declared -- but the base-rate-INVARIANT comparison
    # has to be printed beside it, or the reader draws the wrong conclusion.
    ref_auc = 0.682
    print()
    print("  Base-rate-invariant comparison (ROC-AUC is unaffected by prevalence):")
    print(f"    this grid {auc:.3f}   vs   Conakry {ref_auc:.3f}   "
          f"({auc - ref_auc:+.3f})")
    print(f"    lift ceiling here is 1/{base:.3f} = {1.0 / base:.2f}x; "
          f"at Conakry's base rate it was {1.0 / 0.261:.2f}x")
    if auc >= ref_auc:
        print("    Discrimination did NOT degrade between towns. Whatever the")
        print("    model ranks on, it ranks on it just as consistently here.")
    print()
    print("  What that ranking IS, however, this script cannot tell you: an AUC")
    print("  above 0.5 is equally consistent with detecting waste and with")
    print("  detecting something merely correlated with it. Run")
    print(f"    python 27_blindgrid_diagnostics.py --town {town}")
    print("  which tests the score against the digitised waste points and")
    print("  separates those two cases. At Boke it found no dose-response.")

    out = RUNS / f"blindgrid_{town}" / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "town": town, "n": len(y), "n_pos": int(y.sum()), "base_rate": base,
        "roc_auc": auc, "avg_precision": ap, "operating_points": rows,
        "end_to_end_recall_high": e2e,
        "criterion": c, "passed": passed,
        "conakry_reference": {"n": 422, "base_rate": 0.261, "roc_auc": 0.682,
                              "avg_precision": 0.477, "precision_high": 0.584,
                              "lift_high": 2.24},
    }, indent=2), encoding="utf-8")
    print(f"\n  Written to {out}")
    return 0 if passed else 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("estimate", help="cost an AOI without spending anything")
    s.add_argument("--town", default="newtown")
    s.add_argument("--bounds", nargs=4, type=float, required=True,
                   metavar=("MINX", "MINY", "MAXX", "MAXY"))
    s.set_defaults(func=cmd_estimate)

    b = sub.add_parser("build", help="run the cascade and emit a review set")
    b.add_argument("--town", required=True)
    b.add_argument("--bounds", nargs=4, type=float, required=True,
                   metavar=("MINX", "MINY", "MAXX", "MAXY"))
    b.add_argument("--budget", type=int, required=True,
                   help="billable stage-2 requests. Run `estimate` FIRST -- a "
                        "2.2 x 1.7 km AOI needs about 560, not 250.")
    b.add_argument("--force", action="store_true",
                   help="overwrite an existing review file even if it "
                        "already contains labels. Back it up first.")
    b.add_argument("--allow-partial", action="store_true",
                   help="proceed even when the budget cannot cover the AOI; "
                        "the grid is then whatever stage 2 reached.")
    b.set_defaults(func=cmd_build)

    c = sub.add_parser("check", help="labelling progress and validity warnings")
    c.add_argument("--town", required=True)
    c.set_defaults(func=cmd_check)

    e = sub.add_parser("evaluate", help="metrics against the pass criterion")
    e.add_argument("--town", required=True)
    e.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
