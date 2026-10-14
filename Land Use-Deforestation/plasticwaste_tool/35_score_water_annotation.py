#!/usr/bin/env python
"""
Turn the water annotations into the comparison the methodology needs.

    python 35_score_water_annotation.py
    python 35_score_water_annotation.py --csv manual_qa/water_annotations.csv

Reads the CSV exported from `manual_qa/annotate_water.html`, compares the
observed composition of Guinea's FDI detections against the MARIDA benchmark
(section 7.1.13) and against the classifier's predictions for the same scene,
and writes manual_qa/runs/marida/water_annotation.json.

THE QUESTION THIS ANSWERS

Section 7.1.13 measured, on a global benchmark, that a mean+3-sigma FDI rule
fires mostly on vessels and only about 19% on annotated debris. Section
7.1.14.2 showed the trained classifier says something similar about Conakry.
Both are inferences from elsewhere. This script answers the same question
with direct human observation of Guinean imagery, which is the only evidence
in the whole chain that does not depend on MARIDA being representative.

WHAT IT REFUSES TO DO

It does not compute a debris precision. There is no debris label in the
annotation scheme, deliberately: what floats in a 10 m pixel cannot be
identified as plastic by eye, and a category that invited that judgement
would manufacture exactly the false certainty this module exists to avoid.

The quantity reported instead is the IDENTIFIED CONFUSER SHARE -- the
fraction of detections a human could positively attribute to something that
is not debris. That is a lower bound on the false-positive rate, which is the
useful direction: it cannot overstate the layer's problems, only understate
them.

The complement, `unknown_floating`, is an upper bound on how much of the
detection set could possibly be debris. Reporting both bounds, rather than a
point estimate between them, is the honest form of this result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

MQ = HERE / "manual_qa"
OUT = MQ / "runs" / "marida"

# Which annotation categories are positive identifications of a NON-debris
# cause, and which are not identifications at all.
CONFUSER = ["vessel", "wake", "sediment", "foam", "cloud", "natural",
            "shoreline", "clear_water"]
UNRESOLVED = ["unknown_floating", "uncertain"]

# Could this category plausibly be debris? Only one.
POSSIBLY_DEBRIS = ["unknown_floating"]


def wilson(k: int, n: int, z: float = 1.96):
    """
    Wilson score interval.

    Used rather than the normal approximation because several categories will
    have few or zero counts, where the normal interval is badly behaved and
    can extend below zero. Wilson stays inside [0, 1] and is well behaved at
    k = 0, which matters here: a category with no observations should give an
    informative upper bound, not a degenerate point.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    import numpy as np
    import pandas as pd

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(MQ / "water_annotations.csv"))
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"{path} not found.\n"
              f"Open {MQ / 'annotate_water.html'} in a browser, label the "
              f"queue, click Export CSV, and save it there.")
        return 1

    df = pd.read_csv(path)

    # The labels must belong to the queue 34 built last. Patch ids recur from
    # one build to the next, so ids alone cannot tell an old CSV from a new
    # one; the position of every row is checked against water_queue.json.
    # When nothing is scored, water_annotation.json is overwritten with the
    # reason, so an earlier result cannot be read as the current one.
    def _not_scored(status, msg, **extra):
        print(msg)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "water_annotation.json").write_text(json.dumps(
            {"status": status, "note": msg, "csv": str(path), **extra}, indent=2),
            encoding="utf-8")
        return 2

    qpath = MQ / "water_queue.json"
    if not qpath.exists():
        return _not_scored(
            "no_queue", f"No water_queue.json: {path.name} was labelled for a queue "
            f"built before queues were tracked, so it cannot be matched to the "
            f"published detections. Run 34_build_water_annotation.py first.")
    q = json.loads(qpath.read_text(encoding="utf-8"))
    if not q["items"]:
        return _not_scored(
            "no_published_detections",
            f"The current queue ({q['source']}) is empty: the published layer has "
            f"no patches there, so there is nothing to score.", source=q["source"])
    pos = {i["patch_id"]: (round(i["lat"], 5), round(i["lon"], 5)) for i in q["items"]}
    mism = [r.patch_id for r in df.itertuples()
            if pos.get(r.patch_id) != (round(r.lat, 5), round(r.lon, 5))]
    if mism:
        return _not_scored(
            "stale_labels",
            f"{len(mism)} of {len(df)} rows in {path.name} do not match the current "
            f"queue ({q['stamp']}, {q['source']}). These labels were made for an "
            f"earlier queue and are NOT scored. Label "
            f"{MQ / 'annotate_water.html'} and export again.", source=q["source"])
    total = len(df)
    df = df[df["category"].notna() & (df["category"].astype(str) != "")]
    n = len(df)
    if n == 0:
        print(f"no labelled rows in {path} ({total} rows present)")
        return 1

    print(f"{n} of {total} queued detections labelled "
          f"({n / total:.0%})\n")

    counts = df["category"].value_counts()
    rows = []
    for cat, k in counts.items():
        lo, hi = wilson(int(k), n)
        rows.append({"category": cat, "n": int(k), "share": k / n,
                     "ci_low": lo, "ci_high": hi,
                     "kind": ("identified non-debris" if cat in CONFUSER
                              else "unresolved")})
    tab = pd.DataFrame(rows).sort_values("n", ascending=False)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    n_conf = int(counts.reindex(CONFUSER).fillna(0).sum())
    n_unres = int(counts.reindex(UNRESOLVED).fillna(0).sum())
    n_maybe = int(counts.reindex(POSSIBLY_DEBRIS).fillna(0).sum())
    c_lo, c_hi = wilson(n_conf, n)
    m_lo, m_hi = wilson(n_maybe, n)

    print(f"\nIDENTIFIED AS SOMETHING OTHER THAN DEBRIS: "
          f"{n_conf}/{n} = {n_conf / n:.1%} "
          f"[{c_lo:.1%}, {c_hi:.1%}]")
    print(f"  -> a LOWER bound on the false-positive rate of the FDI layer")
    print(f"COULD CONCEIVABLY BE DEBRIS (floating, unidentifiable): "
          f"{n_maybe}/{n} = {n_maybe / n:.1%} [{m_lo:.1%}, {m_hi:.1%}]")
    print(f"  -> an UPPER bound on the debris share")
    print(f"UNRESOLVED IN TOTAL: {n_unres}/{n} = {n_unres / n:.1%}")

    # ---- against the MARIDA benchmark -------------------------------------
    mar = OUT / "summary.json"
    comparison = {}
    if mar.exists():
        M = json.loads(mar.read_text(encoding="utf-8"))
        bench = {r["class"]: r["share_of_all_detections"]
                 for r in M["rule"]["detection_composition"]}
        pairs = [("vessel", "Ship"), ("wake", "Wakes"), ("foam", "Foam"),
                 ("natural", "Sparse Sargassum"),
                 ("sediment", "Turbid Water"), ("cloud", "Clouds")]
        print(f"\n{'category':12s} {'Guinea (observed)':>18s} "
              f"{'MARIDA (benchmark)':>20s}")
        for ours, theirs in pairs:
            g = counts.get(ours, 0) / n
            b = bench.get(theirs, 0.0)
            comparison[ours] = {"guinea": g, "marida": b, "marida_class": theirs}
            print(f"{ours:12s} {g:17.1%} {b:19.1%}")
        print("\nThe benchmark is a different coast with a different vessel "
              "density and water type, so exact agreement is not expected. "
              "What matters is whether the RANKING holds -- in particular "
              "whether vessels dominate here as they do there.")

    # ---- against the classifier's prediction for the same scene ------------
    if mar.exists():
        M = json.loads(mar.read_text(encoding="utf-8"))
        named = {r["predicted_class"]: r["share"]
                 for r in M.get("transfer", {}).get("named", [])}
        if named:
            print(f"\nClassifier's own reading of the Conakry composite, for "
                  f"reference:")
            for k, v in sorted(named.items(), key=lambda kv: -kv[1])[:5]:
                print(f"  {k:26s} {v:6.1%}")
            print("  (composite, not the single date annotated here; "
                  "see section 7.1.14.4 for why the two differ)")

    payload = {
        "csv": str(path), "n_queued": total, "n_labelled": n,
        "counts": {k: int(v) for k, v in counts.items()},
        "shares": {k: float(v / n) for k, v in counts.items()},
        "identified_non_debris": {"n": n_conf, "share": n_conf / n,
                                  "ci": [c_lo, c_hi]},
        "possibly_debris_upper_bound": {"n": n_maybe, "share": n_maybe / n,
                                        "ci": [m_lo, m_hi]},
        "unresolved": {"n": n_unres, "share": n_unres / n},
        "vs_marida": comparison,
        "note": ("No debris precision is computed. The annotation scheme has "
                 "no debris category by design: material identity is not "
                 "decidable by eye at 10 m. The identified non-debris share "
                 "is a LOWER bound on the false-positive rate and the "
                 "unidentifiable-floating share is an UPPER bound on the "
                 "debris share."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "water_annotation.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'water_annotation.json'}")

    if n < 60:
        print(f"\nNOTE: {n} labels gives a confidence interval about "
              f"{(c_hi - c_lo) * 100:.0f} points wide. Around 150 would "
              f"halve it; beyond roughly 400 the interval narrows slowly and "
              f"the limiting factor becomes whether the sample is "
              f"representative, not how large it is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
