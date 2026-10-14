"""
39 -- Freeze the MARIDA debris classifier that filters the FDI rule's detections.

WHY. With the Floating Debris Index computed as Biermann et al. (2020) define it,
the deployed mean+3sigma FDI-and-NDVI rule responds to anything that raises
near-infrared above the water baseline, vegetation included: at Conakry
(January 2024) the flagged pixels have a median NDVI of 0.47 and 83% lie within
one 20 m pixel of land, and on MARIDA Sargassum is its largest false-positive
class. Biermann et al. separate debris from vegetation and seaweed with a
classifier trained on labelled spectra, not with a threshold. MARIDA supplies
those labels, so the deployed layer keeps a rule detection only where a
classifier trained on MARIDA calls the pixel Marine Debris.

WHAT. A random forest, debris vs everything else, over the twelve features the
Guinea fetch already provides (marida_tool.domain.PIPELINE_FEATURES), trained on
MARIDA's published train split with the settings script 32 evaluates. A pixel is
kept when P(debris) >= 0.5, the classifier's own decision boundary for a binary
model; no threshold is tuned to Guinea.

It is measured here exactly as deployed -- the FDI-and-NDVI rule first, then
the classifier on the rule's detections -- on MARIDA's test split, all of it and
the geographically unseen tiles alone. Those are the figures to quote for the
published layer. The same sequence with the FDI arm alone (the form section 2
of marida_workflow.ipynb evaluates) is recorded beside it for comparison.

    python 39_freeze_debris_filter.py                 # 400 trees, as script 32
    python 39_freeze_debris_filter.py --trees 200

Writes fdi_tool/models/debris_filter.joblib and debris_filter_meta.json.
Needs the MARIDA archive (see marida_tool/data.py). No Earth Engine call.
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from plasticwaste_tool.marida_tool import baseline as B      # noqa: E402
from plasticwaste_tool.marida_tool import data as D          # noqa: E402
from plasticwaste_tool.marida_tool import domain as DOM      # noqa: E402
from plasticwaste_tool.marida_tool import features as F      # noqa: E402
from plasticwaste_tool.marida_tool import model as M         # noqa: E402

MODELS = HERE / "fdi_tool" / "models"
P_MIN = 0.5          # binary classifier's decision boundary; not tuned


def _scores(yte, pred, scored):
    s = B.binary_scores(yte, pred, scored)
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in s.items()}


def _composition(yte, pred):
    df = B.what_does_it_fire_on(yte, pred)
    df = df[df.flagged > 0]
    return [{"class": r["class"], "flagged": int(r["flagged"]),
             "share": round(float(r["share_of_all_detections"]), 4)}
            for _, r in df.iterrows()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--min-leaf", type=int, default=2)
    args = ap.parse_args()

    import joblib
    import sklearn
    from sklearn.ensemble import RandomForestClassifier

    print("loading MARIDA ...")
    Xtr, ytr, ptr = D.load_pixels("train")
    Xte, yte, pte = D.load_pixels("test")
    tile_tr = {D.normalise_id(p).split("_")[2] for p in ptr}
    tile_te = np.array([D.normalise_id(p).split("_")[2] for p in pte])
    scene_te = np.array([p.rsplit("_", 1)[0] for p in pte])
    unseen = ~np.isin(tile_te, list(tile_tr))

    cols = list(DOM.PIPELINE_FEATURES)
    Ftr, Fte = F.build(Xtr)[cols], F.build(Xte)[cols]

    print(f"training: {len(ytr):,} px, {int((ytr == D.TARGET).sum()):,} debris, "
          f"{args.trees} trees, min_samples_leaf {args.min_leaf} ...")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=args.trees, min_samples_leaf=args.min_leaf,
        class_weight="balanced_subsample", n_jobs=-1, random_state=M.SEED)
    clf.fit(Ftr.values, (ytr == D.TARGET).astype(int))
    print(f"  {time.time() - t0:.0f}s")

    p = clf.predict_proba(Fte.values)[:, list(clf.classes_).index(1)]
    # "rule" is the deployed FDI-and-NDVI rule; "fdi_rule" is its FDI arm alone.
    r = B.predict(Xte, yte, group=scene_te, with_ndvi=True)
    r_fdi = B.predict(Xte, yte, group=scene_te)
    rule = r["pred"] & r["scored"]
    filtered = rule & (p >= P_MIN)
    fdi_rule = r_fdi["pred"] & r_fdi["scored"]
    fdi_filtered = fdi_rule & (p >= P_MIN)

    metrics = {}
    for tag, m in (("all_test", r["scored"]), ("unseen_tiles", r["scored"] & unseen)):
        metrics[tag] = {"rule": _scores(yte, rule, m),
                        "rule_then_classifier": _scores(yte, filtered, m),
                        "fdi_rule": _scores(yte, fdi_rule, m),
                        "fdi_rule_then_classifier": _scores(yte, fdi_filtered, m)}
    metrics["composition"] = {
        "rule": _composition(yte, rule),
        "rule_then_classifier": _composition(yte, filtered),
        "fdi_rule": _composition(yte, fdi_rule),
        "fdi_rule_then_classifier": _composition(yte, fdi_filtered)}

    print("\nMARIDA test split (classifier keeps P >= 0.5):")
    for tag in ("all_test", "unseen_tiles"):
        for key, name in (("rule", "FDI-and-NDVI rule (deployed)"),
                          ("rule_then_classifier", "  then classifier (published)"),
                          ("fdi_rule", "FDI rule alone"),
                          ("fdi_rule_then_classifier", "  then classifier")):
            a = metrics[tag][key]
            print(f"  {tag:<13} {name:<31} precision {a['precision'] or 0:.3f}  "
                  f"recall {a['recall'] or 0:.3f}  ({a['tp'] + a['fp']:,} px)")
    print("\n  what the published layer keeps:")
    for c in metrics["composition"]["rule_then_classifier"][:6]:
        print(f"    {c['class']:<26} {c['share']:6.1%}  ({c['flagged']:,} px)")

    MODELS.mkdir(parents=True, exist_ok=True)
    model_p = MODELS / "debris_filter.joblib"
    joblib.dump(clf, model_p, compress=3)
    meta = {
        "what": "MARIDA debris classifier applied to the FDI rule's detections",
        "frozen": date.today().isoformat(),
        "generated_by": "39_freeze_debris_filter.py",
        "features": cols, "p_min": P_MIN, "positive_class": "Marine Debris",
        "model": {"type": "RandomForestClassifier", "n_estimators": args.trees,
                  "min_samples_leaf": args.min_leaf,
                  "class_weight": "balanced_subsample", "random_state": M.SEED,
                  "sklearn": sklearn.__version__},
        "training_data": {"dataset": "MARIDA (Kikaki et al. 2022)",
                          "doi": "10.5281/zenodo.5151941", "split": "train",
                          "n_px": int(len(ytr)),
                          "n_debris_px": int((ytr == D.TARGET).sum())},
        "fdi_formula": __import__("plasticwaste_tool.config", fromlist=["S2_CONFIG"])
                       .S2_CONFIG["fdi_formula"],
        "metrics_marida_test": metrics,
    }
    (MODELS / "debris_filter_meta.json").write_text(
        json.dumps(meta, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {model_p} ({model_p.stat().st_size / 1e6:.1f} MB) and its meta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
