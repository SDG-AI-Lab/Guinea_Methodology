"""
Would a Feature Pyramid Network fix the terrestrial classifier? Tested, no.

WHY THIS EXISTS
---------------
Section 7.2.8 established that the shipped classifier tracks surface
disturbance rather than waste: its score is independent of how much waste a
window contains, and bare construction ground outscores every waste category.
The stated cause was spatial -- a 14 m dumpsite (7.3.5) occupies about 3% of an
80 m window, and the probe averages features over the whole window, so the
target is diluted by the other 97%.

That diagnosis makes a prediction, and the prediction is cheap to test. If
dilution is the mechanism, then representing the window at a finer scale, or
aggregating over it with an operator that does not average, should recover a
response to waste. This is also exactly what the AerialWaste model (Torres and
Fraternali, 2023) does differently: ResNet-50 plus a Feature Pyramid Network
(Lin et al., 2017), whose contribution is not the classifier head but the
carrying forward of earlier, higher-resolution layers so that small objects are
represented where they still occupy a meaningful share of a unit's receptive
field.

So before recommending that architecture, test whether the mechanism it
addresses is the one that binds here.

WHAT IS VARIED, AND WHAT IS NOT
-------------------------------
Same frozen ImageNet ResNet-50, same tiles, same labels, same spatially blocked
split, same regularised logistic probe. Only the representation changes, so any
difference is attributable to it alone.

  SCALE -- which layer the features are taken from. Receptive field, converted
  to ground distance on an 800 px rendering of an 80 m window:

      layer2   512 ch   100x100   RF ~99 px  ~= 10 m    (dumpsite scale)
      layer3  1024 ch    50x50    RF ~291 px ~= 29 m
      layer4  2048 ch    25x25    RF ~483 px ~= 48 m    (what ships today)

  POOLING -- how positions are aggregated. `gap` is the mean over all
  positions, which is what ships. `topk` is the mean of the k strongest
  positions; k=5 of 625 is 0.8%, the same order as the target's share of the
  window, so it is the natural non-diluting alternative.

  pyramid -- layer2, layer3 and layer4 concatenated: a crude approximation of
  what FPN does properly, and the closest this ablation comes to the AerialWaste
  architecture without retraining anything.

THE DECISIVE READ IS NOT ACCURACY
---------------------------------
ROC-AUC near 0.7 is already achievable by ranking disturbed ground above
vegetation, so a variant scoring 0.74 has not necessarily learned anything
about waste. The test that distinguishes them is the dose-response of 7.2.8,
computed here for every variant: among windows that contain waste, does the
score rise with how much waste is present? That comparison is internal to the
positive class and is therefore unaffected by where the class boundary was
drawn.

RESULT
------
Every variant lands between ROC-AUC 0.68 and 0.74, and NOT ONE shows a
dose-response. Finer scale does not help; non-diluting pooling makes matters
slightly worse; the pyramid is no better than the single layer it replaces.

The diagnosis in 7.2.8 therefore needs amending, and the amendment matters for
what to do next. Dilution is not the binding constraint. Generic ImageNet
features do not encode waste as distinct from disturbed ground AT ANY SCALE, so
there is nothing for a finer representation to recover and nothing for a
localisation mechanism to localise. FPN helps AerialWaste because its features
were trained on roughly ten thousand labelled waste images and already
discriminate waste; the pyramid then tells it where. Adopting the architecture
without the supervision that fills it would import the cost and none of the
benefit.

Usage:
    python 28_architecture_ablation.py --town boke
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

HERE = Path(__file__).resolve().parent
MQ = HERE / "manual_qa"
RUNS = MQ / "runs"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 800
TOPK = 5

# Receptive field per layer, converted to ground metres for an 80 m window
# rendered at 800 px. Approximate, and only used for reporting.
GROUND_M = {"l2": "10", "l3": "29", "l4": "48"}

# The probe is fitted exactly as the shipped one is, so a difference cannot be
# attributed to a different regulariser.
PROBE_C = 0.01


def load_grid(town):
    path = MQ / "regions" / f"{town}_blindgrid_review.geojson"
    cache = MQ / f"blindgrid_cache_{town}" / "confirm"
    feats = json.loads(path.read_text(encoding="utf-8"))["features"]
    rows = []
    for f in feats:
        p = f["properties"]
        if str(p.get("label", "")).strip() == "":
            raise SystemExit(f"{path.name}: unlabelled windows remain.")
        y = int(p["label"])
        # Digitised waste points; a negative keeps the untouched default
        # coordinate, which is not an annotation.
        lats = [t for t in str(p["lat"]).split(";") if t.strip()]
        rows.append({"y": y, "half": p["half"],
                     "score": float(p["model_score"]),
                     "n_points": len(lats) if y == 1 else 0,
                     "path": cache / f"{p['tile_id']}.png"})
    keep = [r for r in rows if r["path"].exists()]
    if not keep:
        raise SystemExit(f"No cached tiles in {cache}.")
    return keep


def extract(paths, k=TOPK):
    """gap and topk vectors from layer2, layer3 and layer4, per image.

    The maps are reduced as they are produced: retaining every 2048x25x25
    layer4 map for 557 windows alone would need about 2.7 GB, and they are
    wanted only long enough to be pooled.
    """
    from torchvision.models import ResNet50_Weights, resnet50
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).eval()
    stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    acc = {f"{lay}_{how}": [] for lay in ("l2", "l3", "l4")
           for how in ("gap", "topk")}
    with torch.no_grad():
        for i, p in enumerate(paths, 1):
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
            h = net.layer1(stem(x))
            outs = {}
            h = net.layer2(h); outs["l2"] = h
            h = net.layer3(h); outs["l3"] = h
            h = net.layer4(h); outs["l4"] = h
            for lay, m in outs.items():
                flat = m[0].reshape(m.shape[1], -1)
                acc[f"{lay}_gap"].append(flat.mean(dim=1).numpy())
                acc[f"{lay}_topk"].append(
                    flat.topk(k, dim=1).values.mean(dim=1).numpy())
            if i % 100 == 0:
                print(f"  features {i}/{len(paths)}", flush=True)
    return {n: np.asarray(v, np.float32) for n, v in acc.items()}


def evaluate(X, y, half, npts, n_flagged):
    """Fit on one spatial half, score the other, both directions.

    The two halves are separate ground, which is the only honest split this
    grid supports; a random split would let the probe train on the
    neighbourhood beside the one it is tested on.
    """
    preds = np.full(len(y), np.nan)
    for tr, te in (("west", "east"), ("east", "west")):
        a, b = half == tr, half == te
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=4000, C=PROBE_C))
        clf.fit(X[a], y[a])
        preds[b] = clf.predict_proba(X[b])[:, 1]
    pm = y == 1
    rho, pv = spearmanr(npts[pm], preds[pm])
    return {
        "roc_auc": float(roc_auc_score(y, preds)),
        "avg_precision": float(average_precision_score(y, preds)),
        "precision_at_top_n": float(y[np.argsort(-preds)[:n_flagged]].mean()),
        "dose_rho": float(rho), "dose_p": float(pv),
        "dose_response": bool(pv < 0.05 and rho > 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="boke")
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore cached feature vectors and re-extract")
    args = ap.parse_args()
    town = args.town.lower()

    rows = load_grid(town)
    y = np.array([r["y"] for r in rows])
    half = np.array([r["half"] for r in rows])
    npts = np.array([r["n_points"] for r in rows])
    shipped = np.array([r["score"] for r in rows])
    print(f"{len(rows)} cached windows, {int(y.sum())} positive, "
          f"base rate {y.mean():.3f}")

    cache = MQ / f"_ablation_vectors_{town}.npz"
    if cache.exists() and not args.rebuild:
        d = np.load(cache)
        F = {k: d[k] for k in d.files}
        print(f"loaded cached vectors from {cache.name}")
    else:
        F = extract([r["path"] for r in rows])
        np.savez_compressed(cache, **F)
        print(f"wrote {cache.name}")

    F["pyramid_gap"] = np.hstack([F["l2_gap"], F["l3_gap"], F["l4_gap"]])
    F["pyramid_topk"] = np.hstack([F["l2_topk"], F["l3_topk"], F["l4_topk"]])

    # Match the count the shipped model flags at its 0.90 tier, so
    # precision-at-N compares like with like rather than at a fixed rate.
    n_flagged = int((shipped >= 0.90).sum()) or 76

    pm = y == 1
    srho, spv = spearmanr(npts[pm], shipped[pm])

    print(f"\n{'representation':16s} {'dim':>5s} {'~RF m':>6s} {'ROC-AUC':>8s} "
          f"{'AP':>7s} {f'prec@{n_flagged}':>9s} {'dose rho':>9s} {'P':>6s}")
    print(f"{'shipped probe':16s} {2048:5d} {'48':>6s} "
          f"{roc_auc_score(y, shipped):8.3f} "
          f"{average_precision_score(y, shipped):7.3f} "
          f"{y[np.argsort(-shipped)[:n_flagged]].mean():9.3f} "
          f"{srho:9.3f} {spv:6.3f}")
    print("  " + "-" * 76)

    order = ["l2_gap", "l2_topk", "l3_gap", "l3_topk", "l4_gap", "l4_topk",
             "pyramid_gap", "pyramid_topk"]
    res = {}
    for name in order:
        r = evaluate(F[name], y, half, npts, n_flagged)
        res[name] = r
        rf = GROUND_M.get(name.split("_")[0], "-")
        flag = "  <-- DOSE-RESPONSE" if r["dose_response"] else ""
        print(f"{name:16s} {F[name].shape[1]:5d} {rf:>6s} "
              f"{r['roc_auc']:8.3f} {r['avg_precision']:7.3f} "
              f"{r['precision_at_top_n']:9.3f} {r['dose_rho']:9.3f} "
              f"{r['dose_p']:6.3f}{flag}")

    any_dose = any(v["dose_response"] for v in res.values())
    best = max(res.items(), key=lambda kv: kv[1]["roc_auc"])
    verdict = (
        "NO VARIANT RECOVERS A DOSE-RESPONSE. Neither a finer scale nor a "
        "non-diluting pool makes the score respond to the quantity of waste "
        "present. Spatial dilution is therefore not the binding constraint, "
        "and an FPN-style architecture would not address the limitation: "
        "generic ImageNet features do not encode waste as distinct from "
        "disturbed ground at any scale, so a pyramid has nothing further to "
        "recover. The constraint is the volume of task-specific supervision, "
        "not the architecture."
        if not any_dose else
        f"{best[0]} shows a dose-response; the representation, not the data "
        f"volume, was the constraint. Re-examine 7.2.8.")

    out = RUNS / f"blindgrid_{town}"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "architecture_ablation.json"
    dest.write_text(json.dumps({
        "town": town, "n": int(len(y)), "base_rate": float(y.mean()),
        "n_flagged_reference": n_flagged, "topk_k": TOPK, "probe_C": PROBE_C,
        "shipped": {"roc_auc": float(roc_auc_score(y, shipped)),
                    "avg_precision": float(average_precision_score(y, shipped)),
                    "dose_rho": float(srho), "dose_p": float(spv)},
        "variants": res, "any_dose_response": any_dose, "verdict": verdict,
    }, indent=2), encoding="utf-8")

    print(f"\nVERDICT: {verdict}")
    print(f"\nWritten to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
