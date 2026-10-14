"""Copied from 02c_feature_probe.py and renamed so it can be imported.

The numbered original stays in place for the notebooks; this copy is
what the package and the backend import. Runtime importlib loads of
sibling scripts have been rewritten as relative imports.
Regenerate with scratchpad/build_package.py -- do not edit by hand
until the originals are retired.
"""

"""
02c -- Diagnostic: does the frozen AerialWaste backbone carry ANY signal that
separates Guinea dumpsites from non-dumpsites?

WHY THIS EXISTS
---------------
Fine-tuning via 02b_finetune_aerialwaste.py produced a completely flat run:
val MCC 0.000 / val BalAcc 0.500 for all 20 epochs. Root cause found by
inspecting Net.forward() in aerialwaste_model/architecture/resnet50_fpn.py:

    out5 = F.relu(self.fc(p5))      # fc is Linear(256 -> num_classes)
    out4 = F.relu(self.fc(p4))      # so with num_classes=1 each is ONE scalar
    out3 = F.relu(self.fc(p3))
    out2 = F.relu(self.fc(p2))
    out  = self.classifier(torch.cat([out5, out4, out3, out2], dim=1))

With num_classes=1 the whole 800x800 image is squeezed into FOUR non-negative
scalars before the final Linear(4 -> 1). When all four `fc` outputs are
negative they are clamped to exactly 0 by the ReLUs, and the network's output
degenerates to `classifier.bias` -- a constant carrying zero information about
the image. That bias is 0.742915689945221, which is precisely the logit
observed on 8 of 10 randomly sampled Guinea tiles regardless of their label.

This also explains why fine-tuning could not recover: the gradient through a
dead ReLU is exactly zero, so for those samples NO gradient reaches `self.fc`
at all. Only `classifier.bias`/`classifier.weight` got updates, which is what
a tiny, perfectly linear loss decrease with zero metric movement looks like.

WHAT THIS SCRIPT MEASURES
-------------------------
The 4-scalar bottleneck is a property of the HEAD, not of the features. This
script taps the pooled pyramid features *upstream* of it (p5/p4/p3/p2, 256-d
each after global average pooling = 1024-d total) and asks whether a simple
cross-validated linear probe can separate the classes from those features.

  * If the probe scores meaningfully above chance -> the backbone does encode
    usable Guinea signal, and the fix is a new head that bypasses the dead
    4-scalar ReLU gate (not more epochs, not a different learning rate).
  * If the probe is at chance -> the frozen features genuinely do not
    transfer to this domain, and the path forward is unfreezing backbone
    layers and/or substantially more labels.

Metrics: ROC-AUC is the headline number because it is threshold-free -- it
measures whether the features RANK positives above negatives, which is what
matters for an annotation-queue use case, and it is unaffected by the
degenerate 0.5-threshold behaviour that made balanced accuracy read exactly
0.500. MCC is reported alongside for the same reason 02b/02_train_baseline
prefer it over F1 under class imbalance (Chicco & Jurman 2020).

Usage:
    python 02c_feature_probe.py \
        --labels manual_qa/manual_labels_template.csv \
        --grid manual_qa/browsing_grid.geojson manual_qa/poi_grid.geojson \
        --images-dir manual_qa/tiles \
        --out runs/feature_probe
"""

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plasticwaste_tool.aerialwaste_model.architecture import torchutils  # noqa: E402
from plasticwaste_tool.aerialwaste_model.architecture.resnet50_fpn import Net  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 800


@torch.no_grad()
def extract(net, x):
    """
    Re-implements Net.forward() up to (but NOT through) the 4-scalar ReLU
    bottleneck, returning both the rich pooled pyramid features and the
    bottleneck's own pre-ReLU values so the dead-gate rate can be counted.

    Kept deliberately as a copy of the upstream forward rather than a hook so
    that any future upstream change shows up as a diff here instead of
    silently changing what this diagnostic measures.
    """
    c1 = net.stage0(x)
    c2 = net.stage1(c1)
    c3 = net.stage2(c2)
    c4 = net.stage3(c3)
    c5 = net.stage4(c4)

    p5 = net.toplayer(c5)
    p4 = net._upsample_cat(p5, net.latlayer1(c4))
    p3 = net._upsample_cat(p4, net.latlayer2(c3))
    p2 = net._upsample_cat(p3, net.latlayer3(c2))

    p4 = net.smooth1(p4)
    p3 = net.smooth2(p3)
    p2 = net.smooth3(p2)

    pooled = [torchutils.gap2d(p, keepdims=True).view(p.size(0), -1)
              for p in (p5, p4, p3, p2)]
    feats = torch.cat(pooled, dim=1)                      # [B, 1024]
    pre_relu = torch.cat([net.fc(p) for p in pooled], 1)  # [B, 4]
    logit = net.classifier(F.relu(pre_relu))              # [B, 1] (as upstream)
    return feats, pre_relu, logit


def to_local_xy(lat, lon):
    """Local equirectangular projection to metres (fine over a city AOI)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = lat.mean()
    return np.column_stack([
        np.radians(lon - lon.mean()) * 6_371_000.0 * np.cos(np.radians(lat0)),
        np.radians(lat - lat0) * 6_371_000.0,
    ])


def spatial_groups(lat, lon, block_m):
    """
    Assign each tile to a square spatial BLOCK of block_m metres; the block id
    is the CV group. This is spatial block cross-validation (Roberts et al.
    2017, "Cross-validation strategies for data with temporal, spatial,
    hierarchical, or phylogenetic structure", Ecography 40:913-929).

    WHY IT IS NEEDED HERE: this project's labels contain contiguous runs of
    40m tiles covering ONE physical dumpsite (r0019c0010/0011/0012,
    r0020c0010..0013, r0010c0016/r0011c0016/r0012c0016). Under plain k-fold
    CV a site's tiles get split across train and test, so the model is scored
    on near-duplicates of what it just trained on -- spatial-autocorrelation
    leakage, which inflates every metric. Same motivation as the
    grouped_split() already used by 02_train_baseline.py.

    WHY BLOCKS AND NOT SINGLE-LINKAGE CLUSTERING: clustering tile centroids
    with single linkage was tried first and chained badly on this dataset --
    at a 250m threshold 118 of 124 tiles collapsed into ONE group (and every
    group then contained positives, leaving folds with a single class and no
    fittable split). Fixed-size blocks partition space without chaining.
    """
    xy = to_local_xy(lat, lon)
    ij = np.floor(xy / float(block_m)).astype(int)
    # pair the (row, col) block indices into a single group id
    _, groups = np.unique(ij, axis=0, return_inverse=True)
    return groups, xy


def cv_probe(X, y, splitter, split_args, label):
    oof = np.zeros(len(y), dtype=float)
    for tr, te in splitter.split(*split_args):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", C=0.1))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    preds = (oof >= 0.5).astype(int)
    return {
        "cv": label,
        "roc_auc": float(roc_auc_score(y, oof)),
        "mcc": float(matthews_corrcoef(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
    }, oof


def bootstrap_auc_ci(y, scores, groups, n_boot=2000, seed=0, alpha=0.05):
    """
    Percentile bootstrap CI for ROC-AUC, resampling whole spatial BLOCKS
    rather than individual tiles.

    Resampling tiles independently would understate the interval here: tiles
    from one site are near-duplicates, so treating them as independent draws
    fakes a larger effective sample. The block is the unit that is actually
    (approximately) independent, so that is the unit resampled -- the cluster
    bootstrap, same logic that motivates the blocked CV above.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    aucs = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        aucs.append(roc_auc_score(yy, scores[idx]))
    aucs = np.array(aucs)
    return (float(np.percentile(aucs, 100 * alpha / 2)),
            float(np.percentile(aucs, 100 * (1 - alpha / 2))),
            int(len(aucs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--grid", required=True, nargs="+")
    ap.add_argument("--images-dir", default="manual_qa/tiles")
    ap.add_argument("--checkpoint", default=str(
        Path(__file__).parent / "aerialwaste_model" / "weights" / "checkpoint.pth"))
    ap.add_argument("--out", default="runs/feature_probe")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-m", type=float, default=200.0,
                    help="edge length of the square spatial blocks used as CV "
                         "groups (spatial block cross-validation)")
    ap.add_argument("--reuse-features", action="store_true",
                    help="skip the (slow) forward passes and load features.npz "
                         "from --out, for re-running just the CV analysis")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = importlib.util.spec_from_file_location(
        "ft", Path(__file__).parent / "02b_finetune_aerialwaste.py")
    ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)

    labels = pd.read_csv(args.labels).dropna(subset=["label"]).copy()
    labels["label"] = labels["label"].astype(int)
    manifest = ft.fetch_labeled_tile_images(labels, args.grid, args.images_dir)
    logger.info("Using %d labeled tiles (%d positive)",
                len(manifest), int(manifest.label.sum()))

    net = Net(num_classes=1, pretrained=False)
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                   weights_only=False), strict=True)
    net.eval()

    cache = out_dir / "features.npz"
    cache_ok = False
    if args.reuse_features and cache.exists():
        d = np.load(cache, allow_pickle=True)
        cached_ids = set(map(str, d["tile_id"]))
        current_ids = set(map(str, manifest.tile_id))
        # Without this check, .loc[d["tile_id"]] below would quietly SUBSET the
        # manifest to whatever was cached -- so adding new labels and re-running
        # with --reuse-features would report the OLD result as if it included
        # them. Fail loud and re-extract instead.
        if cached_ids == current_ids:
            cache_ok = True
        else:
            logger.warning(
                "Cached features cover %d tiles but the label set now has %d "
                "(%d added, %d removed) -- IGNORING --reuse-features and "
                "re-extracting so the result reflects the current labels.",
                len(cached_ids), len(current_ids),
                len(current_ids - cached_ids), len(cached_ids - current_ids))

    if cache_ok:
        X, y, pre_relu, base_logit = d["X"], d["y"], d["pre_relu"], d["base_logit"]
        # keep manifest row order aligned with the cached feature rows
        manifest = manifest.set_index("tile_id").loc[d["tile_id"]].reset_index()
        logger.info("Reusing cached features from %s (%d tiles)", cache, len(y))
    else:
        tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        from PIL import Image
        feats, pre_relus, logits = [], [], []
        for i, row in enumerate(manifest.itertuples(), 1):
            img = tf(Image.open(row.image_path).convert("RGB")).unsqueeze(0)
            f, pr, lg = extract(net, img)
            feats.append(f.numpy()[0])
            pre_relus.append(pr.numpy()[0])
            logits.append(lg.item())
            if i % 25 == 0:
                logger.info("  extracted %d/%d", i, len(manifest))

        X = np.stack(feats)
        pre_relu = np.stack(pre_relus)
        base_logit = np.array(logits)
        y = manifest.label.to_numpy()

        np.savez(cache, X=X, y=y, pre_relu=pre_relu,
                 base_logit=base_logit, tile_id=manifest.tile_id.to_numpy())

    n_dead = int((pre_relu <= 0).all(axis=1).sum())
    print("\n" + "=" * 64)
    print("1. THE DEAD-RELU BOTTLENECK")
    print("=" * 64)
    print(f"  tiles where ALL 4 relu gates are closed : {n_dead}/{len(y)} "
          f"({100*n_dead/len(y):.1f}%)")
    print("  -> those tiles ALL produce the identical output "
          f"{base_logit[(pre_relu <= 0).all(axis=1)][:1]}, which is exactly "
          "classifier.bias")
    print("  -> and contribute ZERO gradient to fc during fine-tuning")
    print(f"  distinct base-model logits              : {len(np.unique(np.round(base_logit,4)))}"
          f" across {len(y)} tiles")

    print("\n" + "=" * 64)
    print("2. BASE MODEL AS A RANKER (threshold-free)")
    print("=" * 64)
    try:
        print(f"  ROC-AUC of raw base logit  : {roc_auc_score(y, base_logit):.3f}"
              "   (0.5 = coin flip)")
    except ValueError as e:
        print(f"  ROC-AUC unavailable: {e}")

    print("\n" + "=" * 64)
    print(f"3. LINEAR PROBE ON POOLED PYRAMID FEATURES ({X.shape[1]}-d)")
    print("=" * 64)

    groups, _ = spatial_groups(manifest.center_lat, manifest.center_lon,
                               args.block_m)
    n_groups = len(np.unique(groups))
    gdf = pd.DataFrame({"g": groups, "y": y})
    pos_groups = int(gdf[gdf.y == 1].g.nunique())
    neg_groups = int(gdf[gdf.y == 0].g.nunique())
    print(f"  {len(y)} tiles fall into {n_groups} spatial blocks of "
          f"{args.block_m:.0f}m ({pos_groups} contain a positive, "
          f"{neg_groups} contain a negative)")

    random_probe, _ = cv_probe(
        X, y, StratifiedKFold(n_splits=args.folds, shuffle=True,
                              random_state=args.seed), (X, y), "random k-fold")

    n_gfolds = int(min(args.folds, n_groups))
    spatial_probe, spatial_oof = cv_probe(
        X, y, StratifiedGroupKFold(n_splits=n_gfolds, shuffle=True,
                                   random_state=args.seed),
        (X, y, groups),
        f"spatial block CV ({args.block_m:.0f}m blocks, k={n_gfolds})")
    lo, hi, n_ok = bootstrap_auc_ci(y, spatial_oof, groups, seed=args.seed)
    spatial_probe["roc_auc_ci95"] = [lo, hi]

    print()
    print(f"  {'split':<38} {'ROC-AUC':>8} {'MCC':>8} {'BalAcc':>8}")
    for p in (random_probe, spatial_probe):
        print(f"  {p['cv']:<38} {p['roc_auc']:>8.3f} {p['mcc']:>+8.3f} "
              f"{p['balanced_accuracy']:>8.3f}")
    print()
    print()
    print(f"  Spatial-block AUC 95% CI (block bootstrap): "
          f"[{lo:.3f}, {hi:.3f}]  (n={n_ok} resamples)")
    print()
    print("  The spatially-blocked row is the HONEST estimate. The random row")
    print("  is inflated whenever adjacent tiles of one physical site land on")
    print("  opposite sides of the split -- report the blocked number.")

    probe = dict(spatial_probe)
    probe["n"] = int(len(y))
    probe["n_positive"] = int(y.sum())

    print("\n" + "=" * 64)
    print("VERDICT (based on the spatially-blocked estimate)")
    print("=" * 64)
    if probe["roc_auc"] >= 0.70:
        print("  The frozen backbone DOES encode usable Guinea signal.")
        print("  The 4-scalar ReLU gate in Net.forward() is what is throwing it")
        print("  away. Fix = train a new head on these 1024-d pooled features")
        print("  (bypassing the gate) rather than more epochs or a new LR.")
    elif probe["roc_auc"] >= 0.60:
        print("  Weak but above-chance signal. Worth bypassing the ReLU gate,")
        print("  but expect modest results until the label count grows toward")
        print("  the SRS's 500-1,000 target.")
    else:
        print("  No usable signal in the frozen features at this sample size.")
        print("  Bypassing the head will not be enough -- this needs backbone")
        print("  unfreezing and/or substantially more labels.")

    (out_dir / "probe_metrics.json").write_text(json.dumps(
        {"spatially_blocked": spatial_probe, "random_kfold": random_probe,
         "n": int(len(y)), "n_positive": int(y.sum()),
         "n_sites": int(n_groups), "block_m": args.block_m,
         "n_dead_relu": n_dead,
         "base_model_roc_auc": float(roc_auc_score(y, base_logit)),
         "classifier_bias": float(net.classifier.bias.item())}, indent=2))
    print(f"\nSaved features + metrics to {out_dir}/")


if __name__ == "__main__":
    main()
