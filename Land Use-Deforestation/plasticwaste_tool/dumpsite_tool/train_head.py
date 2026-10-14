"""Copied from 02e_train_probe_head.py and renamed so it can be imported.

The numbered original stays in place for the notebooks; this copy is
what the package and the backend import. Runtime importlib loads of
sibling scripts have been rewritten as relative imports.
Regenerate with scratchpad/build_package.py -- do not edit by hand
until the originals are retired.
"""

"""
02e -- Train the deployable replacement head: a linear probe on frozen
pretrained features.

WHAT THIS REPLACES, AND WHY THAT IS JUSTIFIED
---------------------------------------------
It does NOT replace the AerialWaste model. It keeps that model's published
backbone and weights untouched and frozen, and replaces only its final
classification head, which was measured to fail on Guinea imagery:

  * the published head scores ROC-AUC 0.287 on this project's Guinea labels
    (BELOW the 0.5 coin-flip line -- it ranks real dumpsites BELOW
    non-dumpsites), see 02d_benchmark_features.py;
  * the mechanism is understood, not guessed: with num_classes=1 the head
    compresses the whole image into 4 non-negative scalars
    (`F.relu(self.fc(p))` for each pyramid level) before a Linear(4 -> 1).
    On 73.4% of this project's labelled tiles all four gates close, the output
    becomes exactly `classifier.bias` (0.742915689945221), and -- because the
    gradient through a dead ReLU is exactly zero -- fine-tuning cannot recover
    it. A 20-epoch run never moved val MCC off 0.000. See 02c_feature_probe.py.

Training a linear classifier on frozen pretrained features ("linear probing",
a.k.a. feature-extraction transfer) is one of the two canonical ways to use a
pretrained network, not a bespoke method invented here:

  * Donahue et al. 2014, "DeCAF: A Deep Convolutional Activation Feature for
    Generic Visual Recognition", ICML.
  * Razavian et al. 2014, "CNN Features off-the-shelf: an Astounding Baseline
    for Recognition", CVPR Workshops.
  * Yosinski et al. 2014, "How transferable are features in deep neural
    networks?", NeurIPS.
  * Kornblith, Shlens & Le 2019, "Do Better ImageNet Models Transfer Better?",
    CVPR -- also documents that feature extraction is preferable to full
    fine-tuning in the LOW-SAMPLE regime, which is exactly this project's
    situation (124 labels, 32 positive).
  * Remote sensing specifically: Penatti et al. 2015 (CVPRW); Nogueira,
    Penatti & dos Santos 2017, "Towards better exploiting convolutional neural
    networks for remote sensing scene classification", Pattern Recognition 61.

WHICH BACKBONE -- DELIBERATELY LEFT OPEN
----------------------------------------
`--backbone aerialwaste` (default) and `--backbone imagenet` are both
supported because the benchmark could NOT establish that AerialWaste's
landfill-specific pretraining is required. Paired block-bootstrap of the AUC
difference gave +0.066 with 95% CI [-0.004, +0.133] -- suggestive, but the
interval includes zero at this sample size. Both backbones DID beat a trivial
colour/texture control by a margin excluding zero (+0.199 and +0.134
respectively), so the signal is real and is not a bare-soil-vs-vegetation
shortcut.

This matters practically: the AerialWaste weights are CC BY-NC-ND 4.0, while
ImageNet weights carry no such restriction. If the gap remains unproven as the
label set grows toward the SRS's 500-1,000 target, `--backbone imagenet` is
the simpler and legally cleaner deployment option. Re-run
02d_benchmark_features.py as labels accumulate to settle it.

HONEST REPORTING
----------------
The metrics printed here come from spatially-blocked cross-validation
(Roberts et al. 2017, Ecography 40:913-929), because this project's labels
contain contiguous runs of 40m tiles covering ONE physical dumpsite; plain
k-fold splits those across train/test and inflates every metric (0.877 vs
0.834 AUC on this dataset). Confidence intervals come from a block bootstrap
for the same reason. MCC is preferred to F1 under class imbalance, following
Chicco & Jurman 2020 and this project's existing convention.

Usage:
    python 02e_train_probe_head.py \
        --labels manual_qa/manual_labels_template.csv \
        --grid manual_qa/browsing_grid.geojson manual_qa/poi_grid.geojson \
        --images-dir manual_qa/tiles \
        --backbone aerialwaste \
        --out runs/probe_head
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import torch BEFORE scikit-learn (and before anything that pulls in the
# GDAL/rasterio/geopandas stack). Both ship their own native Windows DLLs and
# whichever initialises first wins -- importing sklearn first made torch's own
# c10.dll fail its init routine (WinError 1114) when this script later loaded
# 02c_feature_probe.py. Same ordering constraint, and same fix, as documented
# at the top of 05_guinea_inference.py; 02c and 02d already import in this
# order, which is why they run and an earlier draft of this file did not.
import torch  # noqa: F401,E402

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1


def build_probe():
    """
    The probe itself. Regularised (C=0.1) and class-weighted because there are
    ~3x more negatives than positives and only ~1e2 samples against 1-2e3
    feature dimensions -- without regularisation a linear model separates that
    trivially and generalises not at all.
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", C=0.1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--grid", required=True, nargs="+")
    ap.add_argument("--images-dir", default="manual_qa/tiles")
    ap.add_argument("--backbone", default="aerialwaste",
                    choices=["aerialwaste", "imagenet"],
                    help="frozen feature extractor -- see the module docstring "
                         "on why this is deliberately configurable")
    ap.add_argument("--checkpoint", default=str(
        Path(__file__).parent / "aerialwaste_model" / "weights" / "checkpoint.pth"))
    ap.add_argument("--out", default="runs/probe_head")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block-m", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).parent
    import importlib.util

    def load_module(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    from . import feature_probe as fp
    from . import features as bm
    ft = load_module(here / "02b_finetune_aerialwaste.py", "ft")

    labels = pd.read_csv(args.labels).dropna(subset=["label"]).copy()
    labels["label"] = labels["label"].astype(int)
    manifest = ft.fetch_labeled_tile_images(labels, args.grid, args.images_dir)
    y = manifest.label.to_numpy()
    logger.info("Training on %d tiles (%d positive)", len(y), int(y.sum()))

    # ---- features -------------------------------------------------------
    if args.backbone == "aerialwaste":
        import torch
        from torchvision import transforms
        from PIL import Image
        from plasticwaste_tool.aerialwaste_model.architecture.resnet50_fpn import Net

        net = Net(num_classes=1, pretrained=False)
        net.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                       weights_only=False), strict=True)
        net.eval()
        tf = transforms.Compose([
            transforms.Resize((fp.IMG_SIZE, fp.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(fp.IMAGENET_MEAN, fp.IMAGENET_STD),
        ])
        feats = []
        for i, p in enumerate(manifest.image_path, 1):
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
            f, _, _ = fp.extract(net, x)
            feats.append(f.numpy()[0])
            if i % 25 == 0:
                logger.info("  features %d/%d", i, len(manifest))
        X = np.stack(feats)
    else:
        X = bm.imagenet_features(manifest.image_path.tolist())

    # ---- honest evaluation ---------------------------------------------
    groups, _ = fp.spatial_groups(manifest.center_lat, manifest.center_lon,
                                  args.block_m)
    k = int(min(args.folds, len(np.unique(groups))))
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                    random_state=args.seed)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in splitter.split(X, y, groups):
        m = build_probe()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]

    auc = float(roc_auc_score(y, oof))
    lo, hi, _ = fp.bootstrap_auc_ci(y, oof, groups, seed=args.seed)

    # Operating threshold chosen on OUT-OF-FOLD predictions only -- picking it
    # on in-sample scores would bake in the overfit it is meant to guard against.
    grid = np.unique(np.round(oof, 3))
    mccs = [(matthews_corrcoef(y, (oof >= t).astype(int)), float(t)) for t in grid]
    best_mcc, best_thr = max(mccs)

    # ---- fit the deployable model on ALL labels -------------------------
    model = build_probe()
    model.fit(X, y)

    import joblib
    joblib.dump(model, out_dir / "probe.joblib")

    meta = {
        "artifact_version": ARTIFACT_VERSION,
        "backbone": args.backbone,
        "feature_dim": int(X.shape[1]),
        "img_size": fp.IMG_SIZE,
        "normalization": {"mean": list(fp.IMAGENET_MEAN),
                          "std": list(fp.IMAGENET_STD)},
        "base_checkpoint": (str(args.checkpoint)
                            if args.backbone == "aerialwaste" else "torchvision/IMAGENET1K_V1"),
        "n": int(len(y)), "n_positive": int(y.sum()),
        "n_blocks": int(len(np.unique(groups))), "block_m": args.block_m,
        "cv": f"StratifiedGroupKFold(k={k}) over {args.block_m:.0f}m spatial blocks",
        "roc_auc_blocked_cv": auc,
        "roc_auc_ci95": [lo, hi],
        "operating_threshold": best_thr,
        "mcc_at_threshold": float(best_mcc),
        "license_note": (
            "AerialWaste weights are CC BY-NC-ND 4.0: this probe is a "
            "derivative work for evaluation/internal use until the authors "
            "sign off. --backbone imagenet carries no such restriction."
            if args.backbone == "aerialwaste" else
            "ImageNet (torchvision) weights -- no CC BY-NC-ND restriction."),
    }
    (out_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2))

    print("\n" + "=" * 70)
    print(f"PROBE HEAD TRAINED -- backbone: {args.backbone}")
    print("=" * 70)
    print(f"  samples            : {len(y)} ({int(y.sum())} positive) "
          f"in {len(np.unique(groups))} spatial blocks")
    print(f"  spatially-blocked  : ROC-AUC {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  operating threshold: {best_thr:.3f}  (MCC {best_mcc:+.3f}, "
          "chosen on out-of-fold scores)")
    print(f"\n  Saved: {out_dir/'probe.joblib'}")
    print(f"         {out_dir/'probe_meta.json'}")
    print("\n  Use it:  python 05_guinea_inference.py --arch probe "
          f"--checkpoint {out_dir/'probe.joblib'} ...")
    if args.backbone == "aerialwaste":
        print("\n  REMINDER: AerialWaste weights are CC BY-NC-ND 4.0. This probe is")
        print("  a derivative -- evaluation/internal use only until the authors")
        print("  sign off. --backbone imagenet avoids that constraint entirely.")


if __name__ == "__main__":
    main()
