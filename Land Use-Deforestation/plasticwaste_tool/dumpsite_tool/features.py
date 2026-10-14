"""Copied from 02d_benchmark_features.py and renamed so it can be imported.

The numbered original stays in place for the notebooks; this copy is
what the package and the backend import. Runtime importlib loads of
sibling scripts have been rewritten as relative imports.
Regenerate with scratchpad/build_package.py -- do not edit by hand
until the originals are retired.
"""

"""
02d -- Benchmark: is the AerialWaste checkpoint actually earning its place?

WHY THIS EXISTS
---------------
02c_feature_probe.py showed that a linear probe on AerialWaste's frozen
pooled-pyramid features scores ROC-AUC 0.834 (spatially-blocked CV) on Guinea
labels, versus 0.287 for the model's own published head. That is a large
improvement -- but on its own it does NOT establish that the AerialWaste
*pretraining* is what is doing the work. Two cheaper explanations have to be
ruled out first, or any claim that this project needs AerialWaste is
unsupported:

  (a) GENERIC PRETRAINING. Would an ordinary ImageNet-pretrained ResNet50 --
      no landfill supervision whatsoever -- do just as well? If yes, the
      AerialWaste weights add nothing here and the honest thing is to say so.
      Standard ablation; the "off-the-shelf features" baseline of Razavian et
      al. 2014 (CVPRW) and the linear-evaluation protocol of Kornblith et al.
      2019 (CVPR).

  (b) TRIVIAL COLOUR/TEXTURE CUES. Would plain per-channel colour statistics
      separate these classes about as well? Guinea dumpsites often sit on
      bare laterite soil, so a model could score well by detecting "brown,
      not green" while learning nothing about waste. This control guards
      against the same class of spurious shortcut this project already hit
      once, when the INS-trained model's output turned out to track image
      brightness/texture rather than content.

Only if AerialWaste features beat BOTH of these is "keep the AerialWaste
backbone, replace the collapsed head" a justified design choice rather than
an unexamined assumption.

METHOD
------
Identical evaluation for every feature source: the same tiles, the same
labels, the same spatially-blocked CV (Roberts et al. 2017) and the same
regularised logistic-regression probe from 02c. Only the features change, so
differences are attributable to the representation and nothing else.

Usage:
    python 02d_benchmark_features.py \
        --labels manual_qa/manual_labels_template.csv \
        --grid manual_qa/browsing_grid.geojson manual_qa/poi_grid.geojson \
        --images-dir manual_qa/tiles \
        --probe-dir runs/feature_probe \
        --out runs/benchmark
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
from sklearn.model_selection import StratifiedGroupKFold
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 800


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def imagenet_features(image_paths, batch_log=25):
    """
    Global-average-pooled penultimate features (2048-d) from a stock
    ImageNet-pretrained ResNet50 -- no landfill supervision of any kind.
    """
    from torchvision.models import resnet50
    try:
        from torchvision.models import ResNet50_Weights
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    except ImportError:  # older torchvision
        net = resnet50(pretrained=True)
    net.fc = torch.nn.Identity()
    net.eval()

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    from PIL import Image
    out = []
    for i, p in enumerate(image_paths, 1):
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        out.append(net(x).numpy()[0])
        if i % batch_log == 0:
            logger.info("  imagenet %d/%d", i, len(image_paths))
    return np.stack(out)


def load_aw_net(checkpoint):
    """
    Load any of the checkpoint shapes floating around for this architecture.

    CAREFUL -- Net registers every module TWICE: once as an attribute
    (self.stage0, self.toplayer, ...) and again inside the self.backbone /
    self.newly_added ModuleLists. So a complete state_dict legitimately holds
    BOTH namings (990 keys), pointing at the same tensors. Rewriting
    "backbone.0.*" into "stage0.*" therefore does not translate a checkpoint,
    it DELETES half of it -- which is exactly what
    manual_qa/execute_model.ipynb's conversion cell does, and why its output
    guatemala_tam_uyumlu.pth has only 654 of 990 keys and cannot be loaded
    strictly. Do not use that file.

    Strategy: unwrap the container, strip Lightning's "model." prefix, and
    load as-is. The published .pth files already carry both namings.
    """
    from plasticwaste_tool.aerialwaste_model.architecture.resnet50_fpn import Net

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck
    if isinstance(ck, dict):
        if "state_dict" in ck:
            sd = ck["state_dict"]
        elif "model_state" in ck:
            sd = ck["model_state"]
    if any(k.startswith("model.") for k in sd):
        sd = {(k[len("model."):] if k.startswith("model.") else k): v
              for k, v in sd.items()}

    net = Net(num_classes=1, pretrained=False)
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net


@torch.no_grad()
def aerialwaste_features(checkpoint, image_paths, fp, batch_log=25):
    """Pooled FPN pyramid features (1024-d) from any AerialWaste-arch checkpoint."""
    from PIL import Image
    from torchvision import transforms

    net = load_aw_net(checkpoint)

    tf = transforms.Compose([
        transforms.Resize((fp.IMG_SIZE, fp.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(fp.IMAGENET_MEAN, fp.IMAGENET_STD)])
    out = []
    for i, p in enumerate(image_paths, 1):
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        f, _, _ = fp.extract(net, x)
        out.append(f.numpy()[0])
        if i % batch_log == 0:
            logger.info("  %s %d/%d", Path(checkpoint).name, i, len(image_paths))
    return np.stack(out)


def colour_features(image_paths):
    """
    Deliberately trivial control: per-channel mean/std/percentiles plus a
    coarse grey-level gradient-magnitude summary. If this rivals the deep
    features, the task is being solved by colour, not by waste.
    """
    from PIL import Image
    rows = []
    for p in image_paths:
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        g = a.mean(axis=2)
        gy, gx = np.gradient(g)
        grad = np.hypot(gx, gy)
        rows.append(np.concatenate([
            a.mean(axis=(0, 1)), a.std(axis=(0, 1)),
            np.percentile(a.reshape(-1, 3), [10, 50, 90], axis=0).ravel(),
            [grad.mean(), grad.std(), g.mean(), g.std()],
        ]))
    return np.stack(rows)


def paired_auc_diff_ci(y, score_a, score_b, groups, n_boot=4000, seed=0,
                       alpha=0.05):
    """
    Percentile bootstrap CI for the DIFFERENCE in ROC-AUC between two
    representations scored on the SAME tiles, resampling whole spatial blocks.

    WHY A PAIRED TEST: every representation here is evaluated on identical
    samples, so the comparison is paired. Judging it by whether two marginal
    confidence intervals overlap is a known statistical error -- that check is
    overly conservative and routinely misses real differences, because it
    ignores the covariance between two estimates computed on the same data
    (Schenker & Gentleman 2001, The American Statistician 55:182-186; see also
    Cumming & Finch 2005). Bootstrapping the paired difference uses that
    covariance and answers the question actually being asked: does A beat B on
    THIS data, given both saw the same tiles?

    Blocks (not tiles) are the resampling unit for the same reason the CV is
    blocked -- adjacent tiles of one site are near-duplicates.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    diffs = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in drawn])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        diffs.append(roc_auc_score(yy, score_a[idx])
                     - roc_auc_score(yy, score_b[idx]))
    diffs = np.array(diffs)
    return (float(np.percentile(diffs, 100 * alpha / 2)),
            float(np.percentile(diffs, 100 * (1 - alpha / 2))),
            float(diffs.mean()),
            float((diffs <= 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--grid", required=True, nargs="+")
    ap.add_argument("--images-dir", default="manual_qa/tiles")
    ap.add_argument("--probe-dir", default="runs/feature_probe",
                    help="where 02c_feature_probe.py cached features.npz "
                         "(its AerialWaste features are reused here)")
    ap.add_argument("--out", default="runs/benchmark")
    ap.add_argument("--aw-checkpoints", nargs="*", default=[],
                    help="extra AerialWaste-arch checkpoints to benchmark as "
                         "feature extractors, as name=path (e.g. "
                         "guatemala=aerialwaste_model/weights/guatemala.pth). "
                         "The Italian base checkpoint is always included via "
                         "--probe-dir's cache.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--block-m", type=float, default=200.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).parent
    from . import feature_probe as fp
    ft = load_module(here / "02b_finetune_aerialwaste.py", "ft")

    labels = pd.read_csv(args.labels).dropna(subset=["label"]).copy()
    labels["label"] = labels["label"].astype(int)
    manifest = ft.fetch_labeled_tile_images(labels, args.grid, args.images_dir)

    cached = np.load(Path(args.probe_dir) / "features.npz", allow_pickle=True)
    manifest = manifest.set_index("tile_id").loc[cached["tile_id"]].reset_index()
    y = cached["y"]
    X_aw = cached["X"]
    base_logit = cached["base_logit"]
    logger.info("Benchmarking on %d tiles (%d positive)", len(y), int(y.sum()))

    cache = out_dir / "alt_features.npz"
    cached = np.load(cache) if cache.exists() else None
    if cached is not None and len(cached["X_imagenet"]) != len(y):
        # Stale cache from a smaller label set: reusing it would misalign
        # features against labels (or just crash on shape). Re-extract.
        logger.warning("Cached alt features cover %d tiles, label set now has "
                       "%d -- re-extracting.", len(cached["X_imagenet"]), len(y))
        cached = None
    if cached is not None:
        X_in, X_col = cached["X_imagenet"], cached["X_colour"]
        logger.info("Reusing cached alternative features from %s", cache)
    else:
        logger.info("Extracting ImageNet ResNet50 features...")
        X_in = imagenet_features(manifest.image_path.tolist())
        logger.info("Extracting trivial colour/texture features...")
        X_col = colour_features(manifest.image_path.tolist())
        np.savez(cache, X_imagenet=X_in, X_colour=X_col)

    groups, _ = fp.spatial_groups(manifest.center_lat, manifest.center_lon,
                                  args.block_m)
    k = int(min(args.folds, len(np.unique(groups))))
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True,
                                    random_state=args.seed)

    sources = [(f"AerialWaste-Italy FPN ({X_aw.shape[1]}-d)", X_aw)]
    for spec_str in args.aw_checkpoints:
        if "=" not in spec_str:
            raise SystemExit(f"--aw-checkpoints entry must be name=path: {spec_str}")
        nm, path = spec_str.split("=", 1)
        cache_f = out_dir / f"features_{nm}.npz"
        Xn = np.load(cache_f)["X"] if cache_f.exists() else None
        if Xn is not None and len(Xn) != len(y):
            logger.warning("Cached %s features cover %d tiles, label set now "
                           "has %d -- re-extracting.", nm, len(Xn), len(y))
            Xn = None
        if Xn is not None:
            logger.info("Reusing cached %s features", nm)
        else:
            logger.info("Extracting %s features from %s...", nm, path)
            Xn = aerialwaste_features(path, manifest.image_path.tolist(), fp)
            np.savez(cache_f, X=Xn)
        sources.append((f"AerialWaste-{nm} FPN ({Xn.shape[1]}-d)", Xn))
    sources += [
        (f"ImageNet ResNet50 features ({X_in.shape[1]}-d)", X_in),
        (f"Colour/texture stats ({X_col.shape[1]}-d)", X_col),
    ]

    results, oofs = [], {}
    for name, X in sources:
        m, oof = fp.cv_probe(X, y, splitter, (X, y, groups), name)
        lo, hi, _ = fp.bootstrap_auc_ci(y, oof, groups, seed=args.seed)
        m["roc_auc_ci95"] = [lo, hi]
        m["features"] = name
        results.append(m)
        oofs[name] = oof

    # The published head, evaluated identically (no probe -- it IS the model).
    from sklearn.metrics import roc_auc_score
    published = {
        "features": "AerialWaste published head (as shipped)",
        "cv": "none -- direct model output",
        "roc_auc": float(roc_auc_score(y, base_logit)),
        "mcc": None, "balanced_accuracy": None,
        "roc_auc_ci95": list(fp.bootstrap_auc_ci(
            y, base_logit, groups, seed=args.seed)[:2]),
    }

    print("\n" + "=" * 78)
    print(f"FEATURE BENCHMARK -- {len(y)} tiles, {int(y.sum())} positive, "
          f"spatial block CV ({args.block_m:.0f}m, k={k})")
    print("=" * 78)
    print(f"  {'representation':<42} {'ROC-AUC':>8}  {'95% CI':>16}  {'MCC':>7}")
    print("  " + "-" * 74)
    for r in [published] + results:
        ci = f"[{r['roc_auc_ci95'][0]:.3f},{r['roc_auc_ci95'][1]:.3f}]"
        mcc = f"{r['mcc']:+.3f}" if r["mcc"] is not None else "   n/a"
        print(f"  {r['features']:<42} {r['roc_auc']:>8.3f}  {ci:>16}  {mcc:>7}")

    names = [r["features"] for r in results]
    print("\n" + "=" * 78)
    print("PAIRED COMPARISONS (block bootstrap of the AUC DIFFERENCE)")
    print("=" * 78)
    print("  Every representation is scored on the same tiles, so this is a")
    print("  PAIRED comparison -- whether the marginal CIs above overlap is not")
    print("  the right test (Schenker & Gentleman 2001).")
    print()
    # Compare the BEST representation against each of the others. Comparing a
    # fixed reference against all would answer the wrong question once several
    # candidate checkpoints are in play -- what matters is whether the leader's
    # lead is real.
    best_i = int(np.argmax([r["roc_auc"] for r in results]))
    paired = []
    for j in range(len(results)):
        if j == best_i:
            continue
        lo, hi, mean, p_le0 = paired_auc_diff_ci(
            y, oofs[names[best_i]], oofs[names[j]], groups, seed=args.seed)
        sig = "YES" if lo > 0 else "no"
        print(f"  {names[best_i].split(' (')[0]}  -  {names[j].split(' (')[0]}")
        print(f"      dAUC = {mean:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
              f"P(diff<=0) = {p_le0:.3f}   excludes 0: {sig}")
        paired.append({"a": names[best_i], "b": names[j], "delta_auc": mean,
                       "ci95": [lo, hi], "p_le_zero": p_le0,
                       "excludes_zero": bool(lo > 0)})

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    leader = names[best_i].split(" (")[0]
    beaten = [p for p in paired if p["excludes_zero"]]
    tied = [p for p in paired if not p["excludes_zero"]]
    col_cmp = next((p for p in paired if "Colour" in p["b"]), None)

    print(f"  Best representation: {leader} "
          f"(ROC-AUC {results[best_i]['roc_auc']:.3f})")
    if col_cmp is not None and not col_cmp["excludes_zero"]:
        print("  BUT it does not beat the trivial colour/texture control by a")
        print("  margin excluding 0 -- treat the headline number with suspicion.")
        print("  The task may be solved by a bare-soil-vs-vegetation shortcut.")
        print("  More diverse negatives needed before trusting any of this.")
    else:
        print(f"  Beats {len(beaten)} of {len(paired)} alternatives with the paired")
        print("  difference excluding 0 (a real difference at this sample size):")
        for p in beaten:
            print(f"      > {p['b'].split(' (')[0]}   (+{p['delta_auc']:.3f})")
        if tied:
            print("  NOT separable from (paired CI includes 0 -- do not claim a win):")
            for p in tied:
                print(f"      = {p['b'].split(' (')[0]}   ({p['delta_auc']:+.3f})")
            print("  -> Where a tie involves ImageNet, prefer ImageNet in practice:")
            print("     it carries no CC BY-NC-ND restriction, so it needs no")
            print("     permission to deploy.")
    print()
    print(f"  Sample size caveat: n={len(y)}, {int(y.sum())} positives across "
          f"{len(np.unique(groups))} spatial")
    print("  blocks. Intervals are wide; more labelled SITES (not more tiles at")
    print("  sites already covered) is what narrows them.")

    (out_dir / "benchmark.json").write_text(json.dumps(
        {"n": int(len(y)), "n_positive": int(y.sum()),
         "block_m": args.block_m, "folds": k,
         "n_blocks": int(len(np.unique(groups))),
         "results": [published] + results,
         "paired_comparisons": paired}, indent=2))
    print(f"\nSaved to {out_dir}/benchmark.json")


if __name__ == "__main__":
    main()
