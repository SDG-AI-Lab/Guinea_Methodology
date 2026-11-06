"""
Features of the Esri crops used by the dumpsite benchmark: the 651 point-centred
crops and the two exhaustively labelled blind grids (Conakry 422, Boke 557).
Representations: colour histogram and ImageNet ResNet-50 (the deployed one, both
from the tool), the AerialWaste checkpoint (trained on waste imagery, Torres &
Fraternali 2023), and two recent backbones from torchvision. Cached per set.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_HOME", r"D:\Guinea_Methodology\plasticwaste_experiments\2_dumpsites\torch_home")

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, TOOL  # noqa: E402

from plasticwaste_tool.dumpsite_tool import feature_probe as fp  # noqa: E402
from plasticwaste_tool.dumpsite_tool import features as bm  # noqa: E402
from torchvision import models  # noqa: E402

OUT = ROOT / "2_dumpsites" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MQ = TOOL / "manual_qa"
CKPT = TOOL / "aerialwaste_model" / "weights" / "checkpoint.pth"


def grid(town):
    feats = json.loads((MQ / "regions" / f"{town}_blindgrid_review.geojson").read_text(encoding="utf-8"))["features"]
    rows = []
    for f in feats:
        p = dict(f["properties"])
        xs = [c[0] for c in f["geometry"]["coordinates"][0]]
        ys = [c[1] for c in f["geometry"]["coordinates"][0]]
        p["lon"], p["lat"] = float(np.mean(xs)), float(np.mean(ys))
        rows.append(p)
    d = pd.DataFrame(rows)
    d["y"] = d["label"].astype(str).str.strip().astype(int)
    d["path"] = [str(MQ / f"blindgrid_cache_{town}" / "confirm_esri" / f"{t}.png") for t in d["tile_id"]]
    return d


def points():
    d = pd.read_csv(MQ / "_point_samples.csv")
    d["y"] = d["label"].astype(int)
    d["path"] = [str(MQ / "tiles_pt80" / f"{s}.png") for s in d["sid"]]
    return d


def backbone(name):
    if name == "convnext":
        w = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        m = models.convnext_base(weights=w)
        m.classifier = torch.nn.Sequential(torch.nn.Flatten(1))
    else:
        w = models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        m = models.vit_b_16(weights=w)
        m.heads = torch.nn.Identity()
    return m.eval(), w.transforms()


def deep(name, paths, batch=16):
    m, tf = backbone(name)
    out = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
            out.append(m(x).numpy())
            if i % 160 == 0:
                print(name, i, len(paths), flush=True)
    return np.concatenate(out)


def main():
    sets = {"points": points(), "conakry": grid("conakry"), "boke": grid("boke")}
    for key, d in sets.items():
        ok = [Path(p).exists() for p in d.path]
        print(key, len(d), "tiles present", sum(ok), "positives", int(d.y.sum()), flush=True)
        d = d[ok].reset_index(drop=True)
        d.drop(columns=[c for c in d.columns if c in ("geometry",)]).to_csv(OUT / f"set_{key}.csv", index=False)
        paths = d.path.tolist()
        for name, fn in {
            "colour": lambda: bm.colour_features(paths),
            "resnet50": lambda: bm.imagenet_features(paths),
            "aerialwaste": lambda: bm.aerialwaste_features(str(CKPT), paths, fp),
            "convnext": lambda: deep("convnext", paths),
            "vit": lambda: deep("vit", paths),
        }.items():
            f = OUT / f"feat_{key}_{name}.npy"
            if not f.exists():
                try:
                    np.save(f, np.asarray(fn(), "float32"))
                    print("wrote", f.name, flush=True)
                except Exception as e:  # noqa: BLE001
                    print("FAILED", key, name, type(e).__name__, e, flush=True)


if __name__ == "__main__":
    main()
