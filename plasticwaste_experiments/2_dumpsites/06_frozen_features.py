"""Benchmark step 2: frozen features for every crop (80/150/300 m).

Backbones (all frozen, no training here):
  colour     19-d colour/texture summary (lower bar; as dumpsite_tool.features.colour_features)
  imagenet   ImageNet ResNet-50, global-average-pooled 2048-d, input 224 px (standard ImageNet size)
  aw_italy   AerialWaste ResNet-50+FPN (Torres & Fraternali), pooled pyramid 1024-d
  aw_guat    the Guatemala fine-tune of the same network
  dinov3_sat DINOv3 ViT-L/16, satellite-trained (SAT-493M, Maxar 0.6 m), frozen
  dinov3_web DINOv3 ViT-L/16, web-trained (LVD-1689M), frozen
             both: CLS token + mean of patch tokens (2048-d), the DINOv2/v3 linear-probe
             feature; input 224 px; normalisation from each model's own image processor.
             Gated on Hugging Face (DINOv3 License); weights cached in D:\hf_cache.

AerialWaste was trained on 0.2-0.5 m imagery, so its inputs are resized to ~0.3 m/px
(80 m -> 256 px, 150 m -> 512 px, 300 m -> 1024 px); ImageNet and DINOv3 use the standard 224 px.
Batches are small enough for a 4 GB GPU (larger ones spill into shared memory and crawl).

Output: outputs/benchmark/feat_<backbone>_<w>m.npy (float32, N x d)
"""
import os, sys
from pathlib import Path

import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT as EXP, PROJECT, TOOL  # noqa: E402
NATSCAN = Path(r"E:\SDG AI Lab\Guinea\national_scan_2026")  # Esri tile driver + tile cache of the Nov 2026 national scan
WORK = EXP / "2_dumpsites" / "outputs" / "benchmark"; WORK.mkdir(parents=True, exist_ok=True)
PT = PROJECT
sys.path[:0] = [str(PT), str(PT / "plasticwaste_tool")]
from plasticwaste_tool.dumpsite_tool import features as feat  # noqa: E402
from plasticwaste_tool.dumpsite_tool import feature_probe as fp  # noqa: E402

WIN = (80, 150, 300)
AW_PX = {80: 256, 150: 512, 300: 1024}
CKPT = {"aw_italy": PT / "plasticwaste_tool/aerialwaste_model/weights/checkpoint.pth",
        "aw_guat": PT / "plasticwaste_tool/aerialwaste_model/weights/guatemala.pth"}
os.environ.setdefault("HF_HUB_CACHE", r"D:\hf_cache")
DINO = {"dinov3_sat": "facebook/dinov3-vitl16-pretrain-sat493m", "dinov3_web": "facebook/dinov3-vitl16-pretrain-lvd1689m"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor((0.485, 0.456, 0.406), device=DEV).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), device=DEV).view(1, 3, 1, 1)


def colour(A):
    a = A.astype(np.float32) / 255.0
    out = []
    for x in a:
        g = x.mean(2); gy, gx = np.gradient(g); gr = np.hypot(gx, gy)
        out.append(np.concatenate([x.mean((0, 1)), x.std((0, 1)),
                                   np.percentile(x.reshape(-1, 3), [10, 50, 90], axis=0).ravel(),
                                   [gr.mean(), gr.std(), g.mean(), g.std()]]))
    return np.stack(out).astype(np.float32)


@torch.no_grad()
def run(A, px, fn, bs, mean=None, std=None):
    mean = MEAN if mean is None else mean; std = STD if std is None else std
    out = []
    for i in range(0, len(A), bs):
        x = torch.from_numpy(A[i:i + bs]).to(DEV).permute(0, 3, 1, 2).float().div(255)
        x = F.interpolate(x, size=(px, px), mode="bicubic", align_corners=False).clamp(0, 1)
        out.append(fn((x - mean) / std).float().cpu().numpy())
    return np.concatenate(out)


def main():
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    from torchvision.models import resnet50, ResNet50_Weights
    inet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1); inet.fc = torch.nn.Identity(); inet.eval().to(DEV)
    aw = {k: feat.load_aw_net(str(p)).to(DEV) for k, p in CKPT.items()}
    dino = {}
    if "--no-dino" not in sys.argv:
        from transformers import AutoImageProcessor, AutoModel
        for k, repo in DINO.items():
            pr = AutoImageProcessor.from_pretrained(repo)
            dino[k] = (AutoModel.from_pretrained(repo).eval().to(DEV),
                       torch.tensor(pr.image_mean, device=DEV).view(1, 3, 1, 1),
                       torch.tensor(pr.image_std, device=DEV).view(1, 3, 1, 1))

    def dino_feat(m):
        def f(x):
            h = m(pixel_values=x).last_hidden_state          # [B, 1 + 4 registers + patches, 1024]
            return torch.cat([h[:, 0], h[:, 5:].mean(1)], 1)
        return f
    for w in WIN:
        A = np.load(WORK / f"crops_{w}m.npy")
        jobs = {"colour": lambda: colour(A), "imagenet": lambda: run(A, 224, inet, 64)}
        for k, net in aw.items():
            jobs[k] = (lambda net=net: run(A, AW_PX[w], lambda x: fp.extract(net, x)[0], {80: 8, 150: 2, 300: 1}[w]))
        for k, (m, mu, sd) in dino.items():
            jobs[k] = (lambda m=m, mu=mu, sd=sd: run(A, 224, dino_feat(m), 16, mu, sd))
        for name, job in jobs.items():
            f = WORK / f"feat_{name}_{w}m.npy"
            if f.exists():
                continue
            X = job(); np.save(f, X); print(name, w, X.shape, flush=True)


if __name__ == "__main__":
    main()
