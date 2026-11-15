"""Benchmark step 4: the two TRAINED methods, on the same outer folds as 07 (seed 0).

  aw_protocol  Guatemala / Torres & Fraternali protocol (GUA_dumpsites/training, model/aerial_waste_classifier.py):
               ResNet-50+FPN from the AerialWaste (Polimi) weights, ResNet-50 backbone frozen,
               FPN + head trained with Adam lr 5e-4, BCE, batch 4, random h/v flips, up to 50 epochs.
               Input ~0.3 m/px as in their data. Learning rate 5e-5, not their 5e-4: at 5e-4 the head
               collapsed to one constant score in every fold (2026-11-13; each pyramid level ends in a
               single ReLU unit, which a large Adam step can switch off for all inputs). Output kept in
               outputs/benchmark/archive_aw_lr5e-4_collapsed/. Other differences from Guatemala, both to remove known bias:
               the epoch is chosen on a validation fold of TRAINING groups (never the test fold; same balanced group folds as 07), and
               frozen BatchNorm statistics stay fixed.
  finetune     ImageNet ResNet-50 fully fine-tuned (the baseline of Sharma et al. 2026): AdamW lr 1e-4,
               wd 1e-4, batch 16 (fits a 4 GB GPU), 224 px, flips + rot90, BCE weighted for class balance, up to 30 epochs,
               epoch chosen on the same kind of validation split.

Usage: 08_trained_methods.py <aw_protocol|finetune> <80|150|300> [seed, default 0]
Output: outputs/benchmark/oof_main_<method>_<w>m.npy (NaN outside the evaluated rows), results/train_<method>_<w>m.json
"""
import json, sys, time
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import average_precision_score as AP
import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT as EXP, PROJECT, TOOL  # noqa: E402
NATSCAN = Path(r"E:\SDG AI Lab\Guinea\national_scan_2026")  # Esri tile driver + tile cache of the Nov 2026 national scan
WORK = EXP / "2_dumpsites" / "outputs" / "benchmark"; WORK.mkdir(parents=True, exist_ok=True)
RES = WORK
PT = PROJECT
sys.path[:0] = [str(PT), str(PT / "plasticwaste_tool")]
DEV = "cuda"
MEAN = torch.tensor((0.485, 0.456, 0.406), device=DEV).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225), device=DEV).view(1, 3, 1, 1)
AW_PX = {80: 256, 150: 512, 300: 1024}
_spec = importlib.util.spec_from_file_location("ev", Path(__file__).resolve().parent / "07_linear_probe_benchmark.py")
ev = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ev)   # same group folds as 07
CFG = {"aw_protocol": dict(bs=4, lr=5e-5, epochs=50, patience=10),   # Guatemala used 5e-4: collapses here, see docstring
       "finetune": dict(bs=16, lr=1e-4, epochs=30, patience=6)}


def build(method):
    if method == "aw_protocol":
        from plasticwaste_tool.dumpsite_tool import features as feat
        net = feat.load_aw_net(str(PT / "plasticwaste_tool/aerialwaste_model/weights/checkpoint.pth"))
        for p in net.backbone.parameters():
            p.requires_grad = False
        params = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=CFG[method]["lr"])
    else:
        from torchvision.models import resnet50, ResNet50_Weights
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1); net.fc = nn.Linear(2048, 1)
        opt = torch.optim.AdamW(net.parameters(), lr=CFG[method]["lr"], weight_decay=1e-4)
    return net.to(DEV), opt


def set_mode(net, method, train):
    net.train(train)
    if method == "aw_protocol":
        net.backbone.eval()  # frozen BN statistics


def batch(A, idx, px, aug):
    x = torch.from_numpy(A[idx]).to(DEV).permute(0, 3, 1, 2).float().div(255)
    x = F.interpolate(x, size=(px, px), mode="bicubic", align_corners=False).clamp(0, 1)
    if aug:
        if np.random.rand() < .5: x = x.flip(3)
        if np.random.rand() < .5: x = x.flip(2)
        if aug == "rot":
            x = torch.rot90(x, np.random.randint(4), (2, 3))
    return (x - MEAN) / STD


@torch.no_grad()
def predict(net, method, A, idx, px, bs):
    set_mode(net, method, False); out = []
    for i in range(0, len(idx), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(net(batch(A, idx[i:i + bs], px, False)).float().view(-1).cpu())
    return torch.cat(out).sigmoid().numpy()


def main(method, w, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    pool = pd.read_parquet(WORK / "pool.parquet"); keep = (pool.label >= 0).to_numpy()
    df = pool[keep].reset_index(drop=True); y = df.label.to_numpy(); g = df.group.to_numpy()
    A = np.load(WORK / f"crops_{w}m.npy")[keep]
    px = AW_PX[w] if method == "aw_protocol" else 224
    c = CFG[method]; aug = True if method == "aw_protocol" else "rot"
    oof = np.full(len(y), np.nan); log = []
    folds = ev.group_folds(y, g, 5, seed)   # seed 0 = the folds of 07's first seed
    for k, (tr, te) in enumerate(folds):
        t0 = time.time()
        # validation: of 5 balanced group folds of the training rows, the one spanning the most
        # separate places (groups), so the epoch is not chosen on one blind town alone
        inner = ev.group_folds(y[tr], g[tr], 5, 100)
        a, v = max(inner, key=lambda s: len(np.unique(g[tr][s[1]])))
        tr_, va = tr[a], tr[v]
        if len(va) < 150 or y[va].sum() < 20:   # stop at once rather than train on a useless split
            raise SystemExit(f"validation fold too small: {len(va)} windows, {int(y[va].sum())} waste sites")
        net, opt = build(method); scaler = torch.cuda.amp.GradScaler()
        pw = torch.tensor((y[tr_] == 0).sum() / max((y[tr_] == 1).sum(), 1), device=DEV) if method == "finetune" else None
        best, best_state, bad = -1, None, 0
        for ep in range(c["epochs"]):
            set_mode(net, method, True); perm = np.random.permutation(tr_)
            for i in range(0, len(perm), c["bs"]):
                j = perm[i:i + c["bs"]]
                xb = batch(A, j, px, aug); yb = torch.tensor(y[j], device=DEV, dtype=torch.float32)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = F.binary_cross_entropy_with_logits(net(xb).float().view(-1), yb, pos_weight=pw)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            pv = predict(net, method, A, va, px, c["bs"] * 2)
            if pv.std() < 1e-5:   # every window gets the same score: the head has collapsed
                raise SystemExit(f"scores collapsed in fold {k}, epoch {ep}")
            ap = AP(y[va], pv)
            if ap > best:
                best, bad = ap, 0
                best_state = {k_: t.detach().clone() for k_, t in net.state_dict().items()}; best_ep = ep
            else:
                bad += 1
                if bad >= c["patience"]:
                    break
        net.load_state_dict(best_state)
        oof[te] = predict(net, method, A, te, px, c["bs"] * 2)
        log.append({"fold": k, "best_epoch": best_ep, "val_ap": round(best, 3), "minutes": round((time.time() - t0) / 60, 1)})
        print(method, w, log[-1], flush=True)
        del net, opt; torch.cuda.empty_cache()
    full = np.full(len(pool), np.nan); full[keep] = oof
    sfx = "" if seed == 0 else f"_seed{seed}"   # extra seeds: not read by 09, averaged separately
    np.save(RES / f"oof_main_{method}_{w}m{sfx}.npy", full)
    json.dump({"method": method, "window_m": w, "seed": seed, "input_px": px, **c, "folds": log}, open(RES / f"train_{method}_{w}m{sfx}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
