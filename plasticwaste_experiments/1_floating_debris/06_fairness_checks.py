"""
Two checks that the leave-one-tile-out test is fair to the deployed rule.

1. Threshold population. On Guinea the rule's mean and sigma come from every water
   pixel of the scene; on MARIDA the published test used only labelled water. Here
   they come from every pixel of the scene's patches that looks like water
   (NDWI > 0, not labelled cloud), which is nearer the deployed rule.
2. Naive Bayes in the NDVI-FDI plane with equal class priors, as Biermann et al.
   (2020) trained it on a balanced spectral library.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import MARIDA, ROOT  # noqa: E402

import rasterio  # noqa: E402

from plasticwaste_tool.marida_tool import data as D  # noqa: E402
from plasticwaste_tool.marida_tool import features as F  # noqa: E402

from sklearn.naive_bayes import GaussianNB  # noqa: E402

OUT = ROOT / "1_floating_debris" / "outputs"


def scene_stats():
    """Mean and sd of FDI and NDVI over each scene's water-looking pixels."""
    rows = {}
    for scene_dir in sorted((MARIDA / "patches").glob("S2_*")):
        vals = []
        for tif in scene_dir.glob("S2_*.tif"):
            if tif.name.endswith(("_cl.tif", "_conf.tif")):
                continue
            with rasterio.open(tif) as s:
                b = s.read().astype("float32")
            with rasterio.open(str(tif).replace(".tif", "_cl.tif")) as s:
                c = s.read(1)
            X = b.reshape(11, -1).T
            ok = np.isfinite(X).all(axis=1) & ~np.isin(c.ravel(), D.NON_WATER)
            X = X[ok]
            ndwi = F.ndwi(X)
            X = X[ndwi > 0]
            if len(X):
                vals.append(np.c_[F.fdi(X), F.ndvi(X)])
        if vals:
            v = np.concatenate(vals)
            rows[scene_dir.name] = (v[:, 0].mean(), v[:, 0].std(), v[:, 1].mean(), v[:, 1].std(), len(v))
    return rows


def main():
    z = np.load(OUT / "marida_pixels.npz", allow_pickle=True)
    X, y = z["X"], z["y"]
    pid = np.array([D.normalise_id(p) for p in z["pid"]])
    scene = np.array([p.rsplit("_", 1)[0] for p in pid])
    tile = np.array([p.split("_")[2] for p in pid])
    pr = np.load(OUT / "01_loto_probs.npz", allow_pickle=True)
    p_rf = pr["p_rf"]
    fb = F.build(X)
    fdi, ndvi = fb["FDI"].to_numpy(), fb["NDVI"].to_numpy()
    t = y == D.TARGET
    scored = ~np.isin(y, D.NON_WATER)
    held = np.isfinite(p_rf) & scored

    is_water = np.isin(y, D.WATER_CLASSES)
    no_lab = sorted(s for s in set(scene[t]) if not is_water[scene == s].sum())
    print("debris pixels in scenes with fewer than 2 labelled water pixels:",
          int(np.isin(scene, no_lab)[t].sum()), "of", int(t.sum()), "scenes", len(no_lab))

    st = scene_stats()
    tf = np.array([st.get(s, (0, 0, 0, 0, 0))[0] + 3 * st.get(s, (0, 0, 0, 0, 0))[1] for s in scene])
    tv = np.array([st.get(s, (0, 0, 0, 0, 0))[2] + 3 * st.get(s, (0, 0, 0, 0, 0))[3] for s in scene])
    fa, na = fdi > tf, ndvi > tv
    kept = p_rf >= 0.5

    def row(name, pred):
        tp, fp, fn = int((pred & t & held).sum()), int((pred & ~t & held).sum()), int((~pred & t & held).sum())
        return dict(method=name, tp=tp, fp=fp, fn=fn, precision=tp / (tp + fp) if tp + fp else np.nan,
                    recall=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn) if tp else 0.0)

    rows = [row("FDI rule, all water", fa), row("FDI and NDVI rule, all water", fa & na),
            row("FDI rule, all water, then forest", fa & kept),
            row("Deployed chain, all water", fa & na & kept)]

    p_nb = np.full(len(y), np.nan)
    for tl in sorted(set(tile[t])):
        te = tile == tl
        tr = ~te & scored
        nb = GaussianNB(priors=[0.5, 0.5]).fit(np.c_[ndvi[tr], fdi[tr]], t[tr])
        p_nb[te] = nb.predict_proba(np.c_[ndvi[te], fdi[te]])[:, 1]
    rows.append(row("Naive Bayes NDVI-FDI, equal priors", p_nb >= 0.5))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "06_fairness.csv", index=False)
    pd.set_option("display.width", 200)
    print(r.round(3).to_string())
    print(pd.DataFrame(st, index=["fm", "fs", "vm", "vs", "n"]).T.describe().round(3))


if __name__ == "__main__":
    main()
