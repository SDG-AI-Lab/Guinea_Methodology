"""
The candidate chains on the cached river tiles (January 2024, dry season).

Each tile is judged by the documented guards: at least 1,000 water pixels and an
NDVI threshold inside [-1, 1]. For each chain: pixels and patches, and how far they
lie from the bank. A chain that only lights up the bank is reading mixed land and
water pixels, not floating material.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, RASTERS  # noqa: E402

from plasticwaste_tool.geo_env import fix_proj  # noqa: E402
fix_proj()
from plasticwaste_tool import data_loader as DL  # noqa: E402
from plasticwaste_tool.fdi_tool import debris_filter as DF  # noqa: E402
from plasticwaste_tool.fdi_tool import spectral as SP  # noqa: E402
from plasticwaste_tool.marida_tool import features as F  # noqa: E402

from scipy import ndimage as ndi  # noqa: E402

OUT = ROOT / "1_floating_debris" / "outputs"
MIN_PX = 1000


def main():
    model, meta = DF.load()
    k1 = list(model.classes_).index(1)
    rows = []
    paths = sorted(RASTERS.glob("river_*_2024-01-01_2024-02-01_composite.tif"))
    for path in paths:
        aoi = "_".join(path.name.split("_")[:3])
        bands, _ = DL.read_composite(str(path))
        fdi, ndvi = SP.compute_fdi(bands), SP.compute_ndvi(bands)
        w = (np.nan_to_num(bands["WATER"], nan=0.0) > 0) & np.isfinite(fdi) & np.isfinite(ndvi)
        n = int(w.sum())
        if n < 2:
            rows.append({"aoi": aoi, "water_px": n, "valid": False})
            continue
        tf = float(fdi[w].mean() + 3 * fdi[w].std())
        tv = float(ndvi[w].mean() + 3 * ndvi[w].std())
        valid = n >= MIN_PX and abs(tv) <= 1
        idx = np.flatnonzero(w)
        flat = {k: np.asarray(v).ravel()[idx] for k, v in bands.items() if hasattr(v, "shape")}
        X = F.from_composite(flat, meta["features"])
        ok = np.isfinite(X).all(axis=1)
        p = np.zeros(idx.size, "float32")
        p[ok] = model.predict_proba(X[ok])[:, k1]
        P = np.zeros(w.shape, "float32")
        P.ravel()[idx] = p
        dist = ndi.distance_transform_edt(w) * 20.0
        forest = w & (P >= 0.5)
        fa, na = w & (fdi > tf), w & (ndvi > tv)
        r = {"aoi": aoi, "water_px": n, "valid": valid, "t_fdi": tf, "t_ndvi": tv,
             "water_median_dist_m": float(np.median(dist[w]))}
        for tag, det in (("deployed", fa & na & forest), ("fdi_rf", fa & forest), ("forest", forest), ("rule", fa & na)):
            r[f"{tag}_px"] = int(det.sum())
            r[f"{tag}_px_over_40m"] = int((det & (dist > 40)).sum())
            lab, m = ndi.label(det, structure=np.ones((3, 3)))
            r[f"{tag}_patches"] = int((ndi.sum(det, lab, np.arange(1, m + 1)) >= 2).sum()) if m else 0
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "08_river_tiles.csv", index=False)
    v = d[d.valid == True]  # noqa: E712
    print(len(d), "cached tiles,", len(v), "valid, water px", int(v.water_px.sum()))
    for t in ("rule", "deployed", "fdi_rf", "forest"):
        print(t, "px", int(v[f"{t}_px"].sum()), "beyond 40 m of bank", int(v[f"{t}_px_over_40m"].sum()),
              "patches", int(v[f"{t}_patches"].sum()), "tiles", int((v[f"{t}_patches"] > 0).sum()))
    print(v.sort_values("forest_px", ascending=False)[["aoi", "water_px", "t_fdi", "t_ndvi", "forest_px", "forest_patches", "fdi_rf_px", "deployed_px"]].head(8).to_string())


if __name__ == "__main__":
    main()
