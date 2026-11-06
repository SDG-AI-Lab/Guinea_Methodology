"""
The candidate chains on Guinea's own imagery, with and without a tide-safe water mask.

MARIDA holds no tidal flats, so a chain that wins there can still fail here. There is
no labelled debris for Guinea, so three things are measured instead, on the cached
Conakry products:
  how much each chain flags;
  what share of it lies on mapped tidal flats or on water seen less than 90 % of the
  time (Murray et al. 2019; JRC Global Surface Water);
  stability: of what is flagged on 11 January, how much is flagged again, same pixel,
  on 26 January. Drifting material should not repeat; a fixed surface does.
Uses the frozen deployed classifier. The tide-safe mask is deployed water minus
tidal flats, keeping occurrence >= 90.
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, RASTERS  # noqa: E402

from plasticwaste_tool.geo_env import fix_proj  # noqa: E402
fix_proj()
import rasterio  # noqa: E402

from plasticwaste_tool import data_loader as DL  # noqa: E402
from plasticwaste_tool.fdi_tool import debris_filter as DF  # noqa: E402
from plasticwaste_tool.fdi_tool import detection as D  # noqa: E402
from plasticwaste_tool.fdi_tool import spectral as SP  # noqa: E402
from plasticwaste_tool.marida_tool import features as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scipy import ndimage as ndi  # noqa: E402

OUT = ROOT / "1_floating_debris" / "outputs"
MASK = OUT / "07_conakry_tide_mask.tif"
REF = RASTERS / "conakry_a410b95f_2024-01-01_2024-02-01_composite.tif"
ARCH = RASTERS.parent / "archive" / "fdi"
B10 = ["B3", "B4", "B6", "B8", "B8A", "B11", "B12", "FDI", "NDVI", "WATER"]
PRODUCTS = {
    "2023-11 median": (ARCH / "conakry_a410b95f_2023-11_de_africa_10b.tif", B10),
    "2024-01 median": (REF, None),
    "2024-01-11": (RASTERS / "conakry_a410b95f_2024-01-11_2024-01-12_composite.tif", None),
    "2024-01-26": (RASTERS / "conakry_a410b95f_2024-01-26_2024-01-27_composite.tif", None),
}


def fetch_mask():
    if MASK.exists():
        return
    import importlib

    import ee
    spec = importlib.import_module("04_patches_on_tidal_flats")
    ee.Initialize(project=spec.project())
    with rasterio.open(REF) as s:
        t, w, h, crs = s.transform, s.width, s.height, s.crs.to_string()
    tidal = (ee.ImageCollection("UQ/murray/Intertidal/v1_1/global_intertidal").select("classification")
             .mosaic().unmask(0).gt(0).rename("tidal"))
    occ = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).rename("occ")
    img = tidal.addBands(occ).toUint8()
    url = img.getDownloadURL({"crs": crs, "crs_transform": [t.a, t.b, t.c, t.d, t.e, t.f],
                              "dimensions": f"{w}x{h}", "format": "GEO_TIFF"})
    urllib.request.urlretrieve(url, MASK)


def score_all(bands, water):
    """Frozen classifier on every water pixel, in blocks."""
    model, meta = DF.load()
    idx = np.flatnonzero(water)
    flat = {k: np.asarray(v).ravel()[idx] for k, v in bands.items() if hasattr(v, "shape")}
    p = np.zeros(idx.size, "float32")
    k1 = list(model.classes_).index(1)
    for i in range(0, idx.size, 800_000):
        sub = {k: v[i:i + 800_000] for k, v in flat.items()}
        X = F.from_composite(sub, meta["features"])
        ok = np.isfinite(X).all(axis=1)
        q = np.zeros(len(X), "float32")
        if ok.any():
            q[ok] = model.predict_proba(X[ok])[:, k1]
        p[i:i + 800_000] = q
    out = np.zeros(water.shape, "float32")
    out.ravel()[idx] = p
    return out


def n_patches(m):
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    return int((ndi.sum(m, lab, np.arange(1, n + 1)) >= 2).sum()) if n else 0


def main():
    fetch_mask()
    with rasterio.open(MASK) as s:
        tidal, occ = s.read(1) > 0, s.read(2)
    flat = tidal | (occ < 90)
    rows, keep = [], {}
    for name, (path, names) in PRODUCTS.items():
        bands, _ = DL.read_composite(str(path), band_names=names) if names else DL.read_composite(str(path))
        fdi, ndvi = SP.compute_fdi(bands), SP.compute_ndvi(bands)
        water = np.nan_to_num(bands["WATER"], nan=0.0) > 0
        safe = water & ~flat
        p = score_all(bands, water)
        print(name, "scored", flush=True)
        for mask_name, m, w in (("deployed water", None, water), ("tide-safe water", safe, safe)):
            mean_f, sd_f = float(np.nanmean(fdi[w])), float(np.nanstd(fdi[w]))
            mean_v, sd_v = float(np.nanmean(ndvi[w])), float(np.nanstd(ndvi[w]))
            fa = w & (fdi > mean_f + 3 * sd_f)
            na = w & (ndvi > mean_v + 3 * sd_v)
            forest = w & (p >= 0.5)
            for chain, det in (("Deployed: FDI and NDVI, then forest", fa & na & forest),
                               ("FDI, then forest", fa & forest), ("Forest alone", forest)):
                rows.append({"product": name, "mask": mask_name, "chain": chain, "px": int(det.sum()),
                             "patches": n_patches(det), "ha": det.sum() * 0.04,
                             "share_on_flats": float(det[flat].sum() / det.sum()) if det.sum() else np.nan,
                             "t_fdi": mean_f + 3 * sd_f, "t_ndvi": mean_v + 3 * sd_v})
                keep[(name, mask_name, chain)] = det
        del bands, fdi, ndvi, p
    d = pd.DataFrame(rows)
    stab = []
    for (name, mk, ch), det in keep.items():
        if name == "2024-01-11":
            other = keep[("2024-01-26", mk, ch)]
            near = ndi.binary_dilation(other, structure=np.ones((3, 3)))
            stab.append({"mask": mk, "chain": ch, "px_11": int(det.sum()), "px_26": int(other.sum()),
                         "share_11_also_26_within_1px": float((det & near).sum() / det.sum()) if det.sum() else np.nan,
                         "expected_random": float(near[det.shape[0] and np.ones_like(det)].mean())})
    s = pd.DataFrame(stab)
    d.to_csv(OUT / "07_candidates.csv", index=False)
    s.to_csv(OUT / "07_stability.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 40, "display.max_rows", 60)
    print(d.round(3).to_string())
    print(s.round(3).to_string())
    np.savez_compressed(OUT / "07_masks.npz", **{"|".join(k): v for k, v in keep.items()
                                                 if k[1] == "tide-safe water"})


if __name__ == "__main__":
    main()
