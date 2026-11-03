"""Notebook 05: drivers not yet in the model, as predictors (studies/forest_loss_drivers.ipynb, section 3).

  Cropland            GLAD global land cover and land use 2000-2020 (Potapov et al. 2022, v2), class
                      244: annual and perennial herbaceous crops, fallows of up to 4 years; shifting
                      cultivation with longer fallows is not cropland there. Epochs 2010, 2015, 2020.
  Regrown forest      the same product's trees of 5 m or more (classes 27-48 and 127-148) on 30 m
                      pixels that were Hansen forest in 2000 (canopy >= 30 %) and lost at least 5 years
                      before the epoch: the fallow that has grown back.
  Crop suitability    FAO GAEZ v4, suitability index on all land of the cell (0-10000), rain-fed, low
                      inputs, climate 1981-2010, about 9 km: upland rice, cassava, maize, groundnut
                      (the staples of shifting cultivation; their maximum) and oil palm.

The 30 m products are summed onto the 100 m grid as shares (each pixel to the cell holding its centre,
as reporting.py). A window starting in `year` reads the latest epoch at or before it. The test is
model_variants': the improved model with one candidate added, on the same training and test cells,
AUC over the 30 m pixels, three seeds.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as MM                                               # noqa: E402
import model_variants as D                                       # noqa: E402
import reporting as R                                            # noqa: E402

G, M, P = MM.G, MM.M, MM.P
EPOCHS = (2010, 2015, 2020)
GLCLU = G.STORE / "downloads" / "glclu"
GAEZ_DIR = G.STORE / "downloads" / "gaez"
GAEZ_URL = "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/res05/CRUTS32/Hist/8110L/sxLr_{}.tif"
GAEZ_CROPS = {"rice_upland": "rcd", "cassava": "csv", "maize": "mze", "groundnut": "grd", "oil_palm": "olp"}
CANDIDATES = ["Cropland within 5 km", "Distance to cropland", "Regrown forest within 5 km",
              "Staple crop suitability (GAEZ)", "Oil palm suitability (GAEZ)"]


# --------------------------------------------------------------------------
# Layers

def _glclu(year, r0, r1, c0, c1):
    """GLCLU classes of rows r0..r1, columns c0..c1 of the 30 m lattice (from 80 N, 180 W), from its tiles."""
    import rasterio
    from rasterio.windows import Window
    out = np.full((r1 - r0, c1 - c0), 255, dtype="uint8")
    for path in sorted((GLCLU / str(year)).glob("*.tif")):
        with rasterio.open(path) as src:
            t = src.transform
            tr, tc = round((80 - t.f) / R.PIX), round((t.c + 180) / R.PIX)
            a, b, c, d = max(r0, tr), min(r1, tr + src.height), max(c0, tc), min(c1, tc + src.width)
            if a < b and c < d:
                out[a - r0:b - r0, c - c0:d - c0] = src.read(1, window=Window(c - tc, a - tr, d - c, b - a))
    return out


def build_glclu_layers(log=print):
    """Per 100 m cell and epoch, the share (%) of its 30 m pixels inside Guinea that are cropland, trees of
    5 m or more, and regrown forest -> store layers glclu_<epoch> (landcover group)."""
    from rasterio.transform import Affine
    g = G.grid()
    t, width, height = g["transform"], g["width"], g["height"]
    west, south, east, north = G.outline().bounds
    R0, R1 = int(np.floor((80 - north) / R.PIX)), int(np.ceil((80 - south) / R.PIX))
    C0, C1 = int(np.floor((west + 180) / R.PIX)), int(np.ceil((east + 180) / R.PIX))
    lon = -180 + (np.arange(C0, C1) + 0.5) * R.PIX
    cell_col = np.clip(np.floor((lon - t[2]) / t[0]).astype(int), 0, width - 1)
    n = np.zeros(height * width, dtype="uint16")
    sums = {(e, k): np.zeros(height * width, dtype="uint16") for e in EPOCHS for k in ("cropland", "trees5m", "regrown")}
    start = time.time()
    for r0 in range(R0, R1, R.STRIP_ROWS):
        r1 = min(r0 + R.STRIP_ROWS, R1)
        transform = Affine(R.PIX, 0, -180 + C0 * R.PIX, 0, -R.PIX, 80 - r0 * R.PIX)
        inside = R._rasterize([(G.outline(), 1)], transform, (r1 - r0, C1 - C0)).astype(bool)
        if not inside.any():
            continue
        lat = 80 - (np.arange(r0, r1) + 0.5) * R.PIX
        cell_row = np.clip(np.floor((lat - t[5]) / t[4]).astype(int), 0, height - 1)
        ids = (cell_row[:, None] * width + cell_col[None, :])[inside]
        n += np.bincount(ids, minlength=n.size).astype("uint16")
        forest2000 = (R._read("treecover2000", r0, r1, C0, C1) >= 30)[inside]
        lossyear = R._read("lossyear", r0, r1, C0, C1)[inside]
        for e in EPOCHS:
            cls = _glclu(e, r0, r1, C0, C1)[inside]
            trees = ((cls >= 27) & (cls <= 48)) | ((cls >= 127) & (cls <= 148))
            regrown = trees & forest2000 & (lossyear >= 1) & (lossyear <= e - 2000 - 5)
            for k, m in (("cropland", cls == 244), ("trees5m", trees), ("regrown", regrown)):
                sums[e, k] += np.bincount(ids[m], minlength=n.size).astype("uint16")
        log(f"  30 m rows {r1 - R0:,}/{R1 - R0:,}, {(time.time() - start) / 60:.1f} min", flush=True)
    cells = G.cells()
    total = np.maximum(n[cells], 1).astype("float32")
    for e in EPOCHS:
        G.write_layer("landcover", f"glclu_{e}",
                      {k: np.rint(100 * sums[e, k][cells] / total).astype("uint8") for k in ("cropland", "trees5m", "regrown")},
                      source=f"GLAD GLCLU 2000-2020 v2 ({e}); Hansen GFC-2025-v1.13 30 m tiles", scale=0.01,
                      notes="share of the cell's 30 m pixels inside Guinea: cropland (class 244), trees >= 5 m "
                            "(27-48, 127-148), regrown = trees >= 5 m on Hansen forest 2000 (canopy >= 30) lost >= 5 years before")


def build_gaez_layer(log=print):
    """GAEZ v4 suitability (all land, rain-fed, low inputs, 1981-2010) resampled to 100 m -> layer gaez_suitability."""
    import requests
    GAEZ_DIR.mkdir(parents=True, exist_ok=True)
    columns = {}
    for name, code in GAEZ_CROPS.items():
        path = GAEZ_DIR / f"sxLr_{code}.tif"
        if not path.exists():
            r = requests.get(GAEZ_URL.format(code), timeout=300)
            r.raise_for_status()
            path.write_bytes(r.content)
        values = G.resample(path, resampling="bilinear")
        columns[name] = np.rint(np.nan_to_num(np.clip(values, 0, 10000))).astype("uint16")
        log(f"  {name}: mean {np.nanmean(values):.0f}", flush=True)
    G.write_layer("predictors", "gaez_suitability", columns, scale=1e-4,
                  source="FAO GAEZ v4, res05 CRUTS32 Hist 8110L sxLr_* (suitability index, all land)",
                  notes="0-10000; bilinear from ~9 km")


# --------------------------------------------------------------------------
# Predictors

def _epoch(year):
    return max(e for e in EPOCHS if e <= year)


def _within(values, size=MM.WINDOW_CELLS):
    from scipy.ndimage import uniform_filter
    return G.from_grid(uniform_filter(G.to_grid(values.astype("float32"), fill=0.0), size, mode="constant"))


def candidate(name, year):
    """Values over the cells inside Guinea."""
    if name == "Cropland within 5 km":
        return _within(G.read_layer(f"glclu_{_epoch(year)}", ["cropland"], scaled=True)["cropland"])
    if name == "Distance to cropland":
        crop = G.read_layer(f"glclu_{_epoch(year)}", ["cropland"], scaled=True)["cropland"] >= 0.5
        return P._distance_m(G.to_grid(crop, fill=False))
    if name == "Regrown forest within 5 km":
        return _within(G.read_layer(f"glclu_{_epoch(year)}", ["regrown"], scaled=True)["regrown"])
    s = G.read_layer("gaez_suitability", scaled=True)
    if name == "Staple crop suitability (GAEZ)":
        return np.max([s[c] for c in ("rice_upland", "cassava", "maize", "groundnut")], axis=0)
    if name == "Oil palm suitability (GAEZ)":
        return s["oil_palm"]
    raise KeyError(name)


def build(year, names, rows):
    """As model_variants.build, with the candidates."""
    out = D.build(year, [n for n in names if n not in CANDIDATES], rows)
    for n in names:
        if n in CANDIDATES:
            out[n] = np.asarray(candidate(n, year))[MM.hansen()["idx"]][rows]
    return out


VARIANTS = {"Improved model": D.BASE, **{f"With {c}": D.BASE + [c] for c in CANDIDATES},
            "With all five": D.BASE + CANDIDATES}


def run_split(split, seeds=(42, 1, 2), log=print):
    """Each candidate added to the improved model, as model_variants.run_split (same draws and test cells)."""
    t1, t2, t3 = split
    start = time.perf_counter()
    area = MM.hansen()["area"]
    draws = {s: D._draw(t1, t2, s) for s in seeds}
    names = sorted({n for v in VARIANTS.values() for n in v})
    cal_rows = np.unique(np.concatenate([c for c, _ in draws.values()]))
    cal = build(t1, names, cal_rows)
    f2, lt = MM.forest_pct(t2), MM.loss_pct(t2, t3)
    forest2 = np.flatnonzero(f2 > 0)
    test_rows = np.sort(np.random.default_rng(0).choice(forest2, min(D.TEST_CELLS, forest2.size), replace=False))
    w_pos = lt[test_rows] / 100 * area[test_rows]
    w_neg = np.clip(f2[test_rows] - lt[test_rows], 0, None) / 100 * area[test_rows]
    test = build(t2, names, test_rows)
    log(f"{M.split_label(split)}: predictors built {time.perf_counter() - start:.0f} s")
    out = {"split": M.split_label(split), "variants": {}, "importance": {}}
    for label, v in VARIANTS.items():
        out["variants"][label] = []
        for s in seeds:
            cells, y = draws[s]
            X, ok = D._xy(cal, np.searchsorted(cal_rows, cells), v)
            model = D._fit(X[ok], y[ok], s)
            Xt, okt = D._xy(test, np.arange(test_rows.size), v)
            risk = np.full(test_rows.size, np.nan, dtype="float32")
            risk[okt] = model.predict_proba(Xt[okt])[:, 1]
            out["variants"][label].append(MM.weighted_auc(risk, w_pos, w_neg))
            if label == "With all five" and s == seeds[0]:
                out["importance"] = dict(zip(v, map(float, model.feature_importances_)))
        log(f"{out['split']}: {label} {np.mean(out['variants'][label]):.4f} {time.perf_counter() - start:.0f} s")
    out["seconds"] = time.perf_counter() - start
    return out


def split_name(split):
    return "candidates_{}_{}_{}".format(*split)


if __name__ == "__main__":
    if "--layers" in sys.argv:
        build_gaez_layer()
        build_glclu_layers()
    else:
        for split in MM.SPLITS:
            MM.cached(split_name(split), lambda: run_split(split), rerun="--rerun" in sys.argv)
            print("SAVED", split_name(split), flush=True)
