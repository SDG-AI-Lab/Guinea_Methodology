"""Benchmark step 1: one label pool, leakage-safe groups, true crops at 80/150/300 m.

Label pool (early 250 m tiles excluded):
  natsample  651 points from the September national sample (exact dump pins; negatives = cell points)
  blind_*    979 80 m windows, Boke and Conakry blind grids (September)
  round1     400 random national windows (November) -> the town test
  round2     399 enriched windows (November)
  added      25 points added from imagery (November)

Labels: 1 = waste site, 0 = none, -1 = unsure (low confidence; left out of the main score,
counted as a site in the sensitivity run).

Leakage control:
  * duplicates: labels from DIFFERENT sets < 40 m apart are one place; keep one by priority
    (within a set nothing is dropped, so the 651 curated set stays identical);
    round1 > natsample > added > blind > round2 (round1 is the test set and stays intact);
  * groups: 1 km blocks, then any two labels < 100 m apart are forced into the same group,
    so near-neighbours can never sit on both sides of a fold.

Imagery: Esri World Imagery z17 (z16 where z17 is a placeholder), the only zoom available
everywhere in Guinea. Each crop is cut from the tile mosaic to EXACTLY the window and
resampled to a fixed ~1.17 m/px: 80 m -> 68 px, 150 m -> 128 px, 300 m -> 256 px.

Output (outputs/benchmark/): pool.parquet, crops_<w>m.npy (uint8, N x px x px x 3), zoom.npy
"""
import importlib.util, io, json, math, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd, geopandas as gpd, mercantile as mt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT as EXP, PROJECT, TOOL  # noqa: E402
NATSCAN = Path(r"E:\SDG AI Lab\Guinea\national_scan_2026")  # Esri tile driver + tile cache of the Nov 2026 national scan
WORK = EXP / "2_dumpsites" / "outputs" / "benchmark"; WORK.mkdir(parents=True, exist_ok=True)
QA = TOOL / "manual_qa"
NL = QA / "national_labels_2026"

spec = importlib.util.spec_from_file_location("drv", NATSCAN / "scripts" / "02_scan.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
from contextily.tile import _merge_tiles  # noqa: E402

WINDOWS = {80: 68, 150: 128, 300: 256}
PRIORITY = {"round1": 0, "natsample": 1, "added": 2, "blind_boke": 3, "blind_conakry": 3, "round2": 4}
LAB = {"dump": 1, "no": 0, "unsure": -1}


def build_pool():
    a = pd.read_csv(QA / "_point_samples.csv", encoding="latin1")
    a = pd.DataFrame({"src_id": a.sid.astype(str), "set": "natsample", "lat": a.lat, "lon": a.lon,
                      "label": a.label.astype(int), "unit": "point", "note": a.note})
    parts = [a]
    for reg in ("boke", "conakry"):
        g = gpd.read_file(QA / "regions" / f"{reg}_blindgrid_review.geojson")
        c = g.to_crs(3857).geometry.centroid.to_crs(4326)
        parts.append(pd.DataFrame({"src_id": g.tile_id, "set": f"blind_{reg}", "lat": c.y, "lon": c.x,
                                   "label": g.label.astype(int), "unit": "window",
                                   "conf": g.confidence.replace("", np.nan), "note": g.notes,
                                   "score_deployed_sep": g.model_score}))
    for r in ("round1", "round2"):
        w = pd.read_csv(NL / f"{r}_windows.csv"); l = pd.read_csv(NL / f"{r}_labels.csv")
        m = w.merge(l[["label_id", "label", "note"]], on="label_id")
        d = pd.DataFrame({"src_id": m.label_id, "set": r, "lat": m.lat, "lon": m.lon,
                          "label": m.label.map(LAB).astype(int), "unit": "window", "note": m.note,
                          "score_deployed": m.score})
        if r == "round1":
            d["stratum"] = m.stratum
        parts.append(d)
    p = pd.read_csv(NL / "added_points.csv")
    parts.append(pd.DataFrame({"src_id": [f"P{i:02d}" for i in range(len(p))], "set": "added",
                               "lat": p.lat, "lon": p.lon, "label": 1, "unit": "point"}))
    df = pd.concat(parts, ignore_index=True)
    df["prio"] = df["set"].map(PRIORITY)
    return df


def xy_m(lat, lon):
    lat0 = 10.0
    return np.column_stack([np.radians(lon) * 6_371_000 * math.cos(math.radians(lat0)),
                            np.radians(lat) * 6_371_000])


def dedupe_and_group(df):
    from scipy.spatial import cKDTree
    xy = xy_m(df.lat.to_numpy(), df.lon.to_numpy()); t = cKDTree(xy)
    order = np.lexsort((np.arange(len(df)), df.prio.to_numpy()))
    keep = np.ones(len(df), bool); log = {"dropped": 0, "conflicts": 0}
    for i in order:
        if not keep[i]:
            continue
        for j in t.query_ball_point(xy[i], 40.0):
            if j != i and keep[j] and df["set"].iat[j] != df["set"].iat[i] and df.prio.iat[j] >= df.prio.iat[i]:
                keep[j] = False; log["dropped"] += 1
                if {df.label.iat[i], df.label.iat[j]} == {0, 1}:
                    log["conflicts"] += 1
    df = df[keep].reset_index(drop=True); xy = xy[keep]
    # groups: 1 km blocks, merged with any pair < 100 m (union-find)
    _, blk = np.unique(np.floor(xy / 1000.0).astype(int), axis=0, return_inverse=True)
    par = np.arange(blk.max() + 1)

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i, j in cKDTree(xy).query_pairs(100.0):
        ra, rb = f(blk[i]), f(blk[j])
        if ra != rb:
            par[rb] = ra
    df["group"] = np.array([f(b) for b in blk])
    df["uid"] = [f"{s}:{i}" for s, i in zip(df["set"], df.src_id)]
    return df, log


_ph = {}


def tile_arr(t):
    b = drv.fetch_tile(t.z, t.x, t.y)
    if b is None:
        return None, True
    a = np.asarray(Image.open(io.BytesIO(b)).convert("RGBA"))
    k = (t.z, t.x, t.y)
    if k not in _ph:
        _ph[k] = bool(drv.gi._is_placeholder(Image.fromarray(a).convert("RGB").resize((800, 800))))
    return a, _ph[k]


def box_at(lat, lon, m):
    dlat = m / 2 / 111320.0; dlon = m / 2 / (111320.0 * math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def crops(lat, lon):
    big = box_at(lat, lon, max(WINDOWS))
    for z in drv.ZOOMS:
        tiles = drv.tiles_for(big, z); arrs = []
        bad = False
        for t in tiles:
            a, ph = tile_arr(t)
            if a is None or ph:
                bad = True; break
            arrs.append(a)
        if bad:
            continue
        merged, (w_, s_, e_, n_) = _merge_tiles(tiles, arrs)
        full = Image.fromarray(merged).convert("RGB"); H, W = merged.shape[:2]
        X0, Y0 = mt.xy(w_, s_); X1, Y1 = mt.xy(e_, n_)
        out = {"zoom": z}
        for m, px in WINDOWS.items():
            bx = box_at(lat, lon, m); a = mt.xy(bx[0], bx[1]); b = mt.xy(bx[2], bx[3])
            cx = lambda X: (X - X0) / (X1 - X0) * W
            cy = lambda Y: (Y1 - Y) / (Y1 - Y0) * H
            im = full.crop((cx(a[0]), cy(b[1]), cx(b[0]), cy(a[1]))).resize((px, px), Image.BICUBIC)
            out[m] = np.asarray(im, np.uint8)
        return out
    return None


def main():
    df, log = dedupe_and_group(build_pool())
    print("pool after dedupe:", len(df), log)
    print(df.groupby("set").label.value_counts().unstack(fill_value=0))
    print("groups:", df.group.nunique())
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda r: crops(r[0], r[1]), zip(df.lat, df.lon)))
    ok = np.array([r is not None for r in res]); df["imaged"] = ok
    df["zoom"] = [r["zoom"] if r else 0 for r in res]
    print("imaged:", ok.sum(), "/", len(df), "zoom:", df.zoom.value_counts().to_dict())
    for m, px in WINDOWS.items():
        A = np.zeros((len(df), px, px, 3), np.uint8)
        for i, r in enumerate(res):
            if r:
                A[i] = r[m]
        np.save(WORK / f"crops_{m}m.npy", A)
    df.to_parquet(WORK / "pool.parquet")
    json.dump({"dedupe": log, "n": int(len(df)), "imaged": int(ok.sum()),
               "by_set": {k: {str(a): int(b) for a, b in v.items()}
                          for k, v in df.groupby("set").label.value_counts().unstack(fill_value=0).to_dict("index").items()}},
              open(WORK / "pool_meta.json", "w"), indent=1)
    # QA montage: 6 dumps x 3 windows
    s = df[(df.label == 1) & ok].sample(6, random_state=0).index
    tiles = [np.hstack([np.asarray(Image.fromarray(np.load(WORK / f"crops_{m}m.npy", mmap_mode="r")[i]).resize((256, 256))) for m in WINDOWS]) for i in s]
    Image.fromarray(np.vstack(tiles)).save(WORK / "qa_crops.jpg", quality=85)


if __name__ == "__main__":
    main()
