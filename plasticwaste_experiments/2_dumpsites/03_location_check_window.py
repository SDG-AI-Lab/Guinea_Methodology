"""
What the "check this location" menu measures.

The check is justified as the point-centred task. But it does not centre on the
pin: it scores the cell of the global 80 m lattice holding the pin, so the pin can
lie up to about 56 m from the window's centre, and a 14 m dumpsite can sit at its
corner or outside it. This scores the 651 labelled points both ways, on free Esri
imagery with the deployed head's recipe, cross-validated by the spatial blocks of the published evaluation: fit on
point-centred crops of the training cells, then score the held-out cells' centred
crop and their lattice window.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # noqa: F401  # before GDAL on Windows

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, TOOL  # noqa: E402

from plasticwaste_tool.dumpsite_tool import features as bm  # noqa: E402
from plasticwaste_tool.dumpsite_tool import inference as gi  # noqa: E402
from plasticwaste_tool.dumpsite_tool import scan as ds  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = ROOT / "2_dumpsites" / "outputs"
TILES = OUT / "lattice_tiles"
M = 80.0


def window(lat, lon):
    dlat = M / 110_574.0
    j = int(math.floor((lat - ds.ORIGIN_LAT) / dlat))
    dlon = ds._row_dlon(j, dlat, M)
    i = int(math.floor((lon - ds.ORIGIN_LON) / dlon))
    return ds._window(j, i, dlat, M)


def main():
    d = pd.read_csv(OUT / "set_points.csv")
    wins = [window(a, b) for a, b in zip(d.lat, d.lon)]
    d["tile_id"] = [w["tile_id"] for w in wins]
    cx = np.array([(w["minx"] + w["maxx"]) / 2 for w in wins])
    cy = np.array([(w["miny"] + w["maxy"]) / 2 for w in wins])
    d["pin_offset_m"] = np.hypot((d.lon - cx) * 111_320 * np.cos(np.radians(d.lat)), (d.lat - cy) * 110_574)
    tiles = pd.DataFrame(wins).drop_duplicates("tile_id")
    TILES.mkdir(exist_ok=True)
    errors = {}
    gi.fetch_tiles_esri(tiles, str(TILES), size_px=800, zoom=ds.SCREEN_ZOOM, errors=errors)
    print("fetched", len(list(TILES.glob("*.png"))), "of", len(tiles), "errors", len(errors), flush=True)
    d["lat_path"] = [str(TILES / f"{t}.png") for t in d.tile_id]
    ok = np.array([Path(p).exists() for p in d.lat_path])
    Xc = np.load(OUT / "feat_points_resnet50.npy")
    f = OUT / "feat_points_lattice.npy"
    if not f.exists():
        np.save(f, bm.imagenet_features(d.lat_path[ok].tolist()))
    Xl = np.full(Xc.shape, np.nan, "float32")
    Xl[ok] = np.load(f)
    y = d.y.to_numpy()
    # The spatial blocks of the published evaluation: a held-out cell's neighbours
    # are held out with it.
    oof = np.load(TOOL / "manual_qa" / "_oof_pt80_esri.npz", allow_pickle=True)
    blocks = dict(zip(oof["sid"].astype(str), oof["groups"].astype(str)))
    cell = d.sid.astype(str).map(blocks).to_numpy().astype(str)
    sc, sl = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xc, y, cell):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.001, max_iter=3000, class_weight="balanced"))
        m.fit(Xc[tr], y[tr])
        sc[te] = m.predict_proba(Xc[te])[:, 1]
        k = te[ok[te]]
        sl[k] = m.predict_proba(Xl[k])[:, 1]
    v = np.isfinite(sl)
    res = {"n": int(v.sum()), "n_pos": int(y[v].sum()),
           "centred_auc": roc_auc_score(y[v], sc[v]), "centred_ap": average_precision_score(y[v], sc[v]),
           "lattice_auc": roc_auc_score(y[v], sl[v]), "lattice_ap": average_precision_score(y[v], sl[v]),
           "pin_offset_median_m": float(d.pin_offset_m.median()), "pin_offset_max_m": float(d.pin_offset_m.max()),
           "pos_share_at_or_above_0.7_centred": float((sc[v & (y == 1)] >= .7).mean()),
           "pos_share_at_or_above_0.7_lattice": float((sl[v & (y == 1)] >= .7).mean()),
           "neg_share_at_or_above_0.7_centred": float((sc[v & (y == 0)] >= .7).mean()),
           "neg_share_at_or_above_0.7_lattice": float((sl[v & (y == 0)] >= .7).mean()),
           "pos_share_at_or_above_0.5_centred": float((sc[v & (y == 1)] >= .5).mean()),
           "pos_share_at_or_above_0.5_lattice": float((sl[v & (y == 1)] >= .5).mean())}
    rng = np.random.default_rng(1)
    cells = np.unique(cell[v])
    idx = {c: np.flatnonzero(v & (cell == c)) for c in cells}
    dif = []
    for _ in range(2000):
        ii = np.concatenate([idx[c] for c in rng.choice(cells, len(cells))])
        if 0 < y[ii].sum() < len(ii):
            dif.append(roc_auc_score(y[ii], sl[ii]) - roc_auc_score(y[ii], sc[ii]))
    res["auc_diff_p05"], res["auc_diff_p95"] = float(np.quantile(dif, .05)), float(np.quantile(dif, .95))
    (OUT / "03_location_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
