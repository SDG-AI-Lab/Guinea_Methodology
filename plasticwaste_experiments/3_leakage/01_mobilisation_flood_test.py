"""
Does the mobilisation score rank flooded ground higher, on independent points?

The published check uses the labelled dump points: 13 flooded points out of 111,
but they come from only 2 places, so it has about two observations. The score's
terrain terms do not depend on the dump labels, so the same test can be run at
thousands of independent points. Here: settled 100 m cells of Guinea, drawn in
equal numbers flooded and never flooded in the Global Flood Database (Tellman et
al. 2021), permanent water left out. Equal numbers leave the AUC unchanged.

The mass term is held constant: it is unknown at these points and is not part of
the hydrology. Normalised terms are normalised on this sample, as the tool does on
its own. Compared with what a hydrologist would use first: height above nearest
drainage (MERIT Hydro, Nobre et al. 2011) and the JRC flood hazard map (100-year
depth, Baugh et al. 2024). Intervals resample 0.5-degree blocks.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import PROJECT, ROOT  # noqa: E402

import ee  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from plasticwaste_tool.characterisation_tool import mobilisation as MOB  # noqa: E402

OUT = ROOT / "3_leakage" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
N_EACH = 1500
SEED = 11


def project():
    for line in (PROJECT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("GEE_PROJECT_ID="):
            return line.split("=", 1)[1].strip().strip('"')


def sample():
    cache = OUT / "01_points.csv"
    if cache.exists():
        return pd.read_csv(cache)
    guinea = ee.FeatureCollection("WM/geoLab/geoBoundaries/600/ADM0").filter(ee.Filter.eq("shapeGroup", "GIN"))
    region = guinea.geometry()
    pop = (ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("country", "GIN"))
           .filter(ee.Filter.eq("year", 2020)).first())
    gfd = ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1").filterBounds(region)
    flooded = gfd.select("flooded").sum().gt(0)
    perm = gfd.select("jrc_perm_water").max().gt(0)
    cls = flooded.And(perm.Not()).unmask(0).rename("flood").updateMask(pop.gte(1)).updateMask(perm.unmask(0).Not())
    pts = cls.stratifiedSample(numPoints=N_EACH, classBand="flood", region=region, scale=250, seed=SEED,
                               geometries=True, tileScale=8)
    rows = pts.getInfo()["features"]
    d = pd.DataFrame([{"lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1],
                       "flood": f["properties"]["flood"]} for f in rows])
    d["point_id"] = [f"p{i}" for i in range(len(d))]
    d.to_csv(cache, index=False)
    return d


def extra(d):
    hand = ee.Image("MERIT/Hydro/v1_0_1").select("hnd").rename("hand")
    jrc = ee.ImageCollection("JRC/CEMS_GLOFAS/FloodHazard/v1").filter(ee.Filter.eq("return_period", 100))
    jrc = jrc.mosaic().select("depth").unmask(0).rename("jrc_depth")
    st = hand.addBands(jrc)
    rows = []
    for i in range(0, len(d), 400):
        b = d.iloc[i:i + 400]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"point_id": r.point_id})
                                   for r in b.itertuples()])
        for f in st.reduceRegions(fc, ee.Reducer.first(), 90).getInfo()["features"]:
            rows.append(f["properties"])
        print("extra", i + len(b), flush=True)
    return d.merge(pd.DataFrame(rows), on="point_id", how="left")


def main():
    ee.Initialize(project=project())
    cache = OUT / "01_points_terms.csv"
    if cache.exists():
        d = pd.read_csv(cache)
    else:
        d = sample()
        print(d.flood.value_counts().to_dict(), flush=True)
        d = extra(d)
        d = MOB.extract_terms(d)
        d.to_csv(cache, index=False)
    d["mass_proxy"] = 2.0
    s = MOB.score(d)
    s["hand_low"] = -s.hand
    s["acc_log"] = np.log1p(s.upstream_acc.fillna(0))
    s["near_river"] = -s.dist_river_m
    s["flat"] = -s.slope_deg
    y = s.flood.astype(int).to_numpy()
    block = (np.floor(s.lat / 0.5).astype(int).astype(str) + "_" + np.floor(s.lon / 0.5).astype(int).astype(str)).to_numpy()
    names = {"mobilisation_risk": "Mobilisation score (deployed form)",
             "p_inundation": "Inundation term alone (upstream catchment)",
             "p_to_river": "Distance to river term alone",
             "p_washoff": "Wash-off term alone",
             "hand_low": "Height above nearest drainage (HAND)",
             "jrc_depth": "JRC flood hazard map (100-year depth)",
             "flat": "Slope alone"}
    rng = np.random.default_rng(SEED)
    blocks = np.unique(block)
    idx = {b: np.flatnonzero(block == b) for b in blocks}
    rows, boots = [], {}
    for c, name in names.items():
        v = s[c].fillna(s[c].median()).to_numpy()
        b = []
        for _ in range(1000):
            ii = np.concatenate([idx[k] for k in rng.choice(blocks, len(blocks))])
            if 0 < y[ii].sum() < len(ii):
                b.append(roc_auc_score(y[ii], v[ii]))
        boots[c] = np.array(b)
        rows.append({"score": name, "auc": roc_auc_score(y, v), "p05": np.quantile(b, .05), "p95": np.quantile(b, .95)})
    t = pd.DataFrame(rows).sort_values("auc", ascending=False)
    t.to_csv(OUT / "01_flood_auc.csv", index=False)
    print(t.round(3).to_string())
    info = {"n": int(len(y)), "flooded": int(y.sum()), "blocks": int(len(blocks))}
    (OUT / "01_flood_info.json").write_text(json.dumps(info), encoding="utf-8")
    print(info)


if __name__ == "__main__":
    main()
