"""
The published global model's cell terms (Meijer et al. 2021) on the same flood test.

Meijer et al. give each 3-arc-second cell a probability of plastic reaching the ocean,
P(E) = P(M) x P(R) x P(O) (their eq. 1), calibrated on 52 river observations, none in
Guinea. With their best-calibrated values (supplementary tables S9, S10, S15):

    P(M) = min(alpha * P, 1) + min(beta * W, 1), capped at 1     eq. 3
           P annual rainfall (mm), W highest monthly mean wind (m/s)
           alpha = 1.5e-4, beta = 0.02
    P(R) = (mean over the path of nu_i * min(eps * s_i + zeta, eta)) ** D_land     eq. 4
           nu land-use value (table S10), s slope (%), D_land km to the nearest river
           zeta = 0.4, eta = 1
    P(O) close to 1 for every cell here (theta * SO + iota with iota = 0.999), so left out.

Simplifications, all stated: the path mean is taken at the cell itself; land use is
ESA WorldCover mapped to the GLC2000 classes of table S10; rainfall is WorldClim
(Hijmans et al. 2005) and wind TerraClimate (Abatzoglou et al. 2018), not WorldClim2;
the river network is the one the platform's terms already use. The slope coefficient
eps has no value in the paper, so four values are run and all are reported.

Their P(M) x P(R) is the published answer to the same question as the platform's
mobilisation score: how likely waste at a place is to reach water. Scored on the 3,000
flood points of 01, with the same blocks and seed, against the initial score and the
HAND-based score of 03. The flood test covers the inundation route only; Meijer's terms
describe wash-off, so this asks which description better fits where water reaches land
in Guinea, not which model is right in general.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT  # noqa: E402

import ee  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from plasticwaste_tool.characterisation_tool import mobilisation as MOB  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
SPEC = __import__("importlib").import_module("01_mobilisation_flood_test")
IMP = __import__("importlib").import_module("03_improved_mobilisation")
OUT = ROOT / "3_leakage" / "outputs"
ALPHA, BETA, ZETA, ETA = 1.5e-4, 0.02, 0.4, 1.0
EPS = (0.0, 0.005, 0.01, 0.02)
# ESA WorldCover class -> GLC2000 class of Meijer table S10 (calibrated value)
NU = {10: 0.15,   # tree cover -> broadleaved deciduous (most of Guinea's tree cover); evergreen is 0.08
      20: 0.23,   # shrubland -> shrub cover
      30: 0.23,   # grassland -> herbaceous cover
      40: 0.45,   # cropland -> cultivated and managed areas
      50: 0.60,   # built-up -> artificial surfaces
      60: 0.45,   # bare -> bare areas
      80: 0.75,   # water -> water bodies
      90: 0.53,   # herbaceous wetland -> regularly flooded shrub/herbaceous
      95: 0.68,   # mangroves -> tree cover, regularly flooded, saline
      100: 0.23}  # moss and lichen -> sparse herbaceous


def climate(d):
    cache = OUT / "04_points_climate.csv"
    if cache.exists():
        return d.merge(pd.read_csv(cache), on="point_id", how="left")
    rain = ee.Image("WORLDCLIM/V1/BIO").select("bio12").rename("rain_annual_mm")
    tc = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate("1991-01-01", "2021-01-01").select("vs")
    months = ee.List.sequence(1, 12).map(lambda m: tc.filter(ee.Filter.calendarRange(m, m, "month")).mean())
    wind = ee.ImageCollection.fromImages(months).max().multiply(0.01).rename("wind_max_month")
    st = rain.addBands(wind)
    rows = []
    for i in range(0, len(d), 400):
        b = d.iloc[i:i + 400]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"point_id": r.point_id})
                                   for r in b.itertuples()])
        rows += [f["properties"] for f in st.reduceRegions(fc, ee.Reducer.first(), 1000).getInfo()["features"]]
        print("climate", i + len(b), flush=True)
    c = pd.DataFrame(rows)
    if c.rain_annual_mm.isna().mean() > 0.05 or c.wind_max_month.isna().mean() > 0.05:
        raise SystemExit("climate sampling missed more than 5 % of points")
    c.to_csv(cache, index=False)
    return d.merge(c, on="point_id", how="left")


def meijer(s, eps):
    rain = s.rain_annual_mm.fillna(s.rain_annual_mm.median())
    wind = s.wind_max_month.fillna(s.wind_max_month.median())
    p_m = (np.minimum(ALPHA * rain, 1) + np.minimum(BETA * wind, 1)).clip(upper=1)
    nu = s.worldcover.map(NU).fillna(0.23)
    slope_pct = 100 * np.tan(np.radians(s.slope_deg.fillna(s.slope_deg.median())))
    p_step = nu * np.minimum(eps * slope_pct + ZETA, ETA)
    p_r = p_step ** (s.dist_river_m.fillna(1e5) / 1000.0)
    return p_m, p_r


def main():
    ee.Initialize(project=SPEC.project())
    d = climate(pd.read_csv(OUT / "01_points_terms.csv"))
    d["mass_proxy"] = 2.0
    s = MOB.score(d)
    s["improved"] = IMP.improved(s)
    y = s.flood.astype(int).to_numpy()
    block = (np.floor(s.lat / 0.5).astype(int).astype(str) + "_" + np.floor(s.lon / 0.5).astype(int).astype(str)).to_numpy()
    blocks = np.unique(block)
    idx = {b: np.flatnonzero(block == b) for b in blocks}
    scores = {"Initial score (deployed form)": s.mobilisation_risk, "HAND-based score (03)": s.improved}
    for e in EPS:
        p_m, p_r = meijer(s, e)
        scores[f"Meijer P(M) x P(R), eps {e}"] = p_m * p_r
        scores[f"Meijer P(R) alone, eps {e}"] = p_r
    scores["Meijer P(M) alone"] = p_m
    rng = np.random.default_rng(11)
    draws = [np.concatenate([idx[k] for k in rng.choice(blocks, len(blocks))]) for _ in range(1000)]
    draws = [ii for ii in draws if 0 < y[ii].sum() < len(ii)]
    ref_i, ref_h = s.mobilisation_risk.to_numpy(), s.improved.to_numpy()
    rows = []
    for name, v in scores.items():
        v = v.to_numpy()
        if not np.isfinite(v).all():
            raise SystemExit(f"non-finite values in {name}")
        b = np.array([roc_auc_score(y[ii], v[ii]) for ii in draws])
        bi = np.array([roc_auc_score(y[ii], ref_i[ii]) for ii in draws])
        bh = np.array([roc_auc_score(y[ii], ref_h[ii]) for ii in draws])
        rows.append({"score": name, "auc": roc_auc_score(y, v), "p05": np.quantile(b, .05), "p95": np.quantile(b, .95),
                     "vs_initial_p05": np.quantile(b - bi, .05), "vs_initial_p95": np.quantile(b - bi, .95),
                     "vs_hand_p05": np.quantile(b - bh, .05), "vs_hand_p95": np.quantile(b - bh, .95)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "04_meijer_flood_auc.csv", index=False)
    print(t.round(3).to_string())
    info = {"rain_annual_mm": d.rain_annual_mm.describe().round(0).to_dict(),
            "wind_max_month": d.wind_max_month.describe().round(2).to_dict(),
            "p_m_range": [round(float(p_m.min()), 3), round(float(p_m.max()), 3)]}
    (OUT / "04_meijer_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(info)


if __name__ == "__main__":
    main()
