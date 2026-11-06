"""
Replacing the inundation term with height above nearest drainage.

The score's weak part on independent floods is its river geometry: catchment size
within 1 km times exp(-distance to river / 2 km). HAND (Nobre et al. 2011) already
measures how far above its river a point sits. Candidate:

    p_inundation = exp(-HAND / 5 m)
    p_mobilise   = 1 - (1 - p_washoff) * (1 - p_inundation)     as before
    score        = mass * p_mobilise                            no separate river term

5 m is one scale chosen before scoring, not tuned; AUC of HAND alone does not depend
on it. Scored on the same 3,000 points, then on the labelled dump points to see how
much the site ranking moves.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, TOOL  # noqa: E402

import ee  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from plasticwaste_tool.characterisation_tool import mobilisation as MOB  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
SPEC = __import__("importlib").import_module("01_mobilisation_flood_test")
OUT = ROOT / "3_leakage" / "outputs"
H = 5.0


def improved(s):
    p_in = np.exp(-s.hand.fillna(s.hand.median()).clip(lower=0) / H)
    return 1.0 - (1.0 - s.p_washoff) * (1.0 - p_in)


def main():
    d = pd.read_csv(OUT / "01_points_terms.csv")
    d["mass_proxy"] = 2.0
    s = MOB.score(d)
    s["improved"] = improved(s)
    y = s.flood.astype(int).to_numpy()
    block = (np.floor(s.lat / 0.5).astype(int).astype(str) + "_" + np.floor(s.lon / 0.5).astype(int).astype(str)).to_numpy()
    blocks = np.unique(block)
    idx = {b: np.flatnonzero(block == b) for b in blocks}
    rng = np.random.default_rng(11)
    vals = {"improved": s.improved.to_numpy(), "deployed": s.mobilisation_risk.to_numpy()}
    dif, b_imp = [], []
    for _ in range(1000):
        ii = np.concatenate([idx[k] for k in rng.choice(blocks, len(blocks))])
        if 0 < y[ii].sum() < len(ii):
            a, b = roc_auc_score(y[ii], vals["improved"][ii]), roc_auc_score(y[ii], vals["deployed"][ii])
            b_imp.append(a)
            dif.append(a - b)
    res = {"improved_auc": roc_auc_score(y, vals["improved"]), "improved_p05": float(np.quantile(b_imp, .05)),
           "improved_p95": float(np.quantile(b_imp, .95)), "deployed_auc": roc_auc_score(y, vals["deployed"]),
           "diff_p05": float(np.quantile(dif, .05)), "diff_p95": float(np.quantile(dif, .95))}

    ee.Initialize(project=SPEC.project())
    m = pd.read_csv(TOOL / "runs" / "mobilisation" / "mobilisation_risk.csv")
    hand = ee.Image("MERIT/Hydro/v1_0_1").select("hnd").rename("hand")
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"point_id": r.point_id})
                               for r in m.itertuples()])
    got = pd.DataFrame([f["properties"] for f in hand.reduceRegions(fc, ee.Reducer.first(), 90).getInfo()["features"]])
    got = got.rename(columns={"first": "hand"})
    m = m.merge(got, on="point_id", how="left")
    m["cell"] = m.point_id.str.replace(r"_p\d+$", "", regex=True)
    m["improved"] = m.mass_proxy * improved(m)
    res["site_spearman_with_deployed"] = float(spearmanr(m.improved, m.mobilisation_risk).statistic)
    a, b = set(m.nlargest(15, "improved").cell), set(m.nlargest(15, "mobilisation_risk").cell)
    res["top15_cells_shared"] = len(a & b)
    res["regional_mean_improved"] = (100 * m.groupby("region").improved.mean() / m.improved.max()).round(1).to_dict()
    res["regional_mean_deployed"] = m.groupby("region").mobilisation_risk.mean().round(1).to_dict()
    m.to_csv(OUT / "03_sites_improved.csv", index=False)
    (OUT / "03_improved.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
