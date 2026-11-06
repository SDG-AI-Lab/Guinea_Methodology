"""
Is the height-based mobilization score worth adding?

Uses only saved files, no Earth Engine. The rule, fixed before running: adopt it if

  1. its AUC stays at 0.80 or more for every height scale tried,
  2. its gain over the current score has a lower bound above 0 for every scale,
  3. it also beats the current score against a second flood source, the JRC map,
  4. the site order barely depends on the scale (Spearman 0.8 or more between scales).

It tests the same thing as before: can water reach this place. It says nothing about
how much plastic moves.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from plasticwaste_tool.characterisation_tool import mobilisation as MOB  # noqa: E402

IN = ROOT / "3_leakage" / "outputs"
OUT = ROOT / "6_three_changes" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SCALES = (3.0, 5.0, 10.0, 20.0)
N_BOOT = 1000


def height_score(s, scale):
    p_in = np.exp(-s.hand.fillna(s.hand.median()).clip(lower=0) / scale)
    return 1.0 - (1.0 - s.p_washoff) * (1.0 - p_in)


def gain(y, new, old, block, rng):
    blocks = np.unique(block)
    idx = {b: np.flatnonzero(block == b) for b in blocks}
    out = []
    for _ in range(N_BOOT):
        ii = np.concatenate([idx[k] for k in rng.choice(blocks, len(blocks))])
        if 0 < y[ii].sum() < len(ii):
            out.append(roc_auc_score(y[ii], new[ii]) - roc_auc_score(y[ii], old[ii]))
    return float(np.quantile(out, .05)), float(np.quantile(out, .95))


def main():
    d = pd.read_csv(IN / "01_points_terms.csv")
    d["mass_proxy"] = 2.0
    s = MOB.score(d)
    block = (np.floor(s.lat / 0.5).astype(int).astype(str) + "_" + np.floor(s.lon / 0.5).astype(int).astype(str)).to_numpy()
    old = s.mobilisation_risk.to_numpy()
    rng = np.random.default_rng(11)
    targets = {"Global Flood Database": s.flood.astype(int).to_numpy(), "JRC 100-year map": (s.jrc_depth > 0).astype(int).to_numpy()}
    rows = []
    for name, y in targets.items():
        rows.append({"flood_source": name, "score": "Current", "scale_m": None, "auc": roc_auc_score(y, old)})
        for h in SCALES:
            new = height_score(s, h).to_numpy()
            low, high = gain(y, new, old, block, rng)
            rows.append({"flood_source": name, "score": "Height based", "scale_m": h, "auc": roc_auc_score(y, new),
                         "gain_low": low, "gain_high": high})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "01_height_score.csv", index=False)
    print(t.round(3).to_string())

    m = pd.read_csv(IN / "03_sites_improved.csv")
    site = pd.DataFrame({h: m.mass_proxy * height_score(m, h) for h in SCALES})
    between = float(site.corr(method="spearman").min().min())
    ranks = pd.DataFrame({"Current": m.groupby("region").mobilisation_risk.mean().rank(ascending=False)})
    for h in SCALES:
        ranks[f"Height {h:g} m"] = site[h].groupby(m.region).mean().rank(ascending=False)
    ranks.astype(int).to_csv(OUT / "01_region_ranks.csv")
    print(ranks.astype(int).sort_values("Height 5 m").to_string())

    g = t[t.score == "Height based"]
    first = g[g.flood_source == "Global Flood Database"]
    second = g[g.flood_source == "JRC 100-year map"]
    current_jrc = float(t[(t.flood_source == "JRC 100-year map") & (t.score == "Current")].auc.iloc[0])
    rule = {"auc_at_least_0.80": bool((first.auc >= .80).all()), "gain_above_zero": bool((first.gain_low > 0).all()),
            "beats_current_on_jrc": bool((second.auc > current_jrc).all()),
            "site_order_stable": between >= .8, "lowest_spearman_between_scales": between,
            "spearman_with_current_at_5m": float(spearmanr(site[5.0], m.mobilisation_risk).statistic)}
    rule["adopt"] = all(rule[k] for k in ("auc_at_least_0.80", "gain_above_zero", "beats_current_on_jrc", "site_order_stable"))
    (OUT / "01_rule.json").write_text(json.dumps(rule, indent=1), encoding="utf-8")
    print(json.dumps(rule, indent=1))


if __name__ == "__main__":
    main()
