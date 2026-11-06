"""
Is the central leakage estimate worth adding to the platform?

Uses saved files and the platform's own chain, no Earth Engine. The central case uses
18.5 % plastic for urban people and 4 % for rural. This moves those shares across
plausible values, and also lowers rural waste per person, because the chain gives
everyone the same waste rate. The rule, fixed before running: add it if

  1. at least 80 % of the combinations give 9-13 kg of plastic per person, the
     published range (0.03 kg a day, rounded),
  2. the central case itself lies inside that range,
  3. the regions keep their order: Spearman 0.9 or more with the central case for
     every combination.

The result also shows how far Conakry's share of leakage moves, to judge whether the
rise from about 14 % to 28 % can be stated.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT  # noqa: E402

from scipy.stats import spearmanr  # noqa: E402

from plasticwaste_tool.leakage_tool import massbalance as MB  # noqa: E402
from plasticwaste_tool.leakage_tool import parameters as P  # noqa: E402

IN = ROOT / "3_leakage" / "outputs"
OUT = ROOT / "6_three_changes" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
URBAN = (.15, .18, .185, .20, .22)
RURAL = (.03, .04, .05, .08, .10)
RURAL_WASTE = (1.0, .8, .6)  # rural waste per person, against urban
WINDOW = (9.0, 13.0)


def leak(adm, urban, rural, waste, params):
    """To the environment by region, and national plastic per person."""
    u = MB.chain(adm.pop_urban.to_numpy(), 0.0 * adm.pop_urban.to_numpy(), **{**params, "plastic_share": urban})
    r = MB.chain(0.0 * adm.pop_rural.to_numpy(), adm.pop_rural.to_numpy() * waste, **{**params, "plastic_share": rural})
    env = u["to_environment_t_yr"] + r["to_environment_t_yr"]
    kg = (u["plastic_t_yr"].sum() + r["plastic_t_yr"].sum()) * 1000 / adm.pop_total.sum()
    return env, kg


def main():
    adm = pd.read_csv(IN / "02_adm1.csv")
    n = json.loads((IN / "02_mixed_mc.json").read_text())["population"]
    scale = n / adm.pop_total.sum()  # the regions leave a little population unassigned
    for c in ("pop_urban", "pop_rural", "pop_total"):
        adm[c] = adm[c] * scale
    params = {k: v.value for k, v in P.all_parameters("waw_2007").items()}
    base, base_kg = leak(adm, .185, .04, 1.0, params)
    i = adm.index[adm.adm1 == "Conakry"][0]

    rows = []
    for u in URBAN:
        for r in RURAL:
            for w in RURAL_WASTE:
                env, kg = leak(adm, u, r, w, params)
                rows.append({"urban_share": u, "rural_share": r, "rural_waste": w, "to_env_t": env.sum(), "plastic_kg": kg,
                             "in_window": WINDOW[0] <= kg <= WINDOW[1], "conakry_share": env[i] / env.sum(),
                             "spearman_regions": spearmanr(env, base).statistic})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "02_central_grid.csv", index=False)
    rule = {"central_kg": float(base_kg), "central_to_env_t": float(base.sum()), "central_conakry_share": float(base[i] / base.sum()),
            "share_in_window": float(t.in_window.mean()), "to_env_range_t": [float(t.to_env_t.min()), float(t.to_env_t.max())],
            "conakry_share_range": [float(t.conakry_share.min()), float(t.conakry_share.max())],
            "lowest_spearman": float(t.spearman_regions.min())}
    rule["add_it"] = bool(rule["share_in_window"] >= .8 and WINDOW[0] <= base_kg <= WINDOW[1] and rule["lowest_spearman"] >= .9)
    (OUT / "02_rule.json").write_text(json.dumps(rule, indent=1), encoding="utf-8")
    print(json.dumps(rule, indent=1))


if __name__ == "__main__":
    main()
