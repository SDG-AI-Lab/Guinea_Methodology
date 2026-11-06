"""
The leakage chain against published national figures, and its structural variants.

Reproduces the two deployed scenarios from the deployed code, then applies the
plastic share only where it was measured: a mixed scenario uses Conakry's recent
share for urban people and the 2007 national share for rural people. Also: open
burning off (the naive estimate) and collected waste treated as uncontrolled
dumping. Every run is the deployed chain, only inputs change.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT  # noqa: E402

from plasticwaste_tool.leakage_tool import massbalance as MB  # noqa: E402
from plasticwaste_tool.leakage_tool import parameters as P  # noqa: E402
from plasticwaste_tool.leakage_tool import population as POP  # noqa: E402

OUT = ROOT / "3_leakage" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def params(scenario="waw_2007", **over):
    p = {k: v.value for k, v in P.all_parameters(scenario).items()}
    p.update(over)
    return p


def mixed(pu, pr, share_urban=0.185, share_rural=0.04, **over):
    u = MB.chain(pu, 0.0, **params(plastic_share=share_urban, **over))
    r = MB.chain(0.0, pr, **params(plastic_share=share_rural, **over))
    keys = ["msw_t_yr", "plastic_t_yr", "collected_t_yr", "uncollected_t_yr", "burned_t_yr",
            "to_environment_t_yr", "contained_t_yr", "to_water_t_yr"]
    return {k: float(u[k] + r[k]) for k in keys}


def main():
    pop = POP.load_population()
    thr = float(P.URBAN_DENSITY_THRESHOLD.value)
    pu, pr = POP.urban_rural_population(pop, thr)
    n = pu + pr
    print(f"population {n:,.0f}, urban share {pu / n:.3f}")
    rows = []

    def add(name, r):
        rows.append({"variant": name, "plastic_t_yr": float(r["plastic_t_yr"]),
                     "plastic_kg_cap_yr": float(r["plastic_t_yr"]) * 1000 / n,
                     "to_env_t_yr": float(r["to_environment_t_yr"]),
                     "to_env_kg_cap_yr": float(r["to_environment_t_yr"]) * 1000 / n,
                     "to_water_t_yr": float(r["to_water_t_yr"]),
                     "effective_plastic_share": float(r["plastic_t_yr"]) / float(r["msw_t_yr"])})

    add("Deployed low: 4 % everywhere", MB.deterministic(pu, pr, "waw_2007"))
    add("Deployed high: 18.5 % everywhere", MB.deterministic(pu, pr, "conakry_recent"))
    add("Mixed: 18.5 % urban, 4 % rural", mixed(pu, pr))
    add("Mixed, no open burning (naive)", mixed(pu, pr, open_burn_fraction=0.0))
    add("Mixed, collected waste dumped openly", mixed(pu, pr, litter_rate_collected=0.75 * 0.45))
    add("High, no open burning (naive)", MB.chain(pu, pr, **params("conakry_recent", open_burn_fraction=0.0)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "02_variants.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(t.round(3).to_string())

    # Monte Carlo of the mixed scenario, drawn as the deployed uncertainty module draws.
    from plasticwaste_tool.leakage_tool import uncertainty as UNC
    d = UNC.draw_parameters(20000, "waw_2007")
    su = P.PLASTIC_SHARE_SCENARIOS["conakry_recent"].draw(20000, np.random.default_rng(1))
    sr = P.PLASTIC_SHARE_SCENARIOS["waw_2007"].draw(20000, np.random.default_rng(2))
    grid = np.array([300., 500., 750., 1000., 1250., 1500.])
    shares = np.array([(lambda a: a[0] / (a[0] + a[1]))(POP.urban_rural_population(pop, g)) for g in grid])
    us = np.interp(d["urban_density_threshold"], grid, shares)
    o = {k: v for k, v in d.items() if k != "plastic_share"}
    u = MB.chain(n * us, 0 * us, plastic_share=su, **o)
    r = MB.chain(0 * us, n * (1 - us), plastic_share=sr, **o)
    env = u["to_environment_t_yr"] + r["to_environment_t_yr"]
    wat = u["to_water_t_yr"] + r["to_water_t_yr"]
    mc = {"to_env_p05": float(np.quantile(env, .05)), "to_env_p50": float(np.quantile(env, .5)),
          "to_env_p95": float(np.quantile(env, .95)), "to_water_p05": float(np.quantile(wat, .05)),
          "to_water_p50": float(np.quantile(wat, .5)), "to_water_p95": float(np.quantile(wat, .95)),
          "urban_share": pu / n, "population": n}
    (OUT / "02_mixed_mc.json").write_text(json.dumps(mc, indent=1), encoding="utf-8")
    print(mc)

    # Regional shape: how much is the map a population map?
    z = POP.zonal_by_adm1(pop, thr)
    z["env_low"] = [float(MB.chain(a, b, **params())["to_environment_t_yr"]) for a, b in zip(z.pop_urban, z.pop_rural)]
    z["pop_share"] = z.pop_total / z.pop_total.sum()
    z["env_share"] = z.env_low / z.env_low.sum()
    print(np.corrcoef(z.pop_share, z.env_share)[0, 1], (z.env_low * 1000 / z.pop_total).round(2).to_dict())
    z.drop(columns=[c for c in z.columns if c == "geometry"]).to_csv(OUT / "02_adm1.csv", index=False)


if __name__ == "__main__":
    main()
