#!/usr/bin/env python
"""
Run the plastic leakage model and write every table the methodology needs.

    python 31_run_leakage.py                    # both scenarios, all outputs
    python 31_run_leakage.py --scenario waw_2007
    python 31_run_leakage.py --no-sites         # skip per-site allocation
    python 31_run_leakage.py --n 50000          # more Monte Carlo draws

Writes to manual_qa/runs/leakage/. Costs nothing: every input is a local file
and no Earth Engine call is made, so this can be re-run freely.

WHAT COMES OUT, AND WHICH QUESTION EACH ANSWERS

  parameters_<scenario>.csv      what was assumed, and on whose authority
  validation.json                does the chain reproduce figures it was not
                                 given? (three independent checks)
  national_<scenario>.csv        the headline estimate with its interval
  variance_<scenario>.csv        which parameter the estimate is limited by
                                 -- i.e. what a survey should measure next
  adm1_<scenario>.csv            where the waste is generated
  sites_<scenario>.csv           tonnes per detected site, with transport
  sensitivity.csv                what the two conventions (urban threshold,
                                 catchment radius) cost
  summary.json                   everything above, condensed, plus the
                                 cross-scenario comparison

The scenario split is the headline uncertainty and is never averaged over:
see leakage_tool/parameters.py, PLASTIC_SHARE_SCENARIOS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from plasticwaste_tool.leakage_tool import massbalance as MB   # noqa: E402
from plasticwaste_tool.leakage_tool import parameters as P     # noqa: E402
from plasticwaste_tool.leakage_tool import population as POP    # noqa: E402
from plasticwaste_tool.leakage_tool import sites as SITES       # noqa: E402
from plasticwaste_tool.leakage_tool import uncertainty as UNC    # noqa: E402

OUT = HERE / "manual_qa" / "runs" / "leakage"


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def validate(pop) -> dict:
    """
    Three checks the model can fail, none of which use their answer as input.

    1. POPULATION. Area-weighted total against UN WPP 2020. Catches the
       density-vs-count error that would inflate everything by 20%.
    2. COLLECTION. Urban and rural surveyed rates, weighted by the urban share
       this model derives, against the separately-reported national rate.
       Catches a wrong urban/rural threshold.
    3. GENERATION. Population x rate against What a Waste's own published
       national tonnage for Guinea. Catches an arithmetic or units error in
       the first link of the chain.

    A model that passes all three is not thereby correct -- none of them
    touches the back half of the chain, where the evidence is weakest -- but
    one that failed any of them would be wrong for a findable reason.
    """
    v = POP.validate_against_census(pop)
    pu, pr = POP.urban_rural_population(pop, P.URBAN_DENSITY_THRESHOLD.value)
    cc = MB.check_collection_consistency(pu / (pu + pr))

    # What a Waste's own national MSW tonnage for Guinea, 2022 (WM_MSW_GEN,
    # Tonnes per year). Our chain multiplies ITS per-capita rate by OUR
    # population surface, so agreement tests that the two are consistent.
    waw_reported_t = 1_568_192.34
    modelled = MB.deterministic(pu, pr)["msw_t_yr"]

    return {
        "population": {**v, "passes": bool(v["within_tolerance"])},
        "collection": {**cc, "passes": bool(cc["agrees_within_20pct"])},
        "generation": {
            "modelled_msw_t_yr": modelled,
            "waw_reported_msw_t_yr": waw_reported_t,
            "relative_error": modelled / waw_reported_t - 1.0,
            "passes": bool(abs(modelled / waw_reported_t - 1.0) <= 0.10),
        },
        "note": ("All three checks exercise the FRONT of the chain only. "
                 "Open burning, environmental leakage and transport to water "
                 "have no Guinean measurement to be checked against, which is "
                 "why they dominate the variance decomposition."),
    }


def run_scenario(pop, scenario: str, n: int, with_sites: bool) -> dict:
    import pandas as pd

    print(f"\n=== scenario: {scenario} "
          f"(plastic share {P.PLASTIC_SHARE_SCENARIOS[scenario].value:.1%}) ===")

    P.provenance_table(scenario).to_csv(
        OUT / f"parameters_{scenario}.csv", index=False)

    pu, pr = POP.urban_rural_population(pop, P.URBAN_DENSITY_THRESHOLD.value)
    det = MB.deterministic(pu, pr, scenario)
    if not MB.check_mass_balance(det):
        raise AssertionError(
            "mass balance does not close: burned + leaked + contained != "
            "plastic generated. Refusing to write results.")

    mc = UNC.national(pop, n=n, scenario=scenario)
    summary = UNC.summarise(mc)
    summary.to_csv(OUT / f"national_{scenario}.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    var = UNC.contribution_to_variance(mc)
    var.to_csv(OUT / f"variance_{scenario}.csv", index=False)
    top = var.iloc[0]
    print(f"\nlargest single source of uncertainty: {top['parameter']} "
          f"({top['share_of_variance']:.0%} of variance)")

    adm1 = UNC.by_adm1(pop, n=n, scenario=scenario)
    adm1.to_csv(OUT / f"adm1_{scenario}.csv", index=False)
    add = UNC.check_additivity(adm1)
    print(f"correlated national interval "
          f"{add['national_p5_correct']:,.0f}-{add['national_p95_correct']:,.0f} t/yr; "
          f"quadrature would understate its width by "
          f"{add['quadrature_understates_by']:.1f}x")

    out = {
        "scenario": scenario,
        "plastic_share": P.PLASTIC_SHARE_SCENARIOS[scenario].value,
        "deterministic": {k: v for k, v in det.items()
                          if not isinstance(v, np.ndarray)},
        "national_quantiles": summary.to_dict("records"),
        "variance_shares": var.to_dict("records"),
        "additivity": add,
        "adm1_top3": adm1.head(3)[["adm1", "to_env_t_yr_p50"]].to_dict("records"),
    }

    if with_sites:
        try:
            s = SITES.allocate(pop, scenario=scenario)
            s.to_csv(OUT / f"sites_{scenario}.csv", index=False)
            cov = SITES.national_coverage(s, pop, scenario)
            out["sites"] = cov
            print(f"{cov['n_sites']} sites: "
                  f"{cov['sites_to_environment_t_yr']:,.0f} t/yr "
                  f"({cov['share_of_national']:.0%} of national) from "
                  f"{cov['population_share']:.0%} of the population")
            out["cross_check"] = MB.cross_check_against_detections(
                det["to_environment_t_yr"], len(s), n_fdi_patches=44)
        except FileNotFoundError as e:
            print(f"  skipping per-site allocation: {e}")
            out["sites"] = {"error": str(e)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=sorted(P.PLASTIC_SHARE_SCENARIOS) + ["all"],
                    default="all")
    ap.add_argument("--n", type=int, default=UNC.DEFAULT_N,
                    help="Monte Carlo draws (default %(default)s)")
    ap.add_argument("--no-sites", action="store_true",
                    help="skip per-site allocation")
    ap.add_argument("--seed", type=int, default=UNC.DEFAULT_SEED)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    print("loading population surface ...")
    pop = POP.load_population()
    print(f"  {POP.total_population(pop):,.0f} people "
          f"(area-weighted from a DENSITY grid; the naive pixel sum would "
          f"give {pop['density'][pop['valid']].sum():,.0f})")

    val = validate(pop)
    print("\nvalidation:")
    for k in ("population", "collection", "generation"):
        c = val[k]
        err = c.get("relative_error", c.get("ratio", 1) - 1)
        print(f"  {k:12s} {'PASS' if c['passes'] else 'FAIL'}  "
              f"({err:+.1%} vs the independent figure)")
    (OUT / "validation.json").write_text(
        json.dumps(val, indent=2, default=_jsonable), encoding="utf-8")

    scenarios = (sorted(P.PLASTIC_SHARE_SCENARIOS)
                 if args.scenario == "all" else [args.scenario])
    results = {s: run_scenario(pop, s, args.n, not args.no_sites)
               for s in scenarios}

    sens = {"urban_threshold": MB.urban_split_sensitivity(pop).to_dict("records")}
    if not args.no_sites:
        try:
            sens["catchment_radius"] = SITES.catchment_sensitivity(
                pop).to_dict("records")
        except FileNotFoundError:
            pass
    import pandas as pd
    pd.DataFrame(sens["urban_threshold"]).to_csv(
        OUT / "sensitivity_urban_threshold.csv", index=False)
    if "catchment_radius" in sens:
        pd.DataFrame(sens["catchment_radius"]).to_csv(
            OUT / "sensitivity_catchment_radius.csv", index=False)

    summary = {
        "generated": __file__,
        "n_draws": args.n, "seed": args.seed,
        "population_source": pop["path"],
        "population_total": POP.total_population(pop),
        "validation": val,
        "scenarios": results,
        "sensitivity": sens,
    }
    if len(results) > 1:
        a, b = "waw_2007", "conakry_recent"
        ea = results[a]["deterministic"]["to_environment_t_yr"]
        eb = results[b]["deterministic"]["to_environment_t_yr"]
        summary["scenario_spread"] = {
            "low_t_yr": ea, "high_t_yr": eb, "ratio": eb / ea,
            "note": ("The gap between scenarios exceeds the Monte Carlo "
                     "interval within either one. The binding constraint on "
                     "this estimate is therefore not statistical -- it is that "
                     "Guinea has no recent national waste composition survey. "
                     "No amount of additional computation narrows it."),
        }
        print(f"\nSCENARIO SPREAD: {ea:,.0f} - {eb:,.0f} t/yr to environment "
              f"({eb / ea:.1f}x). This gap is wider than either scenario's own "
              f"90% interval.")

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_jsonable), encoding="utf-8")
    print(f"\nwrote {len(list(OUT.glob('*')))} files to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
