"""grid_100m notebook 07, section 1: do the code versions of the tools behave like the tools' own programs?
Checked at 250 m, where the programs were run by hand (BENCHMARK_GUIDE.md), with the 250 m scoring of
notebook 09a (benchmark_scoring.tool_row, like_for_like; the allocator of the 250 m stage). A separate
process from the 100 m runs, which use main's landuse_tool.

  MOLUSCE  molusce_run.py (MOLUSCE's own code) against output_molusce_2 (the GUI run): as it runs and
           with the same amount of change
  FLUS     flus_ca.py (pyflus-ca's port of FLUS_console's cellular automaton) on each of the three FLUS
           GUI runs' own probability maps, three seeds each, against the GUI's simulated maps

    python tools_check250.py   ->  grid_100m/outputs/07_check250_molusce.csv, 07_check250_flus_ca.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

import benchmark_scoring as BS
import esri_methods as E
import flus_ca as F
import nb_helpers as H

B = BS.BENCH
OUT = Path(__file__).resolve().parents[1] / "grid_100m" / "outputs"
MOLUSCE_CODE = {"code, float64, seed 0": "output_molusce_code", "code, float32, seed 0": "output_molusce_code_f32",
                "code, float32, seed 1": "output_molusce_code_f32_s1", "code, float32, seed 2": "output_molusce_code_f32_s2"}
FLUS_GUI = {"run 1": "", "run 2": "run2/", "run 3": "run3/"}


def molusce(method="lr"):
    prof = E.grid(); g = H.guinea_mask(prof)
    b, c = E.esri_cached(2024), E.esri_cached(2025)
    runs = {"GUI": (B / "output_molusce_2", f"molusce_{method}_2025_2.tif")}
    runs |= {k: (B / v, f"molusce_{method}_2025.tif") for k, v in MOLUSCE_CODE.items() if (B / v / f"molusce_{method}_2025.tif").exists()}
    gui = BS.to_esri(H.read_tif(runs["GUI"][0] / runs["GUI"][1])[0])
    ok = g & E.valid_classes(b, c)
    rows = {}
    for who, (folder, f) in runs.items():
        who = f"MOLUSCE {method.upper()}, {who}"
        s = BS.to_esri(H.read_tif(folder / f)[0])
        same, missing = BS.like_for_like(folder, method, f"molusce_check_{method}_{folder.name}")
        as_run = BS.tool_row(b, c, s, g)
        rows[who] = {"method": method.upper(), "changes forecast": as_run["Changes forecast"],
                     "FoM as it runs %": as_run["Figure of Merit %"],
                     "FoM same amount %": BS.tool_row(b, c, same, g)["Figure of Merit %"],
                     "same class as the GUI map, % of pixels": 100 * float((s[ok] == gui[ok]).mean())}
        print(who, rows[who], flush=True)
    return pd.DataFrame(rows).T


def flus_ca():
    prof = E.grid(); g = H.guinea_mask(prof)
    b, c = E.esri_cached(2024), E.esri_cached(2025)
    land = H.read_tif(B / "inputs" / "landcover_2024.tif")[0]
    rows = []
    for run, sub in FLUS_GUI.items():
        prob_file = B / "output_flus" / (sub + ("Probability-of-occurrence.tif" if sub else "probability-of-occurrence.tif"))
        with rasterio.open(prob_file) as src:
            prob = src.read()
        gui = BS.tool_row(b, c, BS.to_esri(H.read_tif(B / "output_flus" / f"{sub}flus_2025.tif")[0]), g, draws=1)
        rows.append({"probability map": f"FLUS {run}", "amount": "observed", "simulation": "FLUS program",
                     "changes forecast": gui["Changes forecast"], "FoM %": gui["Figure of Merit %"]})
        for seed in (0, 1, 2):
            sim, hist = F.simulate(land, prob, BS.LEGEND["pixels_2025_observed"].values, seed=seed)
            r = BS.tool_row(b, c, BS.to_esri(sim), g, draws=1)
            rows.append({"probability map": f"FLUS {run}", "amount": "observed", "simulation": f"code, seed {seed}",
                         "changes forecast": r["Changes forecast"], "FoM %": r["Figure of Merit %"], "iterations": len(hist)})
            print(rows[-1], flush=True)
    with rasterio.open(B / "output_flus" / "probability-of-occurrence.tif") as src:
        prob = src.read()
    gui = BS.tool_row(b, c, BS.to_esri(H.read_tif(B / "output_flus" / "projected" / "flus_2025_projected.tif")[0]), g, draws=1)
    rows.append({"probability map": "FLUS run 1", "amount": "projected", "simulation": "FLUS program",
                 "changes forecast": gui["Changes forecast"], "FoM %": gui["Figure of Merit %"]})
    sim, hist = F.simulate(land, prob, BS.LEGEND["pixels_2025_projected"].values, seed=0)
    r = BS.tool_row(b, c, BS.to_esri(sim), g, draws=1)
    rows.append({"probability map": "FLUS run 1", "amount": "projected", "simulation": "code, seed 0",
                 "changes forecast": r["Changes forecast"], "FoM %": r["Figure of Merit %"], "iterations": len(hist)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    which = sys.argv[1:] or ["molusce", "flus"]
    if "molusce" in which:
        methods = [m for m in ("lr", "ann") if any((B / v / f"molusce_{m}_2025.tif").exists() for v in MOLUSCE_CODE.values())]
        pd.concat([molusce(m) for m in methods]).to_csv(OUT / "07_check250_molusce.csv")
    if "flus" in which:
        flus_ca().to_csv(OUT / "07_check250_flus_ca.csv", index=False)
