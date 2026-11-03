"""Notebook 07: the land-change tools at 100 m, scored as the deployed method is (deployed100.score).

Inputs are the tool export (tools_export100, E:/.../landuse/benchmark): learn 2023 -> 2024, forecast
2025 from the 2024 map. Each tool is run in its own program (GUI) or from code; which one is
recorded with every result.

  PLUS     GUI: LEAS and CARS (output_plus/), the user's runs
  MOLUSCE  code: the plugin's own algorithms without its window (scripts/molusce_run.py, in QGIS's
           Python), with the settings of the 250 m GUI runs; factor maps held in float32 to fit in memory
  CLUE-S   code: the Python port of Dyna-CLUE 2.7 (scripts/clue_s.py; at 250 m it gave the original
           program's map on 100.00 % of pixels)
  FLUS     code: the copy of FLUS's network (scripts/flus_ann.py; at 250 m within 0.25 points of
           FLUS's own maps, notebook 09). FLUS's cellular automaton has no faithful copy (notebook 07,
           section 1), so FLUS is compared on its suitability maps only.

Two comparisons, as at 250 m (BENCHMARK_DETAILS.md, section 4):
  as it runs        the tool's own forecast, with the amount of change it is given
  same amount       the tool's suitability maps place the observed number of each 2024 -> 2025
                    transition with main's allocator, as the backend's "observed" runs
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE.parent / "scripts"))
import deployed100 as D                                             # noqa: E402

P, G, C = D.P, D.G, D.C
BENCH = P.LANDUSE / "benchmark"
CODE = BENCH / "output_code"                                        # everything written here, next to the user's runs
LEGEND = pd.read_csv(BENCH / "legend_and_demand.csv")
TO_ESRI = dict(zip(LEGEND.code, LEGEND.esri_code))
PHASE = "test"                                                      # 2023 -> 2024 -> 2025
QGIS_PYTHON = r"C:\Program Files\QGIS 3.22.9\bin\python-qgis-ltr.bat"
COMPRESS = {"compress": "deflate", "predictor": 3, "tiled": True, "blockxsize": 512, "blockysize": 512}


def read(path, band=1):
    with rasterio.open(path) as src:
        return src.read(band)


def to_esri(tool_codes):
    """Tool codes 1..7 (0 = no data) to Esri classes, per cell inside Guinea."""
    v = G.from_grid(tool_codes) if tool_codes.ndim == 2 else tool_codes
    out = np.zeros(v.shape, np.uint8)
    for k, esri in TO_ESRI.items():
        out[v == k] = esri
    return out


def write_classes(vector, path):
    """Tool codes per cell inside Guinea as a grid GeoTIFF (0 = no data)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = {**G.profile("uint8", 0), "compress": "deflate"}
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(G.to_grid(vector, fill=0, dtype="uint8"), 1)
    return path


def write_probabilities(P_cells, path):
    """(k, cells inside Guinea) probabilities as a k-band float32 grid GeoTIFF, band k = tool code k."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = {**G.profile("float32", -1.0), **COMPRESS, "count": len(P_cells)}
    with rasterio.open(path, "w", **prof) as dst:
        for k, p in enumerate(P_cells, start=1):
            dst.write(G.to_grid(p, fill=-1.0, dtype="float32"), k)
    return path


# --------------------------------------------------------------------------
# Runs from code

def factors_cells(year, flus=False):
    """(12, cells inside Guinea) predictors as exported; -9999 and NaN (17 cells of slope and aspect)
    filled with the layer's mean, as a regression needs complete rows."""
    folder = BENCH / "inputs" / (f"factors_{year}_flus" if flus else f"factors_{year}")
    X = np.stack([G.from_grid(read(f)).astype(np.float32) for f in sorted(folder.glob("*.tif"))])
    X[X == -9999] = np.nan
    for row in X:
        row[np.isnan(row)] = np.nanmean(row)
    return X


def clue_s(demand_kind="observed", log=print):
    """CLUE-S (clue_s.py) as at 250 m: one logistic regression per class on the 2024 map and the 2024
    predictors, conversion elasticity = share of each class unchanged 2023 -> 2024, every conversion
    allowed, demand = the 2025 total of each class; convergence 1 % on average, 5 % for any class."""
    import clue_s as CS
    out = CODE / "clue_s" / demand_kind
    if (out / "clue_s_2025.tif").exists():
        return out
    a, b = (G.from_grid(read(BENCH / "inputs" / f"landcover_{y}.tif")) for y in (2023, 2024))
    domain = b > 0
    classes = list(LEGEND.code)
    X = factors_cells(2024)[:, domain]
    probs_path = CODE / "clue_s" / "clue_s_probability.tif"
    P_dom, auc, _, _ = CS.regressions(b[domain], X, np.ones(domain.sum(), bool), classes)
    del X
    if not probs_path.exists():
        full = np.full((len(classes), b.size), -1.0, np.float32)
        full[:, domain] = P_dom
        write_probabilities(full, probs_path)
        del full
    both = (a > 0) & domain
    stat = np.array([((a == k) & (b == k) & both).sum() / max(((a == k) & both).sum(), 1) for k in classes])
    demand = LEGEND[f"pixels_2025_{demand_kind}"].values.astype(float)
    demand *= domain.sum() / demand.sum()
    new, loops = CS.allocate(P_dom, b[domain].astype(int) - 1, demand, stat, avg_max=1.0, ind_max=5.0, log=log)
    sim = np.zeros(b.size, np.uint8)
    sim[domain] = np.array(classes, np.uint8)[new]
    write_classes(sim, out / "clue_s_2025.tif")
    (out / "clue_s_run.json").write_text(json.dumps({"iterations": int(loops), "elasticity": stat.tolist(),
                                                     "demand": demand.tolist(), "regression_auc": auc}, indent=1))
    return out


def flus_network(seed=42, log=print):
    """The copy of FLUS's network (flus_ann.py) on the 2024 map and FLUS's 0-1 predictors of 2024, as
    FLUS was run: a 7-band probability map like FLUS's Probability-of-occurrence.tif."""
    import flus_ann as FA
    path = CODE / "flus_network" / f"probability_s{seed}.tif"
    if not path.exists():
        # flus_ann.probability_map, reading each layer for the cells inside Guinea only (same cells in the
        # same row-major order as its valid-pixel mask, so the same sample and network)
        land = G.from_grid(read(BENCH / "inputs" / "landcover_2024.tif"))
        valid = land > 0
        X = factors_cells(2024, flus=True)[:, valid].T
        y = land[valid].astype(int) - 1
        sample = FA.proportional_sample(y, 0.02, seed)
        net = FA.OccurrenceNetwork(seed=seed).fit(X[sample], y[sample])
        log(f"[2024, seed {seed}] trained on {sample.size:,} pixels; RMSE {net.rmse(X[sample], y[sample]):.3f}")
        full = np.full((FA.N_CLASSES, land.size), -1.0, np.float32)
        full[:, valid] = net.predict_proba(X).T
        write_probabilities(full, path)
    return path


def molusce(method, seed=0):
    """MOLUSCE's own code (molusce_run.py) in QGIS's Python; returns its output folder."""
    import subprocess
    out = CODE / "molusce" / (method if seed == 0 else f"{method}_s{seed}")
    if not (out / f"molusce_{method}_run.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "log.txt", "w") as log:
            subprocess.run([QGIS_PYTHON, str(HERE.parent / "scripts" / "molusce_run.py"), str(BENCH), method, str(out), str(seed),
                            "--float32"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    return out


# --------------------------------------------------------------------------
# Same amount of change: the tool's suitability maps through main's allocator

def transitions():
    """The observed 2024 -> 2025 transitions (Esri codes) with their counts, as the backend's observed runs."""
    return D.observed_demand(2024, 2025)


def class_band_atlas(path, nodata=None, out=None):
    """{(f, t): file} for a k-band probability map whose band k is tool code k (FLUS, CLUE-S): the
    allocator reads class t's probability on the class-f cells. One file per band is written once,
    in out (default: beside the map)."""
    out = out or path.parent
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    with rasterio.open(path) as src:
        prof = {**src.profile, "count": 1}
        for k in range(1, src.count + 1):
            f = out / f"{path.stem}_band{k}.tif"
            if not f.exists():
                with rasterio.open(f, "w", **prof) as dst:
                    dst.write(src.read(k), 1)
            files[TO_ESRI[k]] = f
    return {(f, t): str(files[t]) for f, t in transitions().stack()[lambda s: s > 0].index if f != t and t in files}


def plus_atlas():
    """PLUS's LEAS development potential of each class (0-255), one file per class."""
    files = {TO_ESRI[k]: BENCH / "output_plus" / f"leas_development_potential_band_{k}.tif" for k in TO_ESRI}
    return {(f, t): str(files[t]) for f, t in transitions().stack()[lambda s: s > 0].index if f != t}


def molusce_atlas(folder, method):
    """MOLUSCE's transition potential maps, potential_<method>__from_<i>_0_to_<j>_0.tif (0-100)."""
    import re
    atlas = {}
    for p in Path(folder).glob(f"potential_{method}__from_*_to_*.tif"):
        i, j = map(int, re.findall(r"from_(\d+)_0_to_(\d+)_0", p.name)[0])
        atlas[(TO_ESRI[i], TO_ESRI[j])] = str(p)
    wanted = set(transitions().stack()[lambda s: s > 0].index)
    return {k: v for k, v in atlas.items() if k in wanted and k[0] != k[1]}


def tool_transitions(path):
    """The 2024 -> 2025 transitions (Esri codes) a tool's forecast made, cells inside Guinea, valid classes."""
    start, pred = C.class_100(2024), to_esri(read(path))
    ok = ~np.isin(start, D.INVALID) & ~np.isin(pred, D.INVALID)
    t = pd.crosstab(start[ok], pred[ok])
    t.index, t.columns = t.index.astype(int), t.columns.astype(int)
    return t


def same_amount(name, atlas, demand=None, folder="same_amount"):
    """Main's run_allocation_simulation with the observed transitions (or `demand`) and the tool's maps
    (cached in output_code/<folder>/<name>)."""
    D.P100.tool()
    from landuse_tool import prediction
    out = CODE / folder / name
    path = out / "predicted_land_cover.tif"
    if not path.exists():
        out.mkdir(parents=True, exist_ok=True)
        demand = transitions() if demand is None else demand
        report = {}
        prediction.run_allocation_simulation(P.write_esri(2024), demand, atlas, str(out), stochastic=False,
                                             allocation_report=report)
        missing = sorted({(int(f), int(t)) for f, t in demand.stack()[lambda s: s > 0].index if f != t} - set(atlas))
        (out / "allocation.json").write_text(json.dumps({"report": report, "transitions_without_map": missing},
                                                        indent=1, default=float))
    return path


# --------------------------------------------------------------------------
# Scores

def score_tool(path):
    """deployed100.score of a tool's forecast in tool codes."""
    return D.score_map(PHASE, to_esri(read(path)))


def score_esri(path):
    return D.score(PHASE, path)


# --------------------------------------------------------------------------
# Notebook 07

def scores():
    """Every score of run07.py (output_code/scores/*.json)."""
    return pd.DataFrame([json.loads(p.read_text()) for p in sorted((CODE / "scores").glob("*.json"))]).set_index("key")


def outcome(pred):
    """Per cell inside Guinea, test window (Pontius et al. 2008): 1 hit, 2 wrong class, 3 miss, 4 false
    alarm; pred: Esri classes per cell (to_esri, or G.from_grid of an Esri-coded file)."""
    start, obs = C.class_100(2024), C.class_100(2025)
    ok = ~np.isin(start, D.INVALID) & ~np.isin(obs, D.INVALID) & ~np.isin(pred, D.INVALID)
    o, s = ok & (obs != start), ok & (pred != start)
    out = np.zeros(start.shape, np.uint8)
    out[o & s & (pred == obs)] = 1
    out[o & s & (pred != obs)] = 2
    out[o & ~s] = 3
    out[~o & s] = 4
    return out
