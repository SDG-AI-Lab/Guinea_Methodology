"""Scoring for the tool comparison (notebook 00): the validation set the literature recommends
(Figure of Merit with producer's and user's accuracy, quantity and allocation disagreement,
random placement, coarser resolutions, AUC of suitability maps), MOLUSCE's own metrics computed
exactly as MOLUSCE computes them, and a like-for-like run that places the observed amount of
change with a tool's own suitability maps through the deployed allocator."""
import re

import numpy as np
import pandas as pd

import esri_methods as E
import nb_helpers as H

BENCH = H.OUTPUTS / "benchmark"
LEGEND = pd.read_csv(BENCH / "legend_and_demand.csv")
TO_ESRI = dict(zip(LEGEND.code, LEGEND.esri_code))
FROM_ESRI = {v: k for k, v in TO_ESRI.items()}


def to_esri(tool_map):
    """Tool codes 1..7 (0 = nodata) back to ESRI codes (0 = nodata)."""
    out = np.zeros(tool_map.shape, np.uint8)
    for k, esri in TO_ESRI.items():
        out[tool_map == k] = esri
    return out


def molusce_metrics(reference, simulated, pixels):
    """% of correctness and the three kappas, as MOLUSCE's
    algorithms/models/correlation/model.py computes them from the cross-table:
    overall (P0 - PE) / (1 - PE), histogram (Pmax - PE) / (1 - PE), location (P0 - PE) / (Pmax - PE)."""
    x, y = reference[pixels], simulated[pixels]
    classes = np.union1d(np.unique(x), np.unique(y))
    p0 = float((x == y).mean())
    rows = np.array([(x == k).mean() for k in classes])
    cols = np.array([(y == k).mean() for k in classes])
    pe, pmax = float((rows * cols).sum()), float(np.minimum(rows, cols).sum())
    return {"Correctness %": 100 * p0, "Kappa (overall)": (p0 - pe) / (1 - pe),
            "Kappa (histogram)": (pmax - pe) / (1 - pe), "Kappa (location)": (p0 - pe) / (pmax - pe)}


def all_metrics(start, observed, simulated, guinea, start_year=2024, random_draws=3):
    """MOLUSCE's metrics and ours for one forecast, on the pixels MOLUSCE scores
    (valid in the start, reference and forecast maps, inside Guinea)."""
    pixels = guinea & E.valid_classes(start, observed, simulated)
    kinds = E.change_kind_scores(start_year, observed, simulated, guinea)
    floor = np.mean([E.change_kind_scores(start_year, observed, E.random_like(start, simulated, pixels, seed=s), guinea)["new"]["FoM"]
                     for s in range(random_draws)]) if (pixels & (simulated != start)).any() else np.nan
    return {"Changes forecast": int((pixels & (simulated != start)).sum()),
            **molusce_metrics(observed, simulated, pixels),
            "All-change FoM %": kinds["FoM"], "New-change FoM %": kinds["new"]["FoM"],
            "× random (new change)": kinds["new"]["FoM"] / floor if floor else np.nan}


def standard_metrics(start, observed, simulated, pixels):
    """The validation set the literature recommends for a land-change forecast, from the
    three maps (start, observed end, forecast end) on `pixels`:

    * hits, misses, wrong hits, false alarms (% of the area), Figure of Merit, producer's
      and user's accuracy (Pontius et al. 2008)
    * quantity and allocation disagreement (% of the area; Pontius & Millones 2011)
    * overall accuracy and kappa, only so other tools' reports can be compared
    """
    s, o, f = start[pixels], observed[pixels], simulated[pixels]
    obs, sim = o != s, f != s
    hits = int((obs & sim & (f == o)).sum())
    wrong = int((obs & sim & (f != o)).sum())
    misses = int((obs & ~sim).sum())
    false = int((~obs & sim).sum())
    n = len(s)
    classes = np.union1d(np.unique(o), np.unique(f))
    p_obs = np.array([(o == k).mean() for k in classes])
    p_sim = np.array([(f == k).mean() for k in classes])
    agree = float((o == f).mean())
    quantity = float(np.abs(p_obs - p_sim).sum() / 2)
    pe = float((p_obs * p_sim).sum())
    pct = lambda x: 100 * x / n
    share = lambda part, whole: 100 * part / whole if whole else np.nan
    return {"Changes observed": int(obs.sum()), "Changes forecast": int(sim.sum()),
            "Hits %": pct(hits), "Misses %": pct(misses), "Wrong hits %": pct(wrong), "False alarms %": pct(false),
            "Figure of Merit %": share(hits, hits + misses + wrong + false),
            "Producer's accuracy %": share(hits, hits + misses + wrong),
            "User's accuracy %": share(hits, hits + wrong + false),
            "Quantity disagreement %": 100 * quantity, "Allocation disagreement %": 100 * (1 - agree - quantity),
            "Overall accuracy %": 100 * agree, "Kappa": (agree - pe) / (1 - pe)}


def all_change_floor(start, observed, simulated, pixels, draws=3):
    """Figure of Merit of random placement with the forecast's own amount of each
    transition (Pontius et al. 2004), averaged over `draws`."""
    if not (pixels & (simulated != start)).any():
        return np.nan
    return float(np.mean([standard_metrics(start, observed, E.random_like(start, simulated, pixels, seed=d), pixels)["Figure of Merit %"]
                          for d in range(draws)]))


fom_by_resolution = E.fom_by_resolution


def tool_row(start, observed, simulated, guinea, draws=3):
    """One column of the comparison table: standard_metrics, × random placement of the
    forecast's own amount of each transition, and the Figure of Merit at 1 and 5 km."""
    pixels = guinea & E.valid_classes(start, observed, simulated)
    out = standard_metrics(start, observed, simulated, pixels)
    floor = all_change_floor(start, observed, simulated, pixels, draws)
    res = E.fom_by_resolution(start, observed, simulated, guinea, (4, 20))
    out.update({"× random": out["Figure of Merit %"] / floor if floor else np.nan,
                "Figure of Merit % at 1 km": res[4], "Figure of Merit % at 5 km": res[20]})
    return out


def auc_by_transition(atlas, start, observed, pixels):
    """Area under the ROC curve of each transition's suitability map: among pixels of the
    from-class, how well the map ranks those that really changed to the to-class above the
    rest. It does not depend on the amount of change (Pontius & Si 2014: the TOC's AUC).
    Returns {(f, t): (AUC, observed changes)}."""
    from scipy.stats import rankdata
    out = {}
    for (f, t), path in atlas.items():
        surface = H.read_tif(path)[0]
        sel = pixels & (start == f) & (surface >= 0)
        y, v = observed[sel] == t, surface[sel]
        n1 = int(y.sum()); n0 = y.size - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(v)
        out[(f, t)] = (float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), n1)
    return out


def weighted_auc(aucs):
    w = np.array([n for _, n in aucs.values()], float)
    return float(np.dot([a for a, _ in aucs.values()], w) / w.sum())


def band_files(path, out_dir):
    """{esri_class: path} with one file per band of a multi-band probability raster whose band k
    is class k (FLUS's Probability-of-occurrence.tif), for class_probability_atlas."""
    import rasterio
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    with rasterio.open(path) as src:
        prof = {**src.profile, "count": 1}
        for k in range(1, src.count + 1):
            files[TO_ESRI[k]] = out_dir / f"band_{k}.tif"
            with rasterio.open(files[TO_ESRI[k]], "w", **prof) as dst:
                dst.write(src.read(k), 1)
    return files


def class_probability_atlas(files, transitions, start, guinea, out_dir, profile, name, scale=1.0, nodata=None):
    """Deployed-allocator surfaces from one probability map per destination class
    (PLUS's LEAS, CLUE-S's regressions): for f->t, class t's probability on class-f pixels.
    files: {esri_class: path}."""
    atlas = {}
    for f, t in transitions:
        if t not in files:
            continue
        p = H.read_tif(files[t])[0].astype(np.float32)
        ok = (start == f) & guinea & (p >= 0) & ((p != nodata) if nodata is not None else True)
        atlas[(f, t)] = E.write_float(np.where(ok, p / scale, -1.0), out_dir / "surfaces" / f"{name}_{f}_to_{t}.tif", profile)
    return atlas


def place_observed_amount(atlas, tag):
    """The observed amount of each of the deployed method's 2024->2025 transitions, placed
    by the deployed allocator with the given surfaces. Returns the forecast map."""
    profile = E.grid()
    guinea = H.guinea_mask(profile)
    b, c = E.esri_cached(2024), E.esri_cached(2025)
    _, maps = E.allocate_and_score(b, c, {"Potentials": atlas}, guinea, E.RUNS / tag / "test" / "allocation", profile)
    return maps["Potentials"]


def potentials_atlas(folder, method, transitions, start, guinea, out_dir, profile):
    """Deployed-allocator surfaces from MOLUSCE transition potential maps
    (potential_<method>__from_<i>_0_to_<j>_0.tif, 0-255, 255 = nodata), for the given
    ESRI (from, to) transitions: potential where the pixel is of the from-class, -1 elsewhere."""
    files = {}
    for p in folder.glob(f"potential_{method}__from_*_to_*.tif"):
        i, j = map(int, re.findall(r"from_(\d+)_0_to_(\d+)_0", p.name)[0])
        files[(TO_ESRI[i], TO_ESRI[j])] = p
    atlas = {}
    for f, t in transitions:
        if (f, t) not in files:
            continue
        pot = H.read_tif(files[(f, t)])[0].astype(np.float32)
        surface = np.where((start == f) & guinea & (pot != 255), pot / 254.0, -1.0).astype(np.float32)
        atlas[(f, t)] = E.write_float(surface, out_dir / "surfaces" / f"{method}_{f}_to_{t}.tif", profile)
    return atlas


def like_for_like(folder, method, tag):
    """Observed amount of each of the deployed method's transitions, placed with the tool's
    potentials by the deployed allocator (as every row of notebooks 06-08). Returns the map."""
    profile = E.grid()
    guinea = H.guinea_mask(profile)
    b, c = E.esri_cached(2024), E.esri_cached(2025)
    out = E.RUNS / tag / "test"
    transitions = [tuple(x) for x in E.load_result("R0c_guinea_dated", "test")["transitions"]]
    atlas = potentials_atlas(folder, method, transitions, b, guinea, out, profile)
    _, maps = E.allocate_and_score(b, c, {"Potentials": atlas}, guinea, out / "allocation", profile)
    return maps["Potentials"], sorted(set(transitions) - set(atlas))
