"""Notebook 04: the deployed method as the platform's backend runs it (main), the backend's growth-mode
option, and the methodology document's method, on the three windows at 100 m; fixes to the backend
tested on the same suitability maps.

  backend          platform100.run: one Random Forest per transition with a count above zero, the 12
                   predictors, allocation by demand order (main's run_training / run_prediction)
  backend, growth  the same with use_growth_modes=True (an API option): Expander and Patcher models,
                   combined by dilation in the allocation
  document         Methodology_LandUse_Guinea.docx (§3.3-3.4, §6.1, §7.1): Expander-only, a neighbourhood
                   density layer (share of the destination class within 5 cells) among the predictors
                   (Verburg et al. 2002), transitions with at least 1,000 changed pixels, clouds left out.
                   Built from main's own functions (create_transition_dataset in 'expander' mode,
                   train_rf_model, _generate_single_suitability_map, run_allocation_simulation), so it
                   differs from the backend only in those documented choices. The neighbourhood is read
                   from the start map of each interval (the deployed notebook read the end map, which
                   leaks the answer; 250 m notebook 06).

Fixes on the backend's maps: clouds left out (Esri's legend: cloud is no data), the 1,000-pixel rule,
both. Demand: as the platform projects it (change analysis 2017 to the base year), and the observed
amount (to compare where the change is placed).
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import platform100 as P100                                          # noqa: E402  (sets up landuse_tool)
import classes100 as C                                              # noqa: E402

P, G, H = P100.P, C.G, C.H
PHASES = {"backtest": (2021, 2022, 2023), "validation": (2022, 2023, 2024), "test": (2023, 2024, 2025),
          "fresh": (2020, 2021, 2022)}                               # notebook 05's last check, run once
SEEDS = (42, 1, 2)
INVALID = (0, 10)
MIN_PIXELS = 1000
RADIUS = 5                                                          # the document's neighbourhood, in cells
JOBS = ([("backend", ph, s) for ph in ("test", "validation", "backtest") for s in SEEDS]
        + [("growth", ph, 42) for ph in ("test", "validation", "backtest")]
        + [("document", ph, 42) for ph in ("test", "validation", "backtest")])
FIXES = ("no clouds", "1,000-pixel rule", "both")
CLEAN = ("document", "document3x3", "lr", "blend",                 # our methods: clouds left out (notebooks 04-05)
         "contagion", "flusnet", "blendc", "blend3", "blend3c", "blendw25", "blendw75")


def variants(method, seed):
    """The allocations scored for a run: as it runs, and the fixes on the backend's seed-42 run."""
    return ["as it runs"] + (list(FIXES) if method == "backend" and seed == 42 else [])


def _alloc_dir(run_tag, variant, demand_kind):
    return P100.run_dir(run_tag) / "alloc" / f"{variant.replace(',', '').replace(' ', '_')}_{demand_kind}"


def tag(phase, seed, method="backend"):
    return {"backend": f"{phase}_s{seed}", "growth": f"{phase}_growth_s{seed}"}.get(method, f"{phase}_{method}_s{seed}")


def run_backend(phase, seed, growth=False, log=print):
    a, b, c = PHASES[phase]
    return P100.run([a, b], b, c, seed, tag(phase, seed, "growth" if growth else "backend"), growth_modes=growth, log=log)


# --------------------------------------------------------------------------
# The document's method

def _density_file(year, to_class, kind):
    """Share of `to_class` within RADIUS cells of each cell, on `year`'s map (NaN outside Guinea),
    written as the platform writes predictors; one folder per year and class so that the file has
    the same name in training and prediction."""
    import rasterio
    from scipy.ndimage import correlate
    path = P.INPUTS / "neighbourhood" / f"{kind}_{year}" / str(to_class) / "NEIGHBOURHOOD_DENSITY.tif"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        cls = G.to_grid(C.class_100(year), fill=0, dtype="uint8")
        y, x = np.ogrid[-RADIUS:RADIUS + 1, -RADIUS:RADIUS + 1]
        kernel = (x ** 2 + y ** 2 <= RADIUS ** 2).astype(np.float32)
        dens = correlate((cls == to_class).astype(np.float32), kernel, mode="reflect") / kernel.sum()
        dens[~G.inside_mask()] = np.nan
        with rasterio.open(path, "w", **G.profile("float32", np.nan)) as dst:
            dst.write(dens.astype("float32"), 1)
    return str(path)


def significant(a, b, min_pixels=MIN_PIXELS):
    """Transitions with at least min_pixels changed cells between the maps of years a and b (no
    clouds or no data), the deployed notebook's select_significant_transitions."""
    s, e = C.class_100(a), C.class_100(b)
    ok = ~np.isin(s, INVALID) & ~np.isin(e, INVALID) & (s != e)
    key, n = np.unique(s[ok].astype("int32") * 100 + e[ok], return_counts=True)
    return [(int(k // 100), int(k % 100)) for k, m in zip(key, n) if m >= min_pixels]


def run_document(phase, seed, log=print, method="document", extra=None):
    """The document's method; extra(year, to_class) -> more predictor files (notebook 05)."""
    P100.tool()
    from landuse_tool import change_analysis, prediction, training
    a, b, c = PHASES[phase]
    out = P100.run_dir(tag(phase, seed, method))
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    esri = {y: P.write_esri(y) for y in sorted(set(range(P100.FIRST_YEAR, c + 1)))}
    pred_train, pred_base = P.write_predictors(a), P.write_predictors(b)
    transitions = significant(a, b)
    np.random.seed(seed)
    surfaces, stages = {}, []
    with P100.Peak() as peak:
        for f, t in transitions:
            files_train = pred_train + [_density_file(a, t, "start")] + (extra(a, t) if extra else [])
            ds = training.create_transition_dataset(f, t, esri[a], esri[b], files_train, mode="expander",
                                                    missing_value_handling="exclude")
            if ds[0] is None:
                continue
            X, y, blocks = ds
            schema = [Path(p).name for p in files_train]
            model_path = str(out / "models" / f"model_expander_{f}_{t}.joblib")
            acc, _ = training.train_rf_model(X, y, blocks, model_path, schema)
            if not acc:
                continue
            files_base = pred_base + [_density_file(b, t, "start")] + (extra(b, t) if extra else [])
            surfaces[(f, t)] = prediction._generate_single_suitability_map(
                model_path, files_base, esri[b], str(out), schema, model_type="expander", missing_value_handling="mean")
            if surfaces[(f, t)] is None:
                raise RuntimeError(f"no suitability map for {f}->{t}; see the log")
            log(f"[{out.name}] {f}->{t} done")
    stages.append({"stage": "Train and score (Expander with neighbourhood)", "seconds": peak.seconds, "peak_GB": peak.peak / 1e9})
    history = [{"file": esri[y], "year": y} for y in range(P100.FIRST_YEAR, b + 1)]
    _, demand, _, _ = change_analysis.run_change_analysis(history)
    (demand * (c - b)).to_csv(out / "demand.csv")
    result = {"tag": out.name, "train": [a, b], "base": b, "target": c, "seed": seed, "method": method,
              "transitions": [list(t) for t in transitions], "models": [list(k) for k in surfaces],
              "stages": stages, "platform_commit": (HERE / "data" / "platform_main" / "COMMIT").read_text().strip()}
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    return result


# --------------------------------------------------------------------------
# Allocation variants and scores

def surfaces(run_tag):
    """{key: path} of a run's suitability maps; keys (f, t) or (f, t, mode), as the platform's atlas."""
    out = {}
    for p in P100.run_dir(run_tag).glob("suitability_*.tif"):
        parts = p.stem.split("_")                                    # suitability[_mode]_f_to_t
        f, t = int(parts[-3]), int(parts[-1])
        mode = parts[1] if len(parts) == 5 else None
        if run_tag.split("_")[1] in CLEAN:
            mode = None                                              # Expander-only: used as the standard map
        out[(f, t) if mode is None else (f, t, mode)] = str(p)
    return out


@lru_cache(maxsize=None)
def observed_demand(base, target):
    """The observed transition counts between the two maps inside Guinea, valid classes only."""
    s, e = C.class_100(base), C.class_100(target)
    ok = ~np.isin(s, INVALID) & ~np.isin(e, INVALID)
    t = pd.crosstab(s[ok], e[ok])
    t.index, t.columns = t.index.astype(int), t.columns.astype(int)
    return t


def projected_demand(run_tag):
    d = pd.read_csv(P100.run_dir(run_tag) / "demand.csv", index_col=0)
    d.columns = d.columns.astype(int)
    d.index = d.index.astype(int)
    return d


def variant_inputs(run_tag, phase, variant, demand_kind):
    a, b, c = PHASES[phase]
    atlas = surfaces(run_tag)
    demand = observed_demand(b, c).copy() if demand_kind == "observed" else projected_demand(run_tag)
    if variant in ("no clouds", "both") or run_tag.split("_")[1] in CLEAN:
        atlas = {k: v for k, v in atlas.items() if 10 not in k[:2]}
        keep = [k for k in demand.index if k not in INVALID]
        demand = demand.loc[keep, [k for k in demand.columns if k not in INVALID]]
    if variant in ("1,000-pixel rule", "both"):
        sig = set(significant(a, b))
        atlas = {k: v for k, v in atlas.items() if k[:2] in sig}
    return atlas, demand


def allocate(run_tag, phase, variant="as it runs", demand_kind="projected"):
    """Path of the allocated map (cached): main's run_allocation_simulation, deterministic."""
    P100.tool()
    from landuse_tool import prediction
    out = _alloc_dir(run_tag, variant, demand_kind)
    path = out / "predicted_land_cover.tif"
    if not path.exists():
        out.mkdir(parents=True, exist_ok=True)
        atlas, demand = variant_inputs(run_tag, phase, variant, demand_kind)
        report = {}
        prediction.run_allocation_simulation(P.write_esri(PHASES[phase][1]), demand, atlas, str(out),
                                             stochastic=False, allocation_report=report)
        (out / "allocation.json").write_text(json.dumps(report, indent=1, default=float))
    return path


def _random_like(start, predicted, ok, seed=0):
    """The same predicted transitions placed at random among the cells of each start class."""
    rng = np.random.default_rng(seed)
    out = start.copy()
    change = ok & (predicted != start)
    for f in np.unique(start[change]):
        pool = np.flatnonzero(ok & (start == f))
        dest = predicted[change & (start == f)]
        pick = rng.choice(pool, dest.size, replace=False)
        out[pick] = dest
    return out


def score(phase, path):
    """Figure of Merit over all change and over new change (destination not held in any map since
    2017 before the base year), inside Guinea, valid classes; random placement of the same change;
    and main's own figure_of_merit as the platform reports it (clouds counted, Guinea mask)."""
    import rasterio
    with rasterio.open(path) as src:
        return score_map(phase, G.from_grid(src.read(1)))


def score_map(phase, pred):
    """score() of a forecast given as Esri classes per cell inside Guinea (G.from_grid)."""
    P100.tool()
    from landuse_tool import validation
    a, b, c = PHASES[phase]
    start, obs = C.class_100(b), C.class_100(c)
    ok = ~np.isin(start, INVALID) & ~np.isin(obs, INVALID) & ~np.isin(pred, INVALID)
    held = np.zeros(start.size, dtype=bool)

    def fom(p, held_mask=None):
        o, s = ok & (obs != start), ok & (p != start)
        hit = o & s & (p == obs)
        if held_mask is None:
            return 100 * hit.sum() / (o | s).sum()
        o_new = o & ~held_mask(obs)
        s_new = s & ~held_mask(p)
        return 100 * (hit & o_new).sum() / (o_new | s_new).sum()

    past = [C.class_100(y) for y in range(C.FIRST_YEAR, b)]
    held_mask = lambda dest: np.logical_or.reduce([m == dest for m in past])
    full = validation.figure_of_merit(start, obs, pred)              # all cells inside Guinea, as main
    return {"FoM": fom(pred), "new-change FoM": fom(pred, held_mask), "random FoM": fom(_random_like(start, pred, ok)),
            "predicted change (cells)": int((ok & (pred != start)).sum()), "observed change (cells)": int((ok & (obs != start)).sum()),
            "platform FoM": full["figure_of_merit_pct"]}


def table(rows):
    return pd.DataFrame(rows).set_index(["method", "variant", "demand", "phase", "seed"]).sort_index()


# --------------------------------------------------------------------------
# Notebook 04

METHODS = {"backend": "Backend (main)", "growth": "Backend, growth modes", "document": "Methodology document"}
OUTCOMES = {1: "Hit", 2: "Wrong class", 3: "Miss", 4: "False alarm"}


def scores():
    """Every score of run04.py: one row per method, fix, demand, window and seed."""
    rows = []
    for method, phase, seed in JOBS:
        for v in variants(method, seed):
            for d in ("projected", "observed"):
                s = json.loads((_alloc_dir(tag(phase, seed, method), v, d) / "score.json").read_text())
                rows.append({"method": method, "variant": v, "demand": d, "phase": phase, "seed": seed, **s})
    return pd.DataFrame(rows)


def stage_costs():
    """Minutes and peak memory (GB) of each platform stage in the seed-42 runs. The document method
    times training and scoring together; its demand and allocation are main's, as in the backend."""
    rows = []
    for method in METHODS:
        for phase in ("backtest", "validation", "test"):
            r = json.loads((P100.run_dir(tag(phase, 42, method)) / "result.json").read_text())
            for s in r["stages"]:
                if not s["stage"].startswith("Write inputs"):
                    rows.append({"method": method, "phase": phase, "stage": s["stage"].split(" (")[0],
                                 "minutes": s["seconds"] / 60, "peak_GB": s["peak_GB"]})
    return pd.DataFrame(rows)


def outcome(run_tag, phase, demand_kind="observed", variant="as it runs"):
    """Per cell inside Guinea (Pontius et al. 2008): 1 hit, 2 changed to the wrong class, 3 miss,
    4 false alarm, 0 persistence correctly forecast or no data."""
    import rasterio
    a, b, c = PHASES[phase]
    with rasterio.open(_alloc_dir(run_tag, variant, demand_kind) / "predicted_land_cover.tif") as src:
        pred = G.from_grid(src.read(1))
    start, obs = C.class_100(b), C.class_100(c)
    ok = ~np.isin(start, INVALID) & ~np.isin(obs, INVALID) & ~np.isin(pred, INVALID)
    o, s = ok & (obs != start), ok & (pred != start)
    out = np.zeros(start.shape, np.uint8)
    out[o & s & (pred == obs)] = 1
    out[o & s & (pred != obs)] = 2
    out[o & ~s] = 3
    out[~o & s] = 4
    return out


def outcome_rgba(codes, colours, alpha=0.9):
    from matplotlib.colors import to_rgba
    img = np.zeros(codes.shape + (4,), dtype="float32")
    for k, colour in colours.items():
        img[codes == k] = to_rgba(colour, alpha)
    return img

