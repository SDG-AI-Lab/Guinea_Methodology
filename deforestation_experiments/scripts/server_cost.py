"""Server cost of each change: peak memory and run time of its step, on the full 100 m grid.

Each step runs alone in a fresh Python process, on the latest window (2019-2021 -> 2022-2024), and
reports the process's peak memory (inputs included) and its wall time. The steps:

  load            the Hansen shares over the cells with forest in 2000 (every step starts with it)
  deployed        the deployed model: its predictors, half-cell rule, train, score the half-forest cells
  improved        the improved model: its predictors, trained on the shares, every forest cell scored
  rank_average    riskmapjnr's local loss rate (5 km) and distance threshold, and the rank average
                  with a risk map
  verra           Verra's fit and prediction tests: the benchmark and one model
  areas           forest and loss per prefecture from the shares, border cells given to the nearest unit

The code is the experiments' (values over the forest cells only, scored a million cells at a time),
not the platform's, which holds whole 2D grids: compare a step with "deployed", not with the
platform's own peak.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as MM                                              # noqa: E402

SPLIT = MM.SPLITS[-1]
STEPS = ["load", "deployed", "improved", "rank_average", "verra", "areas"]


def _model(config, rule):
    t1, t2, t3 = SPLIT
    family, names = MM._family(rule), MM.features(config)
    cal = MM.stack(t1, config, (family,))
    X, y = MM.sample(rule, t1, t2, cal[family], 42, names)
    model, _ = MM.M.DR.fit_risk_model(X, y, seed=42)
    del cal, X, y
    test = MM.stack(t2, config, (family,))
    f2 = MM.forest_pct(t2)
    rows = np.flatnonzero(f2 >= 50 if rule == "half" else f2 > 0)
    return MM.score(model, test[family], rows, names=names)


def step(name):
    t1, t2, t3 = SPLIT
    MM.hansen()
    if name == "deployed":
        _model(MM.DEPLOYED, "half")
    elif name == "improved":
        _model(MM.IMPROVED, "weighted")
    elif name == "rank_average":
        import tool_comparison as T
        f2 = MM.forest_pct(t2)
        area = MM.hansen()["area"]
        rate = T.rmj_local_rate(t2 - 3, t2, T.RMJ_WINDOWS[-1])
        distance = T.rmj_distance(t2)
        thresh = T.rmj_threshold(distance[f2 > 0], (MM.loss_pct(t1, t2) / 100 * area)[f2 > 0])
        rmj = np.where(f2 > 0, T.rmj_rescale(np.nan_to_num(rate)), 0).astype("float32")
        rmj[(f2 > 0) & (distance >= thresh)] = 1
        risk = np.random.default_rng(0).random(rmj.size, dtype="float32")   # stands for the improved model's map
        (T._rank(risk, f2 > 0) + T._rank(rmj, f2 > 0)) / 2
    elif name == "verra":
        import verra as V
        V.vt0007_tests(SPLIT)
        rng = np.random.default_rng(0)
        n = MM.hansen()["idx"].size
        V.vt0007_tests(SPLIT, rng.random(n, dtype="float32"), rng.random(n, dtype="float32"))
    elif name == "areas":
        import reporting as R
        unit = MM.G.from_grid(R.nearest_grid("ADM2"))
        area = MM.G.cell_area_ha().astype("float64")
        for band in R.BANDS:
            np.bincount(unit, weights=MM.G.read_layer("hansen_shares", [band])[band] * 0.01 * area)


def run_all(log=print):
    out = {"split": MM.M.split_label(SPLIT), "steps": {}}
    for name in STEPS:
        done = subprocess.run([sys.executable, __file__, "--step", name], capture_output=True, text=True)
        out["steps"][name] = json.loads(done.stdout.strip().splitlines()[-1])
        log(name, out["steps"][name], flush=True)
    return out


if __name__ == "__main__":
    if "--step" in sys.argv:
        name = sys.argv[sys.argv.index("--step") + 1]
        start = time.perf_counter()
        step(name)
        print(json.dumps({"seconds": round(time.perf_counter() - start, 1), "peak_gb": round(MM._peak_gb(), 2)}))
    else:
        MM.cached("server_cost", run_all, rerun="--rerun" in sys.argv)
        print("SAVED server_cost", flush=True)
