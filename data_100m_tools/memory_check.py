"""Time and peak memory of the heavy steps of a 100 m run, on this machine, with dummy data.

Each test runs in a fresh Python process, so its peak memory (Windows: peak working set) is
its own. Forest cells: 9 million (Hansen at 30 % canopy covers about 82,000 km² of Guinea in
2000, about 8.3 million 100 m cells, before loss).

  read_layer     one uint8 column of the whole country from parquet, and its 2D grid (float32)
  distance       distance to the nearest of a sparse set of cells (roads, built-up), full grid
  moving_window  share of a mask within 51 x 51 cells (5 km), full grid (local loss rate)
  stack_predict  16 predictors for the forest cells only, random forest fitted on 10,000
                 samples (deployed settings), scored in chunks of 1 million cells
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FOREST_CELLS = 9_000_000
N_PREDICTORS = 16


def _peak_gb():
    import psutil
    info = psutil.Process().memory_info()
    return getattr(info, "peak_wset", getattr(info, "rss", 0)) / 1e9


def _test(name, workdir):
    import grid100 as G
    g = G.grid()
    shape = (g["height"], g["width"])
    rng = np.random.default_rng(0)
    if name == "read_layer":
        path = Path(workdir) / "dummy_layer.parquet"
        G._write_parquet(path, {"2020": rng.integers(0, 101, G.cells().size, dtype="uint8")})
        import pyarrow.parquet as pq
        start = time.perf_counter()
        values = pq.read_table(path).column("2020").to_numpy()
        G.to_grid(values.astype("float32"))
        return {"seconds": time.perf_counter() - start, "file_mb": path.stat().st_size / 1e6}
    if name == "distance":
        from scipy.ndimage import distance_transform_edt
        mask = np.ones(shape, dtype=bool)
        mask.reshape(-1)[rng.choice(mask.size, 200_000, replace=False)] = False      # the target cells
        start = time.perf_counter()
        distance = distance_transform_edt(mask).astype("float32")
        return {"seconds": time.perf_counter() - start, "max_cells": float(distance.max())}
    if name == "moving_window":
        from scipy.ndimage import uniform_filter
        mask = (rng.random(shape, dtype="float32") < 0.01).astype("float32")
        start = time.perf_counter()
        share = uniform_filter(mask, 51)
        return {"seconds": time.perf_counter() - start, "mean": float(share.mean())}
    if name == "stack_predict":
        from sklearn.ensemble import RandomForestClassifier
        X = rng.random((FOREST_CELLS, N_PREDICTORS), dtype="float32")
        y = (X[:, 0] + 0.3 * rng.random(FOREST_CELLS, dtype="float32") > 0.9).astype(int)
        pick = np.concatenate([rng.choice(np.flatnonzero(y == 1), 5000, replace=False),
                               rng.choice(np.flatnonzero(y == 0), 5000, replace=False)])
        model = RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=5, n_jobs=-1,
                                       random_state=0, class_weight="balanced").fit(X[pick], y[pick])
        start = time.perf_counter()
        risk = np.empty(FOREST_CELLS, dtype="float32")
        for i in range(0, FOREST_CELLS, 1_000_000):
            risk[i:i + 1_000_000] = model.predict_proba(X[i:i + 1_000_000])[:, 1]
        return {"seconds": time.perf_counter() - start, "stack_gb": X.nbytes / 1e9}
    raise ValueError(name)


def run(name, workdir):
    """Run one test in a fresh process -> {"test", "seconds", "peak_gb", ...}."""
    code = ("import sys, json; sys.path.insert(0, {here!r}); import memory_check as M; "
            "out = M._test({name!r}, {workdir!r}); out['peak_gb'] = M._peak_gb(); print('RESULT ' + json.dumps(out))")
    proc = subprocess.run([sys.executable, "-W", "ignore", "-c", code.format(here=str(HERE), name=name, workdir=str(workdir))],
                          capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(f"{name} failed:\n{proc.stderr[-3000:]}")
    return {"test": name, **json.loads(lines[-1][7:])}


TESTS = ["read_layer", "distance", "moving_window", "stack_predict"]
