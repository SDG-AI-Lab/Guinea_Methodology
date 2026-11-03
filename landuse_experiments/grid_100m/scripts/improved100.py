"""Notebook 06: the improved method (notebook 05's blend) against the deployed backend (main) at 100 m,
on the four windows; the flip-back check and the weight check. Runs: run04.py, run05*.py, run06.py."""

import json

import numpy as np
import pandas as pd

import decisions100 as M

D = M.D
C, G = D.C, D.G
WINDOWS = ["backtest", "validation", "test", "fresh"]               # fresh: the window no choice used
LABELS = {"backend": "Deployed (backend, main)", "blend": "Improved (blend)"}


def scores(methods=("backend", "blend", "blendw25", "blendw75")):
    """Every finished score of `methods` as it runs: one row per method, window, seed and amount."""
    rows = []
    for method in methods:
        for phase in WINDOWS:
            for seed in M.SEEDS:
                for demand in ("observed", "projected"):
                    f = D._alloc_dir(D.tag(phase, seed, method), "as it runs", demand) / "score.json"
                    if f.exists():
                        rows.append({"method": method, "phase": phase, "seed": seed, "demand": demand,
                                     **json.loads(f.read_text())})
    return pd.DataFrame(rows)


def returns(phase, method, seed=42, demand="observed"):
    """A forecast's hits split by destination: back to a class the cell held in a map since 2017
    before the base year (a return, which flicker makes easy), or into a class new to the cell."""
    import rasterio
    a, b, c = M.PHASES[phase]
    path = D._alloc_dir(D.tag(phase, seed, method), "as it runs", demand) / "predicted_land_cover.tif"
    with rasterio.open(path) as src:
        pred = G.from_grid(src.read(1))
    start, obs = C.class_100(b), C.class_100(c)
    ok = ~np.isin(start, M.INVALID) & ~np.isin(obs, M.INVALID) & ~np.isin(pred, M.INVALID)
    hit = ok & (obs != start) & (pred == obs)
    held = np.zeros(start.size, dtype=bool)
    for y in range(C.FIRST_YEAR, b):
        held |= C.class_100(y) == obs
    return {"hits": int(hit.sum()), "returns": int((hit & held).sum()), "new": int((hit & ~held).sum())}
