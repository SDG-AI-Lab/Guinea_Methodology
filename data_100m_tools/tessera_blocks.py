"""Sample blocks for the TESSERA tests of the deforestation and land-use modules.

TESSERA's yearly embeddings (Feng et al. 2026, CVPR) are about 300-370 GB per year for Guinea,
so the tests use sample blocks: TESSERA's own 0.1 degree tiles (about 11 x 11 km, 12,000 cells
of 100 m), wholly inside Guinea.

  forest blocks (16)  the deforestation test. Tiles with at least 30 % forest in 2000, shared
                      among the forest ecoregions in proportion to their 2018-2024 loss (at least
                      one each), drawn within an ecoregion with probability proportional to the
                      tile's loss, so each block has losses to learn and test on.
  land blocks (8)     added for the land-use test, which uses all 24: drawn at random among the
                      remaining tiles, shared among all ecoregions in proportion to their area.

Seed 0. Output in STORE/tessera/: blocks.json (the tiles and why each was drawn) and
blocks_cells.npz (for each block: positions in grid100.cells(), cell centres).

Run with landuse_experiments/.venv:  python data_100m_tools/tessera_blocks.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid100 as G                                             # noqa: E402

OUT = G.STORE / "tessera"
N_FOREST, N_LAND, SEED = 16, 8, 0
ECO = {1: "Guinean forest-savanna", 2: "Guinean mangroves", 3: "Guinean montane forests",
       4: "West Sudanian savanna", 5: "Western Guinean lowland forests"}


def _shares(counts, n):
    """Whole numbers summing to n, in proportion to counts, at least 1 where counts > 0."""
    k = {c: 1 for c, v in counts.items() if v > 0}
    rest = n - sum(k.values())
    w = np.array([counts[c] for c in k], float)
    extra = np.floor(rest * w / w.sum()).astype(int)
    order = np.argsort(-(rest * w / w.sum() - extra))
    extra[order[: rest - extra.sum()]] += 1
    return {c: k[c] + int(e) for c, e in zip(k, extra)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lon, lat = G.lon_lat()
    tx, ty = np.floor(lon * 10).astype(int), np.floor(lat * 10).astype(int)   # tile = 0.1 degree, west/south edge x 10
    area = G.cell_area_ha()
    h = G.read_layer("hansen_shares", ["forest2000"] + [f"loss_{y}" for y in range(2018, 2025)], scaled=True)
    forest = h["forest2000"] * area
    loss = sum(h[f"loss_{y}"] for y in range(2018, 2025)) * area
    eco = G.read_layer("ecoregions")["ecoregion"]

    keys, inv, n_cells = np.unique(np.stack([tx, ty], 1), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    t_forest = np.bincount(inv, forest); t_loss = np.bincount(inv, loss); t_area = np.bincount(inv, area)
    t_eco = np.bincount(inv * 6 + eco.astype(int), minlength=len(keys) * 6).reshape(-1, 6).argmax(1)
    full = n_cells >= 0.95 * np.median(n_cells[n_cells > 0.5 * n_cells.max()])   # wholly inside Guinea
    rng = np.random.default_rng(SEED)

    chosen = []
    forest_ok = full & (t_forest >= 0.30 * t_area) & (t_loss > 0)
    eco_loss = {c: float(t_loss[forest_ok & (t_eco == c)].sum()) for c in ECO}
    for c, k in _shares(eco_loss, N_FOREST).items():
        pool = np.flatnonzero(forest_ok & (t_eco == c))
        p = t_loss[pool] / t_loss[pool].sum()
        for i in rng.choice(pool, min(k, pool.size), replace=False, p=p):
            chosen.append((int(i), "forest", c))
    taken = {i for i, _, _ in chosen}
    eco_area = {c: float(t_area[full & (t_eco == c)].sum()) for c in ECO}
    for c, k in _shares(eco_area, N_LAND).items():
        pool = np.array([i for i in np.flatnonzero(full & (t_eco == c)) if i not in taken])
        for i in rng.choice(pool, min(k, pool.size), replace=False):
            chosen.append((int(i), "land", c))

    blocks, arrays = [], {}
    for i, kind, c in chosen:
        w, s = keys[i][0] / 10, keys[i][1] / 10
        name = f"{w:.1f}_{s:.1f}"
        pos = np.flatnonzero(inv == i)
        arrays[f"{name}_pos"] = pos.astype("int32")
        arrays[f"{name}_lon"], arrays[f"{name}_lat"] = lon[pos], lat[pos]
        blocks.append({"name": name, "bbox": [w, s, round(w + 0.1, 1), round(s + 0.1, 1)], "kind": kind,
                       "ecoregion": ECO[c], "cells": int(pos.size),
                       "forest2000_pct": round(100 * t_forest[i] / t_area[i], 1),
                       "loss_2018_2024_ha": round(float(t_loss[i]), 1)})
    np.savez(OUT / "blocks_cells.npz", **arrays)
    (OUT / "blocks.json").write_text(json.dumps({"seed": SEED, "blocks": blocks}, indent=1))
    for b in blocks:
        print(b)


if __name__ == "__main__":
    main()
