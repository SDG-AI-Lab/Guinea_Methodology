"""Notebook 01 of the land-use work at 100 m: the platform's class against the 10 m shares, and how
much the maps change from year to year at 100 m against 250 m.

Reads only files on this laptop: the shared 100 m store (Esri class and the eight class shares,
2017-2025, data_100m_tools/grid100.py) and the 250 m Esri maps of the parent folder
(data/esri/ESRI_LULC_YYYY.tif). No Earth Engine.

Both grids are cut to the cells whose centre is inside Guinea's ADM0 outline (the platform's
create_boundary_mask), so the two are compared over the same area. The 250 m experiments used
all_touched=True for their own scores; that adds a thin border and does not change a rate.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]                        # landuse_experiments/grid_100m
ROOT = HERE.parent                                                 # landuse_experiments
sys.path.insert(0, str(ROOT.parent / "data_100m_tools"))
sys.path.insert(0, str(ROOT / "scripts"))
import grid100 as G                                                # noqa: E402
import nb_helpers as H                                             # noqa: E402

YEARS = list(range(2017, 2026))
FIRST_YEAR = 2017
CLASSES = [1, 2, 4, 5, 7, 8, 11]                                   # Esri codes, clouds (10) left out
SHARE_COLUMNS = {1: "water", 2: "trees", 4: "flooded_vegetation", 5: "crops", 7: "built_area",
                 8: "bare_ground", 10: "clouds", 11: "rangeland"}
NAMES = {1: "Water", 2: "Trees", 4: "Flooded vegetation", 5: "Crops", 7: "Built area",
         8: "Bare ground", 11: "Rangeland"}
MIN_PIXELS = 1000                                                  # config.MIN_CHANGE_PIXELS_LAND_USE
OUT = HERE / "outputs"


# --------------------------------------------------------------------------
# 100 m

@lru_cache(maxsize=None)
def area_100():
    return G.cell_area_ha()


@lru_cache(maxsize=None)
def class_100(year):
    """The platform's class: the mode of the year's Esri tiles read at 100 m."""
    return G.read_layer(f"esri_{year}", ["class"])["class"]


def shares_100(year):
    """(8, n) uint8 percentages in the order of SHARE_COLUMNS."""
    s = G.read_layer(f"esri_{year}", list(SHARE_COLUMNS.values()))
    return np.stack([s[c] for c in SHARE_COLUMNS.values()])


@lru_cache(maxsize=None)
def majority_100(year):
    """The class holding the largest share of the cell's 10 m pixels (0 where no pixel has data)."""
    s = shares_100(year)
    codes = np.array(list(SHARE_COLUMNS), dtype="uint8")
    out = codes[np.argmax(s, axis=0)]
    out[s.sum(axis=0) == 0] = 0
    return out


# --------------------------------------------------------------------------
# 250 m, on the same area

@lru_cache(maxsize=None)
def profile_250():
    import rasterio
    with rasterio.open(H.DATA / "esri" / "ESRI_LULC_2017.tif") as src:
        return src.profile


@lru_cache(maxsize=None)
def inside_250():
    from rasterio.features import rasterize
    p = profile_250()
    return rasterize([(G.outline(), 1)], out_shape=(p["height"], p["width"]), transform=p["transform"],
                     fill=0, dtype="uint8").astype(bool)


@lru_cache(maxsize=None)
def class_250(year):
    """The 250 m map over the cells inside Guinea (1-D, row-major)."""
    import rasterio
    with rasterio.open(H.DATA / "esri" / f"ESRI_LULC_{year}.tif") as src:
        return src.read(1)[inside_250()]


@lru_cache(maxsize=None)
def area_250():
    p = profile_250()
    t = p["transform"]
    lat = t.f + (np.arange(p["height"]) + 0.5) * t.e
    per_row = (abs(t.a) * G.M_PER_DEG_PLATFORM * np.cos(np.radians(lat))) * (abs(t.e) * G.M_PER_DEG_PLATFORM) / 1e4
    return np.broadcast_to(per_row[:, None], (p["height"], p["width"]))[inside_250()].astype("float32")


@lru_cache(maxsize=None)
def index_250_of_100():
    """For each 100 m cell, the 250 m pixel its centre falls in (index into class_250, -1 outside)."""
    p = profile_250()
    t = p["transform"]
    lon, lat = G.lon_lat()
    col = np.floor((lon - t.c) / t.a).astype("int64")
    row = np.floor((lat - t.f) / t.e).astype("int64")
    lookup = np.full(p["height"] * p["width"], -1, dtype="int64")
    lookup[np.flatnonzero(inside_250())] = np.arange(int(inside_250().sum()))
    ok = (row >= 0) & (row < p["height"]) & (col >= 0) & (col < p["width"])
    out = np.full(lon.size, -1, dtype="int64")
    out[ok] = lookup[row[ok] * p["width"] + col[ok]]
    return out


# --------------------------------------------------------------------------
# Section 1: the platform's class against the shares

def class_against_shares():
    rows = []
    for y in YEARS:
        c, s, m = class_100(y), shares_100(y), majority_100(y)
        valid = ~np.isin(c, H.ESRI_INVALID) & ~np.isin(m, H.ESRI_INVALID)
        codes = list(SHARE_COLUMNS)
        own = np.zeros(c.size, dtype="uint8")                       # share of the cell in its own class
        for k, code in enumerate(codes):
            own[c == code] = s[k][c == code]
        top = s.max(axis=0)
        rows.append({"year": y,
                     "class = majority (%)": 100 * float((c == m)[valid].mean()),
                     "cell in its own class (%, mean)": float(own[valid].mean()),
                     "largest share (%, mean)": float(top[valid].mean()),
                     "cells with no class above half (%)": 100 * float((top[valid] <= 50).mean()),
                     "no data or clouds (cells)": int((~valid).sum())})
        del s
    return pd.DataFrame(rows).set_index("year")


def confusion_against_majority(year):
    """Rows: the platform's class; columns: the majority class; % of the platform's class."""
    c, m = class_100(year), majority_100(year)
    t = pd.crosstab(pd.Series(c, name="platform class"), pd.Series(m, name="majority of the shares"))
    t = t.loc[[k for k in CLASSES if k in t.index], [k for k in CLASSES if k in t.columns]]
    t = 100 * t.div(t.sum(axis=1), axis=0)
    return t.rename(index=NAMES, columns=NAMES)


def class_areas(year):
    """km² of each class: from the shares (the 10 m pixels), and from one class per cell."""
    a100, s = area_100(), shares_100(year)
    out = {}
    for k, code in enumerate(SHARE_COLUMNS):
        if code not in CLASSES:
            continue
        out[code] = {"10 m pixels (shares)": float((s[k] * a100).sum() / 1e4),
                     "100 m, platform class": float(a100[class_100(year) == code].sum() / 100),
                     "100 m, majority of the shares": float(a100[majority_100(year) == code].sum() / 100),
                     "250 m, class": float(area_250()[class_250(year) == code].sum() / 100)}
    t = pd.DataFrame(out).T
    for col in list(t.columns[1:]):
        t[f"{col} vs 10 m (%)"] = 100 * (t[col] / t["10 m pixels (shares)"] - 1)
    return t.rename(index=NAMES)


def agreement_100_250(year):
    """% of 100 m cells whose class equals the 250 m class at the cell's centre (both valid)."""
    i = index_250_of_100()
    c100 = class_100(year)
    c250 = np.where(i >= 0, class_250(year)[np.clip(i, 0, None)], 0)
    ok = ~np.isin(c100, H.ESRI_INVALID) & ~np.isin(c250, H.ESRI_INVALID)
    return 100 * float((c100 == c250)[ok].mean())


# --------------------------------------------------------------------------
# Section 2: change from year to year

def _change_rates(maps, area, years=YEARS):
    """For each pair of consecutive years: % of the area that changes class, and of that change,
    the % into a class the cell had not held in any map since FIRST_YEAR before the start year
    (genuinely new change, as notebook 11 counts it), and the % that goes back to the start class
    the year after (a flip)."""
    rows = []
    for i in range(len(years) - 1):
        a, b = maps(years[i]), maps(years[i + 1])
        valid = ~np.isin(a, H.ESRI_INVALID) & ~np.isin(b, H.ESRI_INVALID)
        changed = valid & (a != b)
        row = {"from": years[i], "to": years[i + 1],
               "changed (% of area)": 100 * float(area[changed].sum() / area[valid].sum())}
        if i > 0:
            held = np.zeros(a.size, dtype=bool)
            for y in years[:i]:
                held |= maps(y) == b
            row["of it, new (%)"] = 100 * float(area[changed & ~held].sum() / area[changed].sum())
        if i + 2 < len(years):
            back = maps(years[i + 2]) == a
            row["of it, back the next year (%)"] = 100 * float(area[changed & back].sum() / area[changed].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def share_change_floor(years=YEARS):
    """Least % of the land whose 10 m pixels changed class: half the sum of the absolute changes
    of the eight shares in each cell (a pixel that changes moves one share down and one up; changes
    that cancel inside a cell are not seen, so this is a floor)."""
    a100 = area_100()
    rows, prev = [], shares_100(years[0]).astype("int16")
    for y in years[1:]:
        cur = shares_100(y).astype("int16")
        moved = np.abs(cur - prev).sum(axis=0) / 2                  # % of the cell
        rows.append({"from": y - 1, "to": y, "changed (% of area)": float((moved * a100).sum() / a100.sum())})
        prev = cur
    return pd.DataFrame(rows)


def change_depth(start=2024, end=2025):
    """How much of a 100 m cell's 10 m pixels changed, for cells whose class changed and for the
    others: the share-change floor above, per cell (% of the cell)."""
    moved = np.abs(shares_100(end).astype("int16") - shares_100(start).astype("int16")).sum(axis=0) / 2
    a, b = class_100(start), class_100(end)
    valid = ~np.isin(a, H.ESRI_INVALID) & ~np.isin(b, H.ESRI_INVALID)
    a100, rows = area_100(), {}
    for name, sel in (("class changed", valid & (a != b)), ("class kept", valid & (a == b))):
        m = moved[sel]
        rows[name] = {"cells (%)": 100 * float(sel.sum() / valid.sum()),
                      "10 m pixels changed (% of the cell, median)": float(np.median(m)),
                      "cells with more than half changed (%)": 100 * float((m > 50).mean()),
                      "cells with under a tenth changed (%)": 100 * float((m < 10).mean()),
                      "share of all changed 10 m pixels (%)": 100 * float((m * a100[sel]).sum() / (moved[valid] * a100[valid]).sum())}
    return pd.DataFrame(rows).T


def change_table():
    t250 = _change_rates(class_250, area_250()).set_index(["from", "to"])
    t100 = _change_rates(class_100, area_100()).set_index(["from", "to"])
    tmaj = _change_rates(majority_100, area_100()).set_index(["from", "to"])
    floor = share_change_floor().set_index(["from", "to"])
    parts = {"250 m": t250, "100 m": t100, "100 m majority": tmaj}
    out = pd.concat({k: v for k, v in parts.items()}, axis=1)
    out[("10 m shares", "changed (% of area), floor")] = floor["changed (% of area)"]
    return out


# --------------------------------------------------------------------------
# Section 3: the observed amount of change, 2024 -> 2025

def transitions(start=2024, end=2025):
    """Each transition's area (km²) and cells on both grids, and whether it passes the platform's
    1,000-pixel threshold (1,000 ha at 100 m, 6,250 ha at 250 m)."""
    rows = {}
    for grid, maps, area in (("100 m", class_100, area_100()), ("250 m", class_250, area_250())):
        a, b = maps(start), maps(end)
        valid = ~np.isin(a, H.ESRI_INVALID) & ~np.isin(b, H.ESRI_INVALID) & (a != b)
        key = a[valid].astype("int32") * 100 + b[valid]
        cells = pd.Series(key).value_counts()
        km2 = pd.Series(area[valid], index=key).groupby(level=0).sum() / 100
        for k in cells.index:
            r = rows.setdefault((NAMES[k // 100], NAMES[k % 100]), {})
            r[f"{grid} cells"] = int(cells[k])
            r[f"{grid} km²"] = float(km2[k])
    t = pd.DataFrame(rows).T.fillna(0)
    t.index.names = ["from", "to"]
    for grid in ("100 m", "250 m"):
        t[f"{grid} passes"] = t[f"{grid} cells"] >= MIN_PIXELS
    return t.sort_values("100 m km²", ascending=False)


def busiest_window(size=250, start=2024, end=2025):
    """(row, col, size) of the size x size block of 100 m cells (25 km) with the most change,
    among blocks wholly inside Guinea."""
    changed = G.to_grid(((class_100(start) != class_100(end)) & ~np.isin(class_100(end), H.ESRI_INVALID))
                        .astype("uint8"), fill=0)
    return H.busiest_window(changed, size=size, inside=G.inside_mask(), min_inside=1.0)


def window_maps(window, start=2024, end=2025):
    """The maps of a 100 m window, and the 250 m maps over the same ground with their extent in the
    window's cell coordinates (for drawing on the same axes)."""
    import rasterio
    r, c, n = window
    crop100 = lambda v: G.to_grid(v, fill=0, dtype="uint8")[r:r + n, c:c + n]
    out = {"100 m start": crop100(class_100(start)), "100 m end": crop100(class_100(end)),
           "majority start": crop100(majority_100(start))}
    t, cell = G.grid()["transform"], G.CELL_DEG
    west, north = t[2] + c * cell, t[5] - r * cell
    east, south = west + n * cell, north - n * cell
    p = profile_250()
    t2 = p["transform"]
    c0, c1 = int(np.floor((west - t2.c) / t2.a)), int(np.ceil((east - t2.c) / t2.a))
    r0, r1 = int(np.floor((north - t2.f) / t2.e)), int(np.ceil((south - t2.f) / t2.e))
    for key, y in (("250 m start", start), ("250 m end", end)):
        with rasterio.open(H.DATA / "esri" / f"ESRI_LULC_{y}.tif") as src:
            out[key] = src.read(1)[r0:r1, c0:c1]
    lon0, lon1 = t2.c + c0 * t2.a, t2.c + c1 * t2.a
    lat0, lat1 = t2.f + r0 * t2.e, t2.f + r1 * t2.e
    out["250 m extent"] = ((lon0 - west) / cell - 0.5, (lon1 - west) / cell - 0.5,
                           (north - lat1) / cell - 0.5, (north - lat0) / cell - 0.5)
    return out


def changed_rgba(start, end, alpha=0.9):
    """RGBA image: cells that changed class in the destination class's colour, the rest clear."""
    from matplotlib.colors import to_rgba
    img = np.zeros(start.shape + (4,), dtype="float32")
    changed = (start != end) & ~np.isin(start, H.ESRI_INVALID) & ~np.isin(end, H.ESRI_INVALID)
    for code, colour in H.ESRI_COLOURS.items():
        sel = changed & (end == code)
        img[sel] = to_rgba(colour, alpha)
    return img


def save(name, frame):
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name)
    return frame


if __name__ == "__main__":
    import time
    t0 = time.time()
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(save("01_class_vs_shares.csv", class_against_shares()).round(2)); print(f"{time.time()-t0:.0f}s")
    print(confusion_against_majority(2024).round(1))
    print(save("01_class_areas_2024.csv", class_areas(2024)).round(1))
    print("100 vs 250 agreement 2024:", round(agreement_100_250(2024), 2)); print(f"{time.time()-t0:.0f}s")
    print(save("01_change_by_year.csv", change_table()).round(2)); print(f"{time.time()-t0:.0f}s")
    print(save("01_transitions_2024_2025.csv", transitions()).round(1)); print(f"{time.time()-t0:.0f}s")
