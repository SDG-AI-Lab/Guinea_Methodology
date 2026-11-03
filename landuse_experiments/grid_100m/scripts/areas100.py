"""Notebook 03: land-cover class areas from the 100 m shares, by reporting unit, against one class per
cell (what the platform counts), and the platform's published national composition.

The shares are the means of Esri's 10 m pixels in each 100 m cell, so their sum over a unit is the
unit's 10 m area (to the whole-percent rounding of each cell). One class per cell counts a cell's
whole area in its class: the platform's class areas (LandCoverClassArea, from the published 100 m
land-cover map, landuse_tool/class_areas.py) and every area read off its maps.

Units are those of the deforestation work (deforestation_experiments/scripts/reporting.py): ADM1,
ADM2 (geoBoundaries), ecoregion (RESOLVE 2017), H3 resolution 7, each cell to the unit holding its
centre, and the "+nearest" variants that give border cells outside every polygon to the nearest unit.
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE.parents[1] / "deforestation_experiments" / "scripts"))
import classes100 as C                                              # noqa: E402
import reporting as R                                               # noqa: E402

G = C.G
KINDS = ["ADM1", "ADM2", "ecoregion", "h3", "ADM1+nearest", "ADM2+nearest"]
PLATFORM = "http://172.205.219.243"                                 # the live platform (read only)


def national(year):
    """Guinea's class areas (ha): from the shares, and from one class per cell."""
    a = G.cell_area_ha().astype("float64")
    s, cls = C.shares_100(year), C.class_100(year)
    out = {}
    for k, code in enumerate(C.SHARE_COLUMNS):
        if code in C.CLASSES:
            out[C.NAMES[code]] = {"shares (10 m)": float((s[k] * a).sum() / 100), "one class per cell": float(a[cls == code].sum())}
    return pd.DataFrame(out).T


def by_unit(kind, year):
    """Per unit and class (ha): {"shares": table, "class": table}, units as rows."""
    where = R.cell_units(kind)
    keep = where >= 0
    unit, area = where[keep], G.cell_area_ha()[keep].astype("float64")
    n = len(R.units(kind))
    s, cls = C.shares_100(year), C.class_100(year)[keep]
    shares, one = {}, {}
    for k, code in enumerate(C.SHARE_COLUMNS):
        if code not in C.CLASSES:
            continue
        shares[C.NAMES[code]] = np.bincount(unit, weights=s[k][keep] * area / 100, minlength=n)
        one[C.NAMES[code]] = np.bincount(unit, weights=(cls == code) * area, minlength=n)
    index = R.units(kind).unit
    return {"shares": pd.DataFrame(shares, index=index), "class": pd.DataFrame(one, index=index)}


def unit_errors(kind, year, floor_ha=10.0):
    """Per class: how far one class per cell is from the shares in each unit (R.errors' measures)."""
    t = by_unit(kind, year)
    return R.errors(t["class"], t["shares"], floor_ha)


def all_unit_errors(year):
    rows = []
    for kind in KINDS:
        e = unit_errors(kind, year)
        e.insert(0, "Level", kind)
        rows.append(e)
    return pd.concat(rows).reset_index().rename(columns={"Figure": "Class"})


def unit_totals(kind, year):
    """National total of each class over the units, against Guinea's total from the shares: what the
    units' tables add up to (border cells outside every polygon are lost unless given to the nearest)."""
    t = by_unit(kind, year)["shares"].sum()
    return 100 * (t / national(year)["shares (10 m)"] - 1)


def platform_composition(years=(2017, 2025)):
    """The platform's national composition (GET /api/landuse/composition/, public, read only):
    {year: {class name: ha}} and the no-data area."""
    url = f"{PLATFORM}/api/landuse/composition/?from={years[0]}&to={years[1]}&lang=en"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    out = {}
    for row in data["rows"]:
        out[row["year"]] = {c["label"]["text"]: c["area_ha"] for c in row["classes"]}
        out[row["year"]]["no data"] = row["no_data_ha"]
    return pd.DataFrame(out), data.get("data_date")


def grid_box_ha():
    """Area of the platform's whole 100 m grid (the bounding box of Guinea's outline), and of the
    cells inside Guinea, with the platform's area formula."""
    g = G.grid()
    t = g["transform"]
    lat = t[5] + (np.arange(g["height"]) + 0.5) * t[4]
    row_ha = abs(t[0]) * G.M_PER_DEG_PLATFORM * np.cos(np.radians(lat)) * abs(t[4]) * G.M_PER_DEG_PLATFORM / 1e4
    return float(row_ha.sum() * g["width"]), float(G.cell_area_ha().astype("float64").sum())
