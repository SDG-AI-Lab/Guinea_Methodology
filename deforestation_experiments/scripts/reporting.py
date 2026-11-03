"""Notebook 02: forest and loss areas from the 100 m shares, by reporting unit, against the 30 m pixels.

Units, as the platform reports them:
  ADM1, ADM2  geoBoundaries (the zonal statistics, stats.zonal_deforestation)
  ecoregion   RESOLVE 2017 (studies/output_datasets/ecoregions_guinea.gpkg)
  h3          H3 hexagons at resolution 7, the platform's finest; resolutions 6 to 4 are the sums
              of their children, as the platform derives them (spec 009, GIS-H3-011)
  ADM0        Guinea

100 m: each cell inside Guinea goes wholly to the unit holding its centre (the platform's
rasterize, pixel-centre rule), with its share of each band times its area.
One value per cell: the platform's half-cell masks (HansenShares.display_arrays: forest while at
least half the cell is forest, lost in the year it falls below half). The platform's forest-cover
layer, its hexagon forest cover, its hexagon risk and its prefecture risk summary use these masks.
30 m: the 30 m pixels of Hansen's tiles of the same release (public download, no Earth Engine
compute), each going to the unit holding its centre and counted when its centre is inside Guinea's
outline, with the definitions of the shares (canopy >= 30 % in 2000, loss inside that forest,
datamask 1 for land) and the platform's area formula. So only the size of the counted piece
differs: a 30 m pixel against a 100 m cell.
"""

import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model as MM                                              # noqa: E402
import fetch as F                                               # noqa: E402  (data_100m_tools)

G = MM.G
HERE = MM.HERE
DATA = HERE / "data" / "reports"
ADM = G.REPO / "Land Use-Deforestation" / "data"
ECO = G.REPO / "studies" / "output_datasets" / "ecoregions_guinea.gpkg"
YEARS = F.LOSS_YEARS
BANDS = ["forest2000", "land"] + [f"loss_{y}" for y in YEARS]
H3_RES = 7
TILES = G.STORE / "downloads" / "hansen_30m"
RELEASE = "GFC-2025-v1.13"                                       # F.HANSEN's release
STRIP_ROWS = 1000
KINDS = ["ADM0", "ADM1", "ADM2", "ecoregion", "h3"]
NEAREST = ["ADM1+nearest", "ADM2+nearest", "ecoregion+nearest"]   # border cells outside every polygon to the nearest unit
ALL_KINDS = KINDS + NEAREST


# --------------------------------------------------------------------------
# Units

@lru_cache(maxsize=None)
def units(kind):
    """GeoDataFrame (unit, name, inside_share, geometry): the polygons the platform rasterizes;
    inside_share is the part of the polygon inside Guinea's outline."""
    import geopandas as gpd
    if kind.endswith("+nearest"):
        return units(kind.split("+")[0])
    if kind.startswith("ADM"):
        gdf = gpd.read_file(ADM / f"geoBoundaries-GIN-{kind}.geojson")
        gdf = gdf.rename(columns={"shapeID": "unit", "shapeName": "name"})[["unit", "name", "geometry"]]
    elif kind == "ecoregion":
        gdf = gpd.read_file(ECO, layer="ecoregions_guinea").to_crs(4326).dissolve("ECO_NAME").reset_index()
        gdf = gdf.assign(unit=gdf.ECO_NAME, name=gdf.ECO_NAME)[["unit", "name", "geometry"]]
    else:
        gdf = hexagons()
    path = DATA / f"units_{kind}.parquet"
    if path.exists():
        gdf["inside_share"] = pd.read_parquet(path).loc[gdf.unit, "inside_share"].values
        return gdf.reset_index(drop=True)
    equal_area = gdf.geometry.to_crs(6933)
    inside = gpd.GeoSeries(gdf.geometry.intersection(G.outline()), crs=4326).to_crs(6933)
    gdf["inside_share"] = (inside.area / equal_area.area).values
    DATA.mkdir(parents=True, exist_ok=True)
    gdf[["unit", "inside_share"]].set_index("unit").to_parquet(path)
    return gdf.reset_index(drop=True)


def hexagons():
    """The resolution-7 hexagons that reach into Guinea."""
    import geopandas as gpd
    import h3
    from shapely.geometry import Polygon
    guinea = G.outline()
    near = guinea.buffer(0.05)                                   # 5.5 km, more than a hexagon's radius
    parts = near.geoms if near.geom_type == "MultiPolygon" else [near]
    ids = sorted(set().union(*(h3.geo_to_cells(p.__geo_interface__, H3_RES) for p in parts)))
    gdf = gpd.GeoDataFrame({"unit": ids, "name": ids}, crs=4326,
                           geometry=[Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(h)]) for h in ids])
    return gdf[gdf.intersects(guinea)].reset_index(drop=True)


@lru_cache(maxsize=None)
def cell_units(kind):
    """Position in units(kind) of the unit holding each cell's centre (cells inside Guinea), -1 for none."""
    path = DATA / f"cells_{kind}.npy"
    if path.exists():
        return np.load(path)
    if kind.endswith("+nearest"):
        values = G.from_grid(nearest_grid(kind.split("+")[0]))
    else:
        values = G.from_grid(_unit_grid(kind)) - 1
    DATA.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    return values


def _unit_grid(kind):
    """1 + position of the unit holding each grid cell's centre, 0 for none (the platform's rasterize)."""
    from rasterio.features import rasterize
    p = G.profile()
    gdf = units(kind)
    return rasterize(((g, i + 1) for i, g in enumerate(gdf.geometry)), out_shape=(p["height"], p["width"]),
                     transform=p["transform"], fill=0, dtype="int32")


@lru_cache(maxsize=None)
def nearest_grid(kind):
    """Position of the unit of each grid cell: the unit holding its centre, else the nearest one."""
    from scipy.ndimage import distance_transform_edt
    ids = _unit_grid(kind)
    r, c = distance_transform_edt(ids == 0, return_distances=False, return_indices=True)
    return (ids[r, c] - 1).astype("int32")


# --------------------------------------------------------------------------
# 100 m: the shares, and the half-cell masks

def by_100m(kind):
    """Per unit, in ha: each band from the shares, and from the half-cell masks the forest of 2000,
    the forest left at the end of each year and each year's loss (mask_*)."""
    path = DATA / f"100m_{kind}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    where = cell_units(kind)
    n = len(units(kind))
    keep = where >= 0
    unit, area = where[keep], G.cell_area_ha()[keep].astype("float64")
    total = lambda w: np.bincount(unit, weights=w, minlength=n)
    out = {}
    for band in BANDS:
        out[band] = total(G.read_layer("hansen_shares", [band])[band][keep] * 0.01 * area)
    forest2000 = G.read_layer("hansen_shares", ["forest2000"])["forest2000"][keep].astype("int16")
    forest = forest2000 >= 50
    out["mask_forest2000"] = total(forest * area)
    remaining, standing = forest2000.copy(), forest.copy()
    for y in YEARS:
        remaining -= G.read_layer("hansen_shares", [f"loss_{y}"])[f"loss_{y}"][keep]
        crossed = standing & (remaining < 50)
        standing &= ~crossed
        out[f"mask_loss_{y}"] = total(crossed * area)
        out[f"mask_forest_{y}"] = total(standing * area)
    table = pd.DataFrame(out, index=units(kind).unit)
    table.to_parquet(path)
    return table


# --------------------------------------------------------------------------
# 30 m: Hansen's tiles

PIX = 0.00025                                                    # Hansen's pixel, on a lattice from 80° N, 180° W


@lru_cache(maxsize=None)
def _tiles(layer):
    """[(path, first row, first column, height, width)] of each tile on the global 30 m lattice."""
    import rasterio
    out = []
    for path in sorted(TILES.glob(f"Hansen_{RELEASE}_{layer}_*.tif")):
        with rasterio.open(path) as src:
            t = src.transform
            out.append((path, round((80 - t.f) / PIX), round((t.c + 180) / PIX), src.height, src.width))
    return out


def _read(layer, r0, r1, c0, c1):
    """Rows r0..r1 and columns c0..c1 of the lattice, from whichever tiles hold them."""
    import rasterio
    from rasterio.windows import Window
    out = np.zeros((r1 - r0, c1 - c0), dtype="uint8")
    for path, tr, tc, h, w in _tiles(layer):
        a, b = max(r0, tr), min(r1, tr + h)
        c, d = max(c0, tc), min(c1, tc + w)
        if a < b and c < d:
            with rasterio.open(path) as src:
                out[a - r0:b - r0, c - c0:d - c0] = src.read(1, window=Window(c - tc, a - tr, d - c, b - a))
    return out


def _rasterize(shapes, transform, shape):
    from rasterio.features import rasterize
    shapes = list(shapes)
    if not shapes:
        return np.zeros(shape, dtype="int32")
    return rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int32")


def by_30m(kind, log=print):
    """Per unit, in ha: each band summed over the 30 m pixels whose centres are inside the unit and Guinea.
    With +nearest, a pixel outside every polygon goes to the unit its 100 m cell went to.
    The first call computes every kind in one pass over the tiles."""
    path = DATA / f"30m_{kind}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from rasterio.transform import Affine
    from shapely.geometry import box
    west, south, east, north = G.outline().bounds
    R0, R1 = int(np.floor((80 - north) / PIX)), int(np.ceil((80 - south) / PIX))
    C0, C1 = int(np.floor((west + 180) / PIX)), int(np.ceil((east + 180) / PIX))
    frames = {k: units(k) for k in ALL_KINDS}
    t = G.grid()["transform"]
    nearest = {k: nearest_grid(k.split("+")[0]) for k in NEAREST}
    ny = len(YEARS)
    sums = {k: {"forest2000": np.zeros(len(g)), "land": np.zeros(len(g)), "loss": np.zeros(len(g) * ny)}
            for k, g in frames.items()}
    start = time.time()
    for r0 in range(R0, R1, STRIP_ROWS):
        r1 = min(r0 + STRIP_ROWS, R1)
        shape = (r1 - r0, C1 - C0)
        transform = Affine(PIX, 0, -180 + C0 * PIX, 0, -PIX, 80 - r0 * PIX)
        strip = box(-180 + C0 * PIX, 80 - r1 * PIX, -180 + C1 * PIX, 80 - r0 * PIX)
        inside = _rasterize([(G.outline(), 1)], transform, shape).astype(bool)
        if not inside.any():
            continue
        lat = 80 - (np.arange(r0, r1) + 0.5) * PIX
        ha = np.broadcast_to(((PIX * G.M_PER_DEG_PLATFORM) ** 2 * np.cos(np.radians(lat)) / 1e4)[:, None], shape)
        forest = (_read("treecover2000", r0, r1, C0, C1) >= F.CANOPY) & inside
        land = (_read("datamask", r0, r1, C0, C1) == 1) & inside
        year = _read("lossyear", r0, r1, C0, C1).astype("int32")
        lost = forest & (year >= 1) & (year <= ny)
        cell_row = np.clip(np.floor((lat - t[5]) / t[4]).astype(int), 0, G.grid()["height"] - 1)
        lon = -180 + (np.arange(C0, C1) + 0.5) * PIX
        cell_col = np.clip(np.floor((lon - t[2]) / t[0]).astype(int), 0, G.grid()["width"] - 1)
        base = {}
        for k, gdf in frames.items():
            if k in NEAREST:
                ids = base[k.split("+")[0]].copy()
                gap = ids < 0
                ids[gap] = nearest[k][cell_row[:, None], cell_col[None, :]][gap]
            else:
                near = gdf.sindex.query(strip, predicate="intersects")
                ids = _rasterize(((gdf.geometry.iloc[i], i + 1) for i in near), transform, shape) - 1
                base[k] = ids
            n = len(gdf)
            for name, m in (("forest2000", forest), ("land", land)):
                m = m & (ids >= 0)
                sums[k][name] += np.bincount(ids[m], weights=ha[m], minlength=n)
            m = lost & (ids >= 0)
            sums[k]["loss"] += np.bincount(ids[m] * ny + year[m] - 1, weights=ha[m], minlength=n * ny)
        log(f"  30 m rows {r1 - R0:,}/{R1 - R0:,}, {(time.time() - start) / 60:.1f} min", flush=True)
    for k, gdf in frames.items():
        loss = sums[k]["loss"].reshape(len(gdf), ny)
        table = pd.DataFrame({"forest2000": sums[k]["forest2000"], "land": sums[k]["land"],
                              **{f"loss_{y}": loss[:, i] for i, y in enumerate(YEARS)}}, index=gdf.unit)
        table.to_parquet(DATA / f"30m_{k}.parquet")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# Comparison

def coarser(table, res):
    """Resolution-7 rows summed into their parents at `res` (the platform's coarser hexagons)."""
    import h3
    parent = [h3.cell_to_parent(u, res) for u in table.index]
    return table.groupby(parent).sum()


def interior(res):
    """Hexagons at `res` whose resolution-7 children all lie inside Guinea (and are all there)."""
    import h3
    gdf = units("h3")
    inside = pd.Series(gdf.inside_share.values > 0.999, index=gdf.unit)
    parent = pd.Series([h3.cell_to_parent(u, res) for u in gdf.unit], index=gdf.unit)
    count = inside.groupby(parent.values).agg(["sum", "size"])
    full = count["sum"] == count["size"]
    expected = pd.Series({p: len(h3.cell_to_children(p, H3_RES)) for p in count.index})
    return count.index[full & (count["size"] == expected)]


def metrics(table):
    """The figures the platform reports, per unit (ha)."""
    loss = table[[f"loss_{y}" for y in YEARS]]
    out = pd.DataFrame({"Forest 2000": table.forest2000, "Land": table.land,
                        f"Forest {YEARS[-1]}": table.forest2000 - loss.sum(axis=1),
                        f"Loss {YEARS[0]}–{YEARS[-1]}": loss.sum(axis=1),
                        f"Loss {YEARS[-3]}–{YEARS[-1]}": loss.iloc[:, -3:].sum(axis=1)})
    for y in YEARS:
        out[f"Loss {y}"] = table[f"loss_{y}"]
    return out


def mask_metrics(table):
    """The same figures from the half-cell masks (one value per cell)."""
    loss = table[[f"mask_loss_{y}" for y in YEARS]]
    out = pd.DataFrame({"Forest 2000": table.mask_forest2000, f"Forest {YEARS[-1]}": table[f"mask_forest_{YEARS[-1]}"],
                        f"Loss {YEARS[0]}–{YEARS[-1]}": loss.sum(axis=1),
                        f"Loss {YEARS[-3]}–{YEARS[-1]}": loss.iloc[:, -3:].sum(axis=1)})
    for y in YEARS:
        out[f"Loss {y}"] = table[f"mask_loss_{y}"]
    return out


def errors(estimate, truth, floor_ha=10.0):
    """Per figure: the share of the 30 m total placed in the wrong unit (sum of |error| / total),
    and the median and 95th percentile of |error| in % of the 30 m figure, over units with at least
    `floor_ha` of it; the national bias in %."""
    rows = []
    for figure in estimate.columns.intersection(truth.columns):
        e, t = estimate[figure], truth[figure]
        rel = ((e - t).abs() / t)[t >= floor_ha] * 100
        yearly = figure.startswith("Loss ") and "–" not in figure
        rows.append({"Figure": figure, "Units": int((t >= floor_ha).sum()),
                     "Misplaced (% of total)": 100 * (e - t).abs().sum() / t.sum(),
                     "Median |error| (%)": rel.median(), "95th pct |error| (%)": rel.quantile(0.95),
                     "Bias (%)": 100 * (e.sum() / t.sum() - 1), "yearly": yearly})
    table = pd.DataFrame(rows).set_index("Figure")
    years = table[table.yearly].drop(columns="yearly")
    table = table[~table.yearly].drop(columns="yearly")
    if len(years):
        table.loc["Loss, single years (mean)"] = years.mean()
    return table


def unit_tables():
    """{level: (100 m shares, 100 m masks, 30 m)} as metrics tables, for every reporting level."""
    out = {}
    for kind in ALL_KINDS:
        s, t = by_100m(kind), by_30m(kind)
        if kind != "h3":
            out[kind] = (metrics(s), mask_metrics(s), metrics(t))
            continue
        for res in range(H3_RES, 3, -1):
            keep = interior(res)
            ss, tt = (coarser(x, res) if res < H3_RES else x for x in (s, t))
            out[f"H3 res {res}"] = (metrics(ss).loc[keep], mask_metrics(ss).loc[keep], metrics(tt).loc[keep])
    return out


if __name__ == "__main__":
    for kind in sys.argv[1:] or ALL_KINDS:
        print(kind, len(units(kind)), "units", flush=True)
        by_100m(kind)
        by_30m(kind)
