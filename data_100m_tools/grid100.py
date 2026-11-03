"""The platform's 100 m grid and the shared 100 m data store.

Both experiment folders (deforestation_experiments, landuse_experiments) import this module
for their 100 m runs, so every layer lies on the same cells as the platform's.

The grid
--------
The platform (landuse_tool, both modules) takes its grid from its latest Esri land-cover
download: Earth Engine's getDownloadURL at scale 100 m in EPSG:4326 over the bounds of the
geoBoundaries ADM0 outline, in tiles merged with rasterio. Earth Engine snaps such requests to
one global lattice: cells of 100 / 111,319.49 m per degree (0.000898315°) counted from 0°,
extended outwards to cover the requested area (checked with small downloads, see grid.json).
So the grid follows from the outline's bounds as Earth Engine computes them.

A cell is inside Guinea when its centre is inside the outline, as the platform's
create_boundary_mask does. Cell areas use the platform's formula (stats.pixel_area_ha:
111,320 m per degree, cos(latitude) at the row centre).

The store
---------
STORE (environment variable GUINEA_DATA_100M, else E:\\SDG AI Lab\\Guinea\\data_100m):

  grid.json           the grid, how it was derived, and the checks made
  cells.parquet       the cells inside Guinea as runs along each row (row, first col, last col + 1);
                      cells() expands them to cell_id = row * width + col, row-major
  manifest.json       one entry per layer: file, columns, type, scale, source, date
  <group>/<name>.parquet
                      one row per cell inside Guinea, in the order of cells.parquet; one
                      column per band (e.g. "2013"). Groups: hansen, landcover, predictors.

Layers keep the smallest type that holds them (shares as whole percentages in uint8, with
scale 0.01 in the manifest), so a column of the whole country is 25-100 MB in memory.
"""

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]                        # d:/Guinea_Methodology
BOUNDARY = REPO / "Land Use-Deforestation" / "data" / "geoBoundaries-GIN-ADM0.geojson"
CREDENTIALS = REPO / "Land Use-Deforestation" / "landuse_tool" / "credentials"
STORE = Path(os.environ.get("GUINEA_DATA_100M", r"E:\SDG AI Lab\Guinea\data_100m"))

SCALE_M = 100
M_PER_DEG_EE = 111_319.49079327357                    # Earth Engine: 2 pi 6,378,137 m / 360
CELL_DEG = SCALE_M / M_PER_DEG_EE
M_PER_DEG_PLATFORM = 111_320.0                        # stats.pixel_area_ha
ROW_GROUP = 1_000_000


# --------------------------------------------------------------------------
# Earth Engine (the project's service account; the key file is git-ignored and never printed)

_EE = None


def earth_engine():
    global _EE
    if _EE is None:
        import ee
        key = sorted(CREDENTIALS.glob("tribal*.json"))[0]
        info = json.loads(key.read_text())
        ee.Initialize(ee.ServiceAccountCredentials(info["client_email"], str(key)), project=info["project_id"])
        _EE = ee
    return _EE


def outline():
    """Guinea's ADM0 outline (shapely), the file the platform pins."""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    return unary_union([shape(f["geometry"]) for f in json.loads(BOUNDARY.read_text())["features"]])


# --------------------------------------------------------------------------
# Building the grid (once)

def _lattice_check(ee, image, west, south, east, north):
    """Download two tiny tiles at the grid's corners and report their origin in cells."""
    import requests
    from rasterio.io import MemoryFile
    out = []
    for w, s, e, n in ((west, north - 0.01, west + 0.01, north), (east - 0.01, south, east, south + 0.01)):
        url = image.getDownloadURL({"scale": SCALE_M, "crs": "EPSG:4326", "format": "GEO_TIFF",
                                    "region": ee.Geometry.Rectangle([w, s, e, n], proj="EPSG:4326", geodesic=False)})
        with MemoryFile(requests.get(url, timeout=120).content) as m, m.open() as src:
            t = src.transform
            out.append({"cell_deg": t.a, "west_in_cells": t.c / t.a, "north_in_cells": t.f / -t.e,
                        "east_in_cells": (t.c + src.width * t.a) / t.a, "south_in_cells": (t.f + src.height * t.e) / -t.e})
    return out


def build(overwrite=False):
    """Derive the grid as the platform does and write grid.json and cells.parquet."""
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    if (STORE / "grid.json").exists() and not overwrite:
        return grid()
    ee = earth_engine()
    roi = ee.Geometry(json.loads(BOUNDARY.read_text())["features"][0]["geometry"])
    corners = roi.bounds().getInfo()["coordinates"][0]                  # as api_connector does
    west, east = min(c[0] for c in corners), max(c[0] for c in corners)
    south, north = min(c[1] for c in corners), max(c[1] for c in corners)
    col0, col1 = int(np.floor(west / CELL_DEG)), int(np.ceil(east / CELL_DEG))
    row_top, row_bottom = int(np.ceil(north / CELL_DEG)), int(np.floor(south / CELL_DEG))
    width, height = col1 - col0, row_top - row_bottom
    transform = Affine(CELL_DEG, 0, col0 * CELL_DEG, 0, -CELL_DEG, row_top * CELL_DEG)

    esri = (ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
            .filterDate("2025-01-01", "2025-12-31").select("b1").mode())
    check = _lattice_check(ee, esri, west, south, east, north)

    inside = rasterize([(outline(), 1)], out_shape=(height, width), transform=transform, fill=0, dtype="uint8").astype(bool)
    cell_id = np.flatnonzero(inside).astype("int32")
    STORE.mkdir(parents=True, exist_ok=True)
    _write_parquet(STORE / "cells.parquet", _runs(inside))
    info = {
        "crs": "EPSG:4326", "cell_deg": CELL_DEG, "width": width, "height": height,
        "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
        "first_col": col0, "first_row_from_north": row_top,
        "outline": str(BOUNDARY.relative_to(REPO)), "outline_bounds_ee": [west, south, east, north],
        "cells_inside": int(cell_id.size),
        "derived": "Earth Engine lattice of 100 m in EPSG:4326 (cells counted from 0°), extended outwards "
                   "over the ADM0 bounds, as the platform's getDownloadURL tiles are",
        "lattice_check": check,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (STORE / "grid.json").write_text(json.dumps(info, indent=1))
    grid.cache_clear()
    return grid()


def _runs(inside):
    """Runs of inside cells along each row: {"row", "col0", "col1"} (col1 exclusive)."""
    padded = np.pad(inside, ((0, 0), (1, 1))).astype("int8")
    rows, edges = np.nonzero(np.diff(padded, axis=1))
    starts, ends = edges[0::2], edges[1::2]
    return {"row": rows[0::2].astype("int32"), "col0": starts.astype("int32"), "col1": ends.astype("int32")}


# --------------------------------------------------------------------------
# The grid

@lru_cache(maxsize=1)
def grid():
    return json.loads((STORE / "grid.json").read_text())


def profile(dtype="float32", nodata=None):
    """A rasterio profile of the grid (for writing GeoTIFFs or aligning other rasters)."""
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    g = grid()
    return {"driver": "GTiff", "crs": CRS.from_epsg(4326), "transform": Affine(*g["transform"]),
            "width": g["width"], "height": g["height"], "count": 1, "dtype": dtype, "nodata": nodata,
            "tiled": True, "blockxsize": 512, "blockysize": 512, "compress": "deflate"}


@lru_cache(maxsize=1)
def cells():
    """cell_id of every cell inside Guinea, row-major."""
    import pyarrow.parquet as pq
    runs = pq.read_table(STORE / "cells.parquet")
    row, col0, col1 = (runs.column(c).to_numpy().astype("int64") for c in ("row", "col0", "col1"))
    lengths = col1 - col0
    offsets = np.repeat(row * grid()["width"] + col0 - np.cumsum(lengths) + lengths, lengths)
    return (offsets + np.arange(lengths.sum())).astype("int32")


def inside_mask():
    g = grid()
    mask = np.zeros(g["height"] * g["width"], dtype=bool)
    mask[cells()] = True
    return mask.reshape(g["height"], g["width"])


def row_col(cell_id=None):
    cell_id = cells() if cell_id is None else cell_id
    return np.divmod(cell_id, grid()["width"])


def lon_lat(cell_id=None):
    """Centre of each cell (degrees)."""
    row, col = row_col(cell_id)
    t = grid()["transform"]
    return t[2] + (col + 0.5) * t[0], t[5] + (row + 0.5) * t[4]


def cell_area_ha(cell_id=None):
    """Area of each cell (ha), with the platform's formula."""
    row, _ = row_col(cell_id)
    t = grid()["transform"]
    lat = t[5] + (row + 0.5) * t[4]
    return ((abs(t[0]) * M_PER_DEG_PLATFORM * np.cos(np.radians(lat))) * (abs(t[4]) * M_PER_DEG_PLATFORM) / 1e4).astype("float32")


def to_grid(values, fill=np.nan, dtype=None):
    """Values over the cells inside Guinea -> 2D array of the grid (`fill` outside)."""
    g = grid()
    values = np.asarray(values)
    dtype = dtype or (np.result_type(values.dtype, np.float32) if isinstance(fill, float) and np.isnan(fill) else values.dtype)
    out = np.full(g["height"] * g["width"], fill, dtype=dtype)
    out[cells()] = values
    return out.reshape(g["height"], g["width"])


def from_grid(array):
    """2D array of the grid -> values over the cells inside Guinea."""
    return np.asarray(array).reshape(-1)[cells()]


def row_blocks(rows_per_block=512):
    """(first row, last row + 1, slice into cells()) for processing the grid in bands of rows."""
    g = grid()
    row = cells() // g["width"]
    for r0 in range(0, g["height"], rows_per_block):
        r1 = min(r0 + rows_per_block, g["height"])
        i0, i1 = np.searchsorted(row, [r0, r1])
        yield r0, r1, slice(int(i0), int(i1))


# --------------------------------------------------------------------------
# Layers

def _write_parquet(path, columns, metadata=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({k: np.asarray(v) for k, v in columns.items()})
    if metadata:
        table = table.replace_schema_metadata({k: json.dumps(v) for k, v in metadata.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", row_group_size=ROW_GROUP)


def _manifest():
    path = STORE / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def write_layer(group, name, columns, source, scale=1.0, nodata=None, notes=""):
    """Save a layer: {column: values over the cells inside Guinea, or a 2D grid array}."""
    n = cells().size
    data = {}
    for key, values in columns.items():
        values = np.asarray(values)
        values = from_grid(values) if values.ndim == 2 else values
        if values.size != n:
            raise ValueError(f"{name}/{key}: {values.size} values for {n} cells")
        data[str(key)] = values
    path = STORE / group / f"{name}.parquet"
    entry = {"group": group, "file": str(path.relative_to(STORE)), "columns": list(data),
             "dtype": str(next(iter(data.values())).dtype), "scale": scale, "nodata": nodata,
             "source": source, "notes": notes, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _write_parquet(path, data, {"layer": entry})
    manifest = _manifest()
    manifest[name] = entry
    (STORE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return path


def layers():
    """The manifest: {layer name: entry}."""
    return _manifest()


def rebuild_manifest():
    """Rewrite manifest.json from the entries each layer file carries in its metadata (if two
    processes wrote the manifest at the same moment and one entry was lost)."""
    import pyarrow.parquet as pq
    manifest = {}
    for path in sorted(STORE.glob("*/*.parquet")):
        meta = pq.read_schema(path).metadata or {}
        if b"layer" in meta:
            manifest[path.stem] = json.loads(meta[b"layer"])
    (STORE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def resample(path, band=None, resampling="bilinear"):
    """A band of a raster on another grid (e.g. the 250 m experiments' files), resampled onto this
    grid -> values over the cells inside Guinea (float32). For inputs whose sources are coarser
    than 100 m, which are not stored at 100 m."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    g = grid()
    out = np.full((g["height"], g["width"]), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        index = 1 if band is None else list(src.descriptions).index(str(band)) + 1
        reproject(rasterio.band(src, index), out, dst_transform=profile()["transform"], dst_crs="EPSG:4326",
                  dst_nodata=np.nan, resampling=getattr(Resampling, resampling))
    return from_grid(out)


def read_layer(name, columns=None, as_grid=False, scaled=False, fill=np.nan):
    """{column: values} of a saved layer, over the cells inside Guinea (or as 2D grids).

    scaled=True applies the manifest's scale (e.g. uint8 percent -> 0-1 float32)."""
    import pyarrow.parquet as pq
    entry = layers()[name]
    table = pq.read_table(STORE / entry["file"], columns=None if columns is None else [str(c) for c in columns])
    out = {}
    for key in table.column_names:
        values = table.column(key).to_numpy()
        if scaled and entry["scale"] != 1:
            values = values.astype("float32") * np.float32(entry["scale"])
        out[key] = to_grid(values, fill=fill) if as_grid else values
    return out
