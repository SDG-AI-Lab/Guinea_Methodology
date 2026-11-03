"""Layers read from Earth Engine onto the platform's 100 m grid, saved in the store.

Images are downloaded as the platform downloads them (getDownloadURL, GeoTIFF, scale 100 m,
EPSG:4326), in tiles of whole grid cells, several at a time, and each tile is placed on the
grid by its own transform. Only the cells inside Guinea are kept.

  hansen_shares()      per cell: the share that was forest in 2000 (canopy >= 30 %, as the
                       platform), that is land (datamask 1), and that was forest and lost in
                       each year 2001-2024; the platform's hansen_shares_image (2026-10-26),
                       stored as whole percentages
  esri_landcover(year) per cell: the class the platform gets (the mode of the year's Esri
                       tiles, read at 100 m), and the share of each class among the cell's
                       10 m pixels, as whole percentages
  predictors()         the deforestation predictors that gain from 100 m: elevation, WorldPop,
                       GHSL built-up surface, protected areas (designation year), GRIP4 roads,
                       GAUL regions (read from Earth Engine as the platform reads them);
                       OpenStreetMap roads, railways, waterways and amenities from the Geofabrik
                       extract (the platform's tags, rasterized touching every cell); ecoregions
                       and mines (local polygons). Slope, aspect, distances and the amenity
                       heatmap are made from these in each run, with the platform's functions.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

import grid100 as G

HANSEN = "UMD/hansen/global_forest_change_2025_v1_13"
CANOPY = 30
LOSS_YEARS = list(range(2001, 2025))
ESRI = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"
ESRI_CLASSES = {1: "water", 2: "trees", 4: "flooded_vegetation", 5: "crops", 7: "built_area",
                8: "bare_ground", 10: "clouds", 11: "rangeland"}          # 9 (snow) does not occur
WORKERS = 6


def _tiles(side):
    """(row0, row1, col0, col1) of the tiles that hold cells inside Guinea."""
    g = G.grid()
    inside = G.inside_mask()
    for r0 in range(0, g["height"], side):
        for c0 in range(0, g["width"], side):
            r1, c1 = min(r0 + side, g["height"]), min(c0 + side, g["width"])
            if inside[r0:r1, c0:c1].any():
                yield r0, r1, c0, c1


def _download_tile(ee, image, tile, attempts=4):
    """One tile -> (array bands x rows x cols, row offset, col offset) on the grid."""
    import requests
    import rasterio
    from rasterio.io import MemoryFile
    g = G.grid()
    a, west, north = g["cell_deg"], g["transform"][2], g["transform"][5]
    r0, r1, c0, c1 = tile
    eps = a * 1e-3                                       # stay inside the tile's outer cells
    region = ee.Geometry.Rectangle([west + c0 * a + eps, north - r1 * a + eps, west + c1 * a - eps, north - r0 * a - eps],
                                   proj="EPSG:4326", geodesic=False)
    for attempt in range(attempts):
        try:
            url = image.getDownloadURL({"scale": G.SCALE_M, "crs": "EPSG:4326", "region": region, "format": "GEO_TIFF"})
            response = requests.get(url, timeout=600)
            response.raise_for_status()
            with MemoryFile(response.content) as m, m.open() as src:
                t = src.transform
                data = src.read()
            row_off = int(round((north - t.f) / a))
            col_off = int(round((t.c - west) / a))
            return data, row_off, col_off
        except Exception as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"tile {tile}: {error}") from error
            time.sleep(10 * (attempt + 1))


def download(image, bands, side, dtype="uint8", workers=WORKERS, log=print):
    """Download `bands` of an ee.Image over Guinea -> {band: values over the cells inside Guinea}."""
    ee = G.earth_engine()
    g = G.grid()
    out = np.zeros((len(bands), g["height"], g["width"]), dtype=dtype)
    tiles = list(_tiles(side))
    image = image.select(bands)
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for data, row_off, col_off in pool.map(lambda t: _download_tile(ee, image, t), tiles):
            # clip to the grid (a tile may carry one extra cell at an edge)
            r_start, c_start = max(row_off, 0), max(col_off, 0)
            r_end, c_end = min(row_off + data.shape[1], g["height"]), min(col_off + data.shape[2], g["width"])
            out[:, r_start:r_end, c_start:c_end] = data[:, r_start - row_off:r_end - row_off, c_start - col_off:c_end - col_off]
            done += 1
            if done % 20 == 0 or done == len(tiles):
                log(f"    {done}/{len(tiles)} tiles")
    cells = G.cells()
    return {b: out[i].reshape(-1)[cells] for i, b in enumerate(bands)}


def _side(n_bands, bytes_per_band=1, budget=40e6, largest=2048):
    """Tile side in cells so one request stays under Earth Engine's 48 MB limit, which counts each
    band's bytes plus one byte of mask per cell (2 for uint8, 3 for int16)."""
    return min(largest, int(np.sqrt(budget / (n_bands * (bytes_per_band + 1)))))


def hansen_shares(overwrite=False, log=print):
    name = "hansen_shares"
    if name in G.layers() and not overwrite:
        return G.layers()[name]
    ee = G.earth_engine()
    img = ee.Image(HANSEN)
    forest = img.select("treecover2000").unmask(0).gte(CANOPY)
    loss_year = img.select("lossyear").unmask(0)
    bands = [forest.rename("forest2000"), img.select("datamask").unmask(0).eq(1).rename("land")]
    bands += [forest.And(loss_year.eq(y - 2000)).rename(f"loss_{y}") for y in LOSS_YEARS]
    names = ["forest2000", "land"] + [f"loss_{y}" for y in LOSS_YEARS]
    shares = ee.Image.cat(bands).toFloat().reduceResolution(ee.Reducer.mean(), maxPixels=64)
    image = shares.multiply(100).round().toUint8()
    log(f"Fetching {name} ({len(names)} bands) ...")
    values = download(image, names, _side(len(names)), log=log)
    return G.write_layer("hansen", name, values, scale=0.01,
                         source=f"{HANSEN}, canopy >= {CANOPY} % in 2000, reduceResolution(mean) of the 30 m pixels "
                                f"(the platform's hansen_shares_image), whole percentages",
                         notes="forest2000: share of the cell that was forest in 2000; land: share with datamask 1; "
                               "loss_YYYY: share that was forest and lost in YYYY")


def esri_landcover(year, overwrite=False, log=print):
    """The platform's class of each cell for `year`, and each class's share of the cell."""
    name = f"esri_{year}"
    if name in G.layers() and not overwrite:
        return G.layers()[name]
    ee = G.earth_engine()
    roi = ee.Geometry(__import__("json").loads(G.BOUNDARY.read_text())["features"][0]["geometry"])
    collection = ee.ImageCollection(ESRI).filterBounds(roi).filterDate(f"{year}-01-01", f"{year}-12-31").select("b1")
    platform_class = collection.mode().rename("class")                       # api_connector.build_annual_gee_image
    shares = collection.map(lambda im: ee.Image.cat(
        [im.eq(c).rename(n) for c, n in ESRI_CLASSES.items()]).toFloat()
        .reduceResolution(ee.Reducer.mean(), maxPixels=256)).mosaic()
    image = ee.Image.cat([platform_class.unmask(0).toUint8(), shares.multiply(100).round().unmask(0).toUint8()])
    names = ["class"] + list(ESRI_CLASSES.values())
    log(f"Fetching {name} ({len(names)} bands) ...")
    values = download(image, names, 256, workers=12, log=log)      # 10 m shares: 256 cells per tile (~35 s each)
    return G.write_layer("landcover", name, values, scale=1.0,
                         source=f"{ESRI}, {year}: class = mode of the year's tiles read at 100 m (as the platform); "
                                f"shares = reduceResolution(mean) of each class among the 10 m pixels, whole percentages",
                         notes="class: Esri code (1 water, 2 trees, 4 flooded vegetation, 5 crops, 7 built area, 8 bare ground, "
                               "10 clouds, 11 rangeland; 0 no data); other columns: share of the cell in that class (%)")


# --------------------------------------------------------------------------
# Predictors

def _ee_layer(name, image_fn, bands, dtype, source, scale=1.0, notes="", overwrite=False, log=print):
    """One Earth Engine image, read at 100 m as the platform reads it, saved as a predictors layer."""
    if name in G.layers() and not overwrite:
        return G.layers()[name]
    image = image_fn(G.earth_engine())
    log(f"Fetching {name} ({len(bands)} bands) ...")
    values = download(image, bands, _side(len(bands), np.dtype(dtype).itemsize), dtype=dtype, log=log)
    return G.write_layer("predictors", name, values, scale=scale, source=source, notes=notes)


def _roi(ee):
    return ee.Geometry(json.loads(G.BOUNDARY.read_text())["features"][0]["geometry"])


def ee_predictors(overwrite=False, log=print):
    _ee_layer("dem", lambda ee: ee.Image("USGS/SRTMGL1_003").select("elevation").unmask(0).toInt16(),
              ["elevation"], "int16", "USGS/SRTMGL1_003 elevation (m), read at 100 m as the platform's DEM_GEE",
              overwrite=overwrite, log=log)
    pop_years = [str(y) for y in range(2010, 2021)]
    _ee_layer("worldpop", lambda ee: ee.Image.cat([
        ee.ImageCollection("WorldPop/GP/100m/pop").filterBounds(_roi(ee)).filter(ee.Filter.eq("country", "GIN"))
        .filterDate(f"{y}-01-01", f"{y}-12-31").select("population").mean().unmask(0).rename(y) for y in pop_years]).toFloat(),
        pop_years, "float32", "WorldPop/GP/100m/pop, people per cell, as the platform's POPULATION_DENSITY",
        overwrite=overwrite, log=log)
    ghsl_years = ["2010", "2015", "2020"]
    _ee_layer("ghsl_built", lambda ee: ee.Image.cat([
        ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{y}").select("built_surface").unmask(0).rename(y)
        for y in ghsl_years]).toUint16(),
        ghsl_years, "uint16", "JRC/GHSL/P2023A/GHS_BUILT_S, built-up surface (m² per 100 m GHSL cell)",
        overwrite=overwrite, log=log)
    _ee_layer("wdpa_year", lambda ee: ee.Image().int().paint(
        ee.FeatureCollection("WCMC/WDPA/current/polygons").filter(ee.Filter.eq("ISO3", "GIN"))
        .filter(ee.Filter.inList("STATUS", ["Designated", "Inscribed", "Established"]))
        .map(lambda f: f.set("year", ee.Number(f.get("STATUS_YR")).max(1))), "year")
        .unmask(0).toUint16().rename("status_year"),
        ["status_year"], "uint16", "WCMC/WDPA/current/polygons (Guinea; designated, inscribed, established): "
                                   "year of designation, 0 outside", overwrite=overwrite, log=log)
    _ee_layer("grip4_roads", lambda ee: ee.Image().int().paint(
        ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/Africa").filterBounds(_roi(ee)), 1)
        .unmask(0).toUint8().rename("road"),
        ["road"], "uint8", "GRIP4 roads (Meijer et al. 2018, about 2018): 1 on cells the roads cross",
        overwrite=overwrite, log=log)
    _ee_layer("gaul_admin1", lambda ee: ee.Image().int().paint(
        ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq("ADM0_NAME", "Guinea")), "ADM1_CODE")
        .unmask(0).toUint16().rename("adm1"),
        ["adm1"], "uint16", "FAO/GAUL/2015/level1 ADM1_CODE (the 250 m experiments' Verra regions)",
        overwrite=overwrite, log=log)


GEOFABRIK = "https://download.geofabrik.de/africa/guinea-latest.osm.pbf"
OSM_TAGS = {"roads": {"highway": True}, "railways": {"railway": True},                       # PREDICTOR_VECTORS_CONFIG
            "waterways": {"waterway": True, "natural": "water"}, "poi": {"amenity": True}}
_OSM_LAYERS = ("points", "lines", "multipolygons")                                           # api_connector


def _sql_text(value):
    return "'" + str(value).replace("'", "''") + "'"


def _osm_where(tags, fields):
    """api_connector._osm_where (platform, main 8eb3887): an OGR filter for osmnx-style tags."""
    clauses = []
    for tag, value in tags.items():
        values = None if value is True else ([value] if isinstance(value, str) else list(value))
        if tag in fields:
            clauses.append(f'"{tag}" IS NOT NULL' if values is None
                           else f'"{tag}" IN ({", ".join(_sql_text(v) for v in values)})')
        elif "other_tags" in fields:
            if values is None:
                clauses.append("other_tags LIKE " + _sql_text('%"' + tag + '"=>%'))
            else:
                clauses += ["other_tags LIKE " + _sql_text('%"' + tag + '"=>"' + v + '"%') for v in values]
    return " OR ".join(clauses) or None


def _osm_features(pbf, tags):
    """api_connector._osm_features_from_extract: the features matching `tags` that touch Guinea."""
    import geopandas as gpd
    import pandas as pd
    import pyogrio
    import shapely
    polygon = G.outline()
    shapely.prepare(polygon)
    frames = []
    for layer in _OSM_LAYERS:
        where = _osm_where(tags, set(pyogrio.read_info(pbf, layer=layer)["fields"]))
        if not where:
            continue
        gdf = pyogrio.read_dataframe(pbf, layer=layer, where=where, bbox=polygon.bounds)
        gdf = gdf[shapely.intersects(polygon, gdf.geometry.values)]
        if len(gdf):
            frames.append(gdf)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326") if frames else None


def _rasterize(geometries, values=None, all_touched=True):
    from rasterio.features import rasterize
    g = G.grid()
    values = [1] * len(geometries) if values is None else values
    return rasterize(zip(geometries, values), out_shape=(g["height"], g["width"]), transform=G.profile()["transform"],
                     fill=0, all_touched=all_touched, dtype="uint8")


def osm_masks(overwrite=False, log=print):
    """Cells touched by OpenStreetMap roads, railways, waterways and amenities, rasterized as the
    platform's process_vector_layer does; distances and the amenity heatmap are made from them."""
    import os
    import requests
    name = "osm"
    if name in G.layers() and not overwrite:
        return G.layers()[name]
    pbf = G.STORE / "downloads" / "guinea-latest.osm.pbf"
    if not pbf.exists():
        pbf.parent.mkdir(parents=True, exist_ok=True)
        log("Downloading the Geofabrik extract ...")
        with requests.get(GEOFABRIK, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(str(pbf) + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        os.replace(str(pbf) + ".part", pbf)
    columns, counts = {}, {}
    for key, tags in OSM_TAGS.items():
        gdf = _osm_features(str(pbf), tags)
        counts[key] = 0 if gdf is None else len(gdf)
        g = G.grid()
        columns[key] = _rasterize(list(gdf.geometry)) if gdf is not None else np.zeros((g["height"], g["width"]), "uint8")
        log(f"    {key}: {counts[key]:,} features")
    snapshot = datetime.fromtimestamp(pbf.stat().st_mtime).date().isoformat()
    return G.write_layer("predictors", name, columns,
                         source=f"OpenStreetMap, Geofabrik extract downloaded {snapshot}; the platform's tags "
                                f"(PREDICTOR_VECTORS_CONFIG) and reading (api_connector, main 8eb3887); all_touched",
                         notes="1 on cells touched by the features. "
                               + "; ".join(f"{k}: {v:,} features {OSM_TAGS[k]}" for k, v in counts.items()))


def local_masks(overwrite=False, log=print):
    """Ecoregions (RESOLVE 2017, codes as the 250 m experiments) and mines (Maus et al. 2022), rasterized."""
    import geopandas as gpd
    if "ecoregions" not in G.layers() or overwrite:
        eco = gpd.read_file(G.REPO / "studies" / "output_datasets" / "ecoregions_guinea.gpkg",
                            layer="ecoregions_guinea").to_crs(4326)
        names = sorted(eco["ECO_NAME"].unique())
        codes = _rasterize(list(eco.geometry), [names.index(n) + 1 for n in eco["ECO_NAME"]], all_touched=False)
        G.write_layer("predictors", "ecoregions", {"ecoregion": codes},
                      source="RESOLVE Ecoregions 2017 (Dinerstein et al. 2017), studies/output_datasets/"
                             "ecoregions_guinea.gpkg; cell centre inside",
                      notes="codes: " + json.dumps({i + 1: n for i, n in enumerate(names)}))
    if "mines" not in G.layers() or overwrite:
        path = G.REPO / "landuse_experiments" / "data" / "candidates" / "mining_maus2022" / "global_mining_polygons_v2.gpkg"
        mines = gpd.read_file(path, bbox=G.outline().bounds).to_crs(4326)
        G.write_layer("predictors", "mines", {"mine": _rasterize(list(mines.geometry))},
                      source="Maus et al. 2022, global mining polygons v2 (mines mapped about 2019), Guinea's bounding box",
                      notes=f"1 on cells touched by a mine polygon ({len(mines)} polygons)")
    log("    ecoregions and mines done")


def predictors(overwrite=False, log=print):
    ee_predictors(overwrite, log)
    osm_masks(overwrite, log)
    local_masks(overwrite, log)
