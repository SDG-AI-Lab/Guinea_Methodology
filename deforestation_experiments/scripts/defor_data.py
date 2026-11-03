"""Earth Engine helpers, and the inputs from sources coarser than 100 m (rainfall, land temperature, soil
moisture, night lights, travel time, burned area), fetched once onto the land-use experiments' ~250 m
lattice and resampled to the 100 m grid when used (predictors.py).

Every layer is read once from Earth Engine (or a public download) and saved in
data/; later runs read the saved files. Annual layers are kept for every year, so
each experiment can take a predictor at the start of the window it predicts.

Hansen is read as the deployed notebook read it: downloaded at 100 m, then canopy
resampled bilinearly and loss year by nearest neighbour onto the analysis grid
(deforestation_analysis.ipynb, section 3).

Usage: python defor_data.py            # fetch whatever is missing
"""

import os

# PostGIS sets PROJ_LIB/GDAL_DATA machine-wide and breaks rasterio's CRS lookups.
for _key in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_key, None)

import glob
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling

ROOT = Path(__file__).resolve().parents[1]                 # deforestation_experiments/
REPO = ROOT.parent
DATA = ROOT / "data"
LANDUSE = REPO / "landuse_experiments"
STUDIES = REPO / "studies"
TOOL = REPO / "Land Use-Deforestation"
BOUNDARY = TOOL / "data" / "geoBoundaries-GIN-ADM0.geojson"
ANNUAL_YEARS = list(range(2010, 2025))                      # annual predictors
TILE = 1024


# --------------------------------------------------------------------------
# Grid

def grid():
    """The land-use experiments' grid: 2444 x 3443 cells of 0.002246° (~250 m), EPSG:4326."""
    with rasterio.open(LANDUSE / "data" / "esri" / "ESRI_LULC_2017.tif") as src:
        profile = src.profile
    profile.update(count=1, dtype="float32", nodata=None, tiled=True, blockxsize=512, blockysize=512,
                   compress="deflate")
    return profile


def guinea_mask(profile):
    """True for cells whose centre is inside Guinea (geoBoundaries ADM0), as in the land-use experiments."""
    from rasterio.features import geometry_mask
    shapes = [f["geometry"] for f in json.loads(BOUNDARY.read_text())["features"]]
    return ~geometry_mask(shapes, out_shape=(profile["height"], profile["width"]), transform=profile["transform"])


def fine_grid(profile, factor=2.5):
    """A grid over the same extent with cells `factor` times smaller (100 m for Hansen)."""
    t = profile["transform"]
    width, height = int(round(profile["width"] * factor)), int(round(profile["height"] * factor))
    out = dict(profile)
    out.update(width=width, height=height,
               transform=Affine(t.a * profile["width"] / width, 0, t.c, 0, t.e * profile["height"] / height, t.f))
    return out


def write(path, bands, profile, names, dtype="float32", nodata=None):
    out = dict(profile)
    out.update(count=len(bands), dtype=dtype, nodata=nodata)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out) as dst:
        for i, (band, name) in enumerate(zip(bands, names), start=1):
            dst.write(np.asarray(band).astype(dtype), i)
            dst.set_band_description(i, str(name))
    return path


def read(name, band=None):
    """A saved layer: one band by name (e.g. "2018"), or {name: array} for all bands."""
    with rasterio.open(DATA / f"{name}.tif") as src:
        names = list(src.descriptions)
        if band is not None:
            return src.read(names.index(str(band)) + 1)
        return {n: src.read(i + 1) for i, n in enumerate(names)}


# --------------------------------------------------------------------------
# Earth Engine

_EE = None


def earth_engine():
    """Earth Engine with the project's service account (the key file is git-ignored; never printed)."""
    global _EE
    if _EE is None:
        import ee
        key = glob.glob(str(TOOL / "landuse_tool" / "credentials" / "tribal*.json"))[0]
        info = json.loads(Path(key).read_text())
        ee.Initialize(ee.ServiceAccountCredentials(info["client_email"], key), project=info["project_id"])
        _EE = ee
    return _EE


def compute(image, profile, bands, nodata=0, log=print):
    """Read `bands` of an ee.Image on the grid of `profile`, tile by tile -> {band: array}."""
    ee = earth_engine()
    t = profile["transform"]
    height, width = profile["height"], profile["width"]
    image = image.select(bands).unmask(nodata).toFloat()
    out = {b: np.full((height, width), nodata, dtype="float32") for b in bands}
    side = min(TILE, int(np.sqrt(40e6 / (4 * len(bands)))))      # requests are capped at 48 MB
    tiles = [(r, c) for r in range(0, height, side) for c in range(0, width, side)]
    for k, (r, c) in enumerate(tiles, start=1):
        h, w = min(side, height - r), min(side, width - c)
        request = {"expression": image, "fileFormat": "NUMPY_NDARRAY",
                   "grid": {"dimensions": {"width": w, "height": h}, "crsCode": "EPSG:4326",
                            "affineTransform": {"scaleX": t.a, "shearX": 0, "translateX": t.c + c * t.a,
                                                "shearY": 0, "scaleY": t.e, "translateY": t.f + r * t.e}}}
        for attempt in range(4):
            try:
                block = ee.data.computePixels(request)
                break
            except Exception as error:                      # transient network/quota errors
                if attempt == 3:
                    raise
                log(f"    retry tile {k} ({error})")
        for b in bands:
            out[b][r:r + h, c:c + w] = block[b]
        if k % 10 == 0 or k == len(tiles):
            log(f"    {k}/{len(tiles)} tiles")
    return out


def fetch(name, image_fn, bands, names=None, profile=None, dtype="float32", nodata=0, log=print):
    """Save an Earth Engine layer as data/<name>.tif once; later calls do nothing."""
    path = DATA / f"{name}.tif"
    if path.exists():
        return path
    profile = profile or grid()
    log(f"Fetching {name} ...")
    values = compute(image_fn(earth_engine()), profile, bands, nodata=nodata, log=log)
    return write(path, [values[b] for b in bands], profile, names or bands, dtype=dtype)


def guinea_region(ee):
    t = grid()["transform"]
    w, h = grid()["width"], grid()["height"]
    return ee.Geometry.Rectangle([t.c, t.f + h * t.e, t.c + w * t.a, t.f])


def annual(collection, band, reducer, years, scale=1.0, offset=0.0):
    """One band per year ("2010", ...) of a reduced image collection."""
    def image_fn(ee):
        col = ee.ImageCollection(collection).select(band)
        images = [getattr(col.filterDate(f"{y}-01-01", f"{y + 1}-01-01"), reducer)()
                  .multiply(scale).add(offset).resample("bilinear").rename(str(y)) for y in years]
        return ee.Image.cat(images)
    return image_fn


# --------------------------------------------------------------------------
# Layers

def fetch_hansen(log=print):
    """Hansen canopy 2000, loss year and data mask: 100 m download, then the deployed resampling."""
    path = DATA / "hansen_250m.tif"
    if path.exists():
        return path
    fine = fine_grid(grid())
    raw = fetch("hansen_100m", lambda ee: ee.Image("UMD/hansen/global_forest_change_2025_v1_13"),
                ["treecover2000", "lossyear", "datamask"], profile=fine, dtype="uint8", log=log)
    profile = grid()
    with rasterio.open(raw) as src:
        bands = {d: src.read(i + 1).astype("float32") for i, d in enumerate(src.descriptions)}
    out = []
    for name, method in (("treecover2000", Resampling.bilinear), ("lossyear", Resampling.nearest),
                         ("datamask", Resampling.nearest)):
        dst = np.zeros((profile["height"], profile["width"]), dtype="float32")
        reproject(bands[name], dst, src_transform=fine["transform"], src_crs=fine["crs"],
                  dst_transform=profile["transform"], dst_crs=profile["crs"], resampling=method)
        out.append(dst)
    return write(path, out, profile, ["treecover2000", "lossyear", "datamask"])


def fetch_all(log=print):
    years = [str(y) for y in ANNUAL_YEARS]
    fetch_hansen(log)
    # Yearly predictors of the deployed model, every year, so each window can use its start year
    fetch("lst_annual", annual("MODIS/061/MOD11A2", "LST_Day_1km", "mean", ANNUAL_YEARS, 0.02, -273.15), years, log=log)
    fetch("chirps_annual", annual("UCSB-CHG/CHIRPS/DAILY", "precipitation", "sum", ANNUAL_YEARS), years, log=log)
    fetch("soil_moisture_annual", annual("ECMWF/ERA5_LAND/MONTHLY_AGGR", "volumetric_soil_water_layer_1", "mean",
                                         ANNUAL_YEARS), years, log=log)
    fetch("chirps_normal", lambda ee: ee.ImageCollection([
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(f"{y}-01-01", f"{y + 1}-01-01").sum()
        for y in range(1991, 2021)]).mean().resample("bilinear").rename("1991-2020"), ["1991-2020"], log=log)
    pop_years = [str(y) for y in range(2010, 2021)]
    fetch("worldpop", lambda ee: ee.Image.cat([
        ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.eq("country", "GIN"))
        .filter(ee.Filter.eq("year", int(y))).mosaic().rename(y) for y in pop_years]), pop_years, log=log)
    # Candidate predictors (forest_loss_drivers.ipynb, sections 2 and 3)
    burn_years = [str(y) for y in range(2001, 2025)]
    fetch("burned", lambda ee: ee.Image.cat([
        ee.ImageCollection("MODIS/061/MCD64A1").filterDate(f"{y}-01-01", f"{int(y) + 1}-01-01")
        .select("BurnDate").max().gt(0).rename(y) for y in burn_years]), burn_years, dtype="uint8", log=log)
    viirs_years = [str(y) for y in range(2014, 2025)]
    fetch("nightlights", annual("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG", "avg_rad", "mean", range(2014, 2025)),
          viirs_years, log=log)
    fetch("travel_time_2015", lambda ee: ee.Image("Oxford/MAP/accessibility_to_cities_2015_v1_0")
          .select("accessibility").resample("bilinear").rename("2015"), ["2015"], nodata=-1, log=log)
    ghsl_years = ["2010", "2015", "2020"]
    fetch("ghsl_built", lambda ee: ee.Image.cat([
        ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{y}").select("built_surface").rename(y) for y in ghsl_years]),
        ghsl_years, log=log)
    fetch("wdpa_year", lambda ee: ee.Image().int().paint(
        ee.FeatureCollection("WCMC/WDPA/current/polygons").filter(ee.Filter.eq("ISO3", "GIN"))
        .filter(ee.Filter.inList("STATUS", ["Designated", "Inscribed", "Established"]))
        .map(lambda f: f.set("year", ee.Number(f.get("STATUS_YR")).max(1))), "year").rename("status_year"),
        ["status_year"], log=log)
    fetch("grip4_roads", lambda ee: ee.Image().int().paint(
        ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/Africa").filterBounds(guinea_region(ee)), 1)
        .rename("road"), ["road"], dtype="uint8", log=log)
    fetch("gaul_admin1", lambda ee: ee.Image().int().paint(
        ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq("ADM0_NAME", "Guinea")), "ADM1_CODE")
        .rename("adm1"), ["adm1"], log=log)
    fetch("wri_drivers", lambda ee: ee.Image(
        "projects/landandcarbon/assets/wri_gdm_drivers_forest_loss_1km/v1_2_2001_2024").select([0], ["class"]),
        ["class"], dtype="uint8", log=log)
    fetch_railways(log)
    rasterize_ecoregions(log)


def fetch_railways(log=print):
    """OpenStreetMap railways (Geofabrik extract, today's snapshot as in the deployed model) -> cells crossed."""
    path = DATA / "osm_railways.tif"
    if path.exists():
        return path
    import geopandas as gpd
    import requests
    zip_path = DATA / "downloads" / "guinea-latest-free.shp.zip"
    if not zip_path.exists():
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        log("Downloading the Geofabrik Guinea extract ...")
        with requests.get("https://download.geofabrik.de/africa/guinea-latest-free.shp.zip", stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if "railways" in n and n.endswith(".shp"))
        stem = member[:-4]
        folder = DATA / "downloads" / "railways"
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            if stem + ext in zf.namelist():
                zf.extract(stem + ext, folder)
    rail = gpd.read_file(folder / member).to_crs(4326)
    profile = grid()
    burned = rasterize(((g, 1) for g in rail.geometry), out_shape=(profile["height"], profile["width"]),
                       transform=profile["transform"], fill=0, all_touched=True, dtype="uint8")
    return write(path, [burned], profile, ["railway"], dtype="uint8")


def rasterize_ecoregions(log=print):
    path = DATA / "ecoregions.tif"
    if path.exists():
        return path
    import geopandas as gpd
    eco = gpd.read_file(STUDIES / "output_datasets" / "ecoregions_guinea.gpkg", layer="ecoregions_guinea").to_crs(4326)
    names = sorted(eco["ECO_NAME"].unique())
    profile = grid()
    codes = rasterize(((g, names.index(n) + 1) for g, n in zip(eco.geometry, eco["ECO_NAME"])),
                      out_shape=(profile["height"], profile["width"]), transform=profile["transform"], fill=0, dtype="uint8")
    (DATA / "ecoregions.json").write_text(json.dumps({i + 1: n for i, n in enumerate(names)}, indent=1))
    return write(path, [codes], profile, ["ecoregion"], dtype="uint8")


if __name__ == "__main__":
    fetch_all()
    print("done")
