"""
Configuration for the Land Use Monitoring & Prediction Tool.

Restructured for Azure deployment:
- No secrets in code. All API keys / credentials are read from environment
  variables (which on Azure should be populated from Key Vault).
- All writable paths default to /tmp (safe for Azure Functions / Container Apps)
  and can be overridden, with an optional Blob Storage container for persistence.
- The dataset catalog now carries the exact GEE asset id, the band to select,
  the reducer, the native resolution, and the temporal class of every layer,
  so the fetcher can pull ONE clean layer per year and discover ALL available
  years automatically.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================================
# ENVIRONMENT / PATHS
# ============================================================================
# /tmp is always writable; on Container Apps/ACI any path works. Override w/ env.
_SCRATCH = os.environ.get("APP_SCRATCH_DIR", "/tmp/landuse")

PROCESSING_CONFIG: Dict = {
    "target_resolution": int(os.environ.get("TARGET_RESOLUTION", 100)),  # metres
    "target_crs": os.environ.get("TARGET_CRS", "EPSG:4326"),
    "resampling_method": "bilinear",

    "nodata_value": -9999,
    "invalid_raster_values": [254, 255, -1],

    # Local (ephemeral) working directories
    "temp_dir": os.path.join(_SCRATCH, "temp"),
    "cache_dir": os.path.join(_SCRATCH, "cache"),
    "output_dir": os.path.join(_SCRATCH, "processed"),

    # Persistent storage (Azure Blob). If a container is set, cache/outputs are
    # mirrored to Blob Storage so a fresh container (Container Apps / ACI /
    # Functions) re-uses previously fetched layers instead of re-downloading.
    "azure_container": os.environ.get("AZURE_BLOB_CONTAINER"),        # e.g. "landuse"
    "azure_prefix": os.environ.get("AZURE_BLOB_PREFIX", "landuse"),
    "azure_connection_string": os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
    "azure_account_url": os.environ.get("AZURE_STORAGE_ACCOUNT_URL"),  # for Managed Identity

    # Google Earth Engine
    "gee_authenticated": False,
    "gee_project_id": os.environ.get("GEE_PROJECT_ID"),          # required
    "gee_service_account": os.environ.get("GEE_SERVICE_ACCOUNT"),  # sa@proj.iam...
    # GEE_SA_KEY holds the *contents* of the service-account JSON (a string).
    "gee_sa_key_secret": os.environ.get("GEE_SA_KEY"),
    # Optional GCS bucket for exporting large (country-scale, native-res) images.
    # NOTE: Earth Engine can only export to Google Drive or Google Cloud Storage
    # (not Azure Blob). On Azure, leave this blank to use the tiled download path
    # (writes straight to local/Blob). Set it only if you need native 10 m country
    # exports, in which case GEE -> GCS -> (copy) -> Azure Blob.
    "gcs_export_bucket": os.environ.get("GCS_EXPORT_BUCKET"),
}

# ============================================================================
# API CREDENTIALS  (values come from env; see get_api_key())
# ============================================================================
# Maps a logical key name -> the environment variable that holds it.
# "keyless" APIs are listed so the code and the operator both know they need
# no secret.
API_CREDENTIALS: Dict[str, Dict] = {
    "opentopography": {
        "env": "OPENTOPO_API_KEY",
        "required": True,
        "how": "Free key: create account at portal.opentopography.org "
               "-> 'Request an API Key' in MyOpenTopo dashboard. "
               "Rate limit 250 calls/day (academic) or 50/day (non-academic).",
    },
    "copernicus_client_id": {
        "env": "CDSE_CLIENT_ID",
        "required": False,   # only needed if a Copernicus/Sentinel layer is enabled
        "how": "dataspace.copernicus.eu -> register -> Sentinel Hub dashboard "
               "-> User Settings -> OAuth clients -> Create. "
               "NOTE: client secret expires ~90 days; plan rotation.",
    },
    "copernicus_client_secret": {
        "env": "CDSE_CLIENT_SECRET",
        "required": False,
        "how": "Shown once when the OAuth client is created (see above).",
    },
    "google_maps": {
        "env": "GOOGLE_MAPS_API_KEY",
        "required": False,   # OSM/Geofabrik covers roads for free; only if you insist
        "how": "console.cloud.google.com -> enable Roads/Maps API -> Credentials. "
               "Billed. Prefer OSM/Geofabrik for road networks.",
    },
    "earthengine": {
        "env": "GEE_SA_KEY",
        "required": True,
        "how": "Create a service account in your GEE-registered Google Cloud "
               "project, grant it Earth Engine access, download the JSON key, and "
               "put its *contents* in this env var / Azure Key Vault.",
    },
    # Keyless (documented for clarity)
    "osm_overpass":  {"env": None, "required": False, "how": "No key required."},
    "worldpop":      {"env": None, "required": False, "how": "No key required (REST + GEE)."},
    "geoboundaries": {"env": None, "required": False, "how": "No key required."},
    "stac":          {"env": None, "required": False, "how": "No key required (public STAC)."},
}


def get_api_key(name: str) -> Optional[str]:
    """Return an API key/secret from the environment, or None if unset/keyless."""
    spec = API_CREDENTIALS.get(name)
    if not spec or not spec.get("env"):
        return None
    return os.environ.get(spec["env"])


# ============================================================================
# REGION & ADMINISTRATIVE BOUNDARY  (GeoBoundaries, pinned)
# ============================================================================
# The pinned gbOpen release in the data directory, checksummed in its
# REFERENCE_DATA.md (feature 002, FR-006). Read from disk, never downloaded:
# the "current" API release can change under a stored result, and the units
# every figure is keyed to must be the ones the hub loaded.
#
# `name_field` is the join key, the unit's shapeID, exactly as the hub's
# gis.AdminUnit stores it; `display_field` is the name for people. Names
# never join (Constitution VIII): two units can share one.
_BOUNDARY_DIR = Path(os.environ.get(
    "LANDUSE_DATA_DIR", str(Path(__file__).resolve().parent.parent / "data")))


def _boundary_level(level: str) -> Dict:
    return {
        "path": str(_BOUNDARY_DIR / f"geoBoundaries-GIN-{level}.geojson"),
        "name_field": "shapeID",
        "display_field": "shapeName",
    }


ADMIN_BOUNDARY_CONFIG: Dict = {
    "source": "geoboundaries",
    "country_code": "GIN",              # Guinea ISO3
    "release": "gbOpen 9469f09",
    "adm_level": "ADM0",                # default level: the national analysis mask
    **_boundary_level("ADM0"),
    # ADM0 masks the analysis grid; it cannot answer "which prefecture".
    "levels": {level: _boundary_level(level) for level in ("ADM0", "ADM1", "ADM2", "ADM3")},
    # Levels zonal statistics are produced for. ADM0 is excluded: the national
    # total is already the headline figure and a one-row "breakdown" is noise.
    "zonal_levels": ["ADM1", "ADM2"],
}

# ============================================================================
# TEMPORAL CLASSES
# ============================================================================
# static : fetched once, reused for every snapshot year (terrain, roads...)
# annual : one clean layer per available year
# sparse : irregular availability -> snap to nearest available year
TEMPORAL_STATIC = "static"
TEMPORAL_ANNUAL = "annual"
TEMPORAL_SPARSE = "sparse"

# ============================================================================
# TARGET RASTERS (Land Use / Land Cover)  -- one categorical layer per year
# ============================================================================
# reducer 'mode' is correct for categorical LULC composites (e.g. Dynamic World
# has many scenes per year; ESRI/MODIS/GLC-FCS have one image per year but mode
# is a safe no-op there). Always select the labelled band before reducing.
TARGET_RASTERS_CONFIG: Dict[str, Dict] = {
    "ESRI_LULC": {
        "source": "GEE",
        "asset": "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS",
        "asset_type": "ImageCollection",
        "band": "b1",
        "reducer": "mode",
        "resolution_m": 10,
        "year_hint": (2017, 2025),      # discovery confirms the real list at runtime
        "temporal": TEMPORAL_ANNUAL,
        "description": "ESRI/Impact Observatory 10 m annual LULC (9-class)",
        "enabled": True,
    },
    "DYNAMIC_WORLD": {
        "source": "GEE",
        "asset": "GOOGLE/DYNAMICWORLD/V1",
        "asset_type": "ImageCollection",
        "band": "label",
        "reducer": "mode",              # annual mode of per-scene 'label'
        "resolution_m": 10,
        "year_hint": (2016, 2025),      # 2015 is partial (mission started Jun 2015)
        "temporal": TEMPORAL_ANNUAL,
        "description": "Google Dynamic World near-real-time LULC",
        "enabled": True,
    },
    "MODIS_MCD12Q1": {
        "source": "GEE",
        "asset": "MODIS/061/MCD12Q1",
        "asset_type": "ImageCollection",
        "band": "LC_Type1",             # IGBP scheme
        "reducer": "mode",
        "resolution_m": 500,
        "year_hint": (2001, 2023),
        "temporal": TEMPORAL_ANNUAL,
        "description": "MODIS Land Cover Type (annual, 500 m)",
        "enabled": True,                # coarse but full temporal depth
    },
    "GLC_FCS30D": {
        "source": "GEE",
        "asset": "projects/sat-io/open-datasets/GLC-FCS30D/annual",
        "asset_type": "ImageCollection",
        # Special case: this collection stores each year as a *band* (b1=2000 ...
        # b23=2022) within a small set of tile images, rather than one image per
        # year. The connector handles this via 'band_per_year': True.
        "band_per_year": True,
        "band_base_year": 2000,
        "reducer": "mode",
        "resolution_m": 30,
        "year_hint": (2000, 2022),
        "temporal": TEMPORAL_ANNUAL,
        "description": "GLC_FCS30D 30 m annual LULC, 35 classes (2000-2022)",
        "enabled": True,
    },
    "GLANCE": {
        "source": "GEE",
        "asset": "projects/GLANCE/DATASETS/V001",
        "asset_type": "ImageCollection",
        "band": "LC",                   # verify band name for your continent tiles
        "reducer": "mode",
        "resolution_m": 30,
        "year_hint": (2001, 2021),      # Africa extended to 2021
        "temporal": TEMPORAL_ANNUAL,
        "description": "GLanCE Landsat/CCDC 30 m annual LULC (complex tiling)",
        "enabled": False,               # off by default: continental tiling + CCDC quirks
    },
    # NOTE: the previous 'S2TSLULC' entry was removed -- it is the SAME product
    # as ESRI_LULC (ESRI_Global-LULC_10m_TS) and its 'collection' was a docs URL.
}

# Which TARGET_RASTERS_CONFIG entry the Land Use module should fetch by
# default. Set to ESRI_LULC (rather than MODIS_MCD12Q1) so the Land Use module
# uses the same, higher-resolution (10 m vs. 500 m), officially-documented
# 9-class product already cross-checked in the Deforestation module -- this
# also removes the need for the undocumented ad-hoc 14-class legend that was
# previously reclassified from MODIS outside this pipeline. Trade-off: ESRI's
# temporal depth starts 2017 (vs. MODIS 2001), but it is annual rather than
# 5-year snapshots, which is an improvement for trend-fitting.
LAND_USE_TARGET_DATASET = "ESRI_LULC"

# ============================================================================
# MODULE-SPECIFIC PREDICTOR SETS
# ============================================================================
# The Deforestation and Land Use modules do NOT need identical predictor
# sets -- they answer different questions (forest-loss risk vs. general
# land-cover transition suitability) and should each include only the
# predictors that are scientifically motivated for that question. Both draw
# from the same PREDICTOR_RASTERS_CONFIG / PREDICTOR_VECTORS_CONFIG catalog
# (so fetching stays reproducible and Azure-cached either way); they just
# select different subsets, each justified below with a literature reference.
# This replaces the earlier, unexamined assumption that "consistency" meant
# "identical predictor lists" -- consistency here means each module's set is
# independently defensible, not that the two lists must match.

LAND_USE_PREDICTOR_RASTERS = [
    # Terrain: standard, near-universal constraint on agricultural/urban/mining
    # feasibility in LULC change-allocation modelling (Verburg et al., 2004,
    # Land Use Policy -- CLUE-S framework; aspect's own effect is typically
    # weak but is retained for consistency with slope/elevation and low cost).
    # DEM_GEE (not DEM/OpenTopography): OpenTopography's API has an area limit
    # a full-country bounding box exceeds (confirmed: 400 Client Error on a
    # Guinea-wide request). DEM_GEE is the keyless GEE-hosted equivalent
    # (USGS/SRTMGL1_003, same underlying SRTM data), already used for this
    # exact reason in the Deforestation module's methodology (Section 4.1).
    "DEM_GEE", "SLOPE", "ASPECT",

    # Population density: one of the most consistently significant drivers of
    # settlement, agricultural and urban land-use change across the CA-Markov/
    # RF land-change literature (Veldkamp & Lambin, 2001, Agriculture,
    # Ecosystems & Environment).
    "POPULATION_DENSITY",

    # Precipitation: proxy for the long-run climatic gradient that separates
    # Guinea's vegetation/land-cover zones (forest -> savanna -> grassland);
    # distinct information from soil moisture below (climate signal, not
    # short/medium-term water availability) (Faye et al., 2019, generally on
    # Sahelian/Sudanian LULC climate drivers).
    "PRECIPITATION",

    # Soil moisture: direct proxy for agricultural/fallow viability, which is
    # the dominant transition family in Guinea's observed data (Fallow Land <->
    # Agriculture/Grassland/Bare Soil). More proximate causal variable than
    # precipitation alone for these transitions.
    "SOIL_MOISTURE",

    # Above-ground biomass: relevant to the vegetation-related transitions
    # present in the data (Fallow Land <-> Wooded Savanna) and needed for the
    # conservation/afforestation scenario functions, which modify AGB directly.
    "AGB_NASA",

    # NDVI (Sentinel-2, annual median, cloud-masked): added specifically to
    # address the Crops precision weakness identified in hold-out validation
    # (0.37 precision, substantial over-prediction) -- the most direct,
    # standard spectral discriminator for cropland against spectrally
    # similar classes, not otherwise captured by the existing predictor set.
    "NDVI",

    # NOTE: TEMPERATURE (LST) is deliberately EXCLUDED here. Land-surface
    # temperature over a pixel is largely a *consequence* of its land-cover
    # class (bare/urban surfaces read hotter than vegetated ones), not an
    # independent driver of which class it will transition to -- including it
    # risks circularity/endogeneity (the "predictor" partly encodes the label
    # it's meant to predict). This is a standard land-change-modelling caution
    # (Pontius & Malanson, 2005, Int. J. Geographical Information Science, on
    # avoiding predictors that are effects rather than causes of land change).
    # If a specific analytical need for LST arises later, add it back with a
    # matching justification -- don't include it by default.
]

LAND_USE_PREDICTOR_VECTORS = [
    # Roads: the single most consistently significant accessibility predictor
    # across the LULC-CA/RF literature for virtually every human-driven
    # transition (agriculture, urban, mining) (Pontius et al., 2001,
    # Agriculture, Ecosystems & Environment; also used as the primary
    # accessibility layer in Deforestation module).
    "ROADS",

    # Waterways: classic settlement/agriculture driver (proximity to water for
    # irrigation, transport, domestic use) (Serneels & Lambin, 2001, J.
    # Biogeography).
    "WATERWAYS",

    # Points of interest (density): proxy for existing development intensity,
    # a standard urban-growth-allocation predictor in CA-based models (e.g.
    # Dinamica EGO neighbourhood/attractor layers).
    "POI",

    # Protected areas (WDPA): land-use ALLOCATION CONSTRAINT rather than a
    # causal driver -- standard practice in CA-Markov/Dinamica-style models to
    # include zoning as a predictor so the model learns suppressed transition
    # probability inside protected/conserved areas (Soares-Filho et al., 2002,
    # Ecological Modelling -- Dinamica EGO).
    "PROTECTED_AREAS",

    # NOTE: RAILWAYS is deliberately EXCLUDED from the default Land Use set.
    # Guinea's rail network is a small number of dedicated bauxite-export
    # corridors, not a general accessibility layer -- it is a plausible driver
    # specifically for mining-related transitions, not for the dominant
    # observed transitions (Fallow Land/Urban/Bare Soil/Grassland cycling).
    # Add it back (with justification) if a mining-focused scenario is run.
]

# Deforestation module's predictor set is unchanged and defined/justified in
# its own methodology document (12-predictor driver model, dist_forest_edge
# included) -- listed here only so both modules' choices are visible together.
# DEFORESTATION_PREDICTOR_RASTERS = ["DEM", "SLOPE", "ASPECT", "POPULATION_DENSITY",
#     "PRECIPITATION", "TEMPERATURE", "AGB_NASA"]  # + derived dist_forest_edge
# DEFORESTATION_PREDICTOR_VECTORS = ["ROADS", "RAILWAYS", "WATERWAYS", "POI"]

# ============================================================================
# PREDICTOR RASTERS (Environmental & Socioeconomic)
# ============================================================================
PREDICTOR_RASTERS_CONFIG: Dict[str, Dict] = {
    # ---- Topography (STATIC) ----
    "DEM": {
        "source": "OpenTopography",
        "ot_demtype": "SRTMGL1",        # 30 m; also NASADEM, COP30, AW3D30...
        "resolution_m": 30,
        "reducer": None,
        "temporal": TEMPORAL_STATIC,
        "description": "SRTM 1 Arc-Second DEM",
        "enabled": True,
    },
    "DEM_GEE": {  # keyless alternative to OpenTopography, same data on GEE
        "source": "GEE",
        "asset": "USGS/SRTMGL1_003",
        "asset_type": "Image",
        "band": "elevation",
        "reducer": None,
        "resolution_m": 30,
        "temporal": TEMPORAL_STATIC,
        "description": "SRTM DEM via GEE (no API key needed)",
        "enabled": False,               # enable to skip OpenTopography entirely
                                         # -- left False here (shared config
                                         # default); Land Use enables this at
                                         # RUNTIME in its own notebook cell,
                                         # same pattern Deforestation uses, so
                                         # this shared file's default doesn't
                                         # change either module's behavior
                                         # unless that module's own notebook
                                         # explicitly opts in.
    },
    "SLOPE":  {"source": "Derived", "derived_from": "DEM", "temporal": TEMPORAL_STATIC,
               "description": "Slope (deg) from DEM", "enabled": True},
    "ASPECT": {"source": "Derived", "derived_from": "DEM", "temporal": TEMPORAL_STATIC,
               "description": "Aspect (deg) from DEM", "enabled": True},

    # ---- Population (ANNUAL) ----
    "POPULATION_DENSITY": {
        "source": "GEE",               # WorldPop is on GEE -> avoids a 2nd API
        "asset": "WorldPop/GP/100m/pop",
        "asset_type": "ImageCollection",
        "band": "population",
        "reducer": "mean",
        "resolution_m": 100,
        "year_hint": (2000, 2020),
        "temporal": TEMPORAL_ANNUAL,
        "gee_filter": {"country": "GIN"},  # WorldPop images carry a 'country' prop
        "description": "WorldPop population count (100 m, annual)",
        "enabled": True,
    },

    # ---- Climate (ANNUAL / STATIC) ----
    "PRECIPITATION": {
        "source": "GEE",
        "asset": "UCSB-CHG/CHIRPS/DAILY",
        "asset_type": "ImageCollection",
        "band": "precipitation",
        "reducer": "sum",              # annual total precip
        "resolution_m": 5566,          # ~0.05 deg
        "year_hint": (1981, 2025),
        "temporal": TEMPORAL_ANNUAL,
        "description": "CHIRPS annual precipitation",
        "enabled": True,
    },
    "TEMPERATURE": {
        "source": "GEE",
        "asset": "MODIS/061/MOD11A2",  # 8-day LST
        "asset_type": "ImageCollection",
        "band": "LST_Day_1km",
        "reducer": "mean",             # annual mean LST (scale factor 0.02 -> apply later)
        "resolution_m": 1000,
        "year_hint": (2000, 2025),
        "temporal": TEMPORAL_ANNUAL,
        "description": "MODIS land surface temperature (annual mean)",
        "enabled": True,
    },

    # ---- Biomass (SPARSE -> nearest year) ----
    "AGB_ESA": {
        "source": "GEE",
        # Verify the exact community-catalog path in your project; ESA CCI AGB is
        # published under sat-io. Years available: 2010, 2015-2021 (varies by ver).
        "asset": "projects/sat-io/open-datasets/ESA/ESA_CCI_AGB",
        "asset_type": "ImageCollection",
        "band": "AGB",
        "reducer": "mean",
        "resolution_m": 100,
        "year_hint": (2010, 2021),
        "temporal": TEMPORAL_SPARSE,
        "description": "ESA CCI Above-Ground Biomass",
        "enabled": False,              # enable after confirming the asset path
    },
    "AGB_NASA": {
        "source": "GEE",
        "asset": "NASA/ORNL/biomass_carbon_density/v1",
        "asset_type": "ImageCollection",
        "band": "agb",
        "reducer": "mean",
        "resolution_m": 300,
        "year_hint": (2010, 2010),     # single epoch
        "temporal": TEMPORAL_SPARSE,
        "description": "NASA ORNL biomass carbon density (~2010)",
        "enabled": True,
    },
    "NDVI": {
        # Added specifically to address the Crops precision weakness (0.37
        # in hold-out validation, substantial over-prediction).
        #
        # Uses MODIS MOD13Q1's PRE-COMPUTED NDVI band, not a raw-Sentinel-2
        # computation -- an earlier version of this entry computed NDVI from
        # COPERNICUS/S2_SR_HARMONIZED directly (filter+cloud-mask+compute+
        # reduce ~60-70 images/year/tile), which is genuinely heavy server-
        # side Earth Engine computation, confirmed as the direct cause of
        # repeated "User memory limit exceeded" (400) and 300s read-timeout
        # failures across most tiles/years in a real run. MOD13Q1 already
        # provides a cloud-filtered, pre-reduced 16-day NDVI composite --
        # this predictor now follows the exact same simple ImageCollection/
        # band/reducer pattern as every other dataset in this config, no
        # custom computation branch needed. Trade-off: 250 m native
        # resolution instead of Sentinel-2's 10 m -- comparable to several
        # other predictors already in this set (AGB_NASA at 300 m, SOIL_
        # MOISTURE at ~11 km), not a new departure from precedent.
        "source": "GEE",
        "asset": "MODIS/061/MOD13Q1",
        "asset_type": "ImageCollection",
        "band": "NDVI",
        "reducer": "mean",              # annual mean of the 16-day composites
        "resolution_m": 250,
        "year_hint": (2017, 2024),
        "temporal": TEMPORAL_ANNUAL,
        "description": "MODIS MOD13Q1 annual mean NDVI (16-day composites, pre-cloud-filtered)",
        "enabled": True,
    },
    "GEDI_AGB": {
        "source": "GEE",
        "asset": "LARSE/GEDI/GEDI04_A_002_MONTHLY",
        "asset_type": "ImageCollection",
        "band": "agbd",
        "reducer": "mean",
        "resolution_m": 25,
        "year_hint": (2019, 2023),
        "temporal": TEMPORAL_SPARSE,
        "description": "GEDI L4A above-ground biomass density",
        "enabled": True,
    },

    # ---- Soil moisture (ANNUAL) ----
    # Added for the Land Use module (agricultural/vegetation suitability). Was
    # previously a hand-fetched local file ("Guinea_SoilMoisture.tif") outside
    # this pipeline; now sourced the same reproducible way as every other
    # predictor. ERA5-Land has full historical depth back to 1950, so it can
    # match either MODIS (2001-) or ESRI (2017-) target windows.
    "SOIL_MOISTURE": {
        "source": "GEE",
        "asset": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
        "asset_type": "ImageCollection",
        "band": "volumetric_soil_water_layer_1",  # 0-7cm layer, m3/m3
        "reducer": "mean",              # annual mean of monthly aggregates
        "resolution_m": 11132,          # ~0.1 deg native; resampled to target grid
        "year_hint": (1981, 2025),
        "temporal": TEMPORAL_ANNUAL,
        "description": "ERA5-Land volumetric soil moisture (top layer, annual mean)",
        "enabled": True,
    },
}

# ============================================================================
# PREDICTOR VECTORS (OSM / Geofabrik)  -- STATIC
# ============================================================================
# osmnx tags. For a whole country, prefer a Geofabrik .osm.pbf extract for the
# dense layers (roads/waterways) and reserve Overpass for sparse POIs.
PREDICTOR_VECTORS_CONFIG: Dict[str, Dict] = {
    "POI":        {"source": "OSM", "tags": {"amenity": True},
                   "rasterization_mode": "heatmap",  "temporal": TEMPORAL_STATIC,
                   "enabled": True},
    "ROADS":      {"source": "OSM", "tags": {"highway": True},
                   "rasterization_mode": "distance", "temporal": TEMPORAL_STATIC,
                   "enabled": True},
    "RAILWAYS":   {"source": "OSM", "tags": {"railway": True},
                   "rasterization_mode": "distance", "temporal": TEMPORAL_STATIC,
                   "enabled": True},
    "WATERWAYS":  {"source": "OSM", "tags": {"waterway": True, "natural": "water"},
                   "rasterization_mode": "distance", "temporal": TEMPORAL_STATIC,
                   "enabled": True},
    "GEOFABRIK_URL": "https://download.geofabrik.de/africa/guinea-latest.osm.pbf",

    # Protected/conserved-area zoning (WDPA). Was previously a hand-downloaded
    # local GeoJSON in the Land Use notebook
    # ("protected_conserved_areas_wdpca_polygons (1).geojson"); now sourced from
    # the WDPA table on GEE like every other layer here, so it goes through the
    # same reproducible fetch/cache path. NOT an OSM tag layer -- fetched via
    # Protected/conserved-area zoning (WDPA). Fetched as a real GeoJSON vector
    # via api_connector.fetch_gee_feature_collection() -- a GEE FeatureCollection
    # asset, not an OSM tags-based layer, so it does NOT belong in the OSM
    # dispatch path (fetch_and_load_api_predictors branches on "source" to
    # route this correctly; see data_loader.py). rasterization_mode is
    # "binary_zone", not "zoning" -- WDPA here is a simple in/out mask (the
    # description says so), not a multi-category attribute field the general
    # "zoning" mode expects; using the wrong mode would either crash or
    # silently misinterpret the polygons.
    "PROTECTED_AREAS": {
        "source": "GEE_VECTOR",
        "asset": "WCMC/WDPA/current/polygons",
        "rasterization_mode": "binary_zone",
        "resolution_m": 30,
        "temporal": TEMPORAL_STATIC,
        "description": "WDPA protected/conserved-area zoning (binary)",
        "enabled": True,
    },
}

# ============================================================================
# MULTITEMPORAL / UPDATE POLICY
# ============================================================================
MULTITEMPORAL_CONFIG: Dict = {
    # If None, the connector discovers ALL available years from each source and
    # fetches one layer per year. Set a list (e.g. [2018, 2020]) to restrict.
    "years": None,
    # Discovery clamps to this window to avoid absurd ranges.
    "min_year": 2000,
    "max_year": 2026,
    # Incremental update: skip a (dataset, year) if its file already exists in
    # cache/Blob. Set force_refresh=True to re-fetch everything.
    "force_refresh": False,
    # How to handle sparse layers when an exact year is missing.
    "sparse_strategy": "nearest",   # 'nearest' | 'skip'
}

# ============================================================================
# MISSING VALUE HANDLING  (unchanged)
# ============================================================================
MISSING_VALUE_CONFIG: Dict = {
    "nodata_value": -9999,
    "invalid_raster_values": [254, 255, -1],
    "handling_strategies": {
        "predictors":   {"method": "mean",    "description": "Local mean imputation"},
        "targets":      {"method": "keep",    "description": "Keep nodata in LULC"},
        "derived":      {"method": "mean",    "description": "Local mean imputation"},
        "vector_based": {"method": "mask", "fill_value": 0,
                         "description": "0 = no proximity/heatmap"},
        "training":     {"method": "exclude", "min_valid_predictors": "all",
                         "description": "Drop samples with any missing predictor"},
    },
    "quality_thresholds": {"max_missing_percent": 50, "min_valid_samples": 100},
    "report_stats": True,
}

# ============================================================================
# SAMPLING / MODEL  (unchanged)
# ============================================================================
SAMPLING_CONFIG: Dict = {
    "samples_per_class": 2000, "stratified_sampling": True, "random_seed": 42,
    "train_test_split": 0.8, "cross_validation_folds": 5,
}

# ============================================================================
# TRANSITION SIGNIFICANCE THRESHOLD (Land Use module)
# ============================================================================
# The Deforestation module uses MIN_PIXELS = 200 on its own ~250 m rasters.
# The Land Use module does NOT need to match that number -- the two modules
# ask different questions and can each justify their own threshold. What
# actually matters here is having ENOUGH TRAINING SAMPLES to fit a stable,
# reliable Random Forest classifier per transition, which is a statistical
# criterion, not an ecological area threshold.
#
# Rationale for keeping >= 1,000 changed pixels as the Land Use cutoff:
# each Patcher/Expander RF is trained on a BALANCED sample (positive = min(
# n_positive, n_negative), see training.create_transition_dataset), then
# further split for train/test/CV (SAMPLING_CONFIG: 80/20 split, 5-fold CV).
# General guidance on minimum per-class sample size for stable Random Forest
# probability estimates recommends at least several hundred, and preferably
# >= 1,000, samples per class so that both the held-out test fold and each
# CV fold retain enough positive examples to estimate accuracy without high
# variance (Van Rijn & Hutter, 2018, on RF sample-size sensitivity; also the
# empirical pattern already observed in Table 4: the one transition in this
# dataset below ~100 samples, Bare Soil -> Fallow Land at 94 changed pixels,
# is exactly the one flagged "Moderate" rather than "High" reliability).
# 1,000 pixels is therefore a defensible floor for THIS module independent of
# what the Deforestation module uses on a different resolution and a
# different kind of target variable (binary change/no-change vs. multi-class
# transition).
MIN_CHANGE_PIXELS_LAND_USE = 1000
MODEL_CONFIG: Dict = {
    "model_type": "RandomForest", "n_estimators": 100, "max_depth": 20,
    "min_samples_split": 5, "min_samples_leaf": 2, "random_state": 42, "n_jobs": -1,
}

# ============================================================================
# HELPERS
# ============================================================================
def get_enabled_target_rasters() -> List[str]:
    return [k for k, v in TARGET_RASTERS_CONFIG.items() if v.get("enabled", True)]


def get_enabled_predictor_rasters(keys: Optional[List[str]] = None) -> List[str]:
    """
    All enabled predictor rasters, or -- if `keys` is given -- only the
    requested subset (still filtered to those actually enabled). Lets a
    module select its own predictor set (e.g. LAND_USE_PREDICTOR_RASTERS)
    without changing which layers are globally enabled for other modules.
    """
    enabled = [k for k, v in PREDICTOR_RASTERS_CONFIG.items() if v.get("enabled", True)]
    if keys is None:
        return enabled
    return [k for k in keys if k in enabled]


def get_enabled_predictor_vectors(keys: Optional[List[str]] = None) -> List[str]:
    """Same pattern as get_enabled_predictor_rasters, for vector predictors."""
    enabled = [k for k, v in PREDICTOR_VECTORS_CONFIG.items()
               if isinstance(v, dict) and v.get("enabled", True)]
    if keys is None:
        return enabled
    return [k for k in keys if k in enabled]


def ensure_directories() -> None:
    for d in (PROCESSING_CONFIG["temp_dir"], PROCESSING_CONFIG["cache_dir"],
              PROCESSING_CONFIG["output_dir"]):
        os.makedirs(d, exist_ok=True)


def get_missing_value_strategy(layer_type: str) -> Dict:
    return MISSING_VALUE_CONFIG["handling_strategies"].get(
        layer_type, MISSING_VALUE_CONFIG["handling_strategies"]["predictors"])


def validate_required_keys() -> List[str]:
    """Return a list of human-readable problems for any missing required secret."""
    problems = []
    for name, spec in API_CREDENTIALS.items():
        if spec.get("required") and not get_api_key(name):
            problems.append(f"Missing {spec['env']} ({name}). {spec['how']}")
    if not PROCESSING_CONFIG["gee_project_id"]:
        problems.append("Missing GEE_PROJECT_ID.")
    return problems