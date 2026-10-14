"""
Deforestation module configuration.

Extends the Land Use platform's dataset catalog with forest-specific, READY-MADE
products so nothing has to be recomputed from raw Sentinel:

  * Hansen Global Forest Change  -> authoritative annual forest LOSS (primary)
  * JRC Tropical Moist Forest    -> deforestation/degradation (moist-forest zone)
  * MODIS NDVI (MOD13Q1)         -> vegetation trend / degradation
  * MODIS burned area (MCD64A1)  -> fire driver

Everything here plugs into the SAME api_connector / data_loader used by the Land
Use module, so fetching, alignment and Blob caching are identical. Call
`register()` once at startup to add the annual datasets to the main catalogs;
Hansen (a single multi-band image) has a dedicated fetcher in forest.py.
"""

from .. import config as base

# ---------------------------------------------------------------------------
# Hansen Global Forest Change  (single multi-band Image; version is annual)
# ---------------------------------------------------------------------------
HANSEN_GFC = {
    "source": "GEE",
    "asset": "UMD/hansen/global_forest_change_2025_v1_13",   # override per release
    # UMD/hansen/global_forest_change_2024_v1_12
    "asset_type": "Image",
    "bands": ["treecover2000", "lossyear", "gain", "datamask"],
    "lossyear_base": 2000,          # lossyear value n  ->  calendar year 2000 + n
    "resolution_m": 30,
    "temporal": "static",           # one image; decoded to per-year masks locally
    "description": "Hansen GFC: canopy-2000 (%) + annual loss year (2001-2024)",
    "enabled": True,
}

# ---------------------------------------------------------------------------
# JRC Tropical Moist Forest (complement; covers the moist-forest zone only)
# ---------------------------------------------------------------------------
JRC_TMF_DEFOR = {
    "source": "GEE",
    "asset": "projects/JRC/TMF/v1_2025/DeforestationYear",
    "asset_type": "ImageCollection",   # mosaic -> single band 'DeforestationYear'
    "band": "DeforestationYear",
    "resolution_m": 30,
    "temporal": "static",
    "description": "JRC TMF deforestation year (tropical MOIST forest only)",
    "enabled": False,               # Guinea: only SE (Guinee Forestiere) is covered
}

# ---------------------------------------------------------------------------
# MODIS vegetation index (degradation / trend) -- ready-made, no compositing
# ---------------------------------------------------------------------------
MODIS_NDVI = {
    "source": "GEE",
    "asset": "MODIS/061/MOD13Q1",
    "asset_type": "ImageCollection",
    "band": "NDVI",                 # EVI band also available; both scaled 1e-4
    "scale_factor": 0.0001,
    "reducer": "max",               # annual peak greenness (robust to clouds)
    "resolution_m": 250,
    "temporal": "annual",
    "year_hint": (2000, 2025),
    "description": "MODIS 16-day NDVI, annual peak (degradation & trend)",
    "enabled": False,   # CONFIRMED REJECTED as a risk-model predictor: ablation +
                        # rolling-origin hold-out gave AUC 0.479 (below chance),
                        # Section 6.5. Previously enabled=True here, which meant it
                        # was fetched into shared_predictors for the MAIN pipeline
                        # run too (build_driver_stack does stack=dict(shared_predictors)),
                        # silently including a rejected predictor in the adopted,
                        # deployed model -- confirmed via the real driver-importance
                        # output showing MODIS_NDVI_2024.tif present with weight 0.014.
                        # Only re-enable for a dedicated candidate-predictor test cell,
                        # never for the standard fetch that feeds the adopted model.
}

# ---------------------------------------------------------------------------
# MODIS burned area (fire driver)
# ---------------------------------------------------------------------------
MODIS_BURNED = {
    "source": "GEE",
    "asset": "MODIS/061/MCD64A1",
    "asset_type": "ImageCollection",
    "band": "BurnDate",
    "reducer": "max",               # any burn in the year -> value > 0
    "resolution_m": 500,
    "temporal": "annual",
    "year_hint": (2000, 2025),
    "description": "MODIS burned area (fire driver of deforestation)",
    "enabled": False,   # CONFIRMED REJECTED as a risk-model predictor: ablation +
                        # rolling-origin hold-out gave AUC 0.496 (below chance),
                        # Section 6.5. Same silent-inclusion issue as MODIS_NDVI above --
                        # confirmed present in the real driver-importance output as
                        # MODIS_BURNED_2024.tif with weight 0.004. Only re-enable for a
                        # dedicated candidate-predictor test cell.
}

# ---------------------------------------------------------------------------
# Which LULC class == forest/trees, per product (for LULC-based forest masks)
# ---------------------------------------------------------------------------
FOREST_CLASSES = {
    "ESRI_LULC":     [2],                    # Trees
    "DYNAMIC_WORLD": [1],                    # Trees
    "MODIS_MCD12Q1": [1, 2, 3, 4, 5],        # IGBP forest classes (ENF..MF)
    "GLC_FCS30D":    list(range(51, 92)),    # broadleaf/needleleaf forest codes
    "ESA_WORLDCOVER": [10],                  # Tree cover
}

# ---------------------------------------------------------------------------
# Method parameters
# ---------------------------------------------------------------------------
DEFOR_PARAMS = {
    "canopy_threshold_pct": 30,     # Hansen treecover -> forest (FAO=10, GFW=30)
    "forest_source": "hansen",      # 'hansen' | 'lulc' | 'jrc_tmf'
    "vi_degradation_drop": 0.10,    # NDVI decline within forest -> degradation
    "min_mapping_unit_px": 0,       # optional speckle removal (0 = off)
    "driver_model": "random_forest",  # 'random_forest' | 'logistic'
    "samples_per_class": 5000,      # for the deforestation-risk model
    "analysis_years": None,         # None -> discover from the loss product
    "min_pixels_transition": 200,   # minimum changed pixels to fit a per-transition
                                     # suitability model. Deliberately lower than the
                                     # companion land-cover module's 1000-pixel floor:
                                     # this module targets forest-loss transitions
                                     # specifically, several of which are individually
                                     # rarer than the land-cover module's general
                                     # class-to-class transitions, so a stricter floor
                                     # would exclude real but less-common deforestation
                                     # pathways rather than just noise. Both floors sit
                                     # within the range where RF land-cover classifiers
                                     # are shown to plateau in accuracy (~750 px/class
                                     # in a controlled comparison; performance keeps
                                     # improving up to that point and is markedly less
                                     # stable well below it) -- Thanh Noi & Kappas
                                     # (2018), "Comparison of Random Forest, k-Nearest
                                     # Neighbor, and Support Vector Machine Classifiers
                                     # for Land Cover Classification Using Sentinel-2
                                     # Imagery," Sensors, 18(1), 18. Neither 200 nor
                                     # 1000 is "the" literature-mandated number; both
                                     # are documented, deliberate points on the same
                                     # bias-variance tradeoff (more transitions covered
                                     # vs. more stable per-model estimates), chosen to
                                     # fit each module's specific transition structure.
}


# --- Candidate driver: night-time lights (economic-activity proxy) ---------
# VIIRS DNB monthly composite, cloud-free radiance. Added as a config entry
# here (not just a notebook cell) so it is fetched the same way as every
# other annual predictor via register() below. NOT yet tested through the
# ablation + rolling-origin hold-out protocol (Methodology Section 9.1) --
# do not treat as adopted until it passes that test the way the existing 11
# predictors did (dist_forest_edge failed this test and was excluded; NDVI
# and burned area also failed it and were excluded -- see Section 6.9).
VIIRS_NIGHTLIGHTS = {
    "asset": "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG",
    "band": "avg_rad",
    "resolution_m": 500,        # true native resolution (~15 arc-sec); without this
                                # it inherited the analysis grid's resolution instead,
                                # oversampling ~2-5x beyond real data and contributing
                                # to exceeding GEE's 50MB per-request cap even tiled 2x1
    "temporal": "annual",     # annual composite (mean of monthly bands)
    "enabled": True,
    "reducer": "mean",
    "description": "VIIRS night-time lights, economic-activity proxy",
}

# --- Candidate driver: commodity prices (economic driver proxy) ------------
# WFP Global Food Prices Database (VAM), Guinea subset. This is a TABULAR,
# market-point dataset, not a native raster -- it requires interpolation
# (e.g. inverse-distance-weighting or nearest-market assignment) to become a
# predictor surface, which is NOT yet implemented; this entry documents the
# access path so a developer does not have to re-discover it.
#   Direct bulk download (no auth required):
#   https://data.humdata.org/dataset/4fdcd4dc-5c2f-43af-a1e4-93c9b6539a27/
#   resource/12d7c8e3-eff9-4db0-93b7-726825c4fe9a/download/wfpvam_foodprices.csv
#   Filter to adm0_name == "Guinea". Columns include market name, lat/lon,
#   commodity, date (monthly), price, currency, unit. Build a raster by
#   rasterizing market points per commodity/date and interpolating (IDW is a
#   reasonable default given typically sparse, unevenly distributed markets).
WFP_PRICES_SOURCE = {
    "hdx_dataset": "wfp-food-prices-for-guinea",   # Guinea-specific resource, not the 98-country global file
    "hdx_resource_id": "f8e11cd4-630f-44be-afa0-3e7baa5744ca",
    "format": "csv",
    "requires_interpolation": True,
    "fetch_fn": "api_connector.fetch_wfp_prices",           # implemented, resolves URL via CKAN API
    "rasterize_fn": "data_loader.rasterize_market_points",  # implemented, IDW
    "confirmed_columns": ["date", "admin1", "admin2", "market", "market_id", "latitude",
                          "longitude", "category", "commodity", "commodity_id", "unit",
                          "priceflag", "pricetype", "currency", "price", "usdprice"],
    "commodity": "Oil (palm)",   # confirmed via ablation + rolling-origin hold-out: improved AUC
                                 # on 3/3 historical splits (+0.0081, +0.0063, +0.0021 -- same
                                 # improves-more-for-recent-splits pattern as the dist_forest_edge
                                 # fix), the same bar that fix had to clear before being adopted.
                                 # Chosen originally for a literature reason (palm oil expansion is
                                 # a well-documented direct driver of tropical deforestation), not
                                 # cherry-picked from results -- see Methodology Section 9.1/6.9.
    "reference_year": 2018,     # single fixed snapshot, matching exactly what was tested and
                                 # validated above. Period-specific pricing (nearest-available-year
                                 # per calibration/test window, same pattern as population -- see
                                 # data_loader.fetch_period_specific_predictor) is a natural future
                                 # enhancement but was NOT what was tested; do not silently upgrade
                                 # to period-specific without re-validating through the same protocol.
    "enabled": True,   # ADOPTED -- confirmed stable across all 3 historical hold-out splits.
}


# --- Candidate driver: mining concessions (Guinea mining cadastre) ---------
# Portal: https://guinee.cadastreminier.org/en/ (FlexiCadastre / Spatial
# Dimension platform, same vendor as Cote d'Ivoire, Kenya, Togo, Namibia).
# The underlying ArcGIS Server instance (cadastre.mines.gov.gn/arcgis/rest/
# services/MapPortal/GuineaPortal_ApplicationLayer/MapServer) is real and
# was located, but the permit-boundary layer itself sits behind a login --
# it is not openly queryable the way the top-level services directory is.
# Practical access paths, in order of preference: (1) request the dataset
# directly from the Ministry of Mines and Geology / Centre for Mining
# Promotion and Development (CPDM), the legitimate route for official data;
# (2) use OSM/Geofabrik landuse=quarry and related mining-tagged features
# (already fetched via the existing OSM vector pipeline) as an incomplete
# interim proxy while (1) is pursued. Do not scrape the authenticated portal.
MINING_CADASTRE_SOURCE = {
    "portal_url": "https://guinee.cadastreminier.org/en/",
    "arcgis_root": "https://cadastre.mines.gov.gn/arcgis/rest/services",
    "status": "requires_authentication",
    "interim_proxy": "OSM landuse=quarry via existing Geofabrik vector fetch",
    "enabled": False,   # not yet accessible -- see docstring above
}


def register():
    """
    Add the annual deforestation datasets to the main config catalogs so the
    existing fetch/align/cache pipeline handles them with no extra plumbing.
    Idempotent. Hansen and JRC (static, specially decoded) are fetched directly
    by forest.py, not through the generic annual fetcher.

    CRITICAL -- call this BEFORE the land-use/land-cover module's predictor
    fetch (data_loader.fetch_and_load_api_predictors), not after. This was
    confirmed as the reason MODIS_NDVI and MODIS_BURNED, despite being
    enabled=True by default, were never actually fetched in earlier runs:
    register() was being called from within the deforestation module's own
    import (Section 8 of the notebook), which happens AFTER the land-use
    predictor fetch (Section 3) had already completed using the config as it
    stood at that time. Import `config_-_deforestation` and call register()
    at the very top of the notebook/pipeline entry point, before any predictor
    fetching, so VIIRS_NIGHTLIGHTS does not silently repeat the same bug.
    """
    base.PREDICTOR_RASTERS_CONFIG.setdefault("MODIS_NDVI", MODIS_NDVI)
    base.PREDICTOR_RASTERS_CONFIG.setdefault("MODIS_BURNED", MODIS_BURNED)
    base.PREDICTOR_RASTERS_CONFIG.setdefault("VIIRS_NIGHTLIGHTS", VIIRS_NIGHTLIGHTS)
    # Expose forest-class lookup + params on the base config for other modules.
    base.FOREST_CLASSES = FOREST_CLASSES
    base.DEFOR_PARAMS = DEFOR_PARAMS