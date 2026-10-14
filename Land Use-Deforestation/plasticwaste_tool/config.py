"""
Configuration for the Digital Plastic Waste Management System (DPWMS) —
Mesoscale Hotspot Mapping module.

Alignment note
--------------
This package is a SEPARATE module/screen from `landuse_tool` (per the GEDH SRS,
Module 5 "Plastic Waste Section" is a distinct sidebar section from Module 4
"Land Use and Deforestation Section" — GEDH-FR-0401 vs GEDH-FR-0501). It is
therefore its own top-level package, not a sub-package registered into
`landuse_tool`'s catalog (unlike `deforestation_tool`, which *is* nested inside
`landuse_tool` because those two share one screen/map).

However, it reuses the SAME shared infrastructure the GEDH hub architecture
calls for (HUB-M2 "Shared Geospatial Database", one Azure Blob Storage account,
one GEE service account, one credential-loading path):
  - `landuse_tool.azure_utils`      -> Key Vault secrets + BlobCache
  - `landuse_tool.api_connector`    -> GEE service-account init, Guinea boundary
  - `landuse_tool.utils`            -> raster alignment / missing-value helpers
  - `landuse_tool.deforestation_tool.validation` -> confusion matrix + spatial-block CV

Environment variables are identical to landuse_tool's (GEE_PROJECT_ID,
GEE_SERVICE_ACCOUNT, GEE_SA_KEY, AZURE_STORAGE_CONNECTION_STRING, ...); only the
Blob prefix changes so the two modules' cached files don't collide inside the
same container.
"""

import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# SHARED CREDENTIALS BOOTSTRAP
# ============================================================================
# This package has NO credentials of its own — it authenticates GEE and Azure
# Blob through landuse_tool.api_connector / landuse_tool.azure_utils, using
# the exact same env vars (GEE_PROJECT_ID, GEE_SERVICE_ACCOUNT, GEE_SA_KEY,
# AZURE_STORAGE_CONNECTION_STRING, ...). The only failure mode is those env
# vars not being in os.environ yet when this module is imported.
#
# Real layout:
#   Land Use-Deforestation/
#     .env
#     plasticwaste_analysis.ipynb
#     plasticwaste_tool/   <- this file lives at .../plasticwaste_tool/config.py
#     landuse_tool/
#
# `landuse_tool.azure_utils._hydrate_local_env()` finds a local `.env` via
# `find_dotenv(usecwd=True)`, which searches the CURRENT WORKING DIRECTORY
# upward — that alone is enough whenever the notebook's cwd is
# `Land Use-Deforestation/` (where `.env` actually lives) or below it. The
# extra candidate below is just a safety net for any other cwd (e.g. running
# a script from inside `plasticwaste_tool/` directly), computed relative to
# THIS file rather than assuming a particular launch directory.
def _bootstrap_shared_env() -> None:
    try:
        from dotenv import load_dotenv, find_dotenv
    except ImportError:
        return  # python-dotenv not installed; env vars must already be set

    candidates = []

    # 1. Normal upward search from the current working directory (covers the
    #    case where the notebook IS run from inside landuse_tool/, or from a
    #    shared project root above both landuse_tool/ and plasticwaste/).
    found = find_dotenv(usecwd=True)
    if found:
        candidates.append(found)

    # 2. Explicit fallback: <project_root>/.env, computed relative to THIS
    #    file, where <project_root> is the folder plasticwaste_tool/ lives
    #    directly inside (Land Use-Deforestation/plasticwaste_tool/config.py
    #    -> Land Use-Deforestation/.env). Also check the landuse_tool/ sibling
    #    in case .env ever moves there instead.
    here = os.path.dirname(os.path.abspath(__file__))       # .../Land Use-Deforestation/plasticwaste_tool
    project_root = os.path.dirname(here)                    # .../Land Use-Deforestation
    candidates.append(os.path.join(project_root, ".env"))
    candidates.append(os.path.join(project_root, "landuse_tool", ".env"))

    for path in candidates:
        if path and os.path.exists(path):
            load_dotenv(path, override=False)  # never clobber already-set vars
            logger.info("plasticwaste_tool: loaded shared .env from %s", path)
            return


_bootstrap_shared_env()

# ============================================================================
# ENVIRONMENT / PATHS
# ============================================================================
_SCRATCH = os.environ.get("APP_SCRATCH_DIR", "/tmp/plasticwaste")

PROCESSING_CONFIG: Dict = {
    "target_crs": os.environ.get("TARGET_CRS", "EPSG:4326"),

    "temp_dir": os.path.join(_SCRATCH, "temp"),
    "cache_dir": os.path.join(_SCRATCH, "cache"),
    "output_dir": os.path.join(_SCRATCH, "processed"),

    # Same storage ACCOUNT as landuse_tool (shared geospatial DB / HUB-M2), but
    # its own prefix/container namespace so files never collide.
    "azure_container": os.environ.get("AZURE_BLOB_CONTAINER_PLASTIC",
                                      os.environ.get("AZURE_BLOB_CONTAINER")),
    "azure_prefix": os.environ.get("AZURE_BLOB_PREFIX_PLASTIC", "plasticwaste"),
    "azure_connection_string": os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
    "azure_account_url": os.environ.get("AZURE_STORAGE_ACCOUNT_URL"),

    # NOTE: no gee_project_id / gee_service_account / gee_sa_key_secret here.
    # GEE auth is NOT duplicated in this module — api_connector.initialize_gee()
    # calls landuse_tool.api_connector.initialize_gee() directly, which reads
    # landuse_tool.config.PROCESSING_CONFIG (the single source of truth for
    # GEE credentials). Keeping a second copy here was dead code that could
    # drift out of sync; removed.
}

# ============================================================================
# STUDY AREA — Guinea coastal + riverine plastic accumulation zone
# ============================================================================
# BBox: Conakry peninsula, Konkoure / Fatala river mouths, mangrove fringe,
# Atlantic coastal accumulation zones. Named sub-AOIs mirror the GEE App
# prototype's AOI selector (`getAOI` in plasticwaste_analysis.js).
# Regional AOIs. Two properties matter and neither is automatic:
#
#   WATER SHARE. Each region's detection threshold is mean+3*sigma over its own
#   water pixels, so a box that is mostly land yields a small, heterogeneous
#   population, an inflated threshold, and an uninformative nil return. Measured
#   water shares: conakry 73.4%, konkoure 55.2%, mangrove 33.2%, and the ORIGINAL
#   fatala box (-14.2, 10.1, -13.6, 10.6) just 2.5% -- which is why it detected
#   nothing. Check a candidate with deployment.water_fraction_for_bbox() before
#   committing it.
#
#   NON-OVERLAP. Regions are run independently and mosaicked, so a patch inside
#   two boxes is detected twice and the national count inflates. conakry and
#   mangrove overlapped by ~980 km2 and double-counted 8 patches in the first
#   national run. pipeline.run_national now de-duplicates and warns, but the
#   correct fix is boxes that do not overlap.
AOI_CONFIG: Dict = {
    "full":     {"bbox": (-15.5, 8.8, -13.0, 10.8), "label_fr": "Guinee - zone cotiere complete"},
    "conakry":  {"bbox": (-13.9, 9.3, -13.4, 9.7),  "label_fr": "Presqu'ile de Conakry"},
    "konkoure": {"bbox": (-14.8, 9.7, -14.2, 10.3), "label_fr": "Embouchure du Konkoure"},
    # Retightened onto the Fatala mouth at Boffa. Measured 16.4% water / 249 km2
    # against 2.5% / 93 km2 for the original box, and no overlap with konkoure.
    # Still the least water-rich region, so treat its counts with more caution
    # than Conakry's.
    "fatala":   {"bbox": (-14.2, 10.0, -13.85, 10.35), "label_fr": "Embouchure de la Fatala"},
    # MISNAMED AND MISPLACED — do not report this as a mangrove result until the
    # box is redrawn. Measured on the 2024-01 national run: 364 of its 381
    # detections lie MORE THAN 10 km offshore (median 17.2 km, max 27.0 km), and
    # they attribute to Conakry prefecture rather than any southern coastal one.
    # A mangrove fringe is by definition at the shoreline, so these are open-water
    # detections in the Atlantic, not mangrove-fringe detections. Their likely
    # confounders are sun glint, whitecaps, Sargassum rafts and plume fronts —
    # NOT the mangrove leaf litter originally assumed. The western third of the
    # box also produced nothing at all, consistent with ESA WorldCover carrying
    # no permanent-water class over open ocean.
    "mangrove": {"bbox": (-14.5, 9.0, -13.5, 9.5),  "label_fr": "Frange de mangrove (sud)"},
}
DEFAULT_AOI_KEY = "full"

# ============================================================================
# SENTINEL-2 SOURCE
# ============================================================================
S2_CONFIG: Dict = {
    "collection": "COPERNICUS/S2_SR_HARMONIZED",
    "cloud_property": "CLOUDY_PIXEL_PERCENTAGE",
    "cloud_threshold_pct": float(os.environ.get("PW_CLOUD_THRESHOLD", 70)),
    # Not-vegetated (5, SCL's bare-soil/built-up class) is deliberately excluded:
    # keeping it let urban rooftops/pavement/bare land pass the cloud mask, and
    # since apply_spectral_masks() only strips out vegetation/water (not "land"
    # in general), bare/built surfaces then got FDI-thresholded exactly like
    # open water -- producing a scatter of false hotspots across dry land
    # (e.g. the Conakry peninsula's built-up footprint) instead of a
    # water-restricted floating-debris layer.
    "scl_clear_classes": [4, 6, 11],   # Vegetation, Water, Snow
    "scl_exclude_classes": [2, 3, 5, 7, 8, 9, 10],
    "bands": {"B3": "B3", "B4": "B4", "B6": "B6", "B8": "B8", "B8A": "B8A",
              "B11": "B11", "B12": "B12"},
    # Central wavelengths (nm), used in the FDI formula (Biermann et al. 2020,
    # eq. 2): NIR = B8, RE2 = B6, SWIR1 = B11, and the RED wavelength (B4) in
    # the interpolation ratio. Sentinel-2A values.
    #
    # Before fix/fdi-bands-and-esri-confirm the index used B8A as RE2 and
    # lambda_B8A in the ratio. B8A (864.7 nm) lies ABOVE B8, so that baseline
    # was extrapolated rather than interpolated and the ratio term was -0.43
    # instead of +1.77: a different index, not Biermann's. B8A stays in the
    # band list only for the MARIDA domain-shift comparison.
    "lambda_nm": {"B4": 664.6, "B6": 740.5, "B8": 832.8, "B11": 1613.7},
    # Part of every FDI cache key (api_connector.aoi_cache_token), so a raster
    # cached under an earlier formula is never reused. Change it whenever the
    # index definition changes.
    "fdi_formula": "biermann2020-b6",
    "default_start_date": "2024-01-01",
    "default_end_date": "2024-01-31",
    "native_resolution_m": 10,
    "download_scale_m": int(os.environ.get("PW_DOWNLOAD_SCALE_M", 20)),
}

# --- SIZE_CLASS_CONFIG (written by 37_size_class_breaks.py; do not edit by hand) ---
# Floating-debris size classes (feature 006, research R7; doc 70 S46): small
# area_m2 < b1, medium b1 <= area_m2 < b2, large area_m2 >= b2. The publish
# step classifies patches with it and stores it on the version.
# DEVELOPMENT PLACEHOLDER: the fixture breaks [800, 4000] m2, not computed
# from the reference run. Replace by running 37_size_class_breaks.py on the
# January 2024 national run; the plastic lead approves the values (T087).
SIZE_CLASS_CONFIG: Dict = {
    "breaks_m2": [800, 4000],
    "method": "fisher-jenks",
    "k": 3,
    "reference": "DEVELOPMENT PLACEHOLDER (fixture values, pending T087)",
    "n": None,
    "pixel_m2": 400,
    "computed_at": None,
}
# --- end SIZE_CLASS_CONFIG ---

# ============================================================================
# SPECTRAL / DETECTION PARAMETERS
# ============================================================================
# Scientific basis:
#   FDI  - Biermann, L., Clewley, D., Martinez-Vicente, V. & Topouzelis, K.
#          (2020). "Finding Plastic Patches in Coastal Waters using Optical
#          Satellite Data." Scientific Reports 10, 5364.
#   NDVI/NDWI masks - standard vegetation/water exclusion used by Biermann et
#          al. (2020) and Themistocleous et al. (2020) to isolate the residual
#          floating-debris spectral signature from vegetation and open water.
#   Otsu (1979) thresholding - unsupervised bimodal histogram split, applied
#          to the masked FDI raster per scene (as in the GEE prototype and in
#          Topouzelis et al. 2019/2020 mesoscale plastic-detection workflows).
DETECTION_CONFIG: Dict = {
    "ndvi_threshold": float(os.environ.get("PW_NDVI_THRESHOLD", 0.15)),  # exclude if >
    "ndwi_threshold": float(os.environ.get("PW_NDWI_THRESHOLD", 0.0)),  # exclude if >
    # 'de_africa' (default) | 'legacy'.
    #   de_africa : mask to water, then require BOTH FDI and NDVI to exceed
    #               mean + n*sigma of the water population. Ported from Digital
    #               Earth Africa's operational floating-debris workflow -- see
    #               detection.debris_binary_mask_de_africa, including its scope
    #               note (this detects floating debris, not plastic specifically).
    #   legacy    : the original NDVI/NDWI exclusion masks + a single FDI
    #               threshold, kept to reproduce earlier runs. Its NDVI > 0.15
    #               vegetation exclusion has no recorded source and is the
    #               opposite sign of both DE Africa and Biermann et al., who use
    #               NDVI inclusively as a debris indicator.
    "detection_rule": os.environ.get("PW_DETECTION_RULE", "de_africa"),
    # The de_africa rule's detections are then filtered by a classifier trained
    # on MARIDA (fdi_tool/debris_filter.py): a pixel is kept where
    # P(Marine Debris) >= p_min. With the FDI as Biermann et al. (2020) define
    # it, the rule alone flags vegetation and Sargassum; Biermann et al.
    # separate those with a trained classifier, which this is. p_min 0.5 is the
    # binary classifier's decision boundary, not tuned to Guinea. Measured on
    # MARIDA as deployed: fdi_tool/models/debris_filter_meta.json.
    "debris_filter": {
        "enabled": os.environ.get("PW_DEBRIS_FILTER", "1") == "1",
        "model": "debris_filter.joblib",
        "p_min": 0.5,
    },
    # Applies to the 'legacy' rule only: 'sigma' | 'otsu' | 'manual'. See
    # detection.threshold_with_fallback for why Otsu is not appropriate at
    # whole-AOI scope.
    "threshold_method": os.environ.get("PW_THRESHOLD_METHOD", "sigma"),
    "threshold_n_sigma": float(os.environ.get("PW_THRESHOLD_N_SIGMA", 3.0)),
    # No cited source: this is the original GEE prototype's engineering fallback
    # ("fallback if Otsu fails on sparse scenes"), retained for reproducibility
    # only. Do not present it as a literature value.
    "fdi_manual_threshold": float(os.environ.get("PW_FDI_MANUAL_THRESHOLD", 0.005)),
    "otsu_histogram_bins": 256,
    "otsu_scale_m": 100,            # coarser scale for the histogram search (efficiency)
    # Scale at which mean+n*sigma is computed server-side. MUST match the scale
    # the threshold is applied at (download_scale_m), which is why it does not
    # reuse otsu_scale_m. Coarsening is harmless for Otsu -- it only reshapes a
    # histogram whose split point is a shape property -- but mean+n*sigma is a
    # moment estimate: aggregating 25 pixels into one 100 m pixel averages away
    # within-cell variance, shrinking sigma and pushing the threshold DOWN, so
    # the detector would over-flag at 20 m against statistics gathered at 100 m.
    # None -> fall back to S2_CONFIG["download_scale_m"].
    "sigma_scale_m": None,
    # A region whose AOI is mostly land gives mean+n*sigma a small, heterogeneous
    # water population: sigma inflates, the threshold rises, and the region can
    # return zero patches for want of sensitivity rather than want of debris.
    # Measured on the first national run, Fatala covered 2.2% water against
    # 33-75% for the other three regions, produced a threshold 3.5x Conakry's,
    # and detected nothing. Below this fraction a region's result is reported as
    # NOT ASSESSED rather than as a count.
    "min_water_fraction": float(os.environ.get("PW_MIN_WATER_FRACTION", 0.10)),
    # ABSOLUTE floor on the water population, applied ALONGSIDE the fraction
    # above. Added when the FDI was extended upstream, because fraction turns
    # out to be the wrong statistic for a river.
    #
    # What makes a mean + 3 sigma threshold trustworthy is the NUMBER of pixels
    # behind it: the relative standard error of a standard deviation is about
    # 1/sqrt(2n), so n = 100 gives 7%, n = 500 gives 3.2% and n = 1000 gives
    # 2.2%. Fraction only ever served as a proxy for count on a compact coastal
    # box, where the two move together.
    #
    # On a river they come apart. HydroRIVERS centrelines carry a few hundred
    # metres of positional error (HydroSHEDS 15 arc-sec), so a corridor has to
    # be ~250 m either side to be sure of containing the channel; a 36 m river
    # inside a 500 m corridor is about 7% water and fails the fraction test
    # while carrying hundreds of thousands of water pixels -- an ample
    # population. Judging that reach "not assessed" would be wrong.
    #
    # 1000 is chosen for the 2.2% figure above: the threshold is then stable to
    # well within the margin by which patches exceed it. A region failing
    # EITHER guard is reported as NOT ASSESSED.
    "min_water_px": int(os.environ.get("PW_MIN_WATER_PX", 1000)),
    # Port / harbour exclusion. Structures built OVER water -- quays, jetties,
    # moored vessels -- are correctly classified as water by every water product,
    # so no water mask removes them. They were 6 of 20 confirmed false positives
    # on Conakry, and the class grows with AOI extent (Kamsar, Benty). Sourced
    # from OpenStreetMap rather than hand-drawn so it is reproducible and
    # updatable. Patches are FLAGGED by default, not deleted: an operator can
    # audit what was suppressed, which a hard mask does not allow.
    "port_osm_tags": {
        "harbour": True,          # harbour=yes and variants
        "landuse": "harbour",
        "man_made": ["pier", "breakwater", "quay"],
        "waterway": "dock",
        "amenity": "ferry_terminal",
    },
    "port_buffer_m": float(os.environ.get("PW_PORT_BUFFER_M", 150)),
    # MEASURED: leave this False. Joining the port flag onto the 79 labelled
    # Conakry patches, the zone contains 4 positive / 5 negative / 4 ambiguous.
    # Precision inside the zone is 44.4% against 66.7% outside, so the zone IS
    # enriched for false positives -- but dropping it trades 5 false positives
    # for 4 CONFIRMED debris detections, all four labelled "debris" by the
    # interpreter. Harbours are where floating waste accumulates, so an
    # exclusion that removes them defeats the purpose of the layer. Flag,
    # surface the flag in the UI, and let the analyst filter.
    "port_drop_flagged": os.environ.get("PW_PORT_DROP", "false").lower() == "true",
    "min_patch_area_m2": float(os.environ.get("PW_MIN_PATCH_AREA_M2", 500)),
    "vectorize_scale_m": 20,
    # Positive water constraint. The NDVI/NDWI pair only EXCLUDES vegetation and
    # open water; everything else survives, which on a coastal AOI means bare
    # soil, sand, concrete and port/industrial rooftops. Those have strong
    # NIR/SWIR and therefore a large FDI excess, so they were scoring HIGH --
    # the photo-interpreted validation found HIGH precision (33%) BELOW LOW
    # (71%) for exactly this reason.
    #
    # ESA WorldCover v200 class 80 (permanent water bodies), 10 m, chosen over
    # JRC Global Surface Water on measured separation against the interpreter's
    # labels: WorldCover excluded 3/5 of the infrastructure false positives
    # while retaining 9/9 confirmed debris; GSW occurrence >= 50 matched the
    # 3/5 but lost 2 debris points, and raising it to >= 90 caught 5/5
    # infrastructure while destroying debris recall (2/9).
    "water_mask_asset": "ESA/WorldCover/v200",
    "water_mask_class": 80,    # permanent water bodies
    "built_mask_class": 50,    # built-up
    # 0 by default: a 60 m dilation of the WATER mask was tested first and
    # swallowed the entire shoreline, letting all 5 infrastructure cases back
    # through (they sit within one buffer width of water). No debris point in
    # the sample needed it.
    "water_buffer_m": float(os.environ.get("PW_WATER_BUFFER_M", 0)),
    # The BUILT mask is dilated instead, by one 20 m pixel: a Sentinel-2 pixel
    # straddling a quay edge is contaminated by the built surface even when its
    # centre reads as water. Measured: excluding built-adjacent pixels took
    # infrastructure exclusion from 2/5 to 3/5 with debris retention unchanged
    # at 9/9 and no change in masked AOI fraction (0.734).
    "built_buffer_m": float(os.environ.get("PW_BUILT_BUFFER_M", 20)),
    "max_expected_patch_m2": 1_000_000,  # 1 km^2, kept as the fallback normaliser when
                                          # no FDI-intensity score can be computed
    "sar_fallback_cloud_pct": float(os.environ.get("PW_SAR_FALLBACK_CLOUD_PCT", 70)),
    # Percentile of per-scene excess-over-threshold used to normalise
    # `compute_concentration_score` (see `vectorize_hotspots`'s `excess_norm`).
    # 95th: robust to one extreme outlier pixel while still using this scene's
    # actual FDI dynamic range rather than a fixed/arbitrary divisor.
    "concentration_score_norm_percentile": float(os.environ.get("PW_CONC_NORM_PCTL", 95)),
    # Bands for `detection.compute_concentration_score`'s FDI-intensity score
    # (mean excess-over-threshold within the patch, clipped to [0, 1]). Provisional
    # defaults — recalibrate against real scene score ranges once available, and
    # document the change rather than silently retuning.
    "confidence_band_low_max": float(os.environ.get("PW_CONF_LOW_MAX", 0.33)),
    "confidence_band_high_min": float(os.environ.get("PW_CONF_HIGH_MIN", 0.66)),
}

# ============================================================================
# MACRO-SCALE DUMPSITE DETECTION
# ============================================================================
# Per the DPWMS SRS / Joint Concept Note, the National Hotspot Map combines TWO
# independently-sourced layers (GEDH-FR-0503):
#   1) macro-scale illegal DUMPSITE polygons  <- this config
#   2) meso-scale floating PLASTIC PATCHES    <- the FDI pipeline in this module
#
# WHAT SHIPPED, AND WHY IT IS NOT YOLOv8. The concept note proposed adapting
# the Lab's Guatemala object detector. Guinea has 111 labelled dump points --
# nowhere near enough to train a detector -- so the method is a logistic
# regression over frozen ImageNet ResNet-50 features, run as a two-stage
# cascade: a free Esri 250 m screen, then an 80 m confirm inside the cells
# that survive, on free Esri imagery by default or licensed Google imagery.
# Measured on 651 point-centred samples with spatial block CV: ROC-AUC 0.832
# (Esri) and 0.911 (Google); on the blind grids, see methodology v2 7.2.11.
#
# THRESHOLDS ARE TIERS, NOT A SINGLE CUT-OFF, following Global Forest Watch
# (GLAD/RADD confidence levels) and Global Plastic Watch (sensitivity regimes).
# The authoritative values are published in dumpsite_tool/models/probe_meta.json
# by the `freeze` step, so they can be retuned without editing code or paying
# for imagery again; `default_tier` here only selects which one the backend
# serves by default. Medium (>= 0.70) is the default: on the blind grids with
# Esri at stage 2 the high tier finds only 6-21% of sites, and the lowest tier
# falls close to the base rate (methodology v2 Section 7.2.11).
DUMPSITE_MODEL_CONFIG: Dict = {
    # Off unless the deployment sets PW_DUMPSITE_ENABLED=1, so switching the
    # national scan on is an environment change, not a code change. Switch it
    # on only once the blind-grid figures for the configured confirm_source
    # have been measured (dumpsite_detection_workflow.ipynb, Part 3).
    "enabled": os.environ.get("PW_DUMPSITE_ENABLED", "0") == "1",
    # Stage-2 imagery. "esri" is free (Esri z17, probe_esri.joblib), so the
    # platform can run the scan itself; "google" is licensed and billed per
    # 80 m window (Google z19, probe.joblib) and stays an offline option
    # pending doc 70 D31.
    "confirm_source": os.environ.get("PW_DUMPSITE_CONFIRM_SOURCE", "esri"),
    "model_type": "imagenet_resnet50_linear_probe",
    "task": "binary_scene_classification",
    "architecture": "two-stage cascade (free Esri 250m screen -> 80m confirm on Esri or Google)",
    # The frozen heads ship inside the package; there is no external checkpoint
    # to locate. Kept only so an operator who sets this env var gets a warning
    # rather than silence.
    "checkpoint_path": os.environ.get("PW_DUMPSITE_MODEL_PATH"),
    "default_tier": os.environ.get("PW_DUMPSITE_TIER", "medium"),
    "cache_dir": os.environ.get("PW_DUMPSITE_CACHE", "dumpsite_scan_cache"),
    # A Google stage 2 bills one Static Maps request per 80m window. Leave this
    # False: it forces every caller to pass an explicit budget, which is the
    # only thing standing between a misconfigured Celery task and a four-figure
    # imagery bill.
    "allow_unbounded_spend": False,
    "reference_project": "SDG AI Lab - Guatemala illegal dumpsite detection",
}

# ============================================================================
# TOXICITY / MATERIAL FLAGGING (placeholder interface)
# ============================================================================
# GEDH-FR-0504 requires a material-type distribution + toxicity flag per
# hotspot. Real-time material ID (PVC screening, per the concept note) is a
# separate Collector-app classifier; this module only carries the schema so
# `stats.py` can populate it once that classifier's output is wired in.
MATERIAL_FLAG_SCHEMA: List[str] = [
    "PET", "HDPE", "PVC", "LDPE", "PP", "PS", "OTHER", "UNCLASSIFIED",
]
TOXICITY_FLAG_MATERIALS = {"PVC"}  # flagged per concept note (PVC screening)

# ============================================================================
# VISUALIZATION (kept for parity with the GEE App prototype; used by any
# HTML/Visualizer preview built on top of this pipeline)
# ============================================================================
VIZ_CONFIG: Dict = {
    "FDI":  {"min": -0.05, "max": 0.05,
            "palette": ["#042C53", "#185FA5", "#85B7EB", "#E6F1FB", "#FAEEDA", "#BA7517", "#633806"]},
    "NDVI": {"min": -0.3, "max": 0.7, "palette": ["#4A1B0C", "#E8593C", "#FAEEDA", "#639922", "#173404"]},
    "NDWI": {"min": -0.5, "max": 0.5, "palette": ["#4A1B0C", "#E8593C", "#E6F1FB", "#185FA5", "#042C53"]},
    "HOTSPOT": {"palette": ["#E24B4A"]},
}

# ============================================================================
# MONTHLY FDI ARCHIVE (GEDH-FR-0507)
# ============================================================================
ARCHIVE_CONFIG: Dict = {
    "cache_prefix": "archive/fdi",
    "index_filename": "fdi_archive_index.json",
    # Bands each archive month must carry so the offline side can reproduce the
    # SAME detection rule the main composite path applies. Under the de_africa
    # rule the decision is a CONJUNCTION over two indices, so exporting FDI
    # alone is not enough: FDI > mean+3*sigma on its own flags 102,827 pixels
    # on Conakry 2024-01 where the AND with NDVI flags 6,558. WATER is carried
    # rather than re-derived so both paths take their statistics over exactly
    # the same pixel population. The raw reflectance bands are carried too,
    # because the MARIDA debris classifier that follows the rule
    # (fdi_tool.debris_filter) scores each detection from them; with them the
    # archive month runs through pipeline.run_from_arrays exactly as the main
    # composite does.
    "bands_de_africa": ["B3", "B4", "B6", "B8", "B8A", "B11", "B12",
                        "FDI", "NDVI", "WATER"],
}

# ============================================================================
# HELPERS
# ============================================================================
def get_aoi_bbox(key: str = DEFAULT_AOI_KEY):
    return AOI_CONFIG.get(key, AOI_CONFIG[DEFAULT_AOI_KEY])["bbox"]


def ensure_directories() -> None:
    for d in (PROCESSING_CONFIG["temp_dir"], PROCESSING_CONFIG["cache_dir"],
              PROCESSING_CONFIG["output_dir"]):
        os.makedirs(d, exist_ok=True)


def validate_required_keys() -> List[str]:
    """Re-uses the same required-secret checklist as landuse_tool (one GEE
    service account backs both modules)."""
    from landuse_tool import config as landuse_config
    return landuse_config.validate_required_keys()