# Land Use Monitoring and Prediction Tool — Documentation

## Overview

This tool fetches multitemporal geospatial layers for Guinea, aligns them onto a
common raster grid, and runs a machine-learning pipeline to analyse historical
land-cover change and simulate future land use under different scenarios.

It is built to run **locally** for development and **on Azure** for production
(the platform is deployed by a separate engineer using Azure services).

**Key capabilities**
- Automated, incremental fetching of land-cover (target) and predictor layers from
  open APIs — one clean raster per year, for every year a source provides.
- Vector layers (roads, railways, waterways, POIs) rasterized into meaningful
  continuous predictors (distance-to surfaces, heatmaps).
- Markov-chain change analysis and linear / non-linear trend projection.
- Random-Forest suitability modelling and demand allocation (deterministic or
  stochastic), with scenario/policy overrides.
- Interactive and static visual export (folium, PNG, PDF, GeoTIFF, CSV, JSON).

---

## Table of Contents

1. [Architecture](#architecture)
2. [Core Modules](#core-modules)
3. [Data Acquisition & Multitemporal Design](#data-acquisition--multitemporal-design)
4. [Vector Rasterization](#vector-rasterization)
5. [Dataset Catalog](#dataset-catalog)
6. [API Keys & Secrets](#api-keys--secrets)
7. [Local Installation & Setup](#local-installation--setup)
8. [Testing](#testing)
9. [Azure Deployment](#azure-deployment)
10. [End-to-End Workflow](#end-to-end-workflow)
11. [Advanced Features](#advanced-features)
12. [Troubleshooting](#troubleshooting)
13. [Change Log / Migration Notes](#change-log--migration-notes)

---

## Architecture

The system has two halves: a **data-acquisition layer** (new/restructured) that
produces analysis-ready rasters, and a **three-stage ML pipeline** (unchanged in
concept) that consumes them.

```
   OPEN APIS (GEE, OpenTopography, WorldPop, OSM, GeoBoundaries, ...)
        │
        ▼
   DATA ACQUISITION  (api_connector + data_loader + azure_utils)
     • Guinea boundary (GeoBoundaries, two-step)
     • one LULC raster per year, ALL available years, incrementally updated
     • predictor rasters aligned to a common grid
     • vector layers rasterized (distance-to / heatmap)
     • cached to Azure Blob Storage (reused across runs)
        │
        ▼
   STAGE 1  CHANGE ANALYSIS & TRAINING  (change_analysis + training)
     • Markov transition matrix + trend projection → future demand
     • Random-Forest per-transition suitability models
        │
        ▼
   STAGE 2  SUITABILITY GENERATION  (prediction)
     • P(transition | predictors) maps, window-based, memory-safe
        │
        ▼
   STAGE 3  ALLOCATION & SIMULATION  (prediction + scenarios)
     • allocate demand to most-suitable pixels (deterministic/stochastic)
     • growth modes (expander/patcher), policy overrides
        │
        ▼
   VISUALIZATION & EXPORT  (visualization)
```

---

## Core Modules

### `config.py` — Configuration & Dataset Catalog
Centralises all parameters. Restructured so that:
- **No secrets in code** — every key is read from an environment variable
  (populated from a local `.env` in dev, or Azure Key Vault in production).
- **All writable paths default to `/tmp`** and are overridable; an optional Azure
  Blob container provides persistence.
- The **dataset catalog** carries, per layer: the GEE `asset` id, the `band` to
  select, the `reducer`, native `resolution_m`, the `temporal` class, and a
  `year_hint`. This is what lets the fetcher pull one clean layer per year and
  discover all available years.

Key structures: `PROCESSING_CONFIG`, `API_CREDENTIALS`, `ADMIN_BOUNDARY_CONFIG`,
`TARGET_RASTERS_CONFIG`, `PREDICTOR_RASTERS_CONFIG`, `PREDICTOR_VECTORS_CONFIG`,
`MULTITEMPORAL_CONFIG`, `MISSING_VALUE_CONFIG`, `SAMPLING_CONFIG`, `MODEL_CONFIG`.

Helpers: `get_enabled_target_rasters()`, `get_enabled_predictor_rasters()`,
`get_enabled_predictor_vectors()`, `get_api_key(name)`, `validate_required_keys()`,
`get_missing_value_strategy(layer_type)`, `ensure_directories()`.

### `azure_utils.py` — Secrets & Storage (NEW)
- `load_secrets_into_env()` — loads a local `.env` (dev) and then a single JSON
  secret from Azure Key Vault (prod) into environment variables. Auth uses
  `DefaultAzureCredential` (Managed Identity in Azure, `az login` locally).
- `BlobCache` — a Blob-Storage-backed cache mirroring `{container}/{prefix}/{key}`
  to local files, so a fresh container reuses previously fetched layers. Falls
  back to local-only when no container is configured.

### `api_connector.py` — External Data APIs (rewritten)
- `initialize_gee()` — **headless service-account** auth (not the interactive
  `ee.Authenticate()`), required for servers.
- `initialize_all(strict=True)` — validates required keys and initialises GEE + OSM.
- `get_admin_boundary()` — GeoBoundaries **two-step** fetch (metadata → GeoJSON),
  cached. Handles Guinea's MultiPolygon geometry.
- `build_annual_gee_image(cfg, year, roi)` — selects the band and reduces to **one
  layer for one year** (mode for categorical LULC, mean/sum for continuous;
  handles band-per-year collections like GLC_FCS30D).
- `discover_available_years(cfg, roi)` — queries a collection's real timestamps to
  return every year it holds (this is what makes "all years" + auto-update work).
- `fetch_gee_dataset_all_years(...)` — fetches one layer per available year,
  **skipping years already cached** (incremental).
- `update_all(roi=None)` — top-level entry point: fetches all enabled targets +
  predictors + vectors, only what's missing. Run on a schedule.
- Downloads: tiled `getDownloadURL` + rasterio mosaic by default (works anywhere
  at ~30–500 m); optional GCS export for native 10 m.
- Other fetchers: `fetch_opentopo_dem(...)` (API key), `fetch_osm_vectors(...)`,
  `fetch_worldpop_rest(...)`, `get_copernicus_session()` (OAuth2).

### `data_loader.py` — Loading, Alignment, Rasterization, Sampling (updated)
- `load_admin_boundary()` — via the connector (no longer reads the metadata URL as
  geometry).
- `load_and_align_raster(path, ref_profile, categorical)` — **resamples any raster
  onto the reference grid** (nearest for categorical LULC, bilinear for continuous)
  so mixed native resolutions line up pixel-for-pixel.
- `load_targets(...)`, `load_predictors(...)` — validate + align (predictors are now
  aligned instead of rejected on a dimension mismatch).
- `rasterize_vector_predictors(vector_files, ref_profile, ...)` — turns fetched
  vectors into aligned raster predictors using the mode from config
  (distance / heatmap / zoning).
- `assemble_predictor_stack(year, raster_paths, vector_paths, ref_profile, ...)` —
  builds the **ML-ready stack for one snapshot year**: aligns each raster (nearest
  year for sparse layers), derives slope/aspect from the DEM, and rasterizes the
  vectors; every layer shares the grid and boundary mask.
- `fetch_and_load_api_targets()`, `fetch_and_load_api_predictors()` — orchestrate
  fetch + load; the predictor version also fetches and rasterizes the vectors.
- `sample_training_data_stratified(...)` — balanced per-class sampling (the old
  `logger.progress` crash is fixed).

**Fixes applied:** removed `logger.success`/`logger.progress` (not real logging
methods), unified `progress_callback(frac, text)`, corrected docstring placement,
fixed boundary loading, replaced deprecated `unary_union`.

### `utils.py` — Raster Processing (unchanged)
Reprojection/resampling (`reproject_raster`, `resample_raster`, `align_rasters`),
missing-value handling (`identify_missing_values`, `handle_missing_values`,
`get_missing_value_stats`), masking (`create_mask`, `apply_boundary_mask`),
**vector rasterization** (`process_vector_layer` with `distance`/`heatmap`/`zoning`,
`rasterize_points_as_heatmap`), terrain derivatives (`calculate_slope`,
`calculate_aspect`), and validation/statistics.

### `change_analysis.py` — Historical Trend Analysis
Markov transition matrices (`calculate_transition_matrix`), linear
(`analyze_linear_trends`) and non-linear (`analyze_non_linear_trends`) projection,
per-class statistics, and the `run_change_analysis()` dispatcher (auto-selects
linear vs non-linear by the number of time points). *(Fixed: `logger.success`.)*

### `training.py` — Model Training
Trains a Random-Forest per transition with balanced sampling and optional growth
modes (expander/patcher). Utilizes vectorized, in-memory array slicing to eliminate disk I/O bottlenecks during sample extraction, and enforces Spatial Block Cross-Validation (`GroupShuffleSplit`) to prevent geographic data leakage during training. Reads hyperparameters from `config.MODEL_CONFIG`, writes
models to a config-driven directory, and persists `<model>_features.json` +
`<model>_metrics.json` alongside each `.joblib` — the exact schema `prediction.py`
loads.

Key functions: `create_transition_dataset()`, `train_rf_model()`,
`train_transition_models()`, `load_trained_model()`, `get_model_metrics()`.

### `scenarios.py` — Scenario Building
Predictor modification (`modify_predictor_raster`), neighborhood/spatial-context
predictors (`generate_neighborhood_predictors`), policy demand overrides
(`override_transition_demand`), stochastic config, and scenario packaging.
*(Fixed: a `NameError` in `generate_neighborhood_predictors` where the progress
callback ran before its loop, which previously made it silently return nothing.)*

### `prediction.py` — Two-Stage Prediction
`generate_suitability_atlas()` (Stage 2) and `run_allocation_simulation()`
(Stage 3), plus validation/comparison helpers. Window-based and memory-safe.

### `visualization.py` — Mapping & Export
Interactive folium maps, static PNG/PDF, and GeoTIFF/CSV/JSON export.
*(Fixed: `plt.cm.get_cmap` — removed in matplotlib 3.9 — replaced with the modern
`matplotlib.colormaps[...].resampled()` API.)*

---

## Data Acquisition & Multitemporal Design

Datasets don't share a calendar, so layers are grouped into three temporal classes
and fetched accordingly:

| Class | Meaning | Fetch behaviour |
|---|---|---|
| `static` | time-invariant (DEM, slope, aspect, roads) | fetched once, reused for every year |
| `annual` | one value per year (LULC, population, precipitation) | one clean raster per available year |
| `sparse` | irregular (biomass) | snapped to the nearest available year |

**One layer per year.** `build_annual_gee_image()` filters a collection to the
year, selects the labelled band, and reduces it (mode for categorical, mean/sum for
continuous) — no all-band downloads, no time collapsing.

**All years, not one.** `discover_available_years()` queries each collection's real
timestamps over Guinea and returns every year present, rather than a hardcoded list.

**Incremental / automatic updates.** `update_all()` fetches only the
`(dataset, year)` pairs missing from Blob Storage. When a source publishes a new
year, the next scheduled run picks it up automatically. Run it on a schedule
(Azure Container Apps Job cron). Set `MULTITEMPORAL_CONFIG["force_refresh"]=True`
to re-fetch everything, or set `["years"]` to restrict to specific snapshot years.

---

## Vector Rasterization

Because the ML model needs **all inputs as rasters on one grid**, vector layers are
converted to continuous surfaces. The mode per layer is set in
`PREDICTOR_VECTORS_CONFIG`, and the conversion runs via
`data_loader.rasterize_vector_predictors()` → `utils.process_vector_layer()`:

| Vector layer | Mode | Resulting raster |
|---|---|---|
| ROADS | `distance` | distance-to-nearest-road (metres) |
| RAILWAYS | `distance` | distance-to-nearest-railway |
| WATERWAYS | `distance` | distance-to-nearest-water |
| POI | `heatmap` | density/decay surface around points of interest |
| (admin zones) | `zoning` | one-hot encoded binary masks per zone (e.g., `_zone_1`, `_zone_2`) |

Every output is on the reference grid with NaN outside the Guinea boundary, ready
to be stacked with the raster predictors. `assemble_predictor_stack()` performs
this stacking for a snapshot year.

---

## Dataset Catalog

**Targets (LULC — categorical, one layer/year):**

| Key | GEE asset | Band | Res | Years | Enabled |
|---|---|---|---|---|---|
| ESRI_LULC | `projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS` | `b1` | 10 m | 2017–2025 | ✅ |
| DYNAMIC_WORLD | `GOOGLE/DYNAMICWORLD/V1` | `label` | 10 m | 2016–present | ✅ |
| MODIS_MCD12Q1 | `MODIS/061/MCD12Q1` | `LC_Type1` | 500 m | 2001–2023 | ✅ |
| GLC_FCS30D | `projects/sat-io/open-datasets/GLC-FCS30D/annual` | band/yr | 30 m | 2000–2022 | ✅ |
| GLANCE | `projects/GLANCE/DATASETS/V001` | `LC` | 30 m | 2001–2021 | ⛔ verify band |

> Note: the old `S2TSLULC` entry was the **same** product as ESRI_LULC and stored a
> docs URL instead of an asset id — it was removed.

**Predictors:**

| Key | Source | Band/type | Res | Temporal |
|---|---|---|---|---|
| DEM | OpenTopography `SRTMGL1` (or `DEM_GEE`=`USGS/SRTMGL1_003`) | elevation | 30 m | static |
| SLOPE / ASPECT | derived from DEM | degrees | 30 m | static |
| POPULATION_DENSITY | GEE `WorldPop/GP/100m/pop` | population | 100 m | annual (2000–2020) |
| PRECIPITATION | GEE `UCSB-CHG/CHIRPS/DAILY` (annual sum) | precipitation | ~5.5 km | annual |
| TEMPERATURE | GEE `MODIS/061/MOD11A2` (annual mean) | LST_Day_1km | 1 km | annual |
| AGB_NASA | GEE `NASA/ORNL/biomass_carbon_density/v1` | agb | 300 m | sparse (~2010) |
| GEDI_AGB | GEE `LARSE/GEDI/GEDI04_A_002_MONTHLY` | agbd | 25 m | sparse (2019–2023) |
| AGB_ESA | GEE ESA CCI (verify path) | AGB | 100 m | sparse | ⛔ |

**Vectors:** ROADS, RAILWAYS, WATERWAYS (distance), POI (heatmap) — from OSM /
Geofabrik.

---

## API Keys & Secrets

Only two keys are needed for the default config; the rest are keyless or optional.

| API | Key? | How to get it |
|---|---|---|
| **Google Earth Engine** | Yes (service account) | GCP project (EE-registered) → IAM → create service account → grant Earth Engine access → create JSON key → download. Need JSON contents + SA email + project id. |
| **OpenTopography** | Yes (free) | portal.opentopography.org → "Request an API Key". 250/day academic, 50/day non-academic. Or enable `DEM_GEE` to skip it. |
| **Copernicus (CDSE)** | Optional | Sentinel Hub dashboard → OAuth clients → Create. Secret expires ~90 days. Off by default. |
| **Google Maps/Roads** | Optional (avoid) | Billed; OSM/Geofabrik covers roads free. |
| OSM / WorldPop / GeoBoundaries / STAC | No | Keyless. |

**How secrets are supplied**
- **Locally:** copy `.env.example` → `.env` and fill it in. `.env` is gitignored.
  The service-account JSON is kept as a gitignored file and referenced by
  `GEE_SA_KEY_FILE=./secrets/gee-service-account.json` (no need to paste the JSON).
- **On Azure:** the same values go into **one Key Vault secret** (default name
  `landuse-config`, shape in `keyvault.template.json`), and `AZURE_KEY_VAULT_URL`
  points to the vault. Managed Identity grants access — no keys in the container.

**Committable templates:** `.env.example`, `.gitignore`, `keyvault.template.json`.
**Never committed:** the real `.env`, the service-account JSON.

---

## Local Installation & Setup

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate      # or conda
pip install -r requirements.txt

# 2. Secrets
cp .env.example .env                                   # then edit .env
mkdir secrets                                          # add gee-service-account.json here

# 3. (optional) confirm datasets/keys before a full run
pytest -q                 # offline unit tests
python test_setup.py      # live staged smoke test
```

**Key environment variables** (see `.env.example` for the full list):
`GEE_PROJECT_ID`, `GEE_SERVICE_ACCOUNT`, `GEE_SA_KEY_FILE`, `OPENTOPO_API_KEY`,
`AZURE_BLOB_CONTAINER`, `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_KEY_VAULT_URL`,
`APP_SCRATCH_DIR`, `TARGET_RESOLUTION`.

---

## Testing

**1. Offline unit tests** — no keys, no network (CI-safe):
```bash
pytest -q
```
Checks the config catalog integrity (every dataset has a band + reducer + temporal
class, no URLs-as-assets, GLC_FCS30D band-per-year wiring) and the cache logic.

**2. Live staged smoke test** — checks the real APIs over a tiny box near Conakry:
```bash
python test_setup.py
```
Six independent stages print `PASS`/`FAIL`/`SKIP`: (0) config & keys, (1)
GeoBoundaries boundary, (2) Earth Engine auth, (3) a tiny GEE layer fetch, (4)
OpenTopography DEM, (5) OSM vectors, (6) **vector→raster** conversion. It exits
non-zero on any real failure, so it can also gate CI.

Fastest confidence check: fill `GEE_*` + `OPENTOPO_API_KEY`, drop the GEE JSON in
`secrets/`, run `python test_setup.py` — stages 1–4 green means fetching works
end to end.

---

## Azure Deployment

Split the heavy fetching from the interactive app.

| Concern | Azure service |
|---|---|
| Scheduled data fetch (`update_all()`) | **Container Apps Job** with a cron schedule (or ACI + Logic App / Timer Function) |
| Interactive Streamlit app | **Container Apps** or **App Service (Web App for Containers)** |
| Secrets | **Key Vault** (loaded via Managed Identity) |
| Persistent layers | **Blob Storage** (`AZURE_BLOB_CONTAINER`) |
| Container image | Docker (GDAL-heavy deps: geopandas, rasterio, osmnx) |

Auth in production is Managed Identity: grant the app's identity `get` on the vault
and read/write on the storage account. `/tmp` is scratch only; finished layers live
in Blob.

**Earth Engine + Azure nuance.** GEE can export only to Google Drive or Google
Cloud Storage — never directly to Azure Blob. That's fine: the default download
path tiles and mosaics locally, then uploads to Blob, entirely within Azure (good
for ~30–500 m country scale). Only for native 10 m country-wide layers do you set
`GCS_EXPORT_BUCKET`, adding a GEE → GCS → Blob copy step.

Do **not** use a short-timeout Consumption Function for the fetch job — GEE
downloads can exceed it. A Container Apps Job (or Premium Functions) has no such
limit.

---

## End-to-End Workflow

```python
from landuse_tool import api_connector, data_loader, change_analysis

# 0. Boundary
guinea = api_connector.get_admin_boundary()

# 1. Fetch targets (one raster per year, all years) + build masks
targets, (profiles, masks, boundary_mask) = data_loader.fetch_and_load_api_targets(guinea)
ref_profile = profiles["latest"]

# 2. Fetch predictors (rasters + vectors)
raster_paths, vector_paths, _ = data_loader.fetch_and_load_api_predictors(
    guinea, ref_profile=ref_profile, boundary_mask=boundary_mask)

# 3. Assemble the ML-ready predictor stack for a snapshot year
predictors_2020 = data_loader.assemble_predictor_stack(
    2020, raster_paths, vector_paths, ref_profile, boundary_mask)

# 3b. Write grid-aligned predictor GeoTIFFs (training/prediction read FILES,
#     each indexed by the target grid, so they must be pre-aligned).
predictor_files = data_loader.prepare_predictor_files(
    2020, raster_paths, vector_paths, ref_profile, boundary_mask)

# 4. Change analysis (needs the per-year target rasters)
target_files = [{"file": p, "year": y}
                for d in targets.values() for y, p in d.items()]
matrix, counts, plots, mode = change_analysis.run_change_analysis(target_files)

# 5. Build the significant-transition list from the counts, then train + predict
from landuse_tool import training, prediction
significant = [(int(i), int(j)) for i in counts.index for j in counts.columns
               if i != j and counts.loc[i, j] > 0]     # threshold to taste
model_paths, accuracies = training.train_transition_models(
    significant, target_files, predictor_files, use_growth_modes=False)

suitability = prediction.generate_suitability_atlas(
    predictor_files, target_files[-1]["file"], model_paths)
future_lc = prediction.run_allocation_simulation(
    target_files[-1]["file"], counts, suitability)
# 6. visualization.* to export maps / metrics
```

For production, the scheduled job simply calls `api_connector.update_all()`, and the
app reads the resulting layers from Blob Storage.

---

## Advanced Features

**Missing-value handling** (`config.MISSING_VALUE_CONFIG`): per-layer-type strategy —
predictors use local-mean imputation, targets keep nodata, training excludes samples
with any missing predictor. Methods: `mean`, `median`, `nearest`, `mask`, `keep`.

**Growth modes** (expander/patcher): expander models boundary expansion of existing
patches; patcher models new isolated patches. Allocation combines both.

**Stochastic simulation:** probabilistic allocation weighted by suitability, run
multiple times to estimate uncertainty (`scenarios.configure_stochastic_parameters`).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: Configuration problems` | A required key is missing — run `config.validate_required_keys()`; fill `.env`. |
| GEE auth fails on server | Use a service account (`GEE_SERVICE_ACCOUNT` + `GEE_SA_KEY`/`GEE_SA_KEY_FILE`), not `ee.Authenticate()`. |
| `getDownloadURL` size error | Country-scale export — the tiled path handles it; for native 10 m set `GCS_EXPORT_BUCKET`. |
| Predictor "dimension mismatch" | Now auto-resampled by `load_and_align_raster`; ensure you pass a `ref_profile`. |
| `import landuse_tool` fails on `training` | `training` is optional now; add the module or leave it — the package imports without it. |
| Boundary is empty/None | GeoBoundaries metadata → follow `gjDownloadURL` (handled by `get_admin_boundary`). |
| matplotlib `get_cmap` error | Fixed to the `matplotlib.colormaps` API; upgrade matplotlib ≥ 3.7. |

---

## Change Log / Migration Notes

**v2.0.0 — data layer restructure + Azure**
- New `azure_utils.py` (Key Vault + Blob) replaces AWS/S3 helpers.
- `config.py`: env-driven secrets, Azure paths, corrected dataset catalog
  (per-dataset band/reducer/temporal/year), removed duplicate `S2TSLULC`, added
  `GLC_FCS30D`; `GLANCE`/`AGB_ESA` disabled pending band/asset verification.
- `api_connector.py`: service-account GEE auth, MultiPolygon-safe ROI, band
  selection, per-year single-layer fetch, `discover_available_years`,
  incremental `update_all`, tiled download / optional GCS export, and initializers
  for OpenTopography, WorldPop, OSM, Copernicus, GeoBoundaries.
- `data_loader.py`: boundary via connector, grid alignment, vector rasterization,
  `assemble_predictor_stack`; removed `logger.success`/`logger.progress`, unified
  `progress_callback(frac, text)`.
- Analysis scripts: fixed `logger.success` (`change_analysis`), a silent
  `NameError` in `generate_neighborhood_predictors` (`scenarios`), and the removed
  `plt.cm.get_cmap` (`visualization`).
- `training.py`: fixed a label-alignment bug (skipped NaN samples mismatched `y`
  with `X`); hyperparameters now read from `config.MODEL_CONFIG`; model output dir
  is config-driven (Azure-safe). Its model/schema/metrics naming matches
  `prediction.py`'s loader.
- `data_loader.py`: added `prepare_predictor_files()` / `save_predictor_stack()` to
  write grid-aligned predictor GeoTIFFs — the bridge from the fetch/align layer to
  the file-based `training`/`prediction` modules.
- `__init__.py`: `training` is now a first-class import; `api_connector` and
  `azure_utils` exposed.
- ML Pipeline optimizations: Replaced 1x1 windowed sampling with memory-efficient NumPy vectorization in `training.py` to eliminate I/O bottlenecks on Azure. Added `GroupShuffleSplit` for strict spatial block cross-validation. Enforced one-hot encoding for categorical vector rasterization (`zoning` mode) in `data_loader.py`. Added a minimum 3-period (4-map) statistical validity constraint for non-linear trend regressions in `change_analysis.py`.

**Version:** 2.0.0  •  **Last updated:** July 2026
