# Deforestation Analysis Module — Methodology & Workflow

## Purpose

This module extends the Land Use platform with **forest-loss / deforestation
analysis** for Guinea: detecting deforested areas, tracking them over time,
producing statistics and graphs, attributing drivers, and simulating future
scenarios. It is a **calculation module** — the system engineer wires it into the
platform; we provide the scripts and this workflow.

It deliberately reuses everything we already built. Same data connection
(`api_connector`), same grid alignment (`data_loader`), same raster utilities
(`utils`), same configuration (`config`), same Blob cache. No separate data
pipeline, no re-alignment.

## Design principle

**We use the ready-made products as the primary signal** and keep K-means as an optional
unsupervised cross-check. This is more accurate, far cheaper, and consistent use of available data instead of computing ourselves.

| Need | Ready-made dataset (primary) | Why |
|---|---|---|
| Annual forest **loss** | **Hansen Global Forest Change** `UMD/hansen/global_forest_change_2024_v1_12` | Authoritative, validated, 30 m, annual `lossyear` 2001–2024, global. The reference product. |
| Deforestation/degradation in **moist forest** | **JRC TMF** `projects/JRC/TMF/v1_2025/DeforestationYear` | Tropics-tuned; complements Hansen in SE Guinea (Guinée Forestière). Covers moist forest only. |
| **Vegetation trend / degradation** | **MODIS NDVI** `MODIS/061/MOD13Q1` | 16-day 250 m NDVI/EVI; no compositing needed. |
| **Fire** driver | **MODIS burned area** `MODIS/061/MCD64A1` | Major deforestation driver in Guinea's savanna zone. |
| **Forest → what?** | LULC from the Land Use module (ESRI, Dynamic World, MODIS, GLC_FCS30D) | Their forest classes tell you what forest converted *to* (cropland, built…). |
| Drivers (terrain, population, climate, roads, biomass) | Already in the Land Use catalog | Reused directly via `data_loader`. |

Because Guinea is mostly **dry/savanna**, not moist forest, **Hansen is the
primary loss source** and JRC TMF is an optional complement for the southeast.

## Data flow (how it plugs into the platform)

```
  api_connector.update_all()            # existing fetch: LULC + predictors -> Blob
        +  forest.fetch_hansen(roi)     # NEW: Hansen 2-band image -> Blob (same primitives)
        +  MODIS NDVI / burned area      # NEW config entries -> fetched by the SAME fetcher
        │
        ▼
  data_loader.prepare_predictor_files() # existing: align everything to ONE grid
        │
        ▼
  deforestation.pipeline.run_from_arrays / run_from_files
     forest → detection → stats → drivers → scenario     (all on aligned rasters)
        │
        ▼
  outputs (masks, time series, matrices, risk map, scenario) -> Blob
        │
        ▼
  visualization.render_all_layers(...)  # existing type-aware renderer
```

The only new fetch is Hansen, and it uses the **same** `api_connector` download
stack (tiled/GCS + Blob cache) as every other layer — see `forest.fetch_hansen`.
`config.register()` adds MODIS NDVI/fire to the main catalog so they flow through
the existing annual fetcher automatically.

## Methodology, step by step

### 1. Forest baseline & masks (`forest.py`)
- **Forest definition:** canopy cover ≥ a threshold (config `canopy_threshold_pct`,
  default 30%; FAO uses 10%). From Hansen `treecover2000`.
- **Forest extent per year:** `forest2000 AND NOT (loss on/before year Y)` — the
  Global Forest Watch convention (`forest_extent_by_year`). Hansen `gain` is
  2000–2012 only and is intentionally not added back annually.
- **Alternative:** `forest_mask_from_lulc` uses the Land Use LULC forest classes
  (`FOREST_CLASSES` per product) — useful for the "converted-to" analysis and when
  you want thematic land cover rather than canopy.

### 2. Detection (`detection.py`)
- **Primary — Hansen annual loss:** decode `lossyear` into per-year loss masks
  (`annual_loss_from_hansen`) and cumulative loss (`cumulative_loss`).
- **LULC-transition deforestation:** `forest(t1) ∧ ¬forest(t2)`
  (`deforestation_from_lulc`), plus `conversion_classes` = what each deforested
  pixel became at t2 (cropland/built/…).
- **Degradation:** NDVI decline within pixels that stayed forest
  (`degradation_from_vi`) — thinning, distinct from clearing.
- **K-means (optional, as requested):** `kmeans_landcover` clusters the aligned
  stack; `clusters_to_forest` labels clusters by mean NDVI. Noisier and needs
  label interpretation — offered as an unsupervised cross-check, not the primary.

### 3. Statistics & graphs (`stats.py`)
- **Annual deforested area** (ha) and **forest-cover trajectory** (ha), with
  correct **latitude-weighted pixel area** on the EPSG:4326 grid.
- **Deforestation rate** (%/yr of the prior year's forest).
- **`summary_table`** — tidy `year, forest_ha, loss_ha, rate_pct` DataFrame,
  ready to plot or export as CSV.
- **Conversion matrix** — ha of forest converted to each destination LULC class.
- **VI trend** — per-pixel NDVI slope (least squares) + area-mean slope.
- **Zonal breakdown** — deforested ha per admin unit (GeoBoundaries ADM1/ADM2).

### 4. Driver analysis (`drivers.py`)
- **Driver stack** = shared predictors (slope, elevation, population, precipitation,
  distance-to-roads, biomass — from `data_loader.assemble_predictor_stack`) **plus**
  deforestation-specific proximity layers: distance-to-prior-loss (the *frontier*),
  distance-to-forest-edge, and fire.
- **Risk model:** sample deforested pixels (1) vs stable-forest pixels (0), fit a
  RandomForest (or logistic) whose **feature importances are the driver
  attribution**, and predict a continuous **risk map** P(deforestation) over
  remaining forest.
- Output: ranked driver importances + a risk raster.

### 5. Future scenario (`scenario.py`)
- **How much:** `project_future_loss` extrapolates the historical loss rate
  (mean / linear / %-of-standing-forest).
- **Where:** `allocate_future_loss` clears the projected number of pixels from the
  **highest-risk** standing forest (risk map = the "suitability" of a forest pixel
  to be cleared — the same idea as the Land Use suitability+allocation).
- **Policy levers:** protected areas (risk → 0), new roads/frontiers (risk boosted
  by proximity), and a governance `rate_factor` on demand. `run_scenario` ties it
  together and returns projected loss, a future-forest mask, and a cleared mask.

## Module / function map

| File | Key functions |
|---|---|
| `deforestation/config.py` | dataset catalog (Hansen, JRC TMF, MODIS NDVI/fire), `FOREST_CLASSES`, `DEFOR_PARAMS`, `register()` |
| `deforestation/forest.py` | `fetch_hansen`, `read_hansen`, `forest_mask_from_hansen`, `forest_mask_from_lulc`, `decode_hansen_lossyear`, `forest_extent_by_year` |
| `deforestation/detection.py` | `annual_loss_from_hansen`, `cumulative_loss`, `deforestation_from_lulc`, `conversion_classes`, `degradation_from_vi`, `kmeans_landcover`, `clusters_to_forest` |
| `deforestation/stats.py` | `pixel_area_ha`, `annual_deforestation_area`, `forest_cover_trajectory`, `deforestation_rate`, `summary_table`, `conversion_matrix`, `vi_trend`, `zonal_deforestation` |
| `deforestation/drivers.py` | `distance_to`, `forest_edge`, `build_driver_stack`, `sample_training`, `fit_risk_model`, `driver_importance`, `predict_risk_map` |
| `deforestation/scenario.py` | `project_future_loss`, `apply_policy`, `allocate_future_loss`, `run_scenario` |
| `deforestation/pipeline.py` | `run_from_arrays`, `run_from_files`, `_synthetic_test` |

## How the system operates (for the system engineer)

1. **Scheduled fetch job** (existing Container Apps Job): `api_connector.update_all()`
   fetches LULC + predictors; add `forest.fetch_hansen(boundary)` and call
   `deforestation.config.register()` so MODIS NDVI/fire are fetched too. All land
   in Blob.
2. **Alignment:** `data_loader.prepare_predictor_files(year, …)` produces the
   grid-aligned predictor GeoTIFFs; Hansen and the LULC/NDVI layers are aligned the
   same way via `data_loader.load_and_align_raster`.
3. **Analysis:** call `deforestation.pipeline.run_from_arrays(...)` (or
   `run_from_files`) with the aligned Hansen bands, LULC-by-year, NDVI-by-year, and
   the shared predictor stack. It returns masks, a `summary` DataFrame, a
   conversion matrix, a VI-trend map, driver importances, a risk map, and a
   scenario. Persist these to Blob.
4. **App:** the Streamlit app reads the results from Blob and renders them with the
   existing type-aware `visualization.render_all_layers` (forest masks →
   categorical; loss-year / risk / VI-trend → continuous colorbars).

Everything is raster-file based and window-safe, so it scales to country size the
same way the Land Use module does. No new infrastructure is required beyond what
the Land Use module already needs.

## Honest caveats

- **Hansen ≠ LULC "forest".** Hansen tracks *tree-canopy* loss (including planted
  trees and, in some cases, plantation harvest), not land-use change. Cross-check
  with the LULC "converted-to" matrix before calling loss "deforestation."
- **Dry vs moist forest.** Guinea is largely savanna; canopy thresholds and the
  moist-forest-only JRC TMF behave very differently across the country. Report the
  threshold used and consider zone-specific thresholds.
- **Driver model ≠ causation.** Feature importances show *association* (roads,
  frontier, slope), not proof of cause. Validate with spatial cross-validation
  (random pixel splits inflate accuracy through spatial autocorrelation — the same
  caveat as the Land Use model).
- **Scenario is a projection, not a prediction.** BAU rate extrapolation is a
  planning tool; treat the numbers as scenario envelopes, not forecasts.
- **K-means** is an unsupervised convenience; cluster→forest labelling is
  heuristic. Prefer Hansen/LULC for anything reported.

## Testing

`python -m landuse_tool.deforestation.pipeline` runs a synthetic self-test through
every stage (forest masks → detection → stats → drivers → risk → scenario) with no
API keys, and prints PASS/FAIL. Extend `run_pipeline.py` similarly for a combined
land-use + deforestation integration test once wired into the package.

**Version:** 1.0.0 • deforestation module • built on the Land Use platform.
