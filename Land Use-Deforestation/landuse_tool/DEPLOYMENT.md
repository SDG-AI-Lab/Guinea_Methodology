# Deployment & API Key Guide (Azure)

This tool fetches multitemporal geospatial layers (LULC targets + predictors) for
Guinea, rasterizes vector layers, and produces aligned raster stacks for a
land-use ML model. This guide covers the keys to prepare and how it runs on Azure.

## 1. Which keys do I need?

| API | Key needed? | How to get it |
|---|---|---|
| **Google Earth Engine** | **Yes (service account)** | In your GEE-registered Google Cloud project: IAM → create a service account → grant Earth Engine access → create a JSON key → download it. You need the JSON *contents*, the SA email, and the project id. (GEE is Google's; it works fine from Azure.) |
| **OpenTopography** | **Yes (free)** | portal.opentopography.org → account → "Request an API Key". Rate limit 250/day academic, 50/day non-academic. Only if you fetch DEM from OpenTopography — or enable `DEM_GEE` and skip it. |
| **Copernicus Data Space** | Only if a CDSE layer is enabled | dataspace.copernicus.eu → Sentinel Hub dashboard → OAuth clients → Create. Secret expires ~90 days. Off by default. |
| **Google Maps/Roads** | Optional (avoid) | Billed; OSM/Geofabrik covers roads free. |
| **OSM / WorldPop / GeoBoundaries / STAC** | **No** | Keyless. |

Minimum to run the default config: **GEE service account + OpenTopography key** —
or just **GEE** if you enable `DEM_GEE`.

## 2. Where keys go (secrets)

Never hardcode. Put all keys into **one Azure Key Vault secret** (default name
`landuse-config`) whose value is the JSON in `keyvault.template.json`. Then set
`AZURE_KEY_VAULT_URL=https://<vault>.vault.azure.net/` on the running service.

Authentication uses **Managed Identity** (`DefaultAzureCredential`): grant the
Container App / Function's identity `get` access on the vault. Locally the same
code path uses your `az login`. `azure_utils.load_secrets_into_env()` pulls the
JSON secret into environment variables at startup.

(Key Vault secret *names* can't contain underscores — that's why the real
variables live inside one JSON secret rather than one secret each.)

## 3. Storage

Persistent layers go to **Azure Blob Storage**. Set `AZURE_BLOB_CONTAINER` and
either `AZURE_STORAGE_CONNECTION_STRING` (simplest) or `AZURE_STORAGE_ACCOUNT_URL`
+ Managed Identity (prod). `/tmp` is scratch only. A fresh container reuses layers
a previous run already fetched, so the scheduled update only downloads new years.

## 4. Environment variables

| Var | Purpose | Default |
|---|---|---|
| `AZURE_KEY_VAULT_URL` | Key Vault to load secrets from | (none → env/.env only) |
| `AZURE_KEYVAULT_SECRET_NAME` | secret holding the config JSON | `landuse-config` |
| `AZURE_BLOB_CONTAINER` | persistent cache container | (none → local only) |
| `AZURE_BLOB_PREFIX` | blob key prefix | `landuse` |
| `AZURE_STORAGE_CONNECTION_STRING` | Blob auth (simple) | (none) |
| `AZURE_STORAGE_ACCOUNT_URL` | Blob auth via Managed Identity | (none) |
| `APP_SCRATCH_DIR` | writable scratch dir | `/tmp/landuse` |
| `GCS_EXPORT_BUCKET` | optional GEE→GCS native-res export | (none → tiled download) |
| `TARGET_RESOLUTION` | output resolution (m) | `100` |

## 5. Run modes

**Scheduled data update** (fetches only missing years — the auto-update mechanism):
```bash
python -m landuse_tool.api_connector       # calls update_all()
```
**Streamlit app** (interactive):
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 6. Recommended Azure architecture

Split the heavy fetching from the interactive app:

- **Data fetch job** → **Azure Container Apps Job** with a **cron schedule**
  (or an ACI run triggered by a **Logic App** / **Timer-triggered Function**).
  It runs `update_all()`, which discovers all available years per dataset and
  downloads only the years not already in Blob Storage. When ESRI/MODIS/CHIRPS
  publish a new year, the next run picks it up automatically. Avoid a plain
  Consumption Function for this — GEE downloads can exceed short timeouts; a
  Container Apps Job (or Premium/Container Function) has no such limit.
- **Streamlit app** → **Azure Container Apps** or **App Service (Web App for
  Containers)** — a long-running server. Package as a Docker image because of the
  GDAL-heavy deps (geopandas, rasterio, osmnx). It reads finished layers from Blob.
- **Secrets** → **Key Vault** (loaded at startup via Managed Identity).
- **Persistent layers** → **Blob Storage**. `/tmp` for scratch only.

### Earth Engine + Azure: the one nuance
Earth Engine can export images only to **Google Drive or Google Cloud Storage**,
never directly to Azure Blob. That's fine because the default download path here
**tiles the region and mosaics locally** (no GEE export needed), then uploads the
GeoTIFF to Blob — fully within Azure, good for ~30–500 m at country scale. Only if
you need **native 10 m** country-wide layers do you set `GCS_EXPORT_BUCKET`; then
the flow is GEE → a small GCS bucket → copy to Blob (an extra GCP dependency).

## 7. Data & ML notes

- Every model input ends up a **raster on one grid**. `data_loader.assemble_predictor_stack(year, ...)`
  resamples all raster predictors to the reference grid (nearest year for sparse
  layers), derives slope/aspect from the DEM, and **rasterizes the vector layers**:
  roads/railways/waterways → distance-to surfaces, POIs → heatmap. So the vectors
  become meaningful continuous predictors, not points/lines.
- `ESRI_LULC` and the old `S2TSLULC` were the same product (merged; 2017–2025).
- `GLC_FCS30D` (30 m, 2000–2022) is enabled as a full-depth alternative.
- `GLANCE` and `AGB_ESA` are disabled by default — verify their band/asset before enabling.
