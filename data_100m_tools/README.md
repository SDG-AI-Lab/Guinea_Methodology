# 100 m grid tools

The platform runs both the deforestation and the land-use module on a 100 m grid. This folder
holds that grid and a shared store of layers on it, used by the 100 m runs of
`deforestation_experiments/` and `landuse_experiments/`, so both work on the platform's cells.

| File | What |
|---|---|
| `grid100.py` | The grid (derived as the platform's: Earth Engine's 100 m lattice over the geoBoundaries outline), the cells inside Guinea, cell areas (the platform's formula), and reading and writing layers |
| `fetch.py` | Layers read from Earth Engine as the platform reads them (getDownloadURL at 100 m), OpenStreetMap from the Geofabrik extract with the platform's tags, and local polygons, saved in the store: Hansen shares, Esri land cover (class and class shares), predictors |
| `platform_utils.py` | A verbatim, read-only copy of the platform's `landuse_tool/utils.py` from its main branch (slope, aspect, distance, heatmap), which fixes slope on geographic grids; do not edit |
| `memory_check.py` | Time and peak memory of the heavy steps of a 100 m run on this machine |
| `tessera_blocks.py` | The 24 sample blocks for the TESSERA tests (0.1° tiles: 16 forest blocks drawn by ecoregion in proportion to loss, 8 random land blocks) |
| `tessera_fetch.py` | TESSERA embeddings (Feng et al. 2026, CC0) of the blocks, 2017–2025: each cell's mean of its 10 × 10 pixels. Needs Python 3.12 or later (`geotessera`): `E:\SDG AI Lab\Guinea
ational_scan_2026env_tessera` |
| `tessera_national.py` | The same over all 2,048 tiles with forest, kept as 16 principal components per forest cell (0.4 GB a year; about 9 hours a year with four workers; restartable). Read for 2018 and 2021 |

How the grid was derived and checked, and the memory results: `deforestation_experiments/grid_100m/01_grid_and_store.ipynb`.

## The store

The data are outside git, in `E:\SDG AI Lab\Guinea\data_100m\` (or the folder named by the
environment variable `GUINEA_DATA_100M`):

* `grid.json`, `cells.parquet` (the cells inside Guinea), `manifest.json` (one entry per layer);
* `hansen/`, `landcover/`, `predictors/`: one parquet file per layer, one row per cell inside
  Guinea, one column per band (e.g. a year), in the smallest type that holds it (shares as
  whole percentages, scale 0.01 in the manifest).

Inputs whose sources are coarser than 100 m (rainfall, temperature, soil moisture, night lights,
NDVI) are not stored here; they are read from the 250 m files and resampled when used
(`grid100.resample`). Two processes may write the manifest at once; each layer file carries its
own entry, and `grid100.rebuild_manifest()` rewrites the manifest from the files.

Layers so far (see `manifest.json` for sources): `hansen/hansen_shares`; `landcover/esri_2017`
… `esri_2025`; `predictors/dem`, `worldpop`, `ghsl_built`, `wdpa_year`, `grip4_roads`,
`gaul_admin1`, `osm` (roads, railways, waterways, poi), `ecoregions`, `mines`.

TESSERA (`tessera/`): `blocks.json`, `blocks_cells.npz`, `<year>/<block>.npy` (128 numbers per cell, float16), `glance_points.npz`, and `national/` (`index.npz`, `pca16.npz`, `pc16_2018.npy`, `pc16_2021.npy`).

## Use

```python
import sys
sys.path.insert(0, r"d:\Guinea_Methodology\data_100m_tools")
import grid100 as G

G.grid()                                   # cell size, width, height, transform
G.cells()                                  # cell_id of the 25.1 million cells inside Guinea
G.cell_area_ha()                           # area of each of them (ha)
G.layers()                                 # the manifest
forest = G.read_layer("hansen_shares", ["forest2000"], scaled=True)["forest2000"]   # 0-1, per cell
grid2d = G.to_grid(forest)                 # the same as a 2D grid (NaN outside Guinea)
G.write_layer("predictors", "slope", {"slope": slope_2d}, source="SRTM via Earth Engine")
```

Keep model inputs as values over the cells needed (e.g. forest cells), not as whole grids: 16
predictors for 9 million forest cells take 0.6 GB, 16 whole grids 3.4 GB. Whole-grid steps
(distances, moving windows) are done one layer at a time.

Run with `landuse_experiments/.venv` (numpy, pyarrow, rasterio, scipy, earthengine-api).
