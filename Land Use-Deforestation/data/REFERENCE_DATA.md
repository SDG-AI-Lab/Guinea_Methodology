# Reference data — provenance, licences and checksums

Static reference datasets used by the analysis toolchain and by the GEDH map.
"Static" means obtained once from a pinned source, not re-fetched on a schedule
(see `specs/gis-spec-set/10-data-inventory-and-sources.md` for the full
live/derived/static/on-demand classification).

Checksums are SHA-256 of the file bytes as committed. The pinned files are
marked `-text` in `.gitattributes`, so they are checked out byte-for-byte on
every platform and the checksums hold on Windows as well as in the containers.

Verify:

```bash
cd "Land Use-Deforestation/data"
sha256sum -c <<'EOF'
f48eecbd2ad423f8bf1cc0ef54019bd77b274369b2e2d85c6ac272728fc9881a  geoBoundaries-GIN-ADM0.geojson
6e0f0cc87117280998aad00c183e5fe8ee1e300fdf4bdb8a17f46a7c013de5a2  geoBoundaries-GIN-ADM0_simplified.geojson
60a8c39e2f66d51cd20876a079e076b5fb9bc3c753ae4a7127d1272ccc4083fa  geoBoundaries-GIN-ADM1.geojson
7cf3189bd308c1146a639eb670deb1a5666614ea52b9157ba453fcd350dacccd  geoBoundaries-GIN-ADM1_simplified.geojson
70e81f12c9836a13fc330d16c156cae2413ce65c097caededbdb09e162de1fc8  geoBoundaries-GIN-ADM2.geojson
d5150430f42f765b7418123f2d8f4ea6e1c770c9d23e49494488760a7d9f5680  geoBoundaries-GIN-ADM2_simplified.geojson
b0c1ab8d827012ce6515fd041d27d9a72a29614604925cffd824e7544751a783  geoBoundaries-GIN-ADM3.geojson
fc27f2c48b4d1801d68b85498962b014bb3dc87e251300f40bac5c3a0b6ac95a  geoBoundaries-GIN-ADM3_simplified.geojson
17f628c81f3d349d5eef94c5653774861f09a8af0084950f0b975deeaf83c32f  guinea_mining_polygons.geojson
203221c3f6757da43f8bc0b18bef9991ab2df721d728385a574f984f4d588d0b  gin_pd_2020_1km_UNadj.tif
EOF
```

---

## 1. Administrative boundaries — geoBoundaries gbOpen, Guinea (GIN)

Pinned to geoBoundaries release commit **`9469f09`** (build date 12 Dec 2023),
retrieved 2026-09-24 from `https://www.geoboundaries.org/api/current/gbOpen/GIN/ALL/`.
Download URLs follow `https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/GIN/ADM{n}/{file}`.

| Level | Meaning (FR / EN) | Units | Source year | Upstream source | Licence | Files |
|---|---|---|---|---|---|---|
| ADM0 | Pays / Country | 1 | 2020 | OpenStreetMap | **CC BY-SA 2.0** | `geoBoundaries-GIN-ADM0.geojson`, `_simplified` |
| ADM1 | Région / Region | 8 | 2017 | WFP, OCHA ROWCA | CC BY 3.0 IGO | `geoBoundaries-GIN-ADM1.geojson`, `_simplified` |
| ADM2 | Préfecture / Prefecture | 34 | 2017 | WFP, OCHA ROWCA | CC BY 3.0 IGO | `geoBoundaries-GIN-ADM2.geojson`, `_simplified` |
| ADM3 | Sous-préfecture / Sub-prefecture | 340 | 2021 | WFP, OCHA ROWCA | CC BY 3.0 IGO | `geoBoundaries-GIN-ADM3.geojson`, `_simplified` |

These are all the levels geoBoundaries publishes for Guinea (gbOpen and
gbHumanitarian carry the same four levels and counts).

Properties on every feature: `shapeName`, `shapeISO`, `shapeID`, `shapeGroup`, `shapeType`.

Things to know before joining on these files:

- **Stable identifier is `shapeID`.** `shapeISO` is filled only for ADM0 (`GIN`)
  and ADM1 (ISO 3166-2:GN — `GN-B` Boké, `GN-C` Conakry, `GN-D` Kindia,
  `GN-F` Faranah, `GN-K` Kankan, `GN-L` Labé, `GN-M` Mamou, `GN-N` Nzérékoré).
  ADM2 and ADM3 have empty `shapeISO`.
- **Names carry no diacritics** (`Boke`, `Labe`, `Nzerekore`). French display
  names with accents are a separate, curated field (spec doc 13).
- **No parent codes.** The hierarchy ADM3 → ADM2 → ADM1 is not in the files; it
  is derived by representative-point containment when loading (spec doc 13).
- **ADM0 is ShareAlike.** Anything distributed that is derived from the ADM0
  outline (a clipped raster mask shipped as a download, for example) inherits
  CC BY-SA 2.0. Display on the map is attribution only.
- `geoBoundaries-GIN-ADM1.geojson` was already in this folder and is read by
  `plasticwaste_tool/leakage_tool/population.py` and
  `plasticwaste_tool/29_prepare_river_corridors.py`. The previous copy had the
  same 8 features, IDs and coordinates; it has been replaced with the
  byte-exact pinned release so its checksum matches this manifest.

Required attribution: *"Administrative boundaries: geoBoundaries (Runfola et al.
2020, PLOS ONE 15(4): e0231866), gbOpen GIN build 9469f09; ADM0 © OpenStreetMap
contributors (CC BY-SA 2.0); ADM1–ADM3 WFP / OCHA ROWCA (CC BY 3.0 IGO)."*

## 2. Mining footprint — Maus et al. (2022), Guinea subset

| | |
|---|---|
| File | `guinea_mining_polygons.geojson` (the path `landuse_analysis_improved.ipynb` already expects) |
| Source | Maus, V. et al. (2022): *Global-scale mining polygons (Version 2)*. PANGAEA, https://doi.org/10.1594/PANGAEA.942325 |
| Source file | `global_mining_polygons_v2.gpkg`, SHA-256 `6936e2743a21ba952b4de3e4b6f77099a0aa4479f7e2639d9d1b0787876a3ea1` |
| Subset rule | `ISO3_CODE == 'GIN'` (all 128 features that intersect the ADM0 outline; none from neighbouring countries) |
| Content | 128 polygons, 231.34 km² total (`AREA`, km²); EPSG:4326 |
| Method | Visual interpretation of the 2019 Sentinel-2 cloudless mosaic (10 m), within 10 km of S&P mine coordinates. Reported accuracy: overall 88.3%, user's accuracy (mine) 97.2%, producer's accuracy (mine) 78.9% |
| Licence | **CC BY-SA 4.0** — redistribution allowed with attribution; derivatives must carry the same licence |

**What this is not:** it is the *physical footprint* of mining (pits, tailings,
waste dumps, ponds, plant) as seen in 2019 — not the legal concession
boundaries. The Guinea mining cadastre (concessions / permits) is behind a login
and is not available (`deforestation_tool/config.py` → `MINING_CADASTRE_SOURCE`,
`enabled: False`). The map must label this layer as mining footprint, never as
"concessions", and show its 2019 reference date.

Required attribution: *"Mining footprint: Maus et al. (2022), Global-scale
mining polygons v2, PANGAEA, CC BY-SA 4.0."*

## 3. Other static files already in this folder

| File | What | Source / licence | Used by | Status |
|---|---|---|---|---|
| `gin_pd_2020_1km_UNadj.tif` | Population density 2020, 1 km, unconstrained, UN-adjusted | WorldPop, CC BY 4.0 | `leakage_tool/population.py` (runtime) | **Runtime input, deliberately fixed** — leakage is re-run on parameter change only |
| `hotosm_gin_points_of_interest_points_geojson.geojson`, `hotosm_gin_roads_lines_geojson.zip` | OSM POIs, roads | HOT export of OSM, ODbL | earlier notebooks | Superseded by the Geofabrik/Overpass fetch in `landuse_tool/config.py` |
| `Guinea_LandCover_{2001..2024}.tif`, `Guinea_AGB.tif`, `Guinea_SRTM_DEM_Slope_Aspect.tif`, `Guinea_SoilMoisture.tif`, `cache/lc_*.tif` | Hand-fetched rasters | various | earlier notebooks, `test_run.py` | Superseded by the API fetchers; keep as archive, do not publish as layers |
| `../plasticwaste_tool/data/WB_WAW_WIDEF.csv` | What a Waste global database | World Bank, CC BY 4.0 | `leakage_tool/parameters.py` | Runtime input, fixed |
| `../plasticwaste_tool/data/rivers/*.geojson` | River corridors: 1,929 reaches with Q ≥ 25 m³/s, buffered, tiled | Derived from HydroRIVERS v1.0 by `29_prepare_river_corridors.py` | FDI river runs | Runtime input. **Licence to confirm** — see §4 |

## 4. Deliberately NOT in this repository

| Dataset | Why not committed | How a deployment obtains it |
|---|---|---|
| **HydroRIVERS v1.0, Africa** (`HydroRIVERS_v10_af_shp.zip`, 107.9 MB, SHA-256 `63cc615134d8812b73f27ee9dfd9d790c520af96bc609dfc97bcd47811fec125`) | The HydroSHEDS License Agreement (§2.1.2) allows distribution only incorporated into a derivative product and states "in no event shall Licensee license or distribute the Licensed Materials as a stand-alone product". A clip committed to a public repository is stand-alone redistribution. Showing it as map tiles inside the platform is incorporation, and is allowed with attribution. | Fetched at deploy time from `https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_af_shp.zip`, checksum-verified, clipped to ADM0 (22,739 reaches) and loaded into the spatial store (spec doc 10, layer `river-network`). The same clip must not be offered in the Data Export Hub. |
| **WDPA extract** (`protected_conserved_areas_wdpca_polygons (1).geojson`, removed by GEDH feature 002) | The WDPA terms (UNEP-WCMC) allow display with attribution but not redistribution, and a file committed to a public repository is redistribution. Nothing read it. | The `protected-areas` layer is fetched from `WCMC/WDPA/current/polygons` on Earth Engine into the private input store, keyed by version (`backend/gis/fetchers/wdpa.py`). |
| **MARIDA** (5.5 GB, Zenodo `10.5281/zenodo.5151941`) | A labelled training/benchmark set of marine-debris image patches, not a map layer. Its results are already committed under `plasticwaste_tool/manual_qa/runs/marida/` and served by `/api/plasticwaste/evidence/benchmark/`. | Only needed to re-run the benchmark (`32_run_marida.py`); download from Zenodo. |
| **Guinea mining cadastre** (concessions) | Not openly available (authenticated ArcGIS service). | Request from the Ministry of Mines and Geology / CPDM. Until then, §2 is the mining layer. |
| **AerialWaste checkpoints** (`aerialwaste_model/weights/*.pth`, ~120 MB each) | Licensed CC BY-NC-ND 4.0 (non-commercial, no derivatives), over GitHub's 100 MB file limit, and **not used at runtime**: every deployed dumpsite head (`dumpsite_tool/models/probe*.joblib`, `backbone: imagenet_resnet50_gap2048`) is fitted on stock ImageNet ResNet-50 features. The checkpoints are only used by research benchmark scripts. | Keep ignored. Obtain from the AerialWaste authors only to re-run the benchmarks. |
| **ImageNet ResNet-50 weights** (`resnet50-*.pth`, ~98 MB, torchvision `IMAGENET1K_V1`) | Downloaded by torchvision on first use; not ours to redistribute. | Must be baked into the backend image at build time (or a persistent `TORCH_HOME` volume) so the on-demand location check does not depend on reaching `download.pytorch.org` at runtime (spec doc 41). |

## 5. Retrieval dates

The date each static file was obtained, recorded as the provenance of its layer version
(GEDH feature 002, FR-047). Where the original download date was not written down, the
earliest record is used and named; no date is entered without its evidence.

| File | Retrieved | Evidence |
|---|---|---|
| `geoBoundaries-GIN-ADM{0..3}[_simplified].geojson` | 2026-09-24 | §1 ("retrieved 2026-09-24") |
| `guinea_mining_polygons.geojson` | 2026-09-24 | commit `c616044` (subset extracted and committed that day) |
| `gin_pd_2020_1km_UNadj.tif` | 2026-07-10 | commit `5f2c874`, the earliest record; the download date itself was not recorded |
