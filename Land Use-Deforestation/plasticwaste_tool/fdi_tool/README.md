# fdi_tool — importable package (DPWMS component 1)

Floating-debris detection on **water**, from Sentinel-2. This is the mesoscale
half of the National Hotspot Map (GEDH-FR-0503) and the source of the Monthly
FDI Archive (GEDH-FR-0507). It is the only one of the three DPWMS components
that is fully operational and validated end to end.

| module | what it holds |
|---|---|
| `spectral.py` | offline NumPy FDI / NDVI / NDWI, the positive water constraint, and the Earth Engine cross-check |
| `detection.py` | thresholding (Otsu, mean+3σ, fallback), the Digital Earth Africa conjunction rule, patch vectorisation and scoring |
| `pipeline.py` | `run_from_arrays` / `run_from_gee` / `run_national` / `run_monthly_batch`, plus the offline synthetic self-test |
| `deployment.py` | AOI overlap checks, mosaic de-duplication, OSM port zones, the water-fraction guard, review sampling, admin attribution |
| `plasticwaste_analysis.js` | the Earth Engine App prototype the Python port follows |

---

## These files moved here; they were not copied

Unlike `dumpsite_tool/` and `characterisation_tool/`, whose modules are
generated copies of numbered research scripts, **these four modules are the
originals**. They were moved out of the top level so that each of the three
components is a package of its own, and the old top-level names are now
re-export shims:

```
plasticwaste_tool/spectral.py    ->  re-exports fdi_tool.spectral
plasticwaste_tool/pipeline.py    ->  re-exports fdi_tool.pipeline
plasticwaste_tool/deployment.py  ->  re-exports fdi_tool.deployment
```

There is no second copy of the code, so there is nothing to keep in sync.
**Edit the files here.** The shims exist so that callers written against the
previous flat layout — `plasticwaste_analysis.ipynb` among them — keep working
without modification.

`detection.py` is the one that is not a pure shim. The top-level
`plasticwaste_tool/detection.py` re-exports this package's detection functions
**and** owns `classify_dumpsite_candidates()`, because it is the single stable
interface the backend calls for both layers of the hotspot map. The backend
should not import `fdi_tool` directly.

---

## What the output is, and is not

**Floating-debris candidates, not plastic.** The index responds to any material
that raises near-infrared reflectance above the water baseline, so seaweed,
algal mats, timber and sea foam are flagged identically to macroplastic.
Discriminating them needs the labelled spectral library behind Biermann et
al.'s naïve Bayes classifier, which does not exist for Guinea — this module
uses the index with a threshold rule, which is a deliberate simplification of
the reference method that gives up precisely its ability to identify material.

The scoring column is `floating_debris_score`. It was renamed from
`plastic_concentration_score` because the original name asserted an
identification the method does not make. **Do not rename it back**, and note
that the repo's pre-migration `stats.py` still referred to the old name.

---

## The detection rule, and why each part is there

```
mask to water   (WorldCover class 80, minus built-up dilated 20 m)
t_FDI  = mean(FDI)  + 3σ   over all water pixels
t_NDVI = mean(NDVI) + 3σ   over the same population
debris = (FDI > t_FDI) AND (NDVI > t_NDVI)
```

Three things about this are easy to get wrong, and each was got wrong at some
point during development:

1. **The water constraint is positive, not an exclusion.** Excluding vegetation
   and open water leaves every other non-vegetated surface — bare soil, sand,
   concrete, port and industrial rooftops — all of which have a large FDI
   excess. Before the constraint, photo-interpreted precision was 33% in the
   HIGH band and 71% in LOW: the confidence ranking was fully inverted, because
   bright land beat faint genuine debris.

2. **NDVI is used inclusively, as a debris criterion.** An earlier version
   excluded `NDVI > 0.15` as vegetation. That is the opposite sign to both
   references — Digital Earth Africa require NDVI *above* its own mean + 3σ,
   and Biermann et al. separate debris classes within the NDVI–FDI plane
   precisely because debris sits high on both axes relative to seawater.

3. **The rule runs per region, not once nationally.** mean + 3σ is measured
   against a background population; pooling all coastal water yields one
   compromise threshold, too high offshore and too low in turbid estuaries. The
   highest regional threshold is 4.8× the lowest. Each patch therefore carries
   its own region's thresholds and water-population size as attributes.

A region whose AOI holds little water produces an inflated threshold and can
return zero regardless of what is present. `deployment.water_fraction_for_bbox`
exists for this: regions below 10% water are marked **not assessed** rather
than reported as zero.

---

## Offline self-test

```
python -m plasticwaste_tool.pipeline
```

Runs the full chain — index computation, detection rule, vectorisation,
scoring, summary statistics, sensitivity analysis — against a synthetic fixture
whose debris signature is constructed to match the references. **No network
access and no credentials required.**

The test repairs two environment problems itself before it starts, because a
self-test that fails for reasons unrelated to the code is worse than no test:

* `PROJ_LIB` pointing at another GDAL install (a PostGIS one, typically) makes
  the first `CRS.from_epsg(4326)` raise *"Cannot find proj.db"*. `geo_env.fix_proj`
  repoints it at rasterio's bundled copy, for this process only.
* The tick characters in its own output are not encodable in cp1252, so on a
  default Windows console it used to die in its reporting after the analysis
  had already passed.

---

## Reference

Biermann, L., Clewley, D., Martinez-Vicente, V. & Topouzelis, K. (2020).
Finding Plastic Patches in Coastal Waters using Optical Satellite Data.
*Scientific Reports* 10, 5364.

Digital Earth Africa, *Detection of Floating Marine Debris* — the operational
thresholding workflow this rule follows.
