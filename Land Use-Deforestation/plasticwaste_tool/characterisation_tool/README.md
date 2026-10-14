# characterisation_tool — importable package (DPWMS component 3)

These are **copies** of the numbered research scripts 10–23 in the parent
folder, renamed so Python can import them and with their `sys.path` hacks
replaced by relative imports. The numbered originals stay where they are and
are what `plastic_characterisation_workflow.ipynb` and
`21_build_methodology_doc.py` use; nothing about the notebooks changes.

Regenerate with `python plasticwaste_tool/24_build_component_packages.py
--only characterisation`. Do not hand-edit the copies until the numbered
originals are retired, or the two will drift apart. `__init__.py` and this
README are never generated.

---

## This component returned a negative result, and that is the finding

It asked whether the **plastic content of an already-detected dumpsite** can be
estimated from satellite imagery, which is what the programme concept note
commits to: *"apply more advanced AI and multispectral analysis to these
identified sites to estimate the plastic concentration, distinguishing
plastic-heavy hotspots from organic dumps."*

It was designed as four falsification gates, ordered cheapest first, each able
to end the work. That ordering matters for how the results read: these are
outcomes of tests **designed to be capable of returning them**, not failures
encountered while attempting an implementation.

| gate | question | modules | outcome |
|---|---|---|---|
| 1 | Does hyperspectral imagery of Guinea exist? | `hyperspectral_coverage` | **passed** — EMIT + EnMAP cover 110 of 111 dump points below 20% cloud |
| 2 | Do Sentinel-2 plastic indices separate dumps from non-dumps? | `s2_indices`, `evaluate_indices` | **falsified** |
| 3 | Does the existing FDI code respond on land? | `fdi_on_land` | **falsified** |
| 4 | Do hyperspectral spectra show a C–H signature? | `enmap_download`, `enmap_spectra`, `separability` | **falsified** |

**Gate 2.** Across ten published indices on 646 labelled points, NDVI — which
has no plastic sensitivity whatsoever — scored AP 0.440 against a 0.172
baseline, matching or beating PMLI (0.436), PGI (0.208) and every other plastic
index. At the dumpsite's own scale the plastic indices add nothing over NDVI
(ΔAP −0.003, 95% CI [−0.090, +0.082]). The only footprint at which they help is
the 250 m box — precisely the footprint `dumpsite_tool` established measures
the neighbourhood rather than the dump.

**Gate 3.** FDI is not flat on land — dumps read a median +0.0200 against
−0.0057 — but it never beats NDVI, and the mechanism is documented in this
project's own code: bare soil, sand, concrete and industrial rooftops have a
large FDI excess, which is *why* the marine pipeline carries a positive water
constraint. A dumpsite triggers FDI by the same route as a warehouse roof. The
operational pipeline, with its real masks, yields **0 of 646 land points** —
correctly scoped to water, and it must not be repurposed.

**Gate 4.** Six 0%-cloud EnMAP acquisitions, four regions, three years, 1256
spectra over 30 distinct sites, against ring controls drawn 150–450 m around
each dump so that settlement cannot be the confound. Weak separability exists
(site-blocked out-of-fold lift 1.95) but both diagnostic features rank mid-pack
among 218 bands — 1215 nm at 140, 1730 nm at 128 — and removing brightness cuts
the lift to 1.41. The direction of change is itself evidence: on a single scene
the features ranked 19 and 25; with six scenes they *sank*. A real signal rises
as noise averages out.

**Why, physically.** Plastic identification rests on C–H absorption near
1215 nm and 1730 nm. Neither falls inside a Sentinel-2 band — 1215 nm sits in a
669 nm gap between B9 and B11, 1730 nm in a 589 nm gap between B11 and B12. The
sensors that do resolve them (EnMAP 30 m, EMIT 60 m) are far coarser than the
target: Guinea's dumpsites measure a median **14.0 m** area-equivalent side,
which is 0.49 of a Sentinel-2 SWIR pixel, 0.22 of an EnMAP pixel and 0.05 of an
EMIT pixel. **No pixel anywhere in Guinea contains only dumpsite**, so spectral
library matching is unavailable in principle rather than merely difficult.

### Consequence for deployment

**There is no plastic-fraction layer, and none of these modules should be wired
to a backend endpoint implying one.** Shipping a bare-ground proxy under a
plastic label would render convincingly on a map, colour a legend plausibly,
and be wrong. The FDI code must likewise not be extended to land.

---

## What the component does produce

| module | output | status |
|---|---|---|
| `extent`, `extent_annotator`, `ingest_annotations`, `extent_second_read` | the first measured dumpsite size distribution for Guinea — median 14.0 m, p25 9.9 m, p75 19.5 m (n=46) | measured; report as a **size class**, not a per-site area |
| `fdi_on_land` | waterside flag — 19 of 111 labelled dump points on WorldCover water, wetland or mangrove | measured; links the land layer to the marine FDI layers |
| `mobilisation` | `mobilisation_risk` and `ocean_leakage_risk` per site | `mobilisation_risk` **validated** (AUC 0.771, p=0.001); `ocean_leakage_risk` unvalidated |

**On extent precision.** A blind second read of 12 randomly chosen sites gives
bias −4.0 m, 95% limits of agreement [−19.1, +11.2] m, median relative
difference 42%, Lin's concordance 0.49. Agreement is poor in absolute terms and
moderate in ranking, and the site notes show why: the disagreements are about
*where a diffuse scatter ends*, not about measuring a well-defined object — the
same ambiguity that defeated the automated estimator. The second read was an
independent reproducibility check, **not a second human interpreter**. That
imprecision does not reach the conclusion: a site must reach 30 m to fill one
EnMAP pixel, and the probability that the median site does is under 1% under
either read.

**On mobilisation.** This sidesteps the unmeasurable quantity rather than
depending on it. Characterisation asks what a dumpsite is *made of*, which
needs a per-site material measurement no satellite can supply; mobilisation
asks where waste *goes*, which is a function of terrain, rainfall and drainage
— all directly observable. The first version of the model **failed** its
independent check against observed flooding (AUC 0.469), and that failure
produced its main correction: eleven of thirteen flooded sites lie on the
Niger–Milo floodplain, where local rainfall is *lower* than coastal Conakry's. A
floodplain site is not inundated by local rain but by upstream catchment
discharge, so the corrected model represents wash-off and inundation as two
distinct pathways combined as a union. No weight was tuned against the thirteen
flooded sites — the change is structural, taken from Meijer et al.

The mass term remains a **waste** proxy, not a plastic one. Converting it would
need a Guinean waste-composition figure (World Bank *What a Waste*, or ANASP),
which is a data-sourcing gap rather than a remote-sensing limit.

---

## Paths

Every module resolves against `TOOL`, the parent `plasticwaste_tool`
directory, because `runs/`, `manual_qa/` and the frozen models live one level
up. Saved run outputs under `runs/` are small JSON and CSV and are what the
notebook and the methodology document read; the EnMAP scenes themselves are
not in the repository.

## References

Meijer, L.J.J., van Emmerik, T., van der Ent, R., Schmidt, C. & Lebreton, L.
(2021). More than 1000 rivers account for 80% of global riverine plastic
emissions into the ocean. *Science Advances* 7(18), eaaz5803.

Mellink, Y., van Emmerik, T., Kooi, M., Laufkötter, C. & Niemann, H. (2022).
The Plastic Pathfinder: A macroplastic transport and fate model at river basin
scale. *Frontiers in Environmental Science* 10, 979685.

Tellman, B. et al. (2021). Satellite imaging reveals increased proportion of
population exposed to floods. *Nature* 596, 80–86. (Global Flood Database.)

Wolf, C. et al. Satellite monitoring of terrestrial plastic waste. *PLOS ONE* —
detects waste **sites** from Sentinel-2 and states it cannot determine plastic
fraction within a site.
