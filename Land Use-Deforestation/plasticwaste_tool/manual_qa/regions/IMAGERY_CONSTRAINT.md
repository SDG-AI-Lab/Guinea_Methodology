# Imagery resolution is NOT the constraint — labels are

Status: established 2026-09-14. **This file previously argued the opposite and
recommended procuring commercial imagery. That recommendation was wrong and is
withdrawn.** The reasoning that replaced it is below, along with the evidence
that overturned it, kept deliberately so the mistake is not repeated.

## Conclusion

Free Esri World Imagery is sufficient for this task at every location in
Guinea. No imagery purchase, no Google Maps Platform billing, no ArcGIS
entitlement and no Maxar/Vantor licence is needed. National coverage is
achievable on the free basemap using 250m cells.

What limits performance is the number and spread of labelled scenes.

## The evidence

### The controlled test

Resolution was isolated properly: take the 124 Conakry labels (the largest
clean set, audited 124/124 distinct images), fetched once at z19, then degrade
each image and restore it to the model's 800px input. A 4× degradation of z19
is exactly what z17 delivers. Same labels, same ground, same model, same
protocol — the only variable is pixels.

| condition | m/px | equivalent zoom | AUC | 95% CI |
|---|---|---|---|---|
| native (Esri Conakry) | 0.29 | z19 | 0.795 | [0.652, 0.889] |
| 2× coarser | 0.59 | z18 | 0.773 | [0.621, 0.877] |
| 4× coarser | 1.18 | z17 — what every other town gets | **0.805** | [0.661, 0.896] |

Paired bootstrap over spatial blocks, native vs 4× coarser (valid pairing, the
same tiles in both arms): **ΔAUC = −0.011, 95% CI [−0.053, +0.028],
P(resolution does not help) = 0.70.**

Performance is flat from 0.29 to 1.18 m/px. This makes sense for the target:
an informal dump large enough to matter spans tens of metres, so it is
resolved many times over even at 1.18 m/px. Sub-metre detail resolves
individual sacks and debris, which is not what the scene classifier keys on.

### Why the earlier conclusion was wrong

It rested on comparing Conakry (z19, AUC 0.756) with Boké (z17, 0.595) — but
those are different places, different labelling sessions, n=19 and n=32, with
CIs that overlap heavily ([0.616, 0.923] vs [0.322, 0.837]). Judging a
difference by two overlapping marginal intervals is the exact error this
project already documented against (Schenker & Gentleman 2001, *Am. Stat.*
55:182–186) — and it was made anyway, in the direction of recommending a spend.

Boké's weaker number is therefore **not** explained by resolution. Remaining
candidates, in order of likelihood: small n (32 scenes), the cross-domain gap
(bauxite/laterite terrain unlike coastal Conakry; Tuia, Persello & Bruzzone
2016, IEEE GRSM), and aggregation noise in negatives re-mapped from sparse 40m
labels. All three are addressed by labelling, not by imagery.

## What remains true about Esri coverage

These measurements stand; only their significance changed.

`World_Imagery/MapServer/identify` reports 0.31–0.34m WorldView/Legion source
imagery from early 2025 over every town, with `MaxMapLevel` 17 (18 at
Nzérékoré, and Conakry serves a genuine z19 despite reporting 17). Above the
cap, Esri returns blank "Map data not yet available" placeholders rather than
errors — which silently corrupted the first Boké labelling session until
placeholder detection was added to `fetch_tiles_esri`.

Esri Wayback (41 of 196 historical releases probed) publishes nothing deeper.
Esri Clarity is sharper at some towns and worse at others, and serves z18 only
at Conakry.

**The practical consequence is only about grid size, not image quality.** At
1.17 m/px a 40m cell is 34 real pixels, so neighbouring cells return duplicate
imagery — that is what made 118 Boké labels collapse into 35 distinct pictures.
250m cells give ~214 real pixels and 51/51 distinct images with zero label
conflicts. Use 250m cells outside Conakry; the imagery inside them is fine.

## Imagery options that were surveyed, for the record

- **OpenAerialMap** — queried the full Guinea bbox: 73 scenes, all in Sierra
  Leone or Mali. **Zero coverage of any Guinean town.** (Two CC-BY drone scenes
  of a *décharge* and *collecteur* near Bamako at ~1.6cm exist and are legally
  clean, but the scale gap to ~1m is severe.)
- **Google Earth Engine** — no global sub-metre imagery. NAIP is US-only,
  Sentinel-2 is 10m, Planet NICFI ~4.8m. The programme's GEE access does not
  help with VHR and was never going to; it is the right tool for the FDI layer,
  which is what actually ships at launch.
- **Google Maps Static API** — licensed and paid; would work, and is
  unnecessary. Separately, the key found in the repository returns HTTP 403
  ("enable Billing on the Google Cloud Project"), so it is not usable as-is.
- **Bing Maps** — free Basic keys retired 2025; the platform is being
  deprecated. Not a path.
- **ESA Third Party Missions / Copernicus Contributing Missions** — free VHR
  for approved institutional projects, but application lead times do not fit
  the schedule. Unnecessary given the result above.

## On Google imagery, which no longer matters here

Labelling in QGIS against a Google basemap is ordinary viewing, and the
coordinates produced are the labeller's own data — that workflow is fine and
should continue if it is easier to navigate. Fetching Google tiles as model
input is a different question, and one that is now moot: Esri serves adequate
imagery for free.
