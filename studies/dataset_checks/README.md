# Dataset checks: can the products we use be trusted for Guinea?

The deforestation and land-use work rests on two global products: **Hansen Global Forest Change** (forest loss) and **Esri Land Use/Land Cover** (annual land cover). This folder checks them, and their alternatives, in two tiers and two follow-up checks. It holds only dataset checks; model work is in `landuse_experiments/` (land use), `deforestation_experiments/` (deforestation), and `plasticwaste_experiments/`.

The accepted way to judge a map is its accuracy against independent reference data, using the national definition (Olofsson et al. 2014). That is how Guinea's NERF was made (1,688 plots interpreted in Collect Earth Online). NERF's plots are not public, and no third party is contacted for now, so the checks use published and public data only. Our own sample is recommended for later.

## Order

| Notebook | Tier | Question | Status |
|---|---|---|---|
| `01_desk_check.ipynb` | 1 | What do the products say about Guinea, how do they compare with the official figures and with each other, and how accurate are they reported to be? | Done: Hansen's loss is the size of NERF's deforestation + degradation; TMF covers 3.4 % of Guinea; Esri changes 10–15 % of Guinea a year and a third reverts; no published accuracy is for Guinea |
| `02_public_reference_data.ipynb` | 2 | How accurate are the products against public reference data that already cover Guinea? | Done: at 565 GLanCE points WorldCover agrees best (86 %), Esri 75 %; interpreters' trees have a median Hansen canopy of 28 %; Geo-Wiki unusable in Guinea; change not testable (8 changed points) |
| `03_canopy_threshold.ipynb` | Follow-up | Which canopy threshold matches trees best, in each ecoregion? | Done: 25–30 % in the forest ecoregions (97 % of the loss), 15–25 % in the savanna; keep one national 30 %; 25 % raises the loss by a quarter |
| `04_loss_after_clearing.ipynb` | Follow-up | What does the land become after forest loss: how much is permanent, how much grows back? | Done: most loss grows back; Esri sees tree cover again within 3–4 years, GLAD sees short trees; the permanent part (at most a tenth) is within Esri's noise |

**Tier 1, desk check.** Builds on the stage-1 assessment below and adds what it does not cover:

* forest **loss** products side by side: Hansen, JRC Tropical Moist Forests (deforestation and degradation separated; Vancutsem et al. 2021) and the NERF figures;
* how consistent Esri is from one year to the next (change that reverts the following year);
* the published accuracy of each product.

It ends with one table: for each product, why we use it or not, and its known limits.

**Tier 2, public reference data that include Guinea.** Candidates, to be checked for how many points fall in Guinea and whether their dates and classes fit:

* GLanCE training data (Stanimirova et al. 2023): 565 points in Guinea, mostly labels for 2018, only 8 with change;
* Geo-Wiki crowdsourced land-cover reference data, 2011–2012 (Fritz et al. 2017): 177 areas of about 1 km, but 331 of 341 ratings made without high-resolution imagery;
* Dynamic World expert-labelled tiles (Brown et al. 2022): none in Guinea, and not independent of Esri.

They were made for other purposes and are not a probability sample, so they test the products only roughly, and only for one date.

**What tier 2 shows.**

* **Land cover:** WorldCover agrees best with the interpreters (86 %), then Dynamic World (77 %) and Esri (75 %), the same ranking as globally. Esri's errors are between trees, grassland and cropland, the classes it flips between every year.
* **Forest:** the definition matters more than the product. Where interpreters saw trees, Hansen's canopy has a median of 28 %. At 10 % Hansen calls too much forest (user's accuracy 57 %); at 30 % it misses more than half of the trees (producer's accuracy 41 %). WorldCover balances best (93 % and 96 %).
* **Change:** not testable, only 8 of the 565 points changed.

**Follow-up checks (03–04).** From the same saved tables, no new data.

* **03, canopy threshold:** against WorldCover and GLanCE, the best threshold is 25–30 % in the forest-savanna, lowland and montane forests, and 15–25 % in the savanna, where trees are more open. Between 25 % and 30 % the references hardly differ, but the national loss changes by a quarter (2,235 against 1,770 km² a year).
* **04, after the loss:** of the forest lost in 2013–2016, Esri calls about 80 % tree cover in every later year, close to undisturbed forest. Half is tree cover again one year after the loss, all of it after three to four years. GLAD shows the trees are short: tall trees (10 m or more) fell from 45 % to 16 %. The share that never returns to tree cover is no higher than in undisturbed forest, so permanent clearing cannot be measured with these maps: at most a tenth of the loss, about 180 km² a year. TMF, in the moist south-east only, still calls half of it deforested in 2025.

## Dataset decision (after tiers 1–2)

| Use | Product | How | Why |
|---|---|---|---|
| Forest loss | Hansen Global Forest Change | From 2013, canopy > 30 %, reported as forest loss (tree-cover loss), not deforestation | The only national, annual, 30 m loss record (01 §5); one method since 2013; 30 % is the team's choice in the stage-1 report (Global Forest Watch convention, close to FAO's forest area); its loss is the size of NERF's deforestation + degradation (01 §3) |
| Annual land cover | Esri | With its noise stated wherever change is reported | The only annual 10 m map up to 2025; Dynamic World is no more stable (01 §4); 75 % agreement with interpreters in Guinea (02 §2) |
| Reference for the canopy threshold (check 03) | ESA WorldCover | Only to check Hansen's threshold; not used in the land-use or deforestation models | The most accurate in Guinea (86 %, 02 §2) and globally, but only 2020 and 2021, so it cannot replace Esri's annual maps |
| Moist forest | JRC TMF | To check Hansen in the south-east | Separates deforestation from degradation, but covers only 3.4 % of Guinea (01 §3) |

Not used: GLC_FCS30D (stops in 2022) and Dynamic World (no more stable than Esri, and costly to read every year).

After checks 03–04, the forest-loss line stands:

* **One national threshold of 30 %**, stated with every figure, and 25 % shown as a sensitivity (03). Per-ecoregion thresholds change little: a lower one in the savanna adds 4 % to the loss.
* **All loss, not only permanent loss**, since most of it grows back and the permanent part cannot be told apart from map noise (04). Most of the loss is a fallow cycle, which explains why it is larger than NERF's deforestation.

Still open: how NERF counts fallow clearing, and the accuracy of the products in Guinea, and of change, without our own sample.

**Recommended for later: our own sample.** Not done now, as interpretation takes more time than the project has. Stratified random points interpreted on high-resolution imagery in Collect Earth Online (free, FAO; the tool NERF used), with the national forest definition. The strata come from the stage-1 assessment: agreed forest, agreed non-forest, disagreement, and the loss hotspots, so that the effort goes where the products disagree and where change happens (Olofsson et al. 2014). Gives the accuracy of every product and area estimates with confidence intervals. Interpreters record canopy cover (not only forest / non-forest), so that any forest definition can be scored (tier 2 found the definition matters more than the product), and the land cover of each year, so that change is tested.

## Stage 1 (already done, in `studies/`)

Report: `docs/forest_loss_assessment.qmd` (published at
https://sdg-ai-lab.github.io/Guinea_Methodology/forest_loss_assessment.html).

| Notebook | What it holds |
|---|---|
| `deforestation_hansen.ipynb` | Hansen loss, wall to wall, by ecoregion and year |
| `hansen_annual_maps.ipynb` | Annual maps, the 12 Hansen releases, the 2013 method break |
| `lulc_benchmark.ipynb` | Hansen and 9 other products at 56,712 points, against NERF and FAO FRA |
| `report_1.ipynb` | Hansen per resolution-8 hexagon, concentration and hotspots |
| `forest_loss_drivers.ipynb` | Drivers of Hansen loss (WRI) by ecoregion |
| `ecoregions.ipynb`, `satellite_mosaic.ipynb`, `investigate_tier.ipynb` | Study areas, image mosaics |

Main findings: Hansen's forest area depends on the canopy threshold (at the legal 10%, most wooded savanna is forest); its loss is tree-cover loss, about 3 times NERF's deforestation at 30% canopy and 4.5 times at 10%; its series breaks in 2013; the 10 m land-cover maps disagree by a factor of two to three on cropland and built-up land, and the products agree least in the forest–savanna mosaic, where most loss happens.

## References

* Brown, C.F., Brumby, S.P., Guzder-Williams, B., et al. (2022). Dynamic World, near real-time
  global 10 m land use land cover mapping. *Scientific Data* 9, 251.
* Fritz, S., See, L., Perger, C., et al. (2017). A global dataset of crowdsourced land cover and
  land use reference data. *Scientific Data* 4, 170075.
* Olofsson, P., Foody, G.M., Herold, M., et al. (2014). Good practices for estimating area and
  assessing accuracy of land change. *Remote Sensing of Environment* 148, 42–57.
* Stanimirova, R., Tarrio, K., Turlej, K., et al. (2023). A global land cover training dataset
  from 1984 to 2020. *Scientific Data* 10, 879.
* Vancutsem, C., Achard, F., Pekel, J.-F., et al. (2021). Long-term (1990–2019) monitoring of
  forest cover changes in the humid tropics. *Science Advances* 7(10), eabe1603.
