# Literature notes for the methodology document

Collected during the benchmark. Each entry gives the claim and its source. Items marked VERIFY still need a check against the full text before citing.

## Window size (80, 150, 300 m)

**What other studies use:**

| Source | Footprint per image | Resolution |
|---|---|---|
| AerialWaste (Torres & Fraternali 2023, *Scientific Data* 10:63) | 210 × 210 m, centred on the site: "one square image is created centered at that location and covering an area of 210 × 210 meters" | 0.2–0.5 m |
| UNDP Guatemala 2025 | about 1000 px at about 30 cm, so about 300 m (§4) | 30 cm |
| Sun et al. 2023 (*Nature Communications* 14:1444) | 1024 × 1024 px tiles, "about 614 × 614 m²" | 0.3–1 m |
| Gibellini et al. 2025 (*Waste Management Bulletin* 3:100246; arXiv 2502.06607) | Tested 100, 150 and 210 m context sizes (210 m is AerialWaste's default) | 20, 30 and 50 cm |

**Why surrounding context matters:**
- AerialWaste: "the recognition of the relevant scenes might need a varying degree of context".
- Its experts use clues around the site: "trucks, the isolation of the place, secondary access roads, and stressed vegetation".
- This is why its classifier uses an FPN (several scales).

**Object size:** waste sites in Guinea measure about 14 m (our extent measurement, 2026-09). At 1.17 m per pixel that is about 12 pixels:

| Window | Share of the window taken by a 14 m site | What the window covers |
|---|---|---|
| 80 m | about 3 % | The site plus its immediate surroundings |
| 150 m | about 1 % | Closest to AerialWaste's 210 m on the small side |
| 300 m | about 0.2 % | Matches Guatemala's footprint; brackets AerialWaste from above |

**Window size tested directly for waste sites (Gibellini et al. 2025, §4.1.2, read 2026-11-14):**
- "only a moderate dependency on the context size appears for ResNet-50 models ..., with smaller context areas resulting preferable".
- They explain it by convolution's local dependency, "overcome by Swin-T thanks to its attention module, which increases the transformer robustness to variations in image and context size".
- Resolution between 20 and 50 cm/px made no difference to either network.
- Our benchmark agrees: the ResNet-50 models do best at 80 m, and DINOv3 (a transformer) changes little between 80 and 150 m.

**Our result for 300 m (benchmark 2026-11-13):** tested for seven of the eight methods. Each scored a lower town-test AP at 300 m than at 80 or 150 m. The AerialWaste training protocol was not run at 300 m.

**Evidence that a too-large window fails:** in our September test, a 250 m footprint made the model a settlement detector. Residential negatives scored as high as real sites (0.626 vs 0.618), and moving to point-centred crops fixed it.

**Why 80 m is in the set:** it is the window of the current model and of the labelling (blind grids and November rounds), so it is the reference.

**Patch size is normally chosen by testing several sizes:** Nogueira et al. 2019, "Dynamic Multicontext Segmentation of Remote Sensing Images Based on Convolutional Networks", *IEEE TGRS*. Patch size "is usually set empirically by trying several sizes". VERIFY the exact wording and venue.

**Wording for the methodology:**

> Window size was treated as a parameter to test, not fixed in advance. We tested 80, 150 and 300 m, spanning the context sizes used in published waste-site studies: 100–210 m (Gibellini et al., 2025) and 210 m (Torres & Fraternali, 2023). 80 m is the smallest window certain to contain the identified site, since identification points lie up to 40 m from the waste. Gibellini et al. (2025) report that convolutional networks perform better with smaller context, while transformer networks are robust to it. Our results agree.
>
> 300 m was tested for seven of the eight methods and gave the lowest town-test score for all seven. The eighth (the AerialWaste training protocol at 300 m, about 10–20 hours of computation) was therefore not run.

## Training protocol: learning rate (agreed 2026-11-14)

The published AerialWaste and Guatemala protocol trains with Adam at 5 × 10⁻⁴. On our data the head collapsed to one constant score within the first epoch, in every fold that ran (7 of 7). Each pyramid level ends in a single ReLU unit, and a large update can switch all four off for every input. At 5 × 10⁻⁵ it trained normally. The collapsed runs are kept in `outputs/benchmark/archive_aw_lr5e-4_collapsed/`.

**Source:** Lu, Shin, Su & Karniadakis 2020, "Dying ReLU and initialization: theory and numerical examples", *Communications in Computational Physics* 28(5). VERIFY volume and pages before citing.

> The published training protocol uses Adam with a learning rate of 5 × 10⁻⁴. On our data this setting made the classification head return the same score for every image within the first epoch, in every fold. This is a known failure of ReLU units under large updates (Lu et al., 2020). We therefore used 5 × 10⁻⁵, chosen from training behaviour only, before any test results were seen.

## Validation design

- **Spatial-block cross-validation:**
  - Roberts et al. 2017, *Ecography* 40:913–929.
  - Ploton et al. 2020, *Nature Communications* 11:4540. Random splits overstate accuracy for spatial data.
- **Nested model selection; never selecting the epoch or settings on the test data:**
  - Cawley & Talbot 2010, *JMLR* 11:2079–2107.
  - Varma & Simon 2006, *BMC Bioinformatics* 7:91.
- **Frozen backbone plus linear probe as the protocol for comparing pretrained encoders:**
  - Sharma et al. 2026, arXiv 2611.08003 (VHR-Bench), linear probe for classification.
  - Kornblith et al. 2019, CVPR.

## Model choice

- **ResNet-50 + FPN scene classification for landfills:**
  - Torres & Fraternali 2021, *Remote Sensing* 13:4520.
  - Torres & Fraternali 2023, *Scientific Data* 10:63.
- **Object detection for dumpsites:** Sun et al. 2023, *Nature Communications* (Faster R-CNN + FPN, ResNet-50).
- **Encoder ranking on VHR imagery:** Sharma et al. 2026 (VHR-Bench).
  - Frozen DINOv3 ViT-L (web-trained 64.0 / 69.5; satellite-trained 63.4 / 69.1) and SkySense (65.8 / 69.2) lead.
  - Frozen ImageNet ResNet-50: 57.7 / 63.7.
  - Scores are at 10 % / 100 % of labels.
- **Cross-region drop for waste detectors:** Gibellini et al. 2025, *Waste Management Bulletin* 3:100246, −5.1 % F1 in new territories.
- **Look-alike false positives in VHR waste mapping:**
  - Sun et al. 2023: roofs, sea reflections.
  - Multi-city CNN-Transformer study (*Geocarto International*, 2022/23): parking lots, courtyards.
  - Malaysia ACRS 2024/25: cemeteries, construction plots.
