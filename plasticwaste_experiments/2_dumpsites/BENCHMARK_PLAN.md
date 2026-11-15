# Waste-site model benchmark: plan for review

Status: proposal, nothing run yet. Imagery: Esri (Google only after written permission, see section 6).

## 1. Purpose

Choose the waste-site model for the Guinea location and area checks by comparing published, pretrained methods on the same labels, the same tests and the same imagery. No new method is invented. Every candidate is either the method of a published waste-site study or the standard evaluation protocol of a published foundation-model benchmark.

## 2. What the literature does

| Study | Imagery | Image footprint | Method | Test split | Result | Known failure |
|---|---|---|---|---|---|---|
| Torres & Fraternali 2021, *Remote Sensing*; AerialWaste, *Scientific Data* 2023 | Aerial 20 cm, WorldView-3 30 cm, Google Earth 50 cm (Lombardy) | about 200–500 m | Scene classification: ResNet-50 + FPN | Not geographic | F1 about 0.8–0.9 | |
| Sun et al. 2023, *Nature Communications* | 0.3–1 m | 1024 px, about 614 m | Object detection: Faster R-CNN + FPN, ResNet-50 (BCA-Net) | Random 60/20/20; plus 28 unseen cities | Mean precision 0.70, sensitivity 0.98; 755 of 763 found in unseen cities | Roofs, sea reflections; "require a small amount of manual sifting" |
| Gibellini, Fraternali et al. 2025, *Waste Management Bulletin* | Very high resolution (VHR) | | Scene classification pipeline | Plus tests in other countries | F1 0.92; −5.1 % F1 in new territories | |
| UNDP Guatemala 2025 | Google Static, 30 cm | about 300 m | AerialWaste ResNet-50 + FPN, frozen | Random 20 %, epoch chosen on it | P = R = F1 = 0.68 | No test set |
| Microdumps 2024 (IEEE Access) | Pleiades, GeoEye-1 | | YOLOv8 detection | | F1 0.93 | |
| Malaysia 2024–25 (ACRS) | Pleiades Neo, 0.3 m | | U-Net (ResNet-34) segmentation | | | Look-alikes (cemeteries, construction) needed masking |
| CNN-Transformer, multi-city (China, India, Mexico) | VHR | | Segmentation | | | Parking lots and courtyards read as waste |

**Conclusions from the review:**
- Three families of method exist: scene classification, object detection and segmentation. All use ImageNet-pretrained CNNs, most often ResNet-50.
- Almost all tests are random splits. Very few test on unseen areas, and none tests on a random sample of an urban population, as our town test does.
- The failure we found in towns (bare soil, roofs, yards scored as waste) is the failure most of these papers report.

**Foundation models.** Sharma et al. 2026, "When GeoFMs meet VHR" (arXiv 2611.08003), benchmarked 12 pretrained encoders on 10 VHR RGB datasets with frozen backbones.

| Encoder (frozen) | Score at 10 % labels | Score at 100 % labels |
|---|---|---|
| DINOv3 ViT-L, web-trained | 64.0 | 69.5 |
| DINOv3 ViT-L, satellite-trained | 63.4 | 69.1 |
| SkySense | 65.8 | 69.2 |
| ImageNet ResNet-50 (our current model) | 57.7 | 63.7 |

- The ImageNet ResNet-50 is the weakest of the strong encoders.
- Fully fine-tuning ResNet-50 still stays below the leading frozen encoders.
- For classification, the paper's protocol is a frozen backbone plus a linear probe, which is exactly our setup.

## 3. Candidates

| # | Method | Source | Role |
|---|---|---|---|
| M0 | Colour/texture histogram + logistic regression | Standard baseline | Lower bar |
| M1 | ImageNet ResNet-50, frozen, + logistic regression | Current deployed model | Reference |
| M2 | AerialWaste ResNet-50 + FPN head, frozen backbone | Torres & Fraternali 2021; Guatemala 2025 | Published waste-specific method |
| M3 | ResNet-50 fully fine-tuned | Baseline in Sharma et al. 2026 | Completeness |
| M4 | DINOv3 ViT-L, satellite-trained, frozen, + logistic regression | Siméoni et al. 2025; protocol of Sharma et al. 2026 | Main candidate |
| M5 | DINOv3 ViT-L, web-trained, frozen, + logistic regression | As M4 | Main candidate |
| M6 (phase 2) | Object detector (Faster R-CNN or RT-DETR) | Sun et al. 2023 | Needs boxes drawn on dumps |

**Licences:**
- DINOv3: custom DINOv3 License, commercial use allowed, gated download; UNDP to confirm acceptance.
- AerialWaste weights: the original Politecnico di Milano repository is CC BY-NC-ND 4.0, and that licence governs. The Guatemala copy says CC BY but cannot relicense. The weights are fine for this noncommercial benchmark; deploying M2 on the platform would need Politecnico's permission.
- GUA_dumpsites code: GPL-3, so M2 is reimplemented rather than copied into the platform.

## 4. Data and tests

- **Label pool (about 2,450):**
  - national sample (595 cells, giving the 651 points);
  - blind grids (979 windows);
  - November rounds 1 and 2 (799 windows) plus 25 added points;
  - early 250 m tiles excluded.
- **Deduplication:** labels within 100 m of each other across sets are merged, so the same place is never in both training and test.
- **Images:** true crops centred on the label, at a fixed metres-per-pixel, not tile mosaics. Window sizes tested: 50, 150 and 300 m, plus 80 m as the link to the current model.
- **Split:** spatial-block cross-validation with 1 km blocks. All settings are chosen inside the training folds (nested), and early stopping for M2 and M3 uses a validation set of training blocks only. Folds are balanced by size, not by class share: each blind grid is one indivisible group of 400–560 windows, and class-stratified group folds came out at 38 to 984 rows (corrected 2026-11-13). The Guatemala training protocol was not run at 300 m (decision 2026-11-13): it needs 10–20 h on a 4 GB GPU, and the seven other methods tested at 300 m all scored lower on the town test than at 80 or 150 m.
- **Reported separately on out-of-fold scores:**
  1. Curated (651 set): comparable with Guatemala and the literature. Precision, recall and F1 are also reported at Guatemala's 33 % prevalence.
  2. Blind grids, Boké and Conakry.
  3. **Town test** (round 1, 400 random windows, weighted to the national population). This is the deployment-relevant result.
- **Metrics:**
  - average precision (main metric);
  - ROC-AUC;
  - share of dumps in the top 5 % and top 10 %;
  - F1 at the operating threshold.
- **Paired block-bootstrap 95 % confidence intervals** against M1.

## 5. Decision rule (fixed before running)

1. Choose the method and window with the best town-test average precision whose confidence interval against M1 excludes zero. If none qualifies, keep M1.
2. Report the curated result of the chosen method for comparability with Guatemala and the literature.
3. A national layer is considered only if the town test reaches a precision that the plastic lead accepts for field use. That threshold is set before running.

## 6. Google imagery

The Google Maps Platform Terms §3.2.3(c)(vii) forbid training, testing, validating or fine-tuning AI models on Google Maps content.
- No nonprofit exception exists in these terms.
- The Geo Guidelines permit research and nonprofit *display* of Google Earth content (print, "a handful of static images", video, with attribution). They do not cover API downloads or model training, and they refer back to the Terms of Service.
- Earth Engine (including UNDP's paid plan) does not contain Google's basemap imagery.

The pipeline is written so that the imagery source is a parameter. If Google gives written permission, the same benchmark is rerun on Google imagery without changes.

## 7. Order of work

1. Deduplicate labels, freeze the test sets, build true crops (Esri, sharpest zoom resampled to a fixed metres-per-pixel).
2. Extract features for M0, M1, M4 and M5, and fit linear probes at each window size.
3. Run M2 and M3 (GPU).
4. Write a results table and a recommendation for review.
5. Phase 2 (M6) only if the box drawing is done.
