# Plastic Waste experiments

Benchmark of the Plastic Waste module, November 2026. Results are in
`Report Docs/Methodology_PlasticWaste_Summary.docx`; platform faults and how to fix
them are in `PLATFORM_FINDINGS.md`. The deployed code in `Guinea_Project` was imported
read-only and nothing on the platform was changed.

To follow the work step by step, start with the three walkthrough notebooks in this folder:
`1_floating_debris`, `2_dumpsites` and `3_leakage`. They read the saved outputs, so they run in
seconds. Notebooks 4 to 7 are the follow-up tests.

Scripts run in order inside each folder, with the system Python (scikit-learn 1.4.2,
the version of the frozen models). Outputs are in each `outputs/`.

## 1_floating_debris

| Script | Question | Result |
|---|---|---|
| `01_marida_loto` | All candidates on MARIDA, each of 14 tiles held out in turn | The chain: precision 0.82, recall 0.09. Forest alone: 0.83, 0.84 |
| `02_conakry_persistence` | Do published patches behave like floating material? | 20 of 26 January patches stay above the FDI threshold on both clear dates; 5 of 44 November patches recur, about 100 × chance |
| `03_look_at_clusters` | What is under them? | Tidal flats and turbid fringe, by eye |
| `04_patches_on_tidal_flats` | Share on independent tidal-flat maps (Earth Engine) | At least 54 of 71 patches |
| `05_scenes_per_month` | Clear looks per month | 0–3 June to October, 3–7 November to May |
| `06_fairness_checks` | Is `01` fair to the rule? | With thresholds from all the water: precision 0.88, recall 0.16. Conclusion holds |
| `07_candidates_on_conakry` | Candidates on Guinean imagery, with and without tidal flats | With flats removed the chain leaves 0–1 pixels. Forest alone swings 542–15,518 pixels, on near-shore banks |
| `08_river_candidates` | Candidates on 65 cached river tiles | 16 of 8,673 rule pixels lie over 40 m from the bank; the classifier removes all |
| `09_paper_figure` | Figure 1 | |

## 2_dumpsites

| Script | Question | Result |
|---|---|---|
| `01_features` | Features of the points and the blind grids | |
| `02_benchmark` | Five representations on the blind grids | Frozen model reproduced exactly (0.594, 0.687). Nothing beats ResNet-50 beyond its interval. Other-town labels fall below chance on most models |
| `03_location_check_window` | Cost of the fixed 80 m grid | 0.010 of AUC |
| `04_within_town` | Do same-town labels help? | 0.60–0.64 at Conakry, 0.69–0.72 at Boké, whatever the model |

`03` compares two windows at one fixed C. Its absolute AUC (0.90) is optimistic; the
honest figure is the published nested one, 0.832, reproduced exactly from these features.

**Benchmark against published methods (November 2026).** Plan in `2_dumpsites/BENCHMARK_PLAN.md`, sources in `2_dumpsites/LITERATURE_NOTES.md`. Run everything with `2_dumpsites/run_benchmark.sh`, or from section 7 of the notebook. These scripts use the GPU environment `E:\SDG AI Lab\Guinea\national_scan_2026\venv_gpu`, and the Esri tile cache of the national scan on E:.

| Script | Question | Result |
|---|---|---|
| `05_label_pool_and_crops` | One pool of about 2,440 labels; true 80, 150 and 300 m crops | 2,441 labels, 1,124 leakage-safe groups, all imaged |
| `06_frozen_features` | Colour, ImageNet, AerialWaste (Italy and Guatemala), DINOv3 (satellite and web) | Done, see section 7 |
| `07_linear_probe_benchmark` | Logistic step, nested spatial-block CV; curated, blind-grid and town tests | Done, see section 7 |
| `08_trained_methods` | Guatemala's protocol and a fully fine-tuned ResNet-50 | Done, see section 7 (Guatemala's protocol not run at 300 m) |
| `09_benchmark_report` | One table, intervals against the current network | Done, see section 7 |
| `10_document_table` | The comparison in the methodology document: beat the current network on all four tests, then the best town score | Done, see section 7 |
| `11_geotessera_embeddings` | TESSERA yearly embeddings (Feng et al. 2026, CC0) at the labelled points: the 10 m pixel and the mean and spread over about 150 m. Needs Python 3.12 or later: `E:\SDG AI Lab\Guinea\national_scan_2026\venv_tessera` | 2024 and 2025, all 2,441 points |
| `12_geotessera_benchmark` | TESSERA alone and added to DINOv3 web at 150 m, with section 7's protocol | Alone weak on the blind grids; with DINOv3 town AP 0.18–0.20, AUC 0.82, passes the published mark in both years. Adopted, see section 8 |
| `13_freeze_dinov3_tessera_head` | The adopted model frozen for the platform: fitted on all 2,348 labels (TESSERA 2025), tiers set on the out-of-fold scores to keep the initial tiers' recall | C = 0.001; tiers 0.64 / 0.52 / 0.45; copied to the platform's `dumpsite_tool/models/` (branch `feat/waste-site-dinov3-tessera`) |

## 3_leakage

| Script | Question | Result |
|---|---|---|
| `01_mobilisation_flood_test` | Score against floods on 3,000 independent settled cells | Initial 0.67, HAND 0.86, JRC map 0.80 |
| `02_leakage_variants` | Reproduction and structural variants | 18,166 and 84,017 t reproduced. Central 40,307 t, 10.9 kg per person |
| `03_improved_mobilisation` | Inundation term from HAND | AUC 0.85, +0.19 (0.02 to 0.35) |
| `04_meijer_flood_test` | Meijer et al. (2021) cell terms, best-calibrated values (Table S15 of the supplementary; the PDFs stay local), same flood test | 0.65 to 0.66: same as initial, below HAND-based (−0.32 to −0.08) |

## Follow-up tests

| Folder and notebook | Question | Needs |
|---|---|---|
| `4_plastic_content` | More EnMAP sites, the mixing limit, a positive control, PRISMA coverage | DLR login, 0.5 GB per scene; PRISMA is by hand |
| `5_floating_debris_flags` | Label Conakry patches by tide and movement, without deleting them | Cached imagery only |
| `6_three_changes` | Are the height-based score and the central leakage case worth adding? | Saved outputs only |
| `7_published_detector` | Does the Russwurm et al. (2023) detector stay off the tidal flats? And MariNeXt (Kikaki et al. 2024, MADOS), scripts 03 and 04 | Earth Engine, about 1.5 GB, its own Python environment; MariNeXt in `E:\SDG AI Lab\Guinea\marinext_test` (100 MB). MariNeXt: 96 % of its debris on flats on L2A, 94–95 % on Acolite reflectance (scripts 05, 06; CDSE login kept in `C:\Users\ASUS\.cdse`) |

The rule for each decision is written at the top of the script, before any result exists.
PRISMA is by hand: see `4_plastic_content/PRISMA_coverage_guide.md`.

## Published figures for Guinea

From Our World in Data, read 2026-11-05: 11.59 kg of mismanaged plastic per person and
2,347 t to the ocean in 2019 (Meijer et al. 2021); 0.03 kg of plastic waste per person
per day in 2010 (Jambeck et al. 2015).

## Earth Engine

Scripts 04, 05 and 07 of the first folder and 01 and 03 of the third use the project in
`Land Use-Deforestation/.env`, with small requests only. The live-platform patches
`04` reads are saved as `outputs/live_patches_2026-11-05.json`.
