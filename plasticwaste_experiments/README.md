# Plastic Waste experiments

Benchmark of the Plastic Waste module, November 2026. Results are in
`Report Docs/Methodology_PlasticWaste_Summary.docx`; platform faults and how to fix
them are in `PLATFORM_FINDINGS.md`. The deployed code in `Guinea_Project` was imported
read-only and nothing on the platform was changed.

To follow the work step by step, start with the three walkthrough notebooks in this folder:
`1_floating_debris`, `2_dumpsites` and `3_leakage`. They read the saved outputs, so they run in
seconds. Notebooks 4 to 6 are the follow-up tests, prepared and not yet run.

Scripts run in order inside each folder, with the system Python (scikit-learn 1.4.2,
the version of the frozen models). Outputs are in each `outputs/`.
`scripts/build_paper.py` rebuilds the paper from those outputs.

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

## 3_leakage

| Script | Question | Result |
|---|---|---|
| `01_mobilisation_flood_test` | Score against floods on 3,000 independent settled cells | Initial 0.67, HAND 0.86, JRC map 0.80 |
| `02_leakage_variants` | Reproduction and structural variants | 18,166 and 84,017 t reproduced. Central 40,307 t, 10.9 kg per person |
| `03_improved_mobilisation` | Inundation term from HAND | AUC 0.85, +0.19 (0.02 to 0.35) |

## Follow-up tests, prepared and not yet run

| Folder and notebook | Question | Needs |
|---|---|---|
| `4_plastic_content` | More EnMAP sites, the mixing limit, a positive control, PRISMA coverage | DLR login, 0.5 GB per scene; PRISMA is by hand |
| `5_floating_debris_flags` | Label Conakry patches by tide and movement, without deleting them | Cached imagery only |
| `6_three_changes` | Are the height-based score and the central leakage case worth adding? | Saved outputs only |
| `7_published_detector` | Does the Russwurm et al. (2023) detector stay off the tidal flats? | Earth Engine, about 1.5 GB, its own Python environment |

The rule for each decision is written at the top of the script, before any result exists.
PRISMA is by hand: see `4_plastic_content/PRISMA_coverage_guide.md`. MariNeXt is left out: its
code has no ready way to run on a new image.

## Published figures for Guinea

From Our World in Data, read 2026-11-05: 11.59 kg of mismanaged plastic per person and
2,347 t to the ocean in 2019 (Meijer et al. 2021); 0.03 kg of plastic waste per person
per day in 2010 (Jambeck et al. 2015).

## Earth Engine

Scripts 04, 05 and 07 of the first folder and 01 and 03 of the third use the project in
`Land Use-Deforestation/.env`, with small requests only. The live-platform patches
`04` reads are saved as `outputs/live_patches_2026-11-05.json`.
