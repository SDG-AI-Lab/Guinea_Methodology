# Plastic Waste: baseline metrics

The method as of `45d7b77` (the `fix/fdi-bands-and-esri-confirm` calculation
state). Any change to the Plastic Waste calculation is compared against a row
here. When a change is adopted, update the row and note the commit. Source paths
are relative to `Land Use-Deforestation/`.

### FDI on water + MARIDA debris classifier

Source: `plasticwaste_tool/fdi_tool/models/debris_filter_meta.json`
(`metrics_marida_test`), frozen 2026-10-02 by `39_freeze_debris_filter.py`.
FDI formula id `biermann2020-b6`; classifier threshold `p_min = 0.5`.

"Deployed rule" is the FDI-and-NDVI rule; "FDI alone" is its FDI arm only.
The published chain is deployed rule → classifier.

| Split             | Method                          | Precision | Recall | TP  | FP   |
|-------------------|---------------------------------|-----------|--------|-----|------|
| MARIDA test (all) | deployed rule                   | 0.076     | 0.247  | 94  | 1149 |
| MARIDA test (all) | deployed rule → classifier      | 0.895     | 0.247  | 94  | 11   |
| MARIDA test (all) | FDI alone                       | 0.106     | 0.627  | 239 | 2021 |
| MARIDA test (all) | FDI alone → classifier          | 0.808     | 0.551  | 210 | 50   |
| unseen tiles      | deployed rule                   | 0.654     | 0.886  | 70  | 37   |
| unseen tiles      | deployed rule → classifier      | 0.921     | 0.886  | 70  | 6    |
| unseen tiles      | FDI alone                       | 0.322     | 0.962  | 76  | 160  |
| unseen tiles      | FDI alone → classifier          | 0.892     | 0.937  | 74  | 9    |

The classifier was trained on the MARIDA train split (429,412 px, 1,943
debris px). Its job is to remove false positives (Sargassum, ships). It does not
change recall, which the rule sets.

### Dumpsite scan, stage 2 (blind grids)

Source: `plasticwaste_tool/manual_qa/runs/blindgrid_esri/evaluation.json`,
produced by `38_esri_cascade_evaluation.py`. Head `probe_esri.joblib`, Esri
zoom 17, stage-1 screen retention 0.8545.

| Grid    | Imagery          | n   | Positives | ROC AUC | Avg precision |
|---------|------------------|-----|-----------|---------|---------------|
| Conakry | Esri (free)      | 422 | 110       | 0.594   | 0.359         |
| Conakry | Google (billed)  | 422 | 110       | 0.682   | 0.477         |
| Boke    | Esri (free)      | 557 | 243       | 0.687   | 0.580         |
| Boke    | Google (billed)  | 557 | 243       | 0.738   | 0.640         |

## Open items

| ID  | Item | Status |
|-----|------|--------|
| B03 | Raise the stage-2 Esri head on Conakry (AUC 0.594, close to chance) | todo |
| B04 | Raise recall of the deployed rule (0.247 on the MARIDA test). FDI alone → classifier reaches 0.551 at precision 0.808: weigh switching the published chain | todo |
