# Run: esri, 80m point-centred crops (z17)

Archived 20260915. Imagery not copied; re-fetch with
`python 08_point_model.py fetch --tile-m 80 --source esri`

| metric | value |
|---|---|
| ROC-AUC | 0.832 |
| average precision | 0.543 (no-skill 0.171) |
| mean score, dumps | 0.625 |
| mean score, residential | 0.539 |
| verdict | **waste detector** |
| top-40 precision, per window | 70% |
| top-40 precision, per site | 50% |
| deployment prevalence | 0.021 |

Files archived: _oof_pt80_esri.npz, _metrics_pt80_esri.json, _point_samples.csv, national_random_sample.geojson, results_pt80_esri_flagged.png, results_pt80_esri_missed.png
