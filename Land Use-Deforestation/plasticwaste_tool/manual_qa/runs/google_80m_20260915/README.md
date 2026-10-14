# Run: google, 80m point-centred crops (z19)

Archived 20260915. Imagery not copied; re-fetch with
`python 08_point_model.py fetch --tile-m 80 --source google`

| metric | value |
|---|---|
| ROC-AUC | 0.911 |
| average precision | 0.696 (no-skill 0.171) |
| mean score, dumps | 0.649 |
| mean score, residential | 0.446 |
| verdict | **waste detector** |
| top-40 precision, per window | 85% |
| top-40 precision, per site | 68% |
| deployment prevalence | 0.021 |

Files archived: _oof_pt80_google.npz, _metrics_pt80_google.json, _point_samples.csv, national_random_sample.geojson, results_pt80_google_flagged.png, results_pt80_google_missed.png
