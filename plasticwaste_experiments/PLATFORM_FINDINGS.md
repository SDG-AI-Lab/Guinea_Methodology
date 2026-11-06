# Plastic Waste on the live platform: findings for the developers

Checked 2026-11-05 on http://172.205.219.243, read-only, without logging in (GET
requests on the public API, and the code on `feat/calc-methodology-improvement`).
Nothing on the platform was changed and no job or location check was started.
Paths are relative to the `Guinea_Project` repository.

## Bugs

| # | Symptom on the platform | Cause | Where | Fix |
|---|---|---|---|---|
| 1 | Rivers: "31.7 of 5,970 km assessed" (0.5 %). The methodology says 55 %. 114 of 116 tiles show "too little water" | The tool assesses a corridor by its 1,000-pixel floor and treats the 10 % water share as advisory. The backend then refuses every run under 10 % water anyway, river corridors included | `backend/plasticwaste_api/tasks.py`, `_persist_fdi_run`, `low_water = ... or (water_fraction < 0.10)` | Trust `low_water_warning` from the tool. On the same runs that gives 41 tiles and about 2,700 km instead of 2 tiles and 32 km |
| 2 | `/api/plasticwaste/fdi-runs/` returns HTTP 500, for all filters. The layer manifest points at it. 16 single runs also return 500 (ids 101, 126, 129, 131, 132, 136–139, 141–146, 257) | Most likely NaN: river runs with no water store NaN thresholds, and DRF refuses NaN in JSON (read from the code, not confirmed on the server's log). `threshold_fdi` and `threshold_ndvi` are not passed through `_finite` as the score is | `tasks.py`, `_persist_fdi_run` | Store `_finite(result.get('threshold'))` and the same for NDVI, and clean the 16 rows |
| 3 | The October river run never appeared; only 2026-09 is a period | Job 17 hit the 180-minute limit at tile 99 of 117. Publishing runs only at the end, so 99 finished tiles were lost, 11 patches among them | `run_fdi_analysis`, river branch | Publish what finished, or split the job by river system; raise the limit |
| 4 | Every FDI job since 2026-11-05 06:59 fails: "free disk space below the floor: 1.69 GiB left" | The worker's disk is full | Server | Free disk space. No new month will publish until then |
| 5 | The coastal layer opens empty | The map opens on the latest month (2026-10), and 2026-09 and 2026-10 have 0 patches in all regions. These are rainy months with 1–3 clear observations per pixel, against 3–7 from November to May | `LayerPanel`, default period; `05_scenes_per_month.csv` | Back-run November 2025 – May 2026. Open on the latest month that has clear imagery, or say on the map that the month is cloud-limited |
| 6 | Patches are hard to see on the national map | 800–2,000 m² polygons on a country-wide map. A hexagon variant, `fdi-coastal-h3`, exists | Layer registry | Draw points or the hexagon view at low zoom, and show the AOI outlines so that "not looked at" differs from "nothing found" |

## Text served to users that disagrees with the results

| Where | Says | Measured |
|---|---|---|
| `manifests.py`, `manifests_fr.py`, `fdi_patches.status` | 61 % (Conakry) to 80 % (offshore) confirmed by photo-interpretation | The served evidence endpoint says 36.8 %. Those 7 "positives" were all recorded as low confidence, "could be debris". 61–80 % is left over from an earlier detector |
| `manifests.py`, dumpsite status | Precision 0.584 at the default tier | 0.584 is Google imagery at 0.90. The default is Esri at 0.70: 0.347 Conakry, 0.586 Boké |
| `manifests.py`, `on_demand.why_separate`; `ondemand.py` header | Check uses the model where it scores 0.911 | 0.911 is the Google head. The check uses the Esri head: see the location-check test in `2_dumpsites` |
| `fdi_tool/debris_filter.py` header, `evidence.py` output | Recall 0.886 on unseen tiles | Two tiles. Over all 14 tiles held out in turn: 0.09 to 0.16 (`1_floating_debris`) |
| `leakage` rows, ADM1 | `mass_balance_closes: false` | Not checked at that scope. The field reads as a failure; null would be correct |

## "Check this location"

In order of likelihood, with what the screen would show:

1. `PLASTIC_CHECK_DAILY_LIMIT` is empty, as in `backend/.env.example`. The check
   refuses every request: "Location checks are not available yet on this platform."
   Set a positive number in `backend/.env` and restart the API.
2. The account is a viewer: the button shows a lock.
3. Memory: the API container is limited to 2 GB across its workers, and one ResNet-50
   pass at 800 × 800 pixels with torch loaded is a large share of that. A worker
   killed mid-request shows as a generic failure or a 502.
4. First use downloads the ResNet-50 weights inside the container. If that path is
   not writable, or there is no room (see bug 4), the error is not caught as
   `OnDemandUnavailable`, so the answer is a 500.

The message on screen confirmed cause 1. Running the real check in a
2 GB container then found a fifth cause: `contextily` is not in `backend/requirements.txt`,
so the check would crash once the limit is set. With it installed the check works: 16 s the
first time, 2.5 s after, 0.88 GiB per process. Fixes and steps are in
`Guinea_Project/Docs/PlasticWaste_Platform_Update_2026-11.md`.

Two notes on what a score means, once the check runs:

- The window is not centred on the pin, as the code comment says: it is the cell of
  the 80 m lattice holding the pin, 32 m off-centre at the median. This costs little:
  0.010 of AUC (90 % interval 0.004 to 0.018) on the 651 labelled points.
- The published 0.832 was measured on labelled points whose negatives are mostly
  rural. A pin dropped inside a town is closer to the blind-grid task, where the Esri
  head scores 0.594 at Conakry and 0.687 at Boké. The text served with the result
  should say so, and quote neither 0.911 (Google) nor 0.832 alone.
