# Files kept outside git

These files are on the original machine but are not in this repo, because
GitHub rejects files over 100 MB or because of their size, license or
secrecy. Fill in **Stored at** once each file is uploaded. All paths are
relative to `Land Use-Deforestation/plasticwaste_tool/`.

## Files over 100 MB (13 files, 3.35 GB)

| File | Size | What it is | Stored at |
|------|------|------------|-----------|
| `runs/enmap_scenes/ENMAP01-____L2A-DT0000055956_20231222T121314Z_001_V010502_20260118T095258Z/image.TIF` | 495 MB | EnMAP L2A scene, 2023-12-22 | |
| `runs/enmap_scenes/ENMAP01-____L2A-DT0000055957_20231225T115153Z_002_V010502_20260119T112900Z/image.TIF` | 406 MB | EnMAP L2A scene, 2023-12-25 | |
| `runs/enmap_scenes/ENMAP01-____L2A-DT0000113031_20250125T114334Z_016_V010502_20250128T204638Z/image.TIF` | 406 MB | EnMAP L2A scene, 2025-01-25 | |
| `runs/enmap_scenes/ENMAP01-____L2A-DT0000069186_20240415T115423Z_010_V010502_20260314T155804Z/image.TIF` | 406 MB | EnMAP L2A scene, 2024-04-15 | |
| `runs/enmap_scenes/ENMAP01-____L2A-DT0000060721_20240209T114334Z_002_V010502_20260211T173447Z/image.TIF` | 400 MB | EnMAP L2A scene, 2024-02-09 | |
| `runs/enmap_scenes/ENMAP01-____L2A-DT0000055958_20231229T115536Z_002_V010502_20260120T200034Z/image.TIF` | 371 MB | EnMAP L2A scene, 2023-12-29 | |
| `aerialwaste_model/weights/checkpoint.pth` | 114 MB | AerialWaste weights | |
| `aerialwaste_model/weights/guatemala.pth` | 114 MB | AerialWaste Guatemala fine-tune | |
| `runs/aerialwaste_guinea_finetune/best.pt` | 114 MB | AerialWaste Guinea fine-tune | |
| `runs/baseline_seed0_mcc/best.pt` | 104 MB | Dumpsite baseline checkpoint | |
| `runs/baseline_seed0/best.pt` | 104 MB | Dumpsite baseline checkpoint | |
| `runs/baseline_no_gsd_aug/best.pt` | 104 MB | Dumpsite baseline checkpoint | |
| `runs/baseline_gsd_aug/best.pt` | 104 MB | Dumpsite baseline checkpoint | |

The EnMAP scenes can be downloaded again from the DLR EOWEB GeoPortal using
the scene IDs above.

## Other excluded material (under 100 MB per file)

| What | Size | Why excluded |
|------|------|--------------|
| `manual_qa/` imagery: region chips, `tiles_*`, `blindgrid_cache_*`, `scan_demo_cache`, `_covmap` (~28,300 PNGs) | 5.8 GB | Exceeds GitHub's 2 GB push limit. Much of it is Google Maps imagery, which Google's terms do not allow us to redistribute. The labels, QA metrics and GeoJSONs **are** committed. |
| `runs/` rest of the scene folders (XML, metadata), `runs/hyperspectral_coverage/enmap_items.json` and `emit_granules.json`, `runs/dumpsite_extent/second_read_crops/`, `runs/benchmark/`, `runs/feature_probe/` | ~90 MB | Raw catalogue dumps and intermediates, all regenerable. Summary JSON/CSV and figures **are** committed. |
| `aerialwaste_model/` (code + LICENSE) | <1 MB | CC BY-NC-ND 4.0: no derivatives, non-commercial. |

## Never upload anywhere shared

These are credentials. Share them through a password manager, never through
git or a public drive.

- `Land Use-Deforestation/.env`
- `Land Use-Deforestation/landuse_tool/credentials/` (Google Cloud service-account key and OAuth client secret)
- `Land Use-Deforestation/plasticwaste_tool/manual_qa/regions/._google_api_key.txt`
