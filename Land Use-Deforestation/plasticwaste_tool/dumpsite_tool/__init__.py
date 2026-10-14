"""
Dumpsite detection (DPWMS component 2) — importable package.

Modules here are COPIES of the numbered research scripts in the parent folder,
renamed so Python can import them. The numbered originals remain in place and
are what the notebooks use; these are what the backend imports. `scan` is the
exception: it is written here, not copied, because it is deployment code with
no research original.

    scan           TWO-STAGE NATIONAL SCAN -- the deployment path
    inference      tile fetchers (Esri + Google Static Maps), placeholder
                   detection, tile grids, scoring
    feature_probe  feature extraction, spatial blocking, bootstrap CIs
    features       ImageNet / colour representations, paired bootstrap
    point_model    build samples, fetch, evaluate, freeze, contact sheets
    footprint      national scan footprint (GHSL SMOD) and random sampling
    train_head     freeze a deployable probe head

    models/        frozen heads: probe.joblib (80m confirm) and
                   probe_screen.joblib (250m screen), each with _meta.json

The backend should NOT import this package directly. It calls
`plasticwaste_tool.detection.classify_dumpsite_candidates()`, which is the
stable interface; this package is the implementation behind it.
"""

# Torch MUST load before the GEE/GDAL stack on Windows: both ship native DLLs
# and whichever initialises first wins, otherwise torch's own c10.dll fails with
# WinError 1114. The parent package eagerly imports api_connector (GDAL), so any
# caller that reaches this subpackage through `import plasticwaste_tool...` has
# already lost the race -- see the note in this package's README about importing
# torch first in backend tasks. Kept here so a direct import of this subpackage
# is safe, and wrapped because an FDI-only environment need not have torch.
try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    pass
