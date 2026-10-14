# plasticwaste_tool/__init__.py
"""
Digital Plastic Waste Management System (DPWMS) — Guinea Environment Digital Hub.

A SEPARATE module/screen from `landuse_tool` (GEDH Module 5, distinct from
Module 4's Land Use & Deforestation section — see SRS_GuineaDigitalHub_v2.docx,
GEDH-FR-0401 vs GEDH-FR-0501), but built on the same shared infrastructure
(GEE service account, Azure Blob cache, Key Vault secrets, OSM/Geofabrik
ingestion) that `landuse_tool` already established, imported directly rather
than duplicated.

THREE COMPONENTS, ONE SHARED BASE
---------------------------------
Shared by all three:

    config        AOI, Sentinel-2 / spectral-index parameters, dataset config
    api_connector GEE Sentinel-2 fetch, server-side FDI/Otsu, OSM ancillary
                  layers, monthly archive fetch (reuses landuse_tool infra)
    data_loader   raster loading/alignment, Digital Waste Bank point layers
    validation    reference-point accuracy, threshold sensitivity, spatial-CV
    stats         patch / monthly / waste-bank statistics
    gee_auth      Earth Engine project-convention authentication
    geo_env       PROJ data-path repair (call before the first rasterio CRS op)

The components themselves:

    fdi_tool/               1 — floating debris on WATER. Sentinel-2, the
                                Floating Debris Index, the Digital Earth Africa
                                mean+3σ rule. Operational. FR-0503, FR-0507.
    dumpsite_tool/          2 — waste accumulation sites on LAND. A two-stage
                                cascade over high-resolution RGB basemaps.
                                Operational as a screening aid. FR-0503.
    characterisation_tool/  3 — whether per-site PLASTIC CONTENT is estimable
                                from satellite. Four falsification gates;
                                three returned negative. Research outputs, not
                                a published layer.

THE ONE INTERFACE THE BACKEND CALLS
-----------------------------------
    detection.vectorize_hotspots(...)            marine patches
    detection.classify_dumpsite_candidates(...)  terrestrial polygons

`detection` re-exports the FDI functions from `fdi_tool.detection` and owns the
dumpsite entry point, so the backend's surface stays two functions wide and
does not move when either component changes behind it. Backend code should not
import `fdi_tool`, `dumpsite_tool` or `characterisation_tool` directly.

`spectral`, `pipeline` and `deployment` at this level are re-export shims for
`fdi_tool` — kept so callers written against the previous flat layout, notably
`plasticwaste_analysis.ipynb`, continue to work unchanged.

WHAT THESE LAYERS ARE, AND ARE NOT
----------------------------------
Neither layer measures plastic. The marine layer detects floating-debris
CANDIDATES; the terrestrial layer detects WASTE ACCUMULATION SITES from
appearance. Material composition requires ground survey — see
`docs/Methodology_PlasticWasteAnalysis_Guinea.docx` §1.3 and §8. Interface copy
should follow the same wording; "plastic hotspot" overstates both.

NOTHING HEAVY IS IMPORTED AT PACKAGE LEVEL, AND THAT IS DELIBERATE
------------------------------------------------------------------
torch and the GEE/GDAL stack both ship their own native Windows DLLs, and
whichever initialises first wins. This package used to import `api_connector`
eagerly, which pulls in GDAL — so *any* route to `dumpsite_tool` through
`plasticwaste_tool` had already lost that race, and torch then failed with:

    OSError: [WinError 1114] A dynamic link library (DLL) initialization
    routine failed. Error loading "...torch\\lib\\c10.dll"

which reads like a broken torch installation rather than an ordering problem.
Even `python -m plasticwaste_tool.dumpsite_tool.footprint` failed, because
`-m` imports the parent package first.

Every submodule is therefore loaded on first attribute access instead. Nothing
native is touched by `import plasticwaste_tool`, so whichever component a
caller actually asks for initialises first and wins the race on its own. The
alternative — importing torch at the top of this file — would fix it globally
but force a torch dependency on FDI-only deployments that have no use for it.

`from plasticwaste_tool import api_connector` and
`import plasticwaste_tool.config as cfg` both still work unchanged: Python
falls back to importing the submodule when the attribute is absent. Existing
notebooks and callers need no edit.

Belt and braces for backend code: `import torch` on the first line of the
Celery task module, above any `plasticwaste_tool` import, costs one line and
removes the question entirely. See `dumpsite_tool/README.md`.
"""

_SUBMODULES = frozenset({
    # shared
    "config", "api_connector", "data_loader", "stats", "validation",
    "gee_auth", "geo_env",
    # the backend interface
    "detection",
    # FDI, under both its new and its historical names
    "fdi_tool", "spectral", "pipeline", "deployment",
    # the other two components
    "dumpsite_tool", "characterisation_tool",
})

__version__ = "0.2.0"
__author__ = "SDG AI Lab"

__all__ = [
    # shared
    "config", "api_connector", "data_loader", "stats", "validation",
    # the backend interface
    "detection",
    # FDI, under both its new and its historical names
    "fdi_tool", "spectral", "pipeline", "deployment",
    # the other two components
    "dumpsite_tool", "characterisation_tool",
]


def __getattr__(name):
    """
    Import submodules on first access (PEP 562).

    Two reasons, and the first is the load-bearing one:

    1. Native DLL ordering on Windows. See the module docstring -- an eager
       `api_connector` import made GDAL win the race against torch, breaking
       every path into `dumpsite_tool`.
    2. Dependency weight. `dumpsite_tool` needs torch, `characterisation_tool`
       needs scikit-learn, `deployment` needs geopandas and the OSM path. An
       FDI-only deployment should not be made to install any of them just to
       call `compute_fdi`.
    """
    if name in _SUBMODULES:
        import importlib
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)
