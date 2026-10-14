"""
Floating Debris Index detection (DPWMS component 1) — importable package.

This is the mesoscale marine/estuarine layer: Sentinel-2 at 10-20 m, the
Floating Debris Index of Biermann et al. (2020), thresholded by the Digital
Earth Africa mean + 3σ rule over a positively-constrained water population.
It answers GEDH-FR-0503 (the marine half of the National Hotspot Map) and
GEDH-FR-0507 (the Monthly FDI Archive).

    spectral    offline numpy FDI / NDVI / NDWI, and the GEE cross-check
    detection   thresholding (Otsu, mean+3σ, fallback), the DE Africa
                conjunction rule, vectorisation and patch scoring
    pipeline    orchestration: run_from_arrays / run_from_gee / run_national /
                run_monthly_batch, plus the offline synthetic self-test
    deployment  AOI overlap checks, mosaic de-duplication, OSM port zones,
                water-fraction guard, review sampling, admin attribution

WHAT THE OUTPUT IS, AND IS NOT
------------------------------
Floating-debris CANDIDATES, not plastic. The index responds to any material
that raises near-infrared reflectance above the water baseline, so seaweed,
algal mats, timber and sea foam are flagged identically to macroplastic.
Separating them needs the labelled spectral library behind Biermann et al.'s
naïve Bayes classifier, which does not exist for Guinea. The scoring column is
named `floating_debris_score` for exactly this reason -- it was renamed from
`plastic_concentration_score`, which asserted an identification the method does
not make. Do not rename it back.

THIS PACKAGE IS CANONICAL; THE TOP LEVEL RE-EXPORTS IT
------------------------------------------------------
`plasticwaste_tool.spectral`, `.pipeline` and `.deployment` are thin shims that
re-export these modules, so callers written against the old layout -- including
`plasticwaste_analysis.ipynb` -- keep working unchanged. New code should import
from here.

`plasticwaste_tool.detection` is the one exception and is NOT a pure shim: it
re-exports this package's `detection` module AND owns
`classify_dumpsite_candidates()`, because it is the single stable interface the
backend calls for both layers of the hotspot map.
"""

from . import spectral
from . import detection
from . import pipeline

__all__ = ["spectral", "detection", "pipeline", "deployment"]


def __getattr__(name):
    # `deployment` pulls in geopandas and, on some paths, the GEE connector.
    # Importing it lazily keeps `import plasticwaste_tool` cheap for callers
    # that only need the offline index maths.
    #
    # MUST use importlib, not `from . import deployment`. Under PEP 562,
    # `from <pkg> import <name>` resolves via getattr on the package -- which
    # lands back in this function -- so the plain import form recurses until
    # the stack blows. `import_module` goes to the import system directly and
    # does not re-enter here. The same pattern is used in the parent package
    # and in characterisation_tool.
    if name == "deployment":
        import importlib
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod      # cache, so __getattr__ runs once
        return mod
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
