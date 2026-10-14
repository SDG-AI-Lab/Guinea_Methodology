"""
Plastic characterisation (DPWMS component 3) — importable package.

Modules here are COPIES of the numbered research scripts in the parent folder,
renamed so Python can import them and with their `sys.path` hacks replaced by
ordinary relative imports. The numbered originals stay where they are and are
what `plastic_characterisation_workflow.ipynb` and the figure builders use;
nothing about the notebooks changes. Regenerate with
`24_build_component_packages.py`, and do not hand-edit these copies or the two
will drift apart.

READ THIS BEFORE USING ANYTHING HERE
------------------------------------
This component asked whether the PLASTIC CONTENT of an already-detected
dumpsite can be estimated from satellite imagery, as the programme concept
note commits. It was designed as four falsification gates, ordered cheapest
first, each able to end the work. Gate 1 passed; gates 2, 3 and 4 returned
negative, and the negative is the finding.

    gate 1  does hyperspectral imagery of Guinea exist?      PASSED
            hyperspectral_coverage   EMIT + EnMAP cover 110 of 111 dump points
    gate 2  do Sentinel-2 plastic indices separate dumps?    FALSIFIED
            s2_indices, evaluate_indices   NDVI matches or beats every
            purpose-built plastic index; at the dump's own scale they add
            nothing (dAP -0.003, 95% CI [-0.090, +0.082])
    gate 3  does FDI respond on land?                        FALSIFIED
            fdi_on_land   responds, but as a bare-ground proxy, and the
            operational pipeline yields 0 land pixels by design
    gate 4  is there a C-H signature in EnMAP spectra?       FALSIFIED
            enmap_download, enmap_spectra, separability   1215 nm and 1730 nm
            rank 140 and 128 of 218 bands over six scenes

So there is NO plastic-fraction layer to deploy, and none of these modules
should be wired to a backend endpoint that implies one. Publishing a
bare-ground proxy under a plastic label would render convincingly on a map,
colour a legend plausibly, and be wrong.

WHAT THIS COMPONENT DOES PRODUCE
--------------------------------
Three measured outputs, all of which are real and none of which is a plastic
measurement:

    extent, extent_annotator, ingest_annotations, extent_second_read
        the first measured dumpsite size distribution for Guinea -- median
        area-equivalent side 14.0 m (n=46). Every Guinean dumpsite is
        sub-pixel at every sensor that can see the diagnostic wavelengths,
        which is WHY gate 4 could not have succeeded.

    fdi_on_land (secondary finding)
        a waterside flag: sampling ESA WorldCover at the labelled dump points
        places 19 of 111 on water, wetland or mangrove classes -- the
        land-to-ocean pathway, linking this layer to the marine FDI layers.

    mobilisation
        waste mobilisation and ocean-leakage risk, following Meijer et al.
        (2021) and the Plastic Pathfinder. `mobilisation_risk` is VALIDATED
        against observed flooding from the Global Flood Database (AUC 0.771,
        p = 0.001); `ocean_leakage_risk` is not testable against that data and
        is explicitly unvalidated. This sidesteps the unmeasurable quantity
        rather than depending on it: it asks where waste GOES, which is a
        function of terrain, rainfall and drainage, all directly observable.

None of these is currently a field on the dumpsite output schema. They exist
as analysis outputs under `runs/`, and wiring them into the published layer is
a separate, signed-off decision -- see the methodology document §10.

PATHS
-----
Every module resolves its inputs and outputs against `TOOL`, the parent
`plasticwaste_tool` directory, because `runs/`, `manual_qa/` and the frozen
models all live one level up from this package.
"""

from __future__ import annotations

__all__ = [
    # gate 1
    "hyperspectral_coverage",
    # gate 2
    "s2_indices", "evaluate_indices",
    # gate 3
    "fdi_on_land",
    # extent
    "extent", "extent_annotator", "ingest_annotations", "extent_second_read",
    # gate 4
    "enmap_download", "enmap_spectra", "separability",
    # deliverable
    "mobilisation",
]


def __getattr__(name):
    # Imported lazily: several of these pull in Earth Engine, rasterio or
    # scikit-learn, and a caller that only wants the extent statistics should
    # not pay for the GEE client.
    if name in __all__:
        import importlib
        mod = importlib.import_module("." + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
