"""
MARIDA: does machine learning beat the FDI threshold, and what does the
threshold actually fire on?

    data.py       load the benchmark, its official splits, its class codes
    features.py   spectral features, using the DEPLOYED FDI function
    baseline.py   the shipped mean+3sigma rule, evaluated as a classifier
    model.py      random forest, binary and multiclass
    evaluate.py   matched-recall comparison and the confuser table

Driver: ../32_run_marida.py    Notebook: ../../marida_workflow.ipynb

Kikaki et al. (2022) PLoS ONE 17(1): e0262247. doi:10.5281/zenodo.5151941
"""

from . import baseline, data, features, model  # noqa: F401

__all__ = ["data", "features", "baseline", "model", "evaluate"]
