"""
Plastic leakage: from population to tonnes in the environment.

The fourth component of the plastic waste module, and the one that turns the
other three from detections into quantities. Detection says WHERE; this says
HOW MUCH the surrounding population generates and how much of it escapes.

    parameters.py   every number, its range, and its citation
    population.py   WorldPop 1 km -> people, classified urban/rural
    massbalance.py  the deterministic chain, in tonnes per year
    uncertainty.py  Monte Carlo with correlated national parameters
    sites.py        per-site allocation; gives mobilisation_risk real units

Driver: ../31_run_leakage.py    Notebook: ../../leakage_workflow.ipynb
"""

from . import massbalance, parameters, population, sites, uncertainty  # noqa: F401

__all__ = ["parameters", "population", "massbalance", "uncertainty", "sites"]
