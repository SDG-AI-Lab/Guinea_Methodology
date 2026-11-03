"""Temporal validation of satellite features, CatBoost and covariate weighting."""

import argparse
import logging

from .advanced import run_advanced
from .comparison import DEFAULT_DATA
from .phenology_models import HistoricalPhenology
from .spectral_models import HistoricalSpectra
from .vegetation_context import VegetationContext


def candidates(prefix):
    templates = [
        ("control_rf", "rf", "two_stage", 150, 16, False, False, False, 1.),
        ("joint_xgb", "xgb", "joint", 500, 4, True, False, False, 1.),
        ("two_stage_rf", "rf", "two_stage", 150, 16, True, False, False, 1.),
        ("joint_catboost", "catboost", "joint", 500, 6, True, False, False, 1.),
        ("action_catboost", "catboost", "action", 500, 6, True, False, False, 1.),
        ("two_stage_catboost", "catboost", "two_stage", 500, 6, True, False, False, 1.),
        ("adapt_xgb", "xgb", "joint", 500, 4, True, True, False, 1.),
        ("adapt_rf", "rf", "two_stage", 150, 16, True, True, False, 1.),
        ("adapt_half_xgb", "xgb", "joint", 500, 4, True, True, False, .5),
        ("expanded_xgb", "xgb", "joint", 500, 4, True, False, True, 1.),
        ("expanded_adapt_xgb", "xgb", "joint", 500, 4, True, True, True, .5),
    ]
    return [dict(name=f"{prefix}_{name}", family=family, structure=structure, trees=trees,
        depth=depth, spectral=spectral, adapt=adapt, expanded=expanded, adapt_shrink=shrink,
        weight=1., geometry=True, multiscale=True)
        for name, family, structure, trees, depth, spectral, adapt, expanded, shrink in templates]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenology-dir", required=True)
    parser.add_argument("--spectral-dir", required=True)
    parser.add_argument("--feature-set", choices=["endpoints", "annual"], default="endpoints")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    reference = DEFAULT_DATA / "Guinea_LandCover_2010.tif"
    spectra = HistoricalSpectra(args.spectral_dir, reference)
    if args.feature_set == "annual":
        from .annual_vegetation import AnnualVegetation
        provider = AnnualVegetation(args.phenology_dir, reference, spectra=spectra)
    else:
        provider = VegetationContext(HistoricalPhenology(args.phenology_dir, reference), spectra)
    report = run_advanced(output_dir=args.output_dir, candidates=candidates(f"transfer_{args.feature_set}"),
        feature_provider=provider, feature_provenance=provider.manifests)
    name = report["selection"]["selected_model"]
    print("Selected:", name)
    print("Final FoM:", report["results"][name]["summary"]["figure_of_merit_pct"])


if __name__ == "__main__":
    main()
