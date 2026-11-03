"""Test vegetation neighborhood texture and optional MODIS/Landsat fusion."""

import argparse
import logging

import numpy as np
from scipy.ndimage import uniform_filter

from .comparison import DEFAULT_DATA
from .phenology_models import HistoricalPhenology
from .spectral_models import HistoricalSpectra
from .temporal_memory import run_memory


def masked_moments(values, valid, size):
    support = uniform_filter(valid.astype(np.float32), size, mode="constant")
    clean = np.where(valid, values, 0).astype(np.float32)
    mean = np.divide(uniform_filter(clean, size, mode="constant"), support,
        out=np.zeros_like(clean), where=support > 1e-8)
    second = np.divide(uniform_filter(clean**2, size, mode="constant"), support,
        out=np.zeros_like(clean), where=support > 1e-8)
    return mean, np.sqrt(np.maximum(0, second-mean**2)), support


class VegetationContext:
    def __init__(self, phenology, spectra=None):
        self.phenology, self.spectra = phenology, spectra
        self.summaries = {}
        self.manifests = dict(phenology=phenology.manifests,
            landsat=spectra.manifests if spectra else None,
            context_windows_pixels=[3, 15], missing_pixels="retained with explicit support")

    def annual_grids(self, year):
        if year in self.summaries:
            return self.summaries[year]
        if year < min(self.phenology.years):
            return {}
        # Force validated raster loading without copying its full pixel matrix.
        self.phenology.read_rows(year, np.array([], dtype=int))
        cube = self.phenology.cache[year].reshape(24, *self.phenology.shape)
        grids = {}
        for i, variable in enumerate(("ndvi", "evi")):
            values = cube[i::2]
            for label, periods in (("annual", np.arange(12)), ("wet", np.arange(4, 10)),
                                   ("dry", np.array([0, 1, 2, 3, 10, 11]))):
                selected = values[periods]
                valid = np.isfinite(selected)
                count = valid.sum(axis=0)
                mean = np.where(valid, selected, 0).sum(axis=0) / np.maximum(count, 1)
                grids[f"{variable}_{label}"] = (mean.astype(np.float32), count > 0)
        self.summaries[year] = grids
        return grids

    def __call__(self, year, indices):
        base, names = self.phenology(year, indices)
        rows = [base]
        if self.spectra is not None:
            spectral, spectral_names = self.spectra(year, indices)
            rows.append(spectral)
            names += spectral_names
        current, previous = self.annual_grids(year), self.annual_grids(year-5)
        extra = []
        for key in (f"{v}_{s}" for v in ("ndvi", "evi") for s in ("annual", "wet", "dry")):
            values, valid = current[key]
            for size in (3, 15):
                mean, std, support = masked_moments(values, valid, size)
                for label, grid in (("mean", mean), ("std", std), ("support", support),
                                    ("local_difference", np.where(valid, values-mean, 0))):
                    names.append(f"vegetation_{key}_{label}_{size}")
                    extra.append(grid.ravel()[indices])
                if key in previous:
                    past, past_valid = previous[key]
                    known = valid & past_valid
                    trend = (values-past)/5
                    trend_mean, _, trend_support = masked_moments(trend, known, size)
                else:
                    trend_mean = trend_support = np.zeros_like(mean)
                for label, grid in (("annual_trend_mean", trend_mean), ("annual_trend_support", trend_support)):
                    names.append(f"vegetation_{key}_{label}_{size}")
                    extra.append(grid.ravel()[indices])
        rows.append(np.column_stack(extra))
        return np.column_stack(rows).astype(np.float32), names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenology-dir", required=True)
    parser.add_argument("--spectral-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    reference = DEFAULT_DATA / "Guinea_LandCover_2010.tif"
    provider = VegetationContext(HistoricalPhenology(args.phenology_dir, reference),
        HistoricalSpectra(args.spectral_dir, reference) if args.spectral_dir else None)
    prefix = "fused" if args.spectral_dir else "vegetation_context"
    specs = [dict(name=f"{prefix}_{name}", family=family, structure=structure,
                  spectral=True, memory=True, pooled=False, trees=trees, depth=depth,
                  expanded=expanded) for name, family, structure, trees, depth, expanded in (
        ("two_stage_rf", "rf", "two_stage", 150, 16, False),
        ("joint_xgb", "xgb", "joint", 500, 4, False),
        ("deep_xgb", "xgb", "joint", 600, 6, False),
        ("expanded_xgb", "xgb", "joint", 500, 4, True))]
    report = run_memory(args.output_dir, candidates=specs, feature_provider=provider,
                        feature_provenance=provider.manifests)
    print("Selected:", report["selected_model"])
    print("Final FoM:", report["results"][report["selected_model"]]["summary"]["figure_of_merit_pct"])


if __name__ == "__main__":
    main()
