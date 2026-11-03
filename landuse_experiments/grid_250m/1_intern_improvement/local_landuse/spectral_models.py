"""Fit change models with dated Landsat features and fixed evaluation coverage."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from .comparison import DEFAULT_DATA
from .landsat_features import FEATURE_NAMES
from .temporal_memory import run_memory


CANDIDATES = [
    dict(name="spectral_control_rf", family="rf", structure="two_stage", spectral=False,
         memory=False, pooled=False, trees=150, depth=16),
    dict(name="spectral_two_stage_rf", family="rf", structure="two_stage", spectral=True,
         memory=True, pooled=False, trees=150, depth=16),
    dict(name="spectral_action_rf", family="rf", structure="action", spectral=True,
         memory=True, pooled=False, trees=150, depth=16),
    dict(name="spectral_joint_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=False, trees=400, depth=4),
    dict(name="spectral_two_stage_xgb", family="xgb", structure="two_stage", spectral=True,
         memory=True, pooled=False, trees=400, depth=4),
    dict(name="spectral_action_xgb", family="xgb", structure="action", spectral=True,
         memory=True, pooled=False, trees=400, depth=4),
    dict(name="spectral_deep_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=False, trees=600, depth=6),
    dict(name="spectral_flexible_rf", family="rf", structure="two_stage", spectral=True,
         memory=True, pooled=False, trees=300, depth=20),
    dict(name="pooled_spectral_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=True, trees=400, depth=4),
    dict(name="expanded_spectral_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=False, trees=400, depth=4, expanded=True),
]


class HistoricalSpectra:
    """Starting-year and five-year-past spectra; missingness never removes pixels."""
    def __init__(self, root, reference_path, years=(2005, 2010, 2015)):
        self.root, self.years = Path(root), tuple(years)
        with rasterio.open(reference_path) as src:
            self.profile = src.profile
            self.shape = src.shape
        self.names = (["mean_" + n for n in FEATURE_NAMES] +
                      ["std_" + n for n in FEATURE_NAMES] + ["clear_observation_count"])
        self.cache, self.manifests = {}, {}
        for year in self.years:
            manifest = json.loads((self.root / str(year) / "manifest.json").read_text())
            if manifest["year"] != year or any(int(d[:4]) > year for d in manifest["date_window"]):
                raise ValueError("Spectral features include acquisitions after the requested year")
            if manifest["features"] != self.names:
                raise ValueError("Spectral feature schema differs across years")
            if manifest["land_coverage_fraction"] < .90:
                raise ValueError(f"Year {year} has insufficient spectral coverage; resume acquisition first")
            self.manifests[str(year)] = manifest

    def read_year(self, year):
        if year not in self.years:
            if year < min(self.years):
                return np.full((len(self.names), *self.shape), np.nan, dtype=np.float32)
            raise ValueError("Unexpected spectral feature year")
        if year not in self.cache:
            with rasterio.open(self.root / str(year) / "landsat_dry_season.tif") as src:
                if any(src.profile[k] != self.profile[k] for k in ("crs", "transform", "height", "width")):
                    raise ValueError("Spectral and land-cover grids differ")
                if list(src.descriptions) != self.names:
                    raise ValueError("Spectral band order differs from manifest")
                self.cache[year] = src.read()
        return self.cache[year]

    def __call__(self, year, indices):
        current, previous = self.read_year(year), self.read_year(year - 5)
        rows, names = [], []

        def add(name, values):
            rows.append(np.asarray(values, dtype=np.float32).ravel()[indices])
            names.append(name)

        for label, cube in (("current", current), ("previous", previous)):
            for i, name in enumerate(self.names):
                missing = ~np.isfinite(cube[i])
                add(f"landsat_{label}_{name}", np.where(missing, 0, cube[i]))
                add(f"landsat_{label}_{name}_missing", missing)
        for i, name in enumerate(FEATURE_NAMES):
            known = np.isfinite(current[i]) & np.isfinite(previous[i])
            add(f"landsat_annual_trend_{name}", np.where(known, (current[i] - previous[i]) / 5, 0))
            add(f"landsat_annual_trend_{name}_available", known)
        for name in ("ndvi", "ndmi", "nbr"):
            values = current[self.names.index("mean_" + name)]
            valid = np.isfinite(values)
            for size in (3, 15):
                support = uniform_filter(valid.astype(np.float32), size, mode="constant")
                total = uniform_filter(np.where(valid, values, 0), size, mode="constant")
                mean = np.divide(total, support, out=np.zeros_like(total), where=support > 0)
                add(f"landsat_{name}_neighborhood_{size}", mean)
                add(f"landsat_{name}_neighborhood_support_{size}", support)
                add(f"landsat_{name}_local_difference_{size}", np.where(valid, values - mean, 0))
        return np.column_stack(rows), names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    provider = HistoricalSpectra(args.spectral_dir, DEFAULT_DATA / "Guinea_LandCover_2010.tif")
    report = run_memory(args.output_dir, candidates=CANDIDATES, feature_provider=provider,
                        feature_provenance=provider.manifests)
    print("Selected:", report["selected_model"])
    print("Final FoM:", report["results"][report["selected_model"]]["summary"]["figure_of_merit_pct"])


if __name__ == "__main__":
    main()
