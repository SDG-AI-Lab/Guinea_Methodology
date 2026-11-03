"""Seasonal vegetation trajectories for the unchanged five-year forecast task."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio

from .comparison import DEFAULT_DATA
from .modis_phenology import TARGET_DAYS
from .temporal_memory import run_memory


CANDIDATES = [
    dict(name="phenology_control_rf", family="rf", structure="two_stage", spectral=False,
         memory=False, pooled=False, trees=150, depth=16),
    dict(name="phenology_two_stage_rf", family="rf", structure="two_stage", spectral=True,
         memory=True, pooled=False, trees=150, depth=16),
    dict(name="phenology_action_rf", family="rf", structure="action", spectral=True,
         memory=True, pooled=False, trees=200, depth=20),
    dict(name="phenology_joint_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=False, trees=500, depth=4),
    dict(name="phenology_two_stage_xgb", family="xgb", structure="two_stage", spectral=True,
         memory=True, pooled=False, trees=500, depth=4),
    dict(name="phenology_action_xgb", family="xgb", structure="action", spectral=True,
         memory=True, pooled=False, trees=500, depth=4),
    dict(name="phenology_deep_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=False, trees=600, depth=6),
    dict(name="pooled_phenology_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=True, trees=500, depth=4),
    dict(name="expanded_phenology_rf", family="rf", structure="two_stage", spectral=True,
         memory=True, pooled=False, trees=150, depth=16, expanded=True),
    dict(name="expanded_phenology_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=False, trees=500, depth=4, expanded=True),
]


def trajectory_statistics(values):
    """Per-pixel statistics over 12 dated periods; no interpolation from future years."""
    valid = np.isfinite(values)
    clean = np.where(valid, values, 0)
    count = valid.sum(axis=0)
    mean = clean.sum(axis=0) / np.maximum(count, 1)
    variance = (clean ** 2).sum(axis=0) / np.maximum(count, 1) - mean ** 2
    minimum = np.where(count > 0, np.min(np.where(valid, values, np.inf), axis=0), 0)
    maximum = np.where(count > 0, np.max(np.where(valid, values, -np.inf), axis=0), 0)
    out = dict(mean=mean, std=np.sqrt(np.maximum(variance, 0)), minimum=minimum,
               maximum=maximum, observed_amplitude=maximum-minimum, valid_periods=count)
    for label, periods in (("wet_season", np.arange(4, 10)), ("dry_season", np.array([0, 1, 2, 3, 10, 11]))):
        total = valid[periods].sum(axis=0)
        out[label] = clean[periods].sum(axis=0) / np.maximum(total, 1)
    phase = 2 * np.pi * np.asarray(TARGET_DAYS) / 365.25
    design = np.column_stack([np.ones(12), np.cos(phase), np.sin(phase)]).astype(np.float32)
    # Small ridge term keeps coefficients finite when only a few periods exist.
    gram = np.einsum('ti,tj,tn->nij', design, design, valid.astype(np.float32))
    gram += np.diag([.001, .1, .1]).astype(np.float32)
    rhs = np.einsum('ti,tn->ni', design, clean)
    coefficients = np.linalg.solve(gram, rhs[..., None])[..., 0]
    coefficients[count < 3] = 0
    out['harmonic_cos'] = coefficients[:, 1]
    out['harmonic_sin'] = coefficients[:, 2]
    out['harmonic_amplitude'] = np.hypot(coefficients[:, 1], coefficients[:, 2]) * 2
    return out


class HistoricalPhenology:
    def __init__(self, root, reference_path, years=(2005, 2010, 2015)):
        self.root, self.years = Path(root), tuple(years)
        with rasterio.open(reference_path) as src:
            self.profile, self.shape = src.profile, src.shape
        self.names = [f"{name}_period_{p+1:02d}" for p in range(12) for name in ('ndvi', 'evi')]
        self.cache, self.manifests = {}, {}
        for year in years:
            directory = self.root / str(year)
            manifest = json.loads((directory / 'manifest.json').read_text())
            catalog = json.loads((directory / 'catalog.json').read_text())
            if manifest['year'] != year or manifest['features'] != self.names:
                raise ValueError('Phenology year/schema mismatch')
            for entry in catalog['selected']:
                properties = entry['item']['properties']
                if properties['start_datetime'][:4] != str(year) or properties['end_datetime'][:4] != str(year):
                    raise ValueError('Phenology composite crosses the permitted year boundary')
            if manifest['coverage_any'] < .95 or manifest['coverage_six_periods'] < .80:
                raise ValueError(f'Insufficient phenology coverage for {year}; resume acquisition')
            self.manifests[str(year)] = manifest

    def read_rows(self, year, indices):
        if year < min(self.years):
            return np.full((24, len(indices)), np.nan, dtype=np.float32)
        if year not in self.years:
            raise ValueError('Unexpected phenology year')
        if year not in self.cache:
            with rasterio.open(self.root / str(year) / 'seasonal_vegetation.tif') as src:
                if any(src.profile[k] != self.profile[k] for k in ('crs', 'transform', 'height', 'width')):
                    raise ValueError('Phenology and land-cover grids differ')
                if list(src.descriptions) != self.names:
                    raise ValueError('Phenology band order differs from manifest')
                self.cache[year] = src.read().reshape(24, -1)
        return self.cache[year][:, indices]

    def __call__(self, year, indices):
        current, previous = self.read_rows(year, indices), self.read_rows(year-5, indices)
        rows, names = [], []
        for label, cube in (('current', current), ('previous', previous)):
            for i, name in enumerate(self.names):
                valid = np.isfinite(cube[i])
                rows.extend([np.where(valid, cube[i], 0), (~valid).astype(np.float32)])
                names.extend([f'modis_{label}_{name}', f'modis_{label}_{name}_missing'])
            for i, variable in enumerate(('ndvi', 'evi')):
                statistics = trajectory_statistics(cube[i::2])
                for key, values in statistics.items():
                    rows.append(values)
                    names.append(f'modis_{label}_{variable}_{key}')
        for i, name in enumerate(self.names):
            valid = np.isfinite(current[i]) & np.isfinite(previous[i])
            rows.extend([np.where(valid, (current[i]-previous[i])/5, 0), valid.astype(np.float32)])
            names.extend([f'modis_annual_trend_{name}', f'modis_annual_trend_{name}_available'])
        return np.column_stack(rows).astype(np.float32), names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phenology-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    provider = HistoricalPhenology(args.phenology_dir, DEFAULT_DATA/'Guinea_LandCover_2010.tif')
    report = run_memory(args.output_dir, candidates=CANDIDATES, feature_provider=provider,
                        feature_provenance=provider.manifests)
    print('Selected:', report['selected_model'])
    print('Final FoM:', report['results'][report['selected_model']]['summary']['figure_of_merit_pct'])


if __name__ == '__main__':
    main()
