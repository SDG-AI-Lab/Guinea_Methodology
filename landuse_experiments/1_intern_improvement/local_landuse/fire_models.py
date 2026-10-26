"""Fire-history features for the unchanged five-year forecast task.

Only years at or before the forecast's starting year are read. Burn extent,
burn frequency, time since the last fire, burn timing within the season and the
same quantities over neighbourhoods describe the fire regime a pixel sits in.
Years without usable burned-area mapping keep explicit support columns and never
become unburned zeros.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from .comparison import DEFAULT_DATA
from .phenology_models import HistoricalPhenology
from .spectral_models import HistoricalSpectra
from .temporal_memory import run_memory
from .vegetation_context import VegetationContext

WINDOWS = (3, 15)
HORIZONS = (3, 5, 10)
UNBURNED_HORIZON = 20.


def masked_mean(values, valid, size):
    support = uniform_filter(valid.astype(np.float32), size, mode="constant")
    clean = uniform_filter(np.where(valid, values, 0).astype(np.float32), size, mode="constant")
    return np.divide(clean, support, out=np.zeros_like(clean), where=support > 1e-8), support


class HistoricalFire:
    """Annual MCD64A1 burned area, read only for years up to the starting year."""

    def __init__(self, root, reference_path, years=None):
        self.root = Path(root)
        with rasterio.open(reference_path) as src:
            self.profile, self.shape = src.profile, src.shape
        available = sorted(int(p.name) for p in self.root.iterdir()
                           if p.is_dir() and p.name.isdigit() and (p / "annual_fire.tif").exists())
        self.years = tuple(years) if years else tuple(available)
        missing = set(self.years) - set(available)
        if missing:
            raise ValueError(f"Missing acquired burned-area years: {sorted(missing)}")
        self.cache, self.manifests = {}, {}
        for year in self.years:
            manifest = json.loads((self.root / str(year) / "manifest.json").read_text())
            if manifest["year"] != year or manifest["features"] != ["burned_fraction", "first_burn_day", "mapped_fraction"]:
                raise ValueError("Burned-area year/schema mismatch")
            self.manifests[str(year)] = manifest

    def grids(self, year):
        if year not in self.cache:
            path = self.root / str(year) / "annual_fire.tif"
            with rasterio.open(path) as src:
                if src.shape != self.shape:
                    raise ValueError("Burned-area grid differs from the land-cover grid")
                values = src.read()
            self.cache[year] = (np.nan_to_num(values[0]), values[1], np.isfinite(values[0]))
        return self.cache[year]

    def history(self, year):
        """Burned share, burn day and mapping support for each year up to `year`."""
        years = [y for y in self.years if y <= year]
        if not years:
            raise ValueError(f"No burned-area years at or before {year}")
        burned = np.stack([self.grids(y)[0] for y in years])
        day = np.stack([self.grids(y)[1] for y in years])
        known = np.stack([self.grids(y)[2] for y in years])
        return np.array(years), burned, day, known


class FireContext:
    """Wrap a feature provider and append fire-regime columns."""

    def __init__(self, base, fire):
        self.base, self.fire = base, fire
        self.manifests = dict(base=getattr(base, "manifests", None), fire=fire.manifests,
                              horizons_years=list(HORIZONS), windows_pixels=list(WINDOWS),
                              unburned_horizon_years=UNBURNED_HORIZON,
                              future_observations_used=False)

    def columns(self, year):
        years, burned, day, known = self.fire.history(year)
        out = {}
        recent = years >= year - 4
        for label, mask in [("latest", years == year)] + [
                (f"last_{h}", years >= year - h + 1) for h in HORIZONS]:
            support = known[mask].sum(axis=0).astype(np.float32)
            share = np.where(support > 0, (burned[mask] * known[mask]).sum(axis=0) / np.maximum(support, 1), 0)
            out[f"fire_{label}_burned_share"] = share.astype(np.float32)
            out[f"fire_{label}_observed_years"] = support
            if label != "latest":
                counted = ((burned[mask] > .5) & known[mask]).sum(axis=0).astype(np.float32)
                out[f"fire_{label}_burn_count"] = counted
        burnt = (burned > .5) & known
        ages = np.where(burnt, (year - years)[:, None, None], np.inf)
        since = ages.min(axis=0).astype(np.float32)
        out["fire_years_since_last_burn"] = np.minimum(since, UNBURNED_HORIZON)
        out["fire_ever_burned"] = np.isfinite(since).astype(np.float32)
        dated = burnt & (day > 0)
        seasons = np.where(dated, day, 0)
        counted = dated.sum(axis=0)
        mean_day = seasons.sum(axis=0) / np.maximum(counted, 1)
        variance = (seasons ** 2).sum(axis=0) / np.maximum(counted, 1) - mean_day ** 2
        out["fire_mean_burn_day"] = np.where(counted > 0, mean_day, 0).astype(np.float32)
        out["fire_burn_day_spread"] = np.where(counted > 0, np.sqrt(np.maximum(variance, 0)), 0).astype(np.float32)
        out["fire_burn_day_observed"] = (counted > 0).astype(np.float32)
        latest_day = np.where(burnt[-1] & (day[-1] > 0), day[-1], 0)
        out["fire_latest_burn_day"] = latest_day.astype(np.float32)
        recent_share = np.where(known[recent].sum(axis=0) > 0,
            (burned[recent] * known[recent]).sum(axis=0) / np.maximum(known[recent].sum(axis=0), 1), 0)
        earlier = years < year - 4
        if earlier.any():
            earlier_share = np.where(known[earlier].sum(axis=0) > 0,
                (burned[earlier] * known[earlier]).sum(axis=0) / np.maximum(known[earlier].sum(axis=0), 1), 0)
            out["fire_frequency_trend"] = (recent_share - earlier_share).astype(np.float32)
        else:
            out["fire_frequency_trend"] = np.zeros(self.fire.shape, np.float32)
        observed = known.any(axis=0)
        for key in ("fire_last_10_burned_share", "fire_years_since_last_burn"):
            for size in WINDOWS:
                mean, support = masked_mean(out[key], observed, size)
                out[f"{key}_neighbourhood_{size}"] = mean.astype(np.float32)
                out[f"{key}_local_difference_{size}"] = (out[key] - mean).astype(np.float32)
                out[f"fire_observed_support_{size}"] = support.astype(np.float32)
        out["fire_observed"] = observed.astype(np.float32)
        return out

    def __call__(self, year, indices):
        matrix, names = self.base(year, indices)
        columns = self.columns(year)
        extra = [values.ravel()[indices] for values in columns.values()]
        return (np.column_stack([matrix, np.column_stack(extra)]).astype(np.float32),
                names + list(columns))


CANDIDATES = [
    dict(name="fire_control_rf", family="rf", structure="two_stage", spectral=False,
         memory=False, pooled=False, trees=150, depth=16),
    dict(name="fire_two_stage_rf", family="rf", structure="two_stage", spectral=True,
         memory=True, pooled=False, trees=150, depth=16),
    dict(name="fire_joint_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=False, trees=500, depth=4),
    dict(name="fire_deep_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=False, trees=600, depth=6),
    dict(name="fire_action_xgb", family="xgb", structure="action", spectral=True,
         memory=True, pooled=False, trees=500, depth=4),
    dict(name="expanded_fire_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=False, trees=500, depth=4, expanded=True),
    dict(name="expanded_fire_rf", family="rf", structure="two_stage", spectral=True,
         memory=True, pooled=False, trees=150, depth=16, expanded=True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenology-dir", required=True)
    parser.add_argument("--spectral-dir")
    parser.add_argument("--fire-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    reference = DEFAULT_DATA / "Guinea_LandCover_2010.tif"
    base = VegetationContext(HistoricalPhenology(args.phenology_dir, reference),
        HistoricalSpectra(args.spectral_dir, reference) if args.spectral_dir else None)
    provider = FireContext(base, HistoricalFire(args.fire_dir, reference))
    report = run_memory(args.output_dir, candidates=CANDIDATES, feature_provider=provider,
                        feature_provenance=provider.manifests)
    print("Selected:", report["selected_model"])
    print("Final FoM:", report["results"][report["selected_model"]]["summary"]["figure_of_merit_pct"])


if __name__ == "__main__":
    main()
