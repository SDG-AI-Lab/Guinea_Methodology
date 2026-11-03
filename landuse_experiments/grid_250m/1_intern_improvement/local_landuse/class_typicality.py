"""How poorly a pixel's measured vegetation matches the class it was labelled.

Most false alarms come from pixels whose label is stable and whose label is not.
These features describe, for each starting year, how atypical the pixel's own
seasonal vegetation signature is for its own assigned class, and which other
class its signature resembles more. Class descriptions are estimated only from
pixels whose label was already unchanged over the previous five years, so no
observation and no label after the starting year is ever read.
"""

import argparse
import logging

import numpy as np
from scipy.ndimage import uniform_filter

from .comparison import DEFAULT_DATA, read_raster
from .phenology_models import HistoricalPhenology
from .spectral_models import HistoricalSpectra
from .temporal_memory import run_memory
from .vegetation_context import VegetationContext

GRID_KEYS = tuple(f"{v}_{s}" for v in ("ndvi", "evi") for s in ("annual", "wet", "dry"))
FLOOR = 1e-4


def class_descriptions(features, valid, labels, classes, minimum=200):
    """Diagonal Gaussian mean/variance and prior for each class, on stable pixels."""
    means, variances, priors = {}, {}, {}
    for c in classes:
        rows = features[valid & (labels == c)]
        if len(rows) < minimum:
            continue
        means[c] = rows.mean(axis=0)
        variances[c] = np.maximum(rows.var(axis=0), FLOOR)
        priors[c] = float(len(rows))
    total = sum(priors.values())
    return means, variances, {c: v / total for c, v in priors.items()}


def log_likelihoods(features, means, variances, priors, classes):
    out = np.full((len(features), len(classes)), -np.inf, dtype=np.float32)
    for j, c in enumerate(classes):
        if c not in means:
            continue
        delta = features - means[c]
        out[:, j] = (-.5 * np.sum(delta ** 2 / variances[c] + np.log(2 * np.pi * variances[c]), axis=1)
                     + np.log(priors[c]))
    return out


class ClassTypicality:
    """Wrap a feature provider and append per-year label-agreement features."""

    def __init__(self, base, data_dir=DEFAULT_DATA, window=5, minimum=200):
        self.base, self.data_dir, self.window, self.minimum = base, data_dir, window, minimum
        self.grids, self.cache = {}, {}
        self.manifests = dict(base=getattr(base, "manifests", None), typicality_window=window, minimum_reference_pixels=minimum,
            typicality_inputs=list(GRID_KEYS), reference_pixels="label unchanged over the previous five years",
            future_labels_used=False, future_observations_used=False)

    def _stack(self, year):
        """Standardised seasonal vegetation grid stack for one starting year."""
        if year in self.grids:
            return self.grids[year]
        grids = self.base.annual_grids(year)
        if not grids:
            self.grids[year] = (None, None)
            return self.grids[year]
        values = np.stack([grids[key][0].ravel() for key in GRID_KEYS], axis=1).astype(np.float32)
        valid = np.all(np.stack([grids[key][1].ravel() for key in GRID_KEYS], axis=1), axis=1)
        centre = values[valid].mean(axis=0)
        scale = np.maximum(values[valid].std(axis=0), FLOOR)
        self.grids[year] = ((values - centre) / scale, valid)
        return self.grids[year]

    def _year_features(self, year):
        """Full-grid typicality features for one starting year."""
        if year in self.cache:
            return self.cache[year]
        features, valid = self._stack(year)
        labels, _ = read_raster(self.data_dir / f"Guinea_LandCover_{year}.tif", categorical=True)
        labels = labels.ravel()
        classes = np.array(sorted(c for c in np.unique(labels) if c > 0))
        stable = valid & (labels > 0)
        earlier = self.data_dir / f"Guinea_LandCover_{year-5}.tif"
        if earlier.exists():
            previous, _ = read_raster(earlier, categorical=True)
            stable = stable & (previous.ravel() == labels)
        means, variances, priors = class_descriptions(features, stable, labels, classes, self.minimum)
        if not means:
            raise ValueError(f"No class had enough stable reference pixels in {year}")
        scores = log_likelihoods(features, means, variances, priors, classes)
        own_column = np.searchsorted(classes, np.clip(labels, classes[0], classes[-1]))
        rows = np.arange(len(labels))
        known = valid & np.isin(labels, classes) & np.isfinite(scores[rows, own_column])
        own = np.where(known, scores[rows, own_column], 0).astype(np.float32)
        other = scores.copy()
        other[rows, own_column] = -np.inf
        best = other.argmax(axis=1)
        best_score = np.where(known, np.where(np.isfinite(other[rows, best]), other[rows, best], 0), 0)
        shifted = scores - scores.max(axis=1, keepdims=True)
        posterior = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        out = dict(
            own_loglik=own,
            best_other_loglik=best_score.astype(np.float32),
            margin=(own - best_score).astype(np.float32),
            own_posterior=np.where(known, posterior[rows, own_column], 0).astype(np.float32),
            best_other_posterior=np.where(known, posterior[rows, best], 0).astype(np.float32),
            best_other_class=np.where(known, classes[best], 0).astype(np.float32),
            observed=known.astype(np.float32))
        agreement = np.where(known, (classes[best] == labels) | (posterior[rows, own_column] >= .5), 0)
        out["label_agreement"] = agreement.astype(np.float32)
        shape = self.base.phenology.shape
        support = uniform_filter(out["observed"].reshape(shape), self.window, mode="constant")
        for key in ("own_posterior", "margin"):
            grid = out[key].reshape(shape)
            mean = np.divide(uniform_filter(grid, self.window, mode="constant"), support,
                out=np.zeros_like(grid), where=support > 1e-8)
            out[f"neighbourhood_{key}"] = mean.ravel().astype(np.float32)
            out[f"neighbourhood_{key}_difference"] = (out[key] - mean.ravel()).astype(np.float32)
        self.cache[year] = out
        return out

    def __call__(self, year, indices):
        matrix, names = self.base(year, indices)
        columns, extra = [], []
        current = self._year_features(year)
        for key, values in current.items():
            columns.append(f"typicality_current_{key}")
            extra.append(values[indices])
        previous = (self._year_features(year - 5)
                    if year - 5 in getattr(self.base.phenology, "years", ()) else None)
        for key, values in current.items():
            columns.append(f"typicality_change_{key}")
            if previous is None or key in ("best_other_class", "observed"):
                extra.append(np.zeros(len(indices), dtype=np.float32))
            else:
                both = current["observed"][indices] * previous["observed"][indices]
                extra.append(((values[indices] - previous[key][indices]) * both).astype(np.float32))
        return np.column_stack([matrix, np.column_stack(extra)]).astype(np.float32), names + columns


CANDIDATES = [
    dict(name="typicality_control_rf", family="rf", structure="two_stage", spectral=False,
         memory=False, pooled=False, trees=150, depth=16),
    dict(name="typicality_two_stage_rf", family="rf", structure="two_stage", spectral=True,
         memory=True, pooled=False, trees=150, depth=16),
    dict(name="typicality_joint_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=False, trees=500, depth=4),
    dict(name="typicality_deep_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=False, trees=600, depth=6),
    dict(name="typicality_two_stage_xgb", family="xgb", structure="two_stage", spectral=True,
         memory=True, pooled=False, trees=500, depth=4),
    dict(name="typicality_action_xgb", family="xgb", structure="action", spectral=True,
         memory=True, pooled=False, trees=500, depth=4),
    dict(name="expanded_typicality_xgb", family="xgb", structure="joint", spectral=True,
         memory=True, pooled=False, trees=500, depth=4, expanded=True),
    dict(name="expanded_typicality_rf", family="rf", structure="two_stage", spectral=True,
         memory=True, pooled=False, trees=150, depth=16, expanded=True),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenology-dir", required=True)
    parser.add_argument("--spectral-dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    reference = DEFAULT_DATA / "Guinea_LandCover_2010.tif"
    base = VegetationContext(HistoricalPhenology(args.phenology_dir, reference),
        HistoricalSpectra(args.spectral_dir, reference) if args.spectral_dir else None)
    provider = ClassTypicality(base)
    report = run_memory(args.output_dir, candidates=CANDIDATES, feature_provider=provider,
                        feature_provenance=provider.manifests)
    print("Selected:", report["selected_model"])
    print("Final FoM:", report["results"][report["selected_model"]]["summary"]["figure_of_merit_pct"])


if __name__ == "__main__":
    main()
