"""Multiclass next-state RF and an independent transition/allocation baseline.

The baseline reproduces the standard binary training and demand-first allocation
method, not the legacy pipeline's implementation bugs. Both methods share inputs
and RF hyperparameters. Neither uses the evaluation year's land cover to fit.
"""

from dataclasses import asdict, dataclass
import hashlib

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

NODATA = -2147483648


@dataclass(frozen=True)
class Settings:
    n_estimators: int = 100
    max_depth: int = 20
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    random_state: int = 42
    n_jobs: int = 4

    def forest(self):
        return RandomForestClassifier(**asdict(self))


def feature_stack(landcover, covariates, classes, neighborhood=3):
    """One-hot current class + covariates + local class fractions.

    Fractions are recomputed from the START map for each period, excluding
    nodata neighbors. Classes are fixed from calibration data, never test data.
    """
    if neighborhood < 1 or neighborhood % 2 == 0:
        raise ValueError("neighborhood must be a positive odd number")
    lc = np.asarray(landcover)
    layers, names = [], []
    for name, values in covariates.items():
        values = np.asarray(values, dtype=np.float32)
        if values.shape != lc.shape:
            raise ValueError(f"Predictor {name} does not match the land-cover grid")
        layers.append(values)
        names.append(name)
    valid = lc != NODATA
    denominator = uniform_filter(valid.astype(np.float32), neighborhood,
                                 mode="constant", cval=0)
    for cls in classes:
        indicator = (lc == cls).astype(np.float32)
        layers.append(indicator)
        names.append(f"current_class_{int(cls)}")
        fraction = uniform_filter(indicator, neighborhood, mode="constant", cval=0)
        fraction = np.divide(fraction, denominator, out=np.zeros_like(fraction),
                             where=denominator > 0)
        layers.append(fraction)
        names.append(f"fraction_class_{int(cls)}_{neighborhood}x{neighborhood}")
    return np.stack(layers, axis=-1), names


def valid_features(landcover, features):
    return (landcover != NODATA) & np.isfinite(features).all(axis=-1)


def sample_training_indices(start, end, features, max_samples=100000, seed=42):
    """Draw one shared, unweighted calibration sample, without replacement."""
    eligible = np.flatnonzero(valid_features(start, features) & (end != NODATA))
    if max_samples < 1 or len(eligible) < 2:
        raise ValueError("Need positive max_samples and at least two valid training pixels")
    return np.random.default_rng(seed).choice(eligible, min(max_samples, len(eligible)), replace=False)


def validate_training_indices(indices, start, end, features):
    indices = np.asarray(indices)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or len(indices) < 2:
        raise ValueError("Training indices must be a one-dimensional integer sample of at least two pixels")
    if np.any(indices < 0) or np.any(indices >= start.size) or len(np.unique(indices)) != len(indices):
        raise ValueError("Training indices must be unique and within the raster")
    valid = valid_features(start, features) & (end != NODATA)
    if not valid.ravel()[indices].all():
        raise ValueError("Training indices contain invalid pixels")
    return indices


def sample_digest(indices):
    """Hash row order as well as identities; equal hashes mean equal input rows."""
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def spatial_split(indices, shape, block_size=100, test_size=.3, seed=42):
    """Original GroupShuffleSplit rule, applied ONCE and shared by all models."""
    if block_size < 1 or not 0 < test_size < 1:
        raise ValueError("Need positive block_size and 0 < test_size < 1")
    if (shape[1] - 1) // block_size >= 1000:
        raise ValueError("Grid too wide for the original row_block * 1000 + column_block IDs")
    rows, cols = np.divmod(indices, shape[1])
    groups = (rows // block_size) * 1000 + (cols // block_size)
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two sampled spatial blocks are required")
    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(split.split(indices, groups=groups))
    return indices[train], indices[test], np.unique(groups[train]), np.unique(groups[test])


@dataclass
class DestinationScorer:
    """Adapt one multiclass RF to the SAME binary-score allocation interface.

    No fitting or thresholding: score(A->B) = P(next class=B), evaluated on
    source-A pixels by the shared allocator. Class A remains an explicit
    destination during training even though persistence has no allocation job.
    """
    model: object
    destination: int

    @property
    def classes_(self):
        return np.array([0, 1])

    def predict_proba(self, X):
        column = list(self.model.classes_).index(self.destination)
        p = self.model.predict_proba(X)[:, column]
        return np.column_stack([1 - p, p])


def train_multiclass(start, end, features, *, settings=None, max_samples=100000,
                     training_indices=None):
    """Uniform sampling preserves prevalence in expectation; no balancing.

    Every destination, including persistence, is a target class. No spatial
    coordinates are features. Temporal evaluation is performed by the caller.
    """
    settings = settings or Settings()
    mask = valid_features(start, features) & (end != NODATA)
    chosen = (sample_training_indices(start, end, features, max_samples, settings.random_state)
              if training_indices is None else validate_training_indices(training_indices, start, end, features))
    X = features.reshape(-1, features.shape[-1])[chosen]
    y = end.ravel()[chosen]
    model = settings.forest().fit(X, y)
    classes, counts = np.unique(y, return_counts=True)
    info = {"eligible_pixels": int(mask.sum()), "sampled_pixels": int(len(chosen)),
            "sample_sha256": sample_digest(chosen),
            "sampled_destination_counts": dict(zip(map(int, classes), map(int, counts))),
            "unsampled_destination_classes": list(map(int, np.setdiff1d(
                np.unique(end[mask]), classes)))}
    return model, info


def train_shared_transitions(start, end, features, training_indices, *, settings=None,
                             min_change_pixels=200):
    """Strict same-row comparison: binary one-vs-rest transition models.

    EVERY model sees the entire shared X, in the same order, with equal weights.
    For A->B, positives are A->B; negatives include persistence and all other
    transitions/source classes. No balancing, filtering or additional sampling.
    This differs from the original A->B versus A->A binary target.

    Demand is estimated from this SAME sample, expanded by eligible/sample size.
    The threshold is applied to estimated full-footprint counts; at least ten
    sampled positives and negatives are required. No unsampled transition labels
    are used to determine demands or which transition models to train.
    """
    if min_change_pixels < 1:
        raise ValueError("min_change_pixels must be positive")
    settings = settings or Settings()
    chosen = validate_training_indices(training_indices, start, end, features)
    X = features.reshape(-1, features.shape[-1])[chosen]
    sources, destinations = start.ravel()[chosen], end.ravel()[chosen]
    pairs, counts = np.unique(np.column_stack([sources, destinations]), axis=0, return_counts=True)
    expansion = int((valid_features(start, features) & (end != NODATA)).sum()) / len(chosen)
    models, demands, diagnostics = {}, {}, []
    digest = sample_digest(chosen)
    for (source, destination), count in zip(pairs, counts):
        if source == destination:
            continue
        key = (int(source), int(destination))
        demand = int(round(int(count) * expansion))
        negative = len(chosen) - int(count)
        row = {"from_class": key[0], "to_class": key[1],
               "sampled_positive_pixels": int(count), "sampled_negative_pixels": negative,
               "training_pixels": int(len(chosen)), "sample_sha256": digest,
               "estimated_calibration_demand_pixels": demand}
        if demand < min_change_pixels or min(int(count), negative) < 10:
            row["status"] = "skipped_insufficient_samples"
        else:
            y = ((sources == source) & (destinations == destination)).astype(np.uint8)
            models[key] = settings.forest().fit(X, y)
            demands[key] = demand
            row["status"] = "trained"
        diagnostics.append(row)
    return models, demands, diagnostics


def predict_multiclass(model, start, features, batch_size=50000):
    """Return argmax classes and their model scores; invalid pixels stay nodata."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    mask = valid_features(start, features)
    indices = np.flatnonzero(mask)
    result = np.full(start.shape, NODATA, dtype=np.int32)
    confidence = np.full(start.shape, np.nan, dtype=np.float32)
    flat = features.reshape(-1, features.shape[-1])
    for offset in range(0, len(indices), batch_size):
        selected = indices[offset:offset + batch_size]
        probabilities = model.predict_proba(flat[selected])
        best = probabilities.argmax(axis=1)
        result.ravel()[selected] = model.classes_[best]
        confidence.ravel()[selected] = probabilities[np.arange(len(best)), best]
    return result, confidence


def train_transitions(start, end, features, *, settings=None, max_per_class=50000,
                      min_change_pixels=200):
    """Existing method: one balanced RF per non-persistent transition.

    A->B positives compete against A->A negatives, exactly as in training.py.
    Demand is measured on the same valid calibration footprint as training.
    """
    settings = settings or Settings()
    if max_per_class < 1 or min_change_pixels < 1:
        raise ValueError("Sampling limits must be positive")
    mask = valid_features(start, features) & (end != NODATA)
    pairs, counts = np.unique(np.column_stack([start[mask], end[mask]]),
                              axis=0, return_counts=True)
    flat = features.reshape(-1, features.shape[-1])
    rng = np.random.default_rng(settings.random_state)
    models, demands, diagnostics = {}, {}, []
    for (source, destination), count in zip(pairs, counts):
        if source == destination:
            continue
        key = (int(source), int(destination))
        positive = np.flatnonzero(mask & (start == source) & (end == destination))
        negative = np.flatnonzero(mask & (start == source) & (end == source))
        n = min(len(positive), len(negative), max_per_class)
        row = {"from_class": key[0], "to_class": key[1], "observed_pixels": int(count),
               "samples_per_binary_class": int(n)}
        if count < min_change_pixels or n < 10:
            row["status"] = "skipped_insufficient_samples"
        else:
            chosen = np.concatenate([rng.choice(positive, n, replace=False),
                                     rng.choice(negative, n, replace=False)])
            y = np.concatenate([np.ones(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)])
            models[key] = settings.forest().fit(flat[chosen], y)
            demands[key] = int(count)
            row["status"] = "trained"
        diagnostics.append(row)
    return models, demands, diagnostics


def allocate_transitions(models, demands, start, features, *, calibration_years,
                         forecast_years, batch_size=50000):
    """Demand-first greedy baseline, with explicit horizon and validity handling.

    Only original source pixels may change, once each. No invalid scores are
    allocated. These fix legacy repeated-conversion/nodata issues while keeping
    its model/priority design. Integer ties use class code / row-major order.
    """
    if calibration_years <= 0 or forecast_years <= 0 or batch_size < 1:
        raise ValueError("Durations and batch_size must be positive")
    valid = valid_features(start, features)
    result = np.where(valid, start, NODATA).astype(np.int32)
    used = np.zeros(start.shape, dtype=bool)
    flat = features.reshape(-1, features.shape[-1])
    diagnostics = []
    order = sorted(demands, key=lambda key: (-demands[key], key))
    for key in order:
        source, destination = key
        if source == destination or key not in models:
            continue
        demand = max(0, int(round(demands[key] / calibration_years * forecast_years)))
        indices = np.flatnonzero(valid & ~used & (start == source))
        scores = np.empty(len(indices), dtype=np.float32)
        model = models[key]
        positive_column = list(model.classes_).index(1)
        for offset in range(0, len(indices), batch_size):
            batch = indices[offset:offset + batch_size]
            scores[offset:offset + len(batch)] = model.predict_proba(flat[batch])[:, positive_column]
        good = np.isfinite(scores)
        indices, scores = indices[good], scores[good]
        n = min(demand, len(indices))
        selected = indices[np.argsort(-scores, kind="stable")[:n]]
        result.ravel()[selected] = destination
        used.ravel()[selected] = True
        diagnostics.append({"from_class": source, "to_class": destination,
                            "requested_pixels": demand, "allocated_pixels": n,
                            "unfulfilled_pixels": demand - n})
    return result, diagnostics
