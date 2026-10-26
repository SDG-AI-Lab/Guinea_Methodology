"""Run a reproducible temporal comparison using only local GeoTIFFs."""

import csv
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import rasterio
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .models import (NODATA, Settings, allocate_transitions, feature_stack,
                     predict_multiclass, train_multiclass, train_transitions,
                     sample_training_indices, sample_digest, train_shared_transitions)

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data"   # landuse_experiments/data (was parents[1]; see PROVENANCE.md)


def read_raster(path, reference=None, categorical=False, exclude_codes=(0, 255)):
    """Require identical grids; never silently treat different locations as equal."""
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        if src.crs is None:
            raise ValueError(f"Missing coordinate system: {path}")
        if reference is not None and any(profile[k] != reference[k]
                                         for k in ("crs", "transform", "height", "width")):
            raise ValueError(f"Grid mismatch: {path}. Align it to the first land-cover map before running.")
        values = src.read(out_dtype="float32", masked=True).filled(np.nan)
        if categorical:
            if src.count != 1:
                raise ValueError(f"Land cover must have exactly one band: {path}")
            values = values[0]
            valid = np.isfinite(values) & ~np.isin(values, exclude_codes)
            if np.any(values[valid] != np.floor(values[valid])):
                raise ValueError(f"Noninteger land-cover labels: {path}")
            result = np.full(values.shape, NODATA, dtype=np.int32)
            result[valid] = values[valid].astype(np.int32)
            return result, profile
        names = [f"{Path(path).stem}_band{i + 1}_{name or 'value'}"
                 for i, name in enumerate(src.descriptions)]
        return dict(zip(names, values)), profile


def evaluate(start, observed, predicted, mask):
    """All models use the same mask; include destination-sensitive change FoM."""
    start, observed, predicted = start[mask], observed[mask], predicted[mask]
    if not len(start):
        raise ValueError("No common valid evaluation pixels")
    labels = np.unique(np.concatenate([start, observed, predicted]))
    cm = confusion_matrix(observed, predicted, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        observed, predicted, labels=labels, zero_division=0)
    actual_change, forecast_change = observed != start, predicted != start
    hits = int(np.sum(actual_change & forecast_change & (predicted == observed)))
    wrong = int(np.sum(actual_change & forecast_change & (predicted != observed)))
    misses = int(np.sum(actual_change & ~forecast_change))
    false_alarms = int(np.sum(~actual_change & forecast_change))
    union = hits + wrong + misses + false_alarms
    quantity = float(np.abs(cm.sum(axis=0) - cm.sum(axis=1)).sum() / (2 * len(start)))
    accuracy = float(np.mean(observed == predicted))
    # Macro F1 uses classes present in either observed or predicted destinations.
    destination_labels = np.unique(np.concatenate([observed, predicted]))
    _, _, macro, _ = precision_recall_fscore_support(
        observed, predicted, labels=destination_labels, average="macro", zero_division=0)
    summary = {"pixels": int(len(start)), "overall_accuracy": accuracy,
               "macro_f1": float(macro), "figure_of_merit_pct": 100 * hits / union if union else 0.,
               "observed_change_pixels": int(actual_change.sum()),
               "predicted_change_pixels": int(forecast_change.sum()),
               "correct_change_pixels": hits, "missed_change_pixels": misses,
               "false_change_pixels": false_alarms, "wrong_destination_pixels": wrong,
               "quantity_disagreement": quantity,
               "allocation_disagreement": max(0., 1 - accuracy - quantity)}
    per_class = [{"class_code": int(c), "precision": float(p), "recall": float(r),
                  "f1": float(f), "observed_pixels": int(n),
                  "predicted_pixels": int(np.sum(predicted == c))}
                 for c, p, r, f, n in zip(labels, precision, recall, f1, support)]
    return {"summary": summary, "per_class": per_class,
            "confusion_matrix": cm.tolist(), "class_codes": labels.tolist()}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_map(path, values, profile, nodata=NODATA):
    profile = profile.copy()
    profile.update(count=1, dtype=str(values.dtype), nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)


def plot_comparison(start, observed, predictions, mask, reports, path,
                    title="Local temporal test — where predictions succeed and fail"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    # Pixel categories show actual changes, missed changes and false alarms.
    cmap = ListedColormap(["#e5e7eb", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"])
    cmap.set_bad("white")
    names = ["Correct persistence", "Correct change", "Missed change", "False change", "Wrong destination"]
    fig, axes = plt.subplots(1, len(predictions), figsize=(15, 5.8), constrained_layout=True)
    actual = start != observed
    stride = max(1, int(np.ceil(max(start.shape) / 1000)))
    for ax, (name, predicted) in zip(axes, predictions.items()):
        change = predicted != start
        categories = np.zeros(start.shape, dtype=np.uint8)
        categories[actual & change & (predicted == observed)] = 1
        categories[actual & ~change] = 2
        categories[~actual & change] = 3
        categories[actual & change & (predicted != observed)] = 4
        display = np.ma.array(categories, mask=~mask)[::stride, ::stride]
        ax.imshow(display, cmap=cmap, norm=BoundaryNorm(np.arange(-.5, 5), 5), interpolation="nearest")
        metrics = reports[name]["summary"]
        ax.set_title(f"{name.replace('_', ' ').title()}\nAccuracy {metrics['overall_accuracy']:.1%} | "
                     f"Change FoM {metrics['figure_of_merit_pct']:.1f}%", fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(cmap.colors, names)],
               loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_comparison(data_dir=DEFAULT_DATA, output_dir=None, years=(2010, 2015, 2020),
                   settings=None, max_samples=100000, min_change_pixels=200,
                   neighborhood=3, exclude_codes=(0, 255), static_predictors=None,
                   make_plot=True, sampling="shared"):
    """Train t0->t1, forecast t1->t2, then open t2 for evaluation only.

    Equal intervals are required because the multiclass target has a fixed
    forecast horizon. Default inputs: local terrain and lagged land cover.
    Supplying static_predictors asserts that they are valid at BOTH start dates.
    This is temporal validation over the same region, not spatial holdout.
    """
    settings = settings or Settings()
    if sampling not in ("shared", "legacy_balanced"):
        raise ValueError("sampling must be shared or legacy_balanced")
    if len(years) != 3 or not years[0] < years[1] < years[2]:
        raise ValueError("Supply three strictly increasing years")
    if years[1] - years[0] != years[2] - years[1]:
        raise ValueError("Multiclass comparison requires equal training and forecast intervals")
    data_dir = Path(data_dir).resolve()
    output = Path(output_dir or data_dir.parent / "outputs" / "local_comparison").resolve()
    # Avoid mixing artifacts from different runs or overwriting a reviewed result.
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty; choose a new run directory: {output}")
    landcover_paths = [data_dir / f"Guinea_LandCover_{year}.tif" for year in years]
    covariate_paths = ([data_dir / "Guinea_SRTM_DEM_Slope_Aspect.tif"]
                       if static_predictors is None else [Path(p).resolve() for p in static_predictors])
    for path in landcover_paths + covariate_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    start, profile = read_raster(landcover_paths[0], categorical=True, exclude_codes=exclude_codes)
    middle, _ = read_raster(landcover_paths[1], profile, categorical=True, exclude_codes=exclude_codes)
    if start.size > 5_000_000:
        raise ValueError("This in-memory local runner supports up to 5 million grid cells; crop/resample first")
    covariates = {}
    for path in covariate_paths:
        layers, _ = read_raster(path, profile)
        if covariates.keys() & layers.keys():
            raise ValueError("Duplicate predictor names")
        covariates.update(layers)
    known = np.concatenate([start[start != NODATA], middle[middle != NODATA]])
    classes = np.unique(known)
    if not len(classes):
        raise ValueError("Calibration maps contain no valid classes")
    LOGGER.info("Building features from starting-year maps and static covariates")
    train_features, names = feature_stack(start, covariates, classes, neighborhood)
    forecast_features, _ = feature_stack(middle, covariates, classes, neighborhood)
    output.mkdir(parents=True, exist_ok=True)
    shared_indices = (sample_training_indices(start, middle, train_features, max_samples,
                                              settings.random_state)
                      if sampling == "shared" else None)
    if shared_indices is not None:
        # Persist actual pixel identities for an independently auditable comparison.
        np.save(output / "shared_training_indices.npy", shared_indices)
    LOGGER.info("Training multiclass RF on %s -> %s", years[0], years[1])
    began = perf_counter()
    multiclass, training_info = train_multiclass(start, middle, train_features,
                                               settings=settings, max_samples=max_samples,
                                               training_indices=shared_indices)
    multiclass_map, confidence = predict_multiclass(multiclass, middle, forecast_features)
    multiclass_seconds = perf_counter() - began
    LOGGER.info("Training transition models (%s sampling) and allocating demand", sampling)
    began = perf_counter()
    if sampling == "shared":
        transitions, demands, transition_info = train_shared_transitions(
            start, middle, train_features, shared_indices, settings=settings,
            min_change_pixels=min_change_pixels)
    else:
        transitions, demands, transition_info = train_transitions(
            start, middle, train_features, settings=settings, min_change_pixels=min_change_pixels)
    allocated, allocation_info = allocate_transitions(
        transitions, demands, middle, forecast_features,
        calibration_years=years[1] - years[0], forecast_years=years[2] - years[1])
    transition_seconds = perf_counter() - began
    # Evaluation labels are not loaded until both predictions have been made.
    observed, _ = read_raster(landcover_paths[2], profile, categorical=True, exclude_codes=exclude_codes)
    predictions = {"no_change": middle.copy(), "multiclass_rf": multiclass_map,
                   "transition_rf_allocation": allocated}
    mask = (middle != NODATA) & (observed != NODATA)
    for predicted in predictions.values():
        mask &= predicted != NODATA
    reports = {name: evaluate(middle, observed, pred, mask) for name, pred in predictions.items()}
    # Identical class order in every confusion matrix, including unseen test classes.
    eval_classes = np.unique(np.concatenate([observed[mask], middle[mask]] +
                                            [pred[mask] for pred in predictions.values()]))
    for name, pred in predictions.items():
        reports[name]["confusion_matrix"] = confusion_matrix(
            observed[mask], pred[mask], labels=eval_classes).tolist()
        reports[name]["class_codes"] = eval_classes.tolist()
    rows = [{"model": name, **result["summary"]} for name, result in reports.items()]
    write_csv(output / "comparison.csv", rows)
    write_csv(output / "per_class.csv", [{"model": name, **row} for name, result in reports.items()
                                        for row in result["per_class"]])
    write_csv(output / "transition_training.csv", transition_info)
    write_csv(output / "allocation.csv", allocation_info)
    for name, pred in predictions.items():
        write_map(output / f"{name}.tif", pred.astype(np.int32), profile)
    write_map(output / "evaluation_mask.tif", mask.astype(np.uint8), profile, nodata=0)
    write_map(output / "multiclass_confidence.tif", confidence, profile, nodata=np.nan)
    common = {"feature_names": names, "landcover_classes": classes.tolist(),
              "neighborhood": neighborhood, "forecast_interval_years": years[1] - years[0],
              "settings": asdict(settings), "exclude_codes": list(exclude_codes),
              "sampling": sampling,
              "shared_sample_sha256": sample_digest(shared_indices) if shared_indices is not None else None}
    joblib.dump({"model": multiclass, **common}, output / "multiclass_model.joblib")
    joblib.dump({"models": transitions, "demands": demands, **common}, output / "transition_models.joblib")
    report = {"years": list(years), "method": "temporal_holdout_same_region",
              "training": training_info, "trained_transition_models": len(transitions),
              "max_multiclass_samples": max_samples,
              "max_binary_samples_per_class": 50000 if sampling == "legacy_balanced" else None,
              "binary_target": "specific_transition_vs_all_other_outcomes" if sampling == "shared"
                               else "specific_transition_vs_source_persistence",
              "demand_source": "shared_sample_counts_expanded_to_eligible_footprint" if sampling == "shared"
                               else "full_valid_calibration_map_counts",
              "minimum_transition_pixels": min_change_pixels, **common,
              "runtime_seconds": {"multiclass_rf": multiclass_seconds,
                                  "transition_rf_allocation": transition_seconds},
              "grid": {"crs": str(profile["crs"]), "width": profile["width"],
                       "height": profile["height"], "transform": list(profile["transform"])},
              "evaluation_classes_absent_from_training": list(map(int, np.setdiff1d(
                  np.unique(observed[mask]), multiclass.classes_))),
              "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in landcover_paths + covariate_paths],
              "results": reports,
              "notes": ["Map-label agreement, not field-verified accuracy.",
                        "Raw class codes retained; no unverified class-name legend applied.",
                        "Default covariates are static terrain; no 2020 population or undated soil/biomass.",
                        "Both methods use terrain, one-hot starting class and starting-map neighborhood fractions.",
                        "Baseline independently implements transition RF + demand-first allocation; "
                        "explicit duration scaling, valid-pixel masking and one-conversion-per-pixel fix legacy issues.",
                        "Rare transitions below cutoff are skipped only in the binary baseline.",
                        "Multiclass samples uniformly; class weighting/calibration is not tuned on the test year.",
                        "Confidence values are uncalibrated model scores, not validated probabilities.",
                        "Temporal holdout over the same geography; no claim of spatial transfer performance."]}
    if sampling == "shared":
        report["notes"].extend([
            "Every fitted forest sees identical shared rows in identical order, without class/sample weighting.",
            "Binary negatives are ALL other outcomes, including other source classes; legacy A->A filtering is disabled.",
            "The same seed and row order also make bootstrap row draws identical across forests.",
            "Demand uses only shared sample transition counts expanded by eligible/sample size; amounts may differ from the old run.",
            "This controls training samples, not model targets, model count or allocation design."])
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    if make_plot:
        plot_comparison(middle, observed, predictions, mask, reports, output / "comparison.png")
    LOGGER.info("Saved comparison to %s", output)
    return report
